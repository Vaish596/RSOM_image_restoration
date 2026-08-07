import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler

from .helper.diffusion_unet import UNet


class Palette(nn.Module):
    """
    Palette / SR3 — pixel-space conditional diffusion for image restoration.

    Training:
        Add noise to HQ at random timestep t.
        Input = concat(LQ, noisy_HQ)  →  (B, 6, H, W)
        UNet predicts the noise.
        loss = ||noise_pred - noise||^2

    Inference (DDIM):
        Start from pure noise.
        Iteratively denoise conditioned on LQ.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers=(1, 2, 4, 8),
        n_res_blocks: int = 2,
        attn_resolutions=(16,),
        dropout: float = 0.0,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()

        self.num_train_timesteps = num_train_timesteps

        # 6 input channels: 3 (noisy HQ) + 3 (LQ condition)
        self.unet = UNet(
            in_channels=in_channels * 2,
            out_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            n_res_blocks=n_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
        )

        # DDPM noise scheduler (training)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            prediction_type="v_prediction",
        )

        # DDIM scheduler (inference, faster)
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            prediction_type="v_prediction",
            steps_offset=1,
        )

    def forward(self, noisy_hq, lq, timesteps):
        """
        Args:
            noisy_hq: (B, C, H, W) noisy target image at timestep t
            lq:       (B, C, H, W) low-quality condition (fixed)
            timesteps: (B,) integer timesteps in [0, num_train_timesteps)
        Returns:
            noise_pred: (B, C, H, W) predicted noise
        """
        inp = torch.cat([noisy_hq, lq], dim=1)
        return self.unet(inp, timesteps.float())

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------

    def train_palette(self, hq: torch.Tensor, lq: torch.Tensor):
        """
        Single training step.

        Args:
            hq: (B, C, H, W) clean target
            lq: (B, C, H, W) low-quality input
        Returns:
            loss: scalar tensor
        """
        B = hq.shape[0]
        device = hq.device

        # Sample random timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps, (B,), device=device
        ).long()

        # Add noise to HQ
        noise = torch.randn_like(hq)
        noisy_hq = self.noise_scheduler.add_noise(hq, noise, timesteps)

        # Predict noise
        pred = self(noisy_hq, lq, timesteps)
        target = self.noise_scheduler.get_velocity(hq, noise, timesteps)

        # loss = F.mse_loss(noise_pred, noise)
        return pred, target

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_ddim(self, lq: torch.Tensor, num_inference_steps: int = 50):
        """
        DDIM sampling conditioned on LQ.

        Args:
            lq:                   (B, C, H, W) low-quality input
            num_inference_steps:  number of DDIM steps
        Returns:
            pred_hq: (B, C, H, W) restored image
        """
        device = lq.device
        B = lq.shape[0]

        # Start from pure noise
        z = torch.randn_like(lq)

        self.inference_scheduler.set_timesteps(num_inference_steps)

        for t in self.inference_scheduler.timesteps:
            t_batch = t.expand(B).to(device)
            noise_pred = self(z, lq, t_batch)
            z = self.inference_scheduler.step(noise_pred, t, z).prev_sample

        return z

    @torch.no_grad()
    def sample_ddpm(self, lq: torch.Tensor):
        """
        Full DDPM sampling (slow, used as reference).  1000 steps.

        Args:
            lq: (B, C, H, W) low-quality input
        Returns:
            pred_hq: (B, C, H, W)
        """
        device = lq.device
        B = lq.shape[0]

        z = torch.randn_like(lq)

        for t in reversed(range(self.num_train_timesteps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self(z, lq, t_batch)
            z = self.noise_scheduler.step(noise_pred, t, z).prev_sample

        return z

    @torch.no_grad()
    def sample_with_intermediates(self, lq: torch.Tensor, num_inference_steps: int = 50):
        """
        DDIM sampling returning intermediate predictions.

        Args:
            lq:                   (B, C, H, W)
            num_inference_steps:  number of DDIM steps
        Returns:
            pred_hq: (B, C, H, W) final restoration
            intermediates: list of (B, C, H, W) at regular intervals
        """
        device = lq.device
        B = lq.shape[0]
        z = torch.randn_like(lq)
        intermediates = []
        log_interval = max(1, num_inference_steps // 10)

        self.inference_scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.inference_scheduler.timesteps):
            t_batch = t.expand(B).to(device)
            noise_pred = self(z, lq, t_batch)
            z = self.inference_scheduler.step(noise_pred, t, z).prev_sample

            if i % log_interval == 0:
                intermediates.append(z.clone())

        intermediates.append(z.clone())
        return z, intermediates
