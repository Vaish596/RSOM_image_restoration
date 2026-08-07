
import torch
import torch.nn.functional as F
import lightning as L
import lpips
from pytorch_msssim import ssim
from model.unet import UNet2D
import model.esrgan as esrgan
import losses
import wandb
from model.hat import HAT, create_hat_model
from model.palette import Palette
from model.dip import DeepImagePrior, dip_optimise

# # ---------------------------------------------------------------------------
# # Helper: unpack the RSOM dict batch if sent as a dict with condition
# # ---------------------------------------------------------------------------

class SRLightningModel(L.LightningModule):
    """
    Unified Lightning module for super-resolution.

    model_type options
    ------------------
    'UNET'      — plain supervised UNet
    'ESRGAN'    — GAN-based (generator + discriminator)
    'HAT'       — Hybrid Attention Transformer
    'LDM_VAE'   — Stage 1: fine-tune the VAE on your dataset
    'LDM_UNET'  — Stage 2: freeze VAE, train denoising UNet
    'PALETTE'   — PALETTE: Patch-based Latent Diffusion for Efficient Image Restoration
    """

    def __init__(
        self,
        lr: float                         = 1e-4,
        # pretrained: bool                  = True,
        model_type: str                   = 'UNET',
        perceptual_weight: float          = 0.01,
        adv_weight: float                 = 0.001,
        in_channels: int                  = 3,
        out_channels: int                 = 3,
        patch_size: int | None           = 128, 
        # ---- Loss options ------------------------------------------------- #
        loss_type: str                    = 'l1_ssim',
        l1_weight: float                  = 0.8,
        ssim_weight: float                = 0.2,
        # ---- Slice dataset options ---------------------------------------- #
        use_slices: bool                  = False,
        use_log_scale: bool               = False,
        log_scale_factor: float           = 100.0,
        # ---- Palette params ------------------------------------------------ #
        palette_base_channels: int = 64,
        palette_channel_multipliers: tuple = (1, 2, 4, 8),
        palette_n_res_blocks: int = 2,
        palette_num_train_timesteps: int = 1000,
        palette_num_inference_steps: int = 50,
        # ---- DIP params (from the paper) ----------------------------------- #
        input_depth: int = 32,
        n_scales: int = 5,
        need_sigmoid: bool = True,
        dip_num_iterations: int = 2400,
        dip_tv_weight: float = 0.0,
        dip_input_noise: bool = True,
        dip_reg_noise_std: float = 0.033,
        dip_ema_weight: float = 0.99,
        dip_backtrack_thresh: float = 1.05,
        dip_num_runs: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_type            = model_type
        self.lr                    = float(lr)
        self.loss_type = loss_type
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.patch_size = patch_size
        self.use_slices = use_slices
        self.log_scale_C = log_scale_factor if use_log_scale and log_scale_factor else 0.0

        # ------------------------------------------------------------------ #
        #  UNET
        # ------------------------------------------------------------------ #
        if model_type == 'UNET':
            self.unet         = UNet2D(in_channels=in_channels, 
                                       out_channels=out_channels)
            
        # ------------------------------------------------------------------ #
        #  ESRGAN
        # ------------------------------------------------------------------ #
        elif model_type == 'ESRGAN':
            self.automatic_optimization = False
            self.gen  = esrgan.RRDBNet(in_nc=in_channels, out_nc=out_channels)
            self.disc = esrgan.UNetDiscriminatorSN(num_in_ch=in_channels)
            self.perceptual_criterion = losses.VGGFeatureLoss()
            self.perceptual_weight    = perceptual_weight
            self.adv_weight           = adv_weight
            self.gan_criterion        = losses.gan_loss

        # ------------------------------------------------------------------ #
        #  HAT (Hybrid Attention Transformer)
        # ------------------------------------------------------------------ #
        elif model_type == 'HAT':
            self.hat = create_hat_model(
                in_channels=in_channels,
                out_channels=out_channels)
            self.perceptual_criterion = losses.VGGFeatureLoss()
            self.perceptual_weight    = perceptual_weight

        # ------------------------------------------------------------------ #
        #  PALETTE — pixel-space diffusion
        # ------------------------------------------------------------------ #
        elif model_type == 'PALETTE':
            self.palette = Palette(
                in_channels=in_channels,
                base_channels=palette_base_channels,
                channel_multipliers=palette_channel_multipliers,
                n_res_blocks=palette_n_res_blocks,
                num_train_timesteps=palette_num_train_timesteps,
            )
            self.palette_num_inference_steps = palette_num_inference_steps

        # ------------------------------------------------------------------ #
        #  DIP — Deep Image Prior (zero-shot per-image optimisation)
        # ------------------------------------------------------------------ #
        elif model_type == 'DIP':
            self.dip = DeepImagePrior(
                out_channels=out_channels,
                input_depth=input_depth,
                n_scales=n_scales,
                need_sigmoid=need_sigmoid,
            )
            self.dip_input_depth = input_depth
            self._dip_kwargs = dict(
                input_depth=input_depth,
                lr=float(lr),
                num_iterations=dip_num_iterations,
                input_noise=dip_input_noise,
                reg_noise_std=dip_reg_noise_std,
                ema_weight=dip_ema_weight,
                tv_weight=dip_tv_weight,
                backtrack_thresh=dip_backtrack_thresh,
                num_runs=dip_num_runs,
            )

        else:
            raise ValueError(f"Unknown model_type='{model_type}'")
        

        self.sr_criterion = losses.SRLoss(loss_type, l1_weight, ssim_weight)
        self.lpips_fn = lpips.LPIPS(net='alex')
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)

    
    def forward(self, x):
        if self.model_type == 'UNET':
            return self.unet(x)
        elif self.model_type == 'ESRGAN':
            return self.gen(x)
        elif self.model_type == 'HAT':
            return self.hat(x)
        elif self.model_type == 'PALETTE':
            return self.palette.sample_ddim(x, num_inference_steps=self.palette_num_inference_steps)
        elif self.model_type == 'DIP':
            return dip_optimise(self.dip, x, **self._dip_kwargs)
        else:
            raise ValueError(f"Unknown model_type='{self.model_type}'")

    # ---------------------------------------------------------------------- #
    #  Metrics
    # ---------------------------------------------------------------------- #
    def _psnr(self, pred, target):
        mse = torch.mean((pred - target) ** 2).clamp(min=1e-10)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    def _ssim(self, pred, target):
        return ssim(pred, target, data_range=1.0, size_average=True)
    
    def _lpips(self, pred, target):
        # Expand single channel to 3-channel RGB
        if pred.shape[1] == 1:
            pred   = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # Rescale [0, 1] → [-1, 1]
        pred   = pred   * 2.0 - 1.0
        target = target * 2.0 - 1.0
        return self.lpips_fn(pred, target).mean()

    # ---------------------------------------------------------------------- #
    #  Slice prediction helper (shared across all model types)
    # ---------------------------------------------------------------------- #
    def _predict_slices(self, batch):
        """
        Predict every slice in a volume and return MIPs.

        Returns:
            x:      (1, 3, H, W) — LQ MIP (batch dim for shared metrics + logging)
            y_hat:  (1, 3, H, W) — predicted MIP (batch dim)
            y:      (1, 3, H, W) — GT MIP from HQ_MIP.npy (batch dim)
        """
        lq_vol, hq_vol, hq_mip_gt = batch
        lq_vol = lq_vol.squeeze(0).float().to(self.device)
        hq_vol = hq_vol.squeeze(0).float().to(self.device)
        hq_mip_gt = hq_mip_gt.squeeze(0).float().to(self.device)
        Y = lq_vol.shape[0]

        pred_frames = []
        for s in range(Y):
            pred_s = self(lq_vol[s].unsqueeze(0)).squeeze(0)
            pred_frames.append(pred_s)
        pred_vol = torch.stack(pred_frames)

        # Undo log scale for all volumes
        if self.log_scale_C:
            pred_vol = torch.expm1(pred_vol) / self.log_scale_C
            hq_vol   = torch.expm1(hq_vol) / self.log_scale_C
            lq_vol   = torch.expm1(lq_vol) / self.log_scale_C
            pred_vol = pred_vol.clamp(0, 1)

        x     = lq_vol.max(dim=0)[0].unsqueeze(0)       # (1, 3, H, W) — LQ MIP
        y_hat = pred_vol.max(dim=0)[0].unsqueeze(0)     # (1, 3, H, W) — predicted MIP
        y     = hq_mip_gt.unsqueeze(0)                   # (1, 3, H, W) — GT MIP from dataset

        return x, y_hat, y

    # ---------------------------------------------------------------------- #
    #  Shared image logging helper
    # ---------------------------------------------------------------------- #
    def _log_images(self, tag, x, y, y_hat, step, model_type=None):

        def _prep(t):
            return (t.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)

        if model_type == 'LDM_VAE':
            self.logger.experiment.log({
                tag: [
                    wandb.Image(_prep(y[0]),     caption="HR Ground Truth"),
                    wandb.Image(_prep(y_hat[0]), caption="VAE Reconstruction"),
                ],
                "trainer/global_step": step
            })
        else:
            self.logger.experiment.log({
                tag: [
                    wandb.Image(_prep(x[0]),     caption="LR Input"),
                    wandb.Image(_prep(y[0]),     caption="HR Ground Truth"),
                    wandb.Image(_prep(y_hat[0]), caption="SR Prediction"),
                ],
                "trainer/global_step": step
            })
        if not torch.isnan(y_hat).any():
            self.logger.experiment.log({
                "histograms/gt_pixels":   wandb.Histogram(y.detach().cpu().float().numpy()),
                "histograms/pred_pixels": wandb.Histogram(y_hat.detach().cpu().float().numpy()),
                "trainer/global_step":    step
            })

    # ---------------------------------------------------------------------- #
    #  Training step
    # ---------------------------------------------------------------------- #
    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.float().to(self.device), y.float().to(self.device)

        # ---- UNET --------------------------------------------------------- #
        if self.model_type == 'UNET':
            y_hat = self(x)
            loss  = self.sr_criterion(y_hat, y)
            self.log("train/train_loss", loss, on_epoch=True, prog_bar=True)

        # ---- ESRGAN ------------------------------------------------------- #
        elif self.model_type == 'ESRGAN':
            opt_g, opt_d = self.optimizers()

            y_hat           = self(x)
            pixel_loss      = self.sr_criterion(y_hat, y)
            perceptual_loss = self.perceptual_weight * self.perceptual_criterion(y_hat, y)
            adv_loss        = self.adv_weight * self.gan_criterion(self.disc(y_hat), True) \
                              if self.disc else 0
            loss = pixel_loss + perceptual_loss + adv_loss
            opt_g.zero_grad()
            self.manual_backward(loss)
            opt_g.step()
            self.log("train/train_loss", loss, on_epoch=True, prog_bar=True)
            self.log("train/pred_mean", y_hat.mean())
            self.log("train/pred_std", y_hat.std())

            if self.disc and batch_idx % 5 == 0:
                y_hat     = self(x).detach()
                pred_real = self.disc(y)
                pred_fake = self.disc(y_hat)
                d_loss    = 0.5 * (self.gan_criterion(pred_real, True) +
                                   self.gan_criterion(pred_fake, False))
                opt_d.zero_grad()
                self.manual_backward(d_loss)
                opt_d.step()
                self.log("train/D_train_loss", d_loss, prog_bar=True)
                self.log("train/D_real_mean", pred_real.mean(),prog_bar=True)
                self.log("train/D_fake_mean", pred_fake.mean(),prog_bar=True)

        # ---- HAT (Hybrid Attention Transformer) ---------------------------- #
        elif self.model_type == 'HAT':
            y_hat = self(x)
            # loss  = self.sr_criterion(y_hat, y)
            pixel_loss      = self.sr_criterion(y_hat, y)
            perceptual_loss = self.perceptual_weight * self.perceptual_criterion(y_hat, y)
            loss = pixel_loss + perceptual_loss
            self.log("train/train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)

        # ---- PALETTE (pixel-space diffusion) ------------------------------- #
        elif self.model_type == 'PALETTE':
            pred, target = self.palette.train_palette(hq=y, lq=x)
            loss = self.sr_criterion(pred, target)
            self.log("train/train_loss", loss, on_epoch=True, prog_bar=True)
            # Sample for visualisation
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    y_hat = self(x)
                    self._log_images("train/visual_comparison",x, y, y_hat, self.global_step)

        # ---- DIP (zero-shot — no dataset training) ------------------------- #
        elif self.model_type == 'DIP':
            # DIP is zero-shot: no dataset training needed.
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            # self.log("train/train_loss", loss, on_epoch=True, prog_bar=True)
            y_hat = x

        # Shared image logging for non-LDM_UNET paths
        if batch_idx % 100 == 0 and self.model_type not in ('PALETTE', 'DIP'):
            self._log_images("train/visual_comparison",
                             x, y, y_hat, self.global_step)
        return loss

    # ---------------------------------------------------------------------- #
    #  Validation step
    # ---------------------------------------------------------------------- #
    def validation_step(self, batch, batch_idx):
        if not self.use_slices:
            x, y = batch
            x, y = x.float().to(self.device), y.float().to(self.device)

        # ---- UNET --------------------------------------------------------- #
        if self.model_type == 'UNET':
            if self.use_slices:
               x, y_hat, y = self._predict_slices(batch)
            else:
                y_hat= self(x)
            loss  = self.sr_criterion(y_hat, y)

        # ---- ESRGAN ------------------------------------------------------- #
        elif self.model_type == 'ESRGAN':
            if self.use_slices:
                x, y_hat, y = self._predict_slices(batch)
            else:
                y_hat = self(x)
            loss = self.sr_criterion(y_hat, y) + \
                    self.perceptual_weight * self.perceptual_criterion(y_hat, y)

        # ---- HAT (Hybrid Attention Transformer) ---------------------------- #
        elif self.model_type == 'HAT':
            if self.use_slices:
                x, y_hat, y = self._predict_slices(batch)
            else:
                y_hat = self(x)
            loss  = self.sr_criterion(y_hat, y) + \
                                self.perceptual_weight * self.perceptual_criterion(y_hat, y)

        # ---- PALETTE (pixel-space diffusion) ------------------------------- #
        elif self.model_type == 'PALETTE':
            if self.use_slices:
                with torch.no_grad():
                    x, y_hat, y = self._predict_slices(batch)
            else:
                with torch.no_grad():
                    y_hat = self(x)
            loss  = self.sr_criterion(y_hat, y)

        # ---- DIP (per-image optimisation) ---------------------------------- #
        elif self.model_type == 'DIP':
            if self.use_slices:
                x, y_hat, y = self._predict_slices(batch)
            else:
                y_hat = self(x)
            loss  = self.sr_criterion(y_hat, y)

        
        if batch_idx % 100 == 0:
            self._log_images("val/visual_comparison",
                                x, y, y_hat, self.global_step)

        psnr_val  = self._psnr(y_hat, y)
        ssim_val  = self._ssim(y_hat, y)
        lpips_val = self._lpips(y_hat, y)
        self.log("val/psnr",  psnr_val,  prog_bar=True)
        self.log("val/ssim",  ssim_val,  prog_bar=True)
        self.log("val/lpips", lpips_val, prog_bar=True)
        self.log("val/loss", loss, prog_bar=True)
    # ---------------------------------------------------------------------- #
    #  Test step
    # ---------------------------------------------------------------------- #
    def test_step(self, batch, batch_idx):
        if not self.use_slices:
            x, y = batch
            x, y = x.float().to(self.device), y.float().to(self.device)

        if self.model_type == 'UNET':
            if self.use_slices:
               x, y_hat, y = self._predict_slices(batch)
            else:
                y_hat = self(x)
            loss  = self.sr_criterion(y_hat, y)

        elif self.model_type == 'ESRGAN':
            if self.use_slices:
                x, y_hat, y = self._predict_slices(batch)
            else:
                y_hat = self(x)
            loss  = self.sr_criterion(y_hat, y) + \
                    self.perceptual_weight * self.perceptual_criterion(y_hat, y)
        elif self.model_type == 'HAT':
            if self.use_slices:
                x, y_hat, y = self._predict_slices(batch)
            else:
                y_hat = self(x)
            loss  = self.sr_criterion(y_hat, y)+ \
                    self.perceptual_weight * self.perceptual_criterion(y_hat, y)
        elif self.model_type == 'PALETTE':
            if self.use_slices:
                with torch.no_grad():
                    x, y_hat, y = self._predict_slices(batch)
            else:
                with torch.no_grad():
                    y_hat = self(x)
            loss  = self.sr_criterion(y_hat, y)

        elif self.model_type == 'DIP':
            if self.use_slices:
                x, y_hat, y = self._predict_slices(batch)
            else:
                y_hat = self(x)
            loss  = self.sr_criterion(y_hat, y)

        psnr_val  = self._psnr(y_hat, y)
        ssim_val  = self._ssim(y_hat, y)
        lpips_val = self._lpips(y_hat, y)
        self.log("test/psnr",  psnr_val,  on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ssim",  ssim_val,  on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/lpips", lpips_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss", loss, prog_bar=True)
        self._log_images(f"test/visual_comparison",
                            x, y, y_hat, self.global_step)

    # ---------------------------------------------------------------------- #
    #  Optimisers
    # ---------------------------------------------------------------------- #
    def configure_optimizers(self):
        if self.model_type == 'UNET':
            return torch.optim.Adam(self.unet.parameters(), lr=self.lr)

        elif self.model_type == 'ESRGAN':
            return (
                [torch.optim.Adam(self.gen.parameters(),  lr=self.lr),
                 torch.optim.Adam(self.disc.parameters(), lr=self.lr)],
                []
            )

        elif self.model_type == 'HAT':
            decay, no_decay = [], []
            for name, p in self.hat.named_parameters():
                if ('relative_position_bias_table' in name
                        or 'norm' in name or 'bias' in name):
                    no_decay.append(p)
                else:
                    decay.append(p)
            return torch.optim.AdamW([
                {'params': decay, 'weight_decay': 1e-4},
                {'params': no_decay, 'weight_decay': 0},
            ], lr=self.lr, betas=(0.9, 0.999))

        elif self.model_type == 'LDM_VAE':
            # Train only the VAE
            return torch.optim.AdamW(self.ldm.vae.parameters(),
                                     lr=self.lr, weight_decay=1e-4)

        elif self.model_type == 'LDM_UNET':
            # Train UNet + LR projector; VAE is frozen
            params = list(self.ldm.unet.parameters()) + \
                     list(self.ldm.lr_projector.parameters())
            return torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)

        elif self.model_type == 'PALETTE':
            return torch.optim.AdamW(self.palette.parameters(),
                                     lr=self.lr, weight_decay=1e-4)

        elif self.model_type == 'DIP':
            return torch.optim.Adam(self.dip.parameters(), lr=self.lr)

        else:
            raise ValueError(f"No optimiser defined for model_type='{self.model_type}'")