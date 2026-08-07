import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# from original DIP common.py
# ---------------------------------------------------------------------------

def act(act_fun: str = "LeakyReLU") -> nn.Module:
    if act_fun == "LeakyReLU":
        return nn.LeakyReLU(0.2, inplace=True)
    elif act_fun == "ELU":
        return nn.ELU()
    elif act_fun == "Swish":

        class Swish(nn.Module):
            def forward(self, x):
                return x * torch.sigmoid(x)

        return Swish()
    elif act_fun == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation: {act_fun}")


def bn(num_features: int) -> nn.Module:
    return nn.BatchNorm2d(num_features)


def conv(
    in_f: int,
    out_f: int,
    kernel_size: int = 3,
    stride: int = 1,
    bias: bool = True,
    pad: str = "reflection",
    downsample_mode: str = "stride",
) -> nn.Module:
    need_pad = (kernel_size - 1) // 2
    layers = []

    if pad == "reflection":
        layers.append(nn.ReflectionPad2d(need_pad))
        need_pad = 0

    layers.append(nn.Conv2d(in_f, out_f, kernel_size, stride, padding=need_pad, bias=bias))

    if stride != 1 and downsample_mode != "stride":
        layers.pop(-1)  # remove conv, we'll use explicit downsampler
        layers.append(nn.Conv2d(in_f, out_f, kernel_size, 1, padding=need_pad, bias=bias))
        if downsample_mode == "avg":
            layers.append(nn.AvgPool2d(stride, stride))
        elif downsample_mode == "max":
            layers.append(nn.MaxPool2d(stride, stride))
        else:
            raise ValueError(f"Unknown downsample_mode: {downsample_mode}")

    return nn.Sequential(*layers)


class Concat(nn.Module):
    """
    Concatenate outputs of multiple submodules along dim,
    aligning spatial sizes to the smallest.
    """

    def __init__(self, dim: int, *modules: nn.Module):
        super().__init__()
        self.dim = dim
        for idx, mod in enumerate(modules):
            self.add_module(str(idx), mod)

    def forward(self, x):
        outputs = [mod(x) for mod in self._modules.values()]
        shapes2 = [o.shape[2] for o in outputs]
        shapes3 = [o.shape[3] for o in outputs]

        if not (np.all(np.array(shapes2) == min(shapes2))
                and np.all(np.array(shapes3) == min(shapes3))):
            target2 = min(shapes2)
            target3 = min(shapes3)
            aligned = []
            for o in outputs:
                d2 = (o.size(2) - target2) // 2
                d3 = (o.size(3) - target3) // 2
                aligned.append(o[:, :, d2:d2 + target2, d3:d3 + target3])
            outputs = aligned

        return torch.cat(outputs, dim=self.dim)

# ---------------------------------------------------------------------------

def skip_net(
    num_input_channels: int = 32,
    num_output_channels: int = 3,
    num_channels_down: list = None,
    num_channels_up: list = None,
    num_channels_skip: list = None,
    filter_size_down: int = 3,
    filter_size_up: int = 3,
    filter_skip_size: int = 1,
    need_sigmoid: bool = True,
    need_bias: bool = True,
    pad: str = "reflection",
    upsample_mode: str = "bilinear",
    downsample_mode: str = "stride",
    act_fun: str = "LeakyReLU",
    need1x1_up: bool = True,
) -> nn.Module:
    """
    Assembles the DIP encoder-decoder with skip connections.

    Architecture (5 scales by default):
        Input
        ├── Skip path: 1x1 conv → BN → LeakyReLU
        ├── Down path: conv stride2 → BN → LeakyReLU → conv → BN → LeakyReLU
        │                  ╰── deeper (recursive)
        └── Concat[skip, deeper] → BN → conv → BN → LeakyReLU
                         → (optional 1x1) → Upsample → ...

    Returns a nn.Module that maps (B, C, H, W) → (B, C_out, H, W).
    """
    n_scales = len(num_channels_down)
    assert len(num_channels_up) == n_scales
    assert len(num_channels_skip) == n_scales

    if isinstance(upsample_mode, str):
        upsample_mode = [upsample_mode] * n_scales
    if isinstance(downsample_mode, str):
        downsample_mode = [downsample_mode] * n_scales
    if isinstance(filter_size_down, int):
        filter_size_down = [filter_size_down] * n_scales
    if isinstance(filter_size_up, int):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1
    model = nn.Sequential()
    model_tmp = model
    input_depth = num_input_channels

    for i in range(n_scales):
        deeper = nn.Sequential()
        skip = nn.Sequential()

        # -- Down block (stride-2 conv) --
        deeper.add_module(
            "down_conv",
            conv(input_depth, num_channels_down[i], filter_size_down[i], 2,
                 bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]),
        )
        deeper.add_module("down_bn", bn(num_channels_down[i]))
        deeper.add_module("down_act", act(act_fun))
        deeper.add_module(
            "down_conv2",
            conv(num_channels_down[i], num_channels_down[i], filter_size_down[i],
                 bias=need_bias, pad=pad),
        )
        deeper.add_module("down_bn2", bn(num_channels_down[i]))
        deeper.add_module("down_act2", act(act_fun))

        # -- Skip path (1×1 conv if used) --
        skip_out = num_channels_skip[i]
        if skip_out:
            skip.add_module(
                "skip_conv",
                conv(input_depth, skip_out, filter_skip_size, bias=need_bias, pad=pad),
            )
            skip.add_module("skip_bn", bn(skip_out))
            skip.add_module("skip_act", act(act_fun))

        # -- Concat skip + deeper --
        deeper_out = num_channels_up[i + 1] if i < last_scale else num_channels_down[i]
        concat_in = skip_out + deeper_out
        if skip_out:
            model_tmp.add_module(f"scale{i}_concat", Concat(1, skip, deeper))
        else:
            model_tmp.add_module(f"scale{i}_deeper", deeper)
        model_tmp.add_module(f"scale{i}_post_bn", bn(concat_in))

        # -- Up block --
        if i != last_scale:
            deeper_main = nn.Sequential()
            deeper.add_module("deeper_main", deeper_main)

        model_tmp.add_module(
            f"scale{i}_upsample",
            nn.Upsample(scale_factor=2, mode=upsample_mode[i]),
        )

        model_tmp.add_module(
            f"scale{i}_up_conv",
            conv(concat_in, num_channels_up[i], filter_size_up[i], bias=need_bias, pad=pad),
        )
        model_tmp.add_module(f"scale{i}_up_bn", bn(num_channels_up[i]))
        model_tmp.add_module(f"scale{i}_up_act", act(act_fun))

        if need1x1_up:
            model_tmp.add_module(
                f"scale{i}_1x1",
                conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad),
            )
            model_tmp.add_module(f"scale{i}_1x1_bn", bn(num_channels_up[i]))
            model_tmp.add_module(f"scale{i}_1x1_act", act(act_fun))

        input_depth = num_channels_down[i]
        if i != last_scale:
            model_tmp = deeper_main

    # -- Final 1×1 conv → output --
    model.add_module("final_conv", conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add_module("final_sigmoid", nn.Sigmoid())

    return model

# ---------------------------------------------------------------------------

def get_noise(
    input_depth: int,
    spatial_size: tuple,
    noise_type: str = "u",
    var: float = 1.0 / 10.0,
) -> torch.Tensor:
    """
    Returns a (1, input_depth, H, W) noise tensor.

    Args:
        input_depth: Number of input channels for the network.
        spatial_size: (H, W) of the target image.
        noise_type: 'u' for uniform, 'n' for normal.
        var: Scaling factor (standard deviation for normal, range for uniform).
    """
    H, W = spatial_size
    t = torch.zeros(1, input_depth, H, W)
    if noise_type == "u":
        t.uniform_()
        t *= var
    elif noise_type == "n":
        t.normal_()
        t *= var
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    return t


def get_noise_input(
    input_depth: int,
    spatial_size: tuple,
    noise_type: str = "u",
    var: float = 1.0 / 10.0,
) -> torch.Tensor:
    """Alias for get_noise — generates the fixed input z for DIP."""
    return get_noise(input_depth, spatial_size, noise_type, var)

# ---------------------------------------------------------------------------

DEFAULT_CHANNELS_DOWN = (128, 128, 128, 128, 128)
DEFAULT_CHANNELS_UP = (128, 128, 128, 128, 128)
DEFAULT_CHANNELS_SKIP = (4, 4, 4, 4, 4)


class DeepImagePrior(nn.Module):
    """
    DIP network: encoder-decoder with skip connections.

    Architecture follows the original paper.  The network is randomly
    initialised and fit to a single degraded image at test time.

    Args:
        out_channels: Number of output image channels (3 for RGB).
        input_depth: Number of input noise channels (default 32 per paper).
        n_scales: Number of encoder-decoder scales (default 5).
        need_sigmoid: Whether to apply Sigmoid at the output.
    """

    def __init__(
        self,
        out_channels: int = 3,
        input_depth: int = 32,
        n_scales: int = 5,
        need_sigmoid: bool = True,
    ):
        super().__init__()
        self.input_depth = input_depth
        self.out_channels = out_channels
        self.need_sigmoid = need_sigmoid

        channels_down = list(DEFAULT_CHANNELS_DOWN)[:n_scales]
        channels_up = list(DEFAULT_CHANNELS_UP)[:n_scales]
        channels_skip = list(DEFAULT_CHANNELS_SKIP)[:n_scales]

        self.net = skip_net(
            num_input_channels=input_depth,
            num_output_channels=out_channels,
            num_channels_down=channels_down,
            num_channels_up=channels_up,
            num_channels_skip=channels_skip,
            need_sigmoid=need_sigmoid,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def reset_weights(self):
        """Re-initialize all network parameters."""
        def _init(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
        self.net.apply(_init)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())


# ---------------------------------------------------------------------------
# Per-image optimisation (the core DIP algorithm)
# ---------------------------------------------------------------------------

def dip_optimise(
    net: nn.Module,
    x: torch.Tensor,
    input_depth: int = 32,
    lr: float = 0.01,
    num_iterations: int = 2400,
    input_noise: bool = True,
    reg_noise_std: float = 0.033,
    ema_weight: float = 0.99,
    tv_weight: float = 0.0,
    backtrack_thresh: float = 1.05,
    num_runs: int = 1,
) -> torch.Tensor:
    """
    Run DIP per-image optimisation on a single LQ image.

    Follows the original paper:
      - Fixed noise input z, noise perturbation each iteration
      - Exponential moving average of outputs (ema_weight)
      - Backtracking when loss spikes (backtrack_thresh)
      - Multi-run averaging (num_runs > 1)

    Args:
        net: DeepImagePrior network (will be reset).
        x: (1, C, H, W) LQ input tensor.
        input_depth: Noise channels (32 in paper).
        lr: Adam learning rate (0.01 in paper).
        num_iterations: Optimisation steps (2400-3000 in paper).
        input_noise: True = classic DIP (noise input), False = LQ input.
        reg_noise_std: Std of noise perturbation (1/30 in paper).
        ema_weight: EMA decay (0.99 in paper).
        tv_weight: TV regularisation weight (0 = off).
        backtrack_thresh: Reload params when loss exceeds best × thresh.
        num_runs: Number of independent runs to average (1 = single).

    Returns:
        (1, C, H, W) restored image in [0, 1].
    """
    B, C, H, W = x.shape
    device = x.device

    # Pad to nearest multiple of 2^5 = 32 (network has 5 down/up scales)
    stride = 2 ** 5
    pad_h = (stride - H % stride) % stride
    pad_w = (stride - W % stride) % stride
    if pad_h or pad_w:
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    else:
        x_pad = x
    Hp, Wp = x_pad.shape[2], x_pad.shape[3]

    def _run_once() -> torch.Tensor:
        net.reset_weights()
        net.train()
        net.to(device)

        # --- Fixed input z (at padded size) ---
        if input_noise:
            z_base = get_noise(input_depth, (Hp, Wp),
                               noise_type="u", var=1.0 / 10.0).to(device)
            noise_buffer = torch.zeros_like(z_base)
        else:
            repeats = input_depth // C
            z_base = x_pad.clone().detach()
            if repeats > 1:
                z_base = z_base.repeat(1, repeats, 1, 1)
            noise_buffer = None

        opt = torch.optim.Adam(net.parameters(), lr=lr)

        out_avg = None
        last_params = None
        best_loss = float("inf")

        for i in range(num_iterations):
            z = z_base
            if input_noise and reg_noise_std > 0:
                noise_buffer.normal_()
                z = z_base + noise_buffer * reg_noise_std

            y_hat = net(z)

            # EMA
            if ema_weight > 0:
                if out_avg is None:
                    out_avg = y_hat.detach()
                else:
                    out_avg = (out_avg * ema_weight
                               + y_hat.detach() * (1 - ema_weight))

            # Loss (compare cropped output with padded input)
            loss = F.l1_loss(y_hat[:, :, :H, :W], x_pad[:, :, :H, :W])
            if tv_weight > 0:
                dy = torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]).mean()
                dx = torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]).mean()
                loss += tv_weight * (dx + dy)

            # Backtracking check
            if i % 100 == 0:
                loss_val = loss.item()
                if loss_val > best_loss * backtrack_thresh and last_params is not None:
                    with torch.no_grad():
                        for p, saved in zip(net.parameters(), last_params):
                            p.copy_(saved.to(device))
                    loss = loss.detach() * 0.0
                    loss.requires_grad_(True)
                else:
                    best_loss = min(loss_val, best_loss)
                    last_params = [p.detach().cpu() for p in net.parameters()]

            opt.zero_grad()
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            result = out_avg if out_avg is not None else net(z)
            return result[:, :, :H, :W]

    # --- Multi-run averaging ---
    outputs = []
    for _ in range(max(1, num_runs)):
        with torch.enable_grad():
            outputs.append(_run_once())

    return torch.stack(outputs).mean(dim=0)
