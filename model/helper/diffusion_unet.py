import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


def exists(x):
    return x is not None


def default(val, d):
    return val if val is not None else d


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, dropout=0.0):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch * 2) if time_emb_dim else None

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.act = SiLU()
        self.dropout = nn.Dropout(dropout)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.act(self.norm1(self.conv1(x)))
        if exists(self.time_mlp) and exists(t_emb):
            time_shift, time_scale = self.time_mlp(t_emb).chunk(2, dim=1)
            h = h * (1 + time_scale.unsqueeze(-1).unsqueeze(-1)) + time_shift.unsqueeze(-1).unsqueeze(-1)
        h = self.dropout(self.act(self.norm2(self.conv2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch, n_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.to_qkv = nn.Conv2d(ch, ch * 3, 1)
        self.to_out = nn.Conv2d(ch, ch, 1)
        self.n_heads = n_heads

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, self.n_heads, c // self.n_heads, h * w).transpose(-2, -1)
        k = k.view(b, self.n_heads, c // self.n_heads, h * w)
        v = v.view(b, self.n_heads, c // self.n_heads, h * w).transpose(-2, -1)

        attn = torch.matmul(q, k) * (c // self.n_heads) ** -0.5
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1).contiguous().view(b, c, h, w)
        return x + self.to_out(out)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, has_attn=False, dropout=0.0):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, time_emb_dim, dropout)
        self.attn = AttentionBlock(out_ch) if has_attn else nn.Identity()

    def forward(self, x, t_emb=None):
        x = self.res(x, t_emb)
        x = self.attn(x)
        return x


class UpBlockNoSkip(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, has_attn=False, dropout=0.0):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, time_emb_dim, dropout)
        self.attn = AttentionBlock(out_ch) if has_attn else nn.Identity()

    def forward(self, x, t_emb=None):
        x = self.res(x, t_emb)
        x = self.attn(x)
        return x


class UpBlockWithSkip(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim, has_attn=False, dropout=0.0):
        super().__init__()
        self.res = ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim, dropout)
        self.attn = AttentionBlock(out_ch) if has_attn else nn.Identity()

    def forward(self, x, skip, t_emb=None):
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=6,
        out_channels=3,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        n_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        time_emb_dim=256,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        ch = base_channels
        self.encoder_blocks = nn.ModuleList()
        self.encoder_down = nn.ModuleList()
        self.decoder_first = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.decoder_up = nn.ModuleList()

        resolution = 128
        n_levels = len(channel_multipliers)

        # Encoder
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            level = nn.ModuleList()
            for _ in range(n_res_blocks):
                has_attn = resolution in attn_resolutions
                level.append(DownBlock(ch, out_ch, time_emb_dim, has_attn, dropout))
                ch = out_ch
            self.encoder_blocks.append(level)
            if i < n_levels - 1:
                self.encoder_down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                resolution //= 2
            else:
                self.encoder_down.append(nn.Identity())

        # Middle
        self.mid_block1 = ResidualBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch, time_emb_dim, dropout)

        # Decoder
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            has_attn = resolution in attn_resolutions
            self.decoder_first.append(
                UpBlockNoSkip(ch, out_ch, time_emb_dim, has_attn, dropout))
            level = nn.ModuleList()
            for _ in range(n_res_blocks):
                level.append(
                    UpBlockWithSkip(out_ch, out_ch, out_ch, time_emb_dim, has_attn, dropout))
            self.decoder_blocks.append(level)
            ch = out_ch
            if i > 0:
                self.decoder_up.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                resolution *= 2
            else:
                self.decoder_up.append(nn.Identity())

        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, ch),
            SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps):
        t_emb = timestep_embedding(timesteps, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        h = self.conv_in(x)
        enc_skips = []

        # Encoder
        for level, down in zip(self.encoder_blocks, self.encoder_down):
            for block in level:
                h = block(h, t_emb)
                enc_skips.append(h)
            h = down(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for first_block, blocks, up in zip(
            self.decoder_first, self.decoder_blocks, self.decoder_up):
            h = first_block(h, t_emb)
            for block in blocks:
                skip = enc_skips.pop()
                h = block(h, skip, t_emb)
            h = up(h)

        return self.conv_out(h)
