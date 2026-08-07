import torch
import torch.nn as nn
import torch.nn.functional as F


def create_conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def create_double_conv(in_channels: int, mid_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        create_conv_bn_relu(in_channels, mid_channels),
        create_conv_bn_relu(mid_channels, out_channels),
    )

def upsample(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
            x,
            size=reference.shape[2:],
            mode='bicubic',
            align_corners=True
        )

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=32, **kwargs):
        super().__init__()

        # self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.down1 = create_double_conv(in_channels, features, features)
        self.down2 = create_double_conv(features, features*2, features*2)
        self.down3 = create_double_conv(features*2, features*4, features*4)
        self.down4 = create_double_conv(features*4, features*8, features*8)

        self.pool = nn.MaxPool2d(2)

        self.up1 = create_double_conv(features*8 + features*4, features*4, features*4)
        self.up2 = create_double_conv(features*4 + features*2, features*2, features*2)
        self.up3 = create_double_conv(features*2 + features, features, features)

        self.final = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))

        x = upsample(x4, x3)
        x = torch.cat([x, x3], dim=1)
        x = self.up1(x)

        x = upsample(x, x2)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)

        x = upsample(x, x1)
        x = torch.cat([x, x1], dim=1)
        x = self.up3(x)

        return self.final(x)
