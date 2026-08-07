import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict

# in_nc/out_nc = 1 for grayscale
# ------------------------
# RRDBNet for grayscale
# ------------------------
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf+gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf+2*gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf+3*gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf+4*gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1),1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2),1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3),1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4),1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block — 3 RDBs with 0.2 residual scaling."""
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.conv_up1(fea))
        fea = self.lrelu(self.conv_up2(fea))
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        return out

#-----------Discriminator-----------

class UNetDiscriminatorSN(nn.Module):
    """U-Net discriminator with spectral normalization — from Real-ESRGAN.

    Args:
        num_in_ch (int): Input channels. Default: 3.
        num_feat (int): Base feature channels. Default: 64.
        skip_connection (bool): Use skip connections. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        norm = nn.utils.spectral_norm

        self.conv0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))

        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), 0.2, inplace=True)

        x1 = F.leaky_relu(self.conv1(x0), 0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), 0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), 0.2, inplace=True)

        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), 0.2, inplace=True)
        if self.skip_connection:
            x4 = x4 + x2

        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), 0.2, inplace=True)
        if self.skip_connection:
            x5 = x5 + x1

        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), 0.2, inplace=True)
        if self.skip_connection:
            x6 = x6 + x0

        out = F.leaky_relu(self.conv7(x6), 0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), 0.2, inplace=True)
        out = self.conv9(out)
        return out


def load_pretrained_disc(path, in_nc=1):
    disc = UNetDiscriminatorSN(num_in_ch=in_nc)
    checkpoint = torch.load(path, map_location='cpu')
    disc.load_state_dict(checkpoint, strict=False)
    return disc

def load_pretrained_gen(path, in_nc=1, out_nc=1):
    gen = RRDBNet(in_nc=in_nc, out_nc=out_nc)
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = OrderedDict()
    if in_nc == 1:
        for k, v in checkpoint.items():
            if 'conv_first.weight' in k and v.shape[1]==3:
                v = v.mean(dim=1, keepdim=True)  # RGB → grayscale
            state_dict[k] = v
    gen.load_state_dict(state_dict, strict=False)
    print("Loaded model pretrained weights")
    return gen