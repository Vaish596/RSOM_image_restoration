from pytorch_msssim import ssim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19



class SRLoss(nn.Module):
    """
        loss_type options:
        - 'mse': MSE loss
        - 'l1': L1 loss
        - 'l1_ssim': l1_weight * L1 + ssim_weight * SSIM loss
    """
    def __init__(self, loss_type='l1_ssim', l1_weight=0.8, ssim_weight=0.2):
        super().__init__()
        self.loss_type = loss_type
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight

    def forward(self, pred, target):
        if self.loss_type == 'mse':
            return F.mse_loss(pred, target)
        elif self.loss_type == 'l1':
            return F.l1_loss(pred, target)
        elif self.loss_type == 'l1_ssim':
            l1 = F.l1_loss(pred, target)
            ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
            return self.l1_weight * l1 + self.ssim_weight * ssim_loss
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

def gan_loss(pred, target_is_real=True):
    target = torch.full_like(pred,0.9) if target_is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)

#-----------------VGG PERPETUAL LOSS-----------------

class VGGFeatureLoss(nn.Module):
    def __init__(self, layer_ids=[2, 7, 12], use_input_norm=True):
        super().__init__()
        vgg = vgg19(weights="IMAGENET1K_V1").features[:35].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def forward(self, x, y):
        # Convert to 3-channel if needed
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
            y = y.repeat(1,3,1,1)

        # Normalize
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x = (x - mean) / std
        y = (y - mean) / std

        feat_x = self.vgg(x)
        feat_y = self.vgg(y)

        return F.mse_loss(feat_x, feat_y)
    
class VAELoss(nn.Module):
    """
    L = L1_recon  +  perceptual_weight * L_perceptual  +  kl_weight * KL

    VGGFeatureLoss is imported from your existing losses module.
    """

    def __init__(self, perceptual_weight: float = 0.1, kl_weight: float = 1e-6):
        super().__init__()
        self.perceptual        = VGGFeatureLoss()
        self.perceptual_weight = perceptual_weight
        self.kl_weight         = kl_weight

    def forward(self, recon, target, posterior):
        l1   = F.l1_loss(recon, target)
        perc = self.perceptual(recon, target)
        kl   = posterior.kl().mean()
        loss = l1 + self.perceptual_weight * perc + self.kl_weight * kl
        return loss, {"l1": l1.detach(), "perceptual": perc.detach(), "kl": kl.detach()}
