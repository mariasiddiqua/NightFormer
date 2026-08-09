# model/diff_refiner.py
# Tiny deterministic diffusion-like refiner for perceptual cleanup.
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyUNet(nn.Module):
    def __init__(self, ch=32, cond_ch=None):
        super().__init__()
        self.enc1 = nn.Conv2d(3, ch, 3, 1, 1)
        self.enc2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.dec1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.out  = nn.Conv2d(ch, 3, 1, 1, 0)

        # Project condition (bottleneck features) to match 'ch'
        if cond_ch is None or cond_ch == ch:
            self.cond_proj = nn.Identity()
        else:
            self.cond_proj = nn.Conv2d(cond_ch, ch, 1, 1, 0)

    def forward(self, x, cond):
        # Upsample and project cond to (B, ch, H, W)
        cond_up = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cond_up = self.cond_proj(cond_up)

        # Refine
        h = F.gelu(self.enc1(x) + cond_up)
        h = F.gelu(self.enc2(h))
        h = F.gelu(self.dec1(h))
        return self.out(h)

class DiffusionRefiner(nn.Module):
    def __init__(self, cond_ch, ch=48, steps=3, sigma=0.02):
        super().__init__()
        self.steps = steps
        self.sigma = sigma
        self.score = TinyUNet(ch=ch, cond_ch=cond_ch)

    def forward(self, coarse_img, cond):
        x = coarse_img
        for _ in range(self.steps):
            eps = self.score(x, cond)
            x = x - self.sigma * eps
        return x.clamp(0, 1)
