# Lightweight degradation-specific adapters with soft routing.
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

class DegradationAdapter(nn.Module):
    """
    1x1 reduce -> 3x3 process -> 1x1 expand, with residual connection.
    Adds minimal parameters per weather type (fog, rain, haze, snow).
    """
    def __init__(self, dim, bottleneck_ratio=0.25):
        super().__init__()
        hid = max(8, int(dim * bottleneck_ratio))
        self.reduce = ConvBNAct(dim, hid, 1, 1, 0)
        self.process = ConvBNAct(hid, hid, 3, 1, 1)
        self.expand = nn.Conv2d(hid, dim, 1, 1, 0)

    def forward(self, x):
        idt = x
        x = self.reduce(x)
        x = self.process(x)
        x = self.expand(x)
        return idt + x

class SoftRouter(nn.Module):
    """
    Produces soft mixture weights over degradation adapters.
    No explicit labels required; learns from gradients.
    """
    def __init__(self, dim, n_types=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim//4, 1), nn.GELU(),
            nn.Conv2d(dim//4, n_types, 1)
        )

    def forward(self, feat):
        w = self.fc(self.pool(feat)).flatten(1)     # [B, n_types]
        return torch.softmax(w, dim=1)

class WeatherAwareBlock(nn.Module):
    """
    Wraps a generic block with a mixture of degradation adapters.
    """
    def __init__(self, block, dim):
        super().__init__()
        self.block = block
        self.adapters = nn.ModuleList([DegradationAdapter(dim) for _ in range(4)])
        self.router = SoftRouter(dim, n_types=4)

    def forward(self, x):
        x = self.block(x)
        w = self.router(x)                          # [B, 4]
        out = 0
        for i, adp in enumerate(self.adapters):
            out = out + adp(x) * w[:, i].view(-1, 1, 1, 1)
        return out


