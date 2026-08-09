# models/aux_heads.py
# Auxiliary heads for transmission map and rain/snow masks.
import torch.nn as nn
import torch

class SimpleHead(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//2, 3, 1, 1), nn.GELU(),
            nn.Conv2d(in_ch//2, out_ch, 1, 1, 0)
        )
    def forward(self, x): return self.net(x)

class AuxMultiTask(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.T  = SimpleHead(in_ch, 1)
        self.Mr = SimpleHead(in_ch, 1)
        self.Ms = SimpleHead(in_ch, 1)

    def forward(self, feat):
        return {
            "T":  torch.sigmoid(self.T(feat)),
            "Mr": torch.sigmoid(self.Mr(feat)),
            "Ms": torch.sigmoid(self.Ms(feat)),
        }


