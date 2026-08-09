# models/illumination.py
# Predicts an illumination map (L) to guide low-light restoration.
import torch
import torch.nn as nn

class IlluminationHead(nn.Module):
    def __init__(self, in_ch, mid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, 1, 1), nn.GELU(),
            nn.Conv2d(mid, mid, 3, 1, 1), nn.GELU(),
            nn.Conv2d(mid, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Returns L in [0,1]
        return self.net(x)
