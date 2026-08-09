# models/attention_cswa.py
# Cross-Scale Window Attention: mixes fine-scale with pooled coarse K/V for global context.
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossScaleWindowAttention(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8, pool=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.win = window_size
        self.pool = nn.AvgPool2d(kernel_size=pool, stride=pool)
        self.qkv = nn.Conv2d(dim, dim*3, 1, 1, 0)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0)

    def _partition(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.win, self.win).unfold(3, self.win, self.win)   # [B,C,nH,nW,win,win]
        return x.contiguous().view(B, C, -1, self.win*self.win), H, W

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)     # [B, C, H, W]
        # pooled coarse K/V
        k_coarse = self.pool(k); v_coarse = self.pool(v)

        # reshape to windows
        qw, _, _ = self._partition(q)
        kw, _, _ = self._partition(k)
        vw, _, _ = self._partition(v)

        # upsample coarse and partition
        kc = F.interpolate(k_coarse, size=(H,W), mode="nearest")
        vc = F.interpolate(v_coarse, size=(H,W), mode="nearest")
        kc, _, _ = self._partition(kc)
        vc, _, _ = self._partition(vc)

        k_all = torch.cat([kw, kc], dim=-1)
        v_all = torch.cat([vw, vc], dim=-1)

        d = (C // self.num_heads) ** -0.5
        qw = qw.view(B, self.num_heads, C//self.num_heads, -1)
        k_all = k_all.view(B, self.num_heads, C//self.num_heads, -1)
        v_all = v_all.view(B, self.num_heads, C//self.num_heads, -1)

        attn = (qw.transpose(2,3) @ k_all) * d   # [B, heads, tokens_q, tokens_k]
        attn = attn.softmax(dim=-1)
        out = attn @ v_all.transpose(2,3)
        out = out.transpose(2,3).contiguous().view(B, C, -1)

        nW = (H//self.win) * (W//self.win)
        out = out.view(B, C, nW, self.win*self.win)
        out = out.view(B, C, H, W)
        return self.proj(out)

class ConvMLP(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()
        hid = dim * expansion
        self.net = nn.Sequential(
            nn.Conv2d(dim, hid, 1), nn.GELU(),
            nn.Conv2d(hid, dim, 1)
        )
    def forward(self, x): return self.net(x)

class TransformerBlockCSWA(nn.Module):
    def __init__(self, dim, heads=4, window_size=8):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn  = CrossScaleWindowAttention(dim, num_heads=heads, window_size=window_size)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp   = ConvMLP(dim, expansion=2)

    def forward(self, x):
        y = self.attn(self.norm1(x)) + x
        y = self.mlp(self.norm2(y)) + y
        return y


