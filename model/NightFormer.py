import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Basic Building Blocks
# =========================================================

class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )


class ResidualBlock(nn.Module):
    """Simple residual conv block used before/after attention blocks."""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvBNAct(ch, ch, 3, 1, 1)
        self.conv2 = ConvBNAct(ch, ch, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class PlainBottleneckBlock(nn.Module):
    """
    Replaces TransformerBlockCSWA for ablation.
    Simple conv block without any window attention.
    """
    def __init__(self, dim):
        super().__init__()
        self.block = ResidualBlock(dim)

    def forward(self, x):
        return self.block(x)


class _NoOpAux(nn.Module):
    """Stub that returns empty dict; keeps forward() API identical."""
    def forward(self, feat):
        return {}


class NightFormer(nn.Module):
    def __init__(self,
                 # Legacy args (from original repo)
                 inp_channels=3,
                 out_channels=3,
                 dim=16,
                 num_blocks=None,
                 num_refinement_blocks=None,
                 # New-style arg
                 base_ch=None,
                 use_weather_adapters=True,   # kept for API compat, ignored here
                 **kwargs):
        """
        Backward-compatible constructor:
          Old: NightFormer(inp_channels=3, out_channels=3, dim=16,
                          num_blocks=[...], num_refinement_blocks=2)
          New: NightFormer(base_ch=48)
        """
        super().__init__()

        if base_ch is None:
            base_ch = max(32, dim * 3)

        self.inp_channels = inp_channels
        self.out_channels = out_channels
        C = base_ch

        # --- Stem ---
        self.stem = ConvBNAct(self.inp_channels, C, 3, 1, 1)

        # [OFF] Illumination head removed
        # self.illum_head = IlluminationHead(C)

        # --- Encoder ---
        # [OFF] WeatherAwareBlock wrapping removed; plain ResidualBlocks used
        self.enc1 = ResidualBlock(C)
        self.down1 = ConvBNAct(C, 2*C, 3, 2, 1)

        self.enc2 = ResidualBlock(2*C)
        self.down2 = ConvBNAct(2*C, 4*C, 3, 2, 1)

        self.enc3 = ResidualBlock(4*C)
        self.down3 = ConvBNAct(4*C, 4*C, 2, 2, 0)

        # --- Bottleneck ---
        # [OFF] TransformerBlockCSWA replaced with plain conv blocks
        self.mid1 = PlainBottleneckBlock(4*C)
        self.mid2 = PlainBottleneckBlock(4*C)

        # [OFF] Auxiliary multi-task heads removed (no T, Mr, Ms)
        self.aux = _NoOpAux()

        # --- Decoder ---
        self.up3 = nn.ConvTranspose2d(4*C, 4*C, 2, 2)
        self.dec3 = ResidualBlock(4*C)

        self.up2 = nn.ConvTranspose2d(4*C, 2*C, 2, 2)
        self.dec2 = ResidualBlock(2*C)

        self.up1 = nn.ConvTranspose2d(2*C, C, 2, 2)
        self.dec1 = ResidualBlock(C)

        # [OFF] Pyramid side heads removed (no side1, side2)
        # self.side1 = nn.Conv2d(4*C, self.out_channels, 3, 1, 1)
        # self.side2 = nn.Conv2d(2*C, self.out_channels, 3, 1, 1)

        # Full-resolution output (only scale kept)
        self.out = nn.Conv2d(C, self.out_channels, 3, 1, 1)

        # [OFF] Diffusion refiner removed
        # self.refiner = DiffusionRefiner(cond_ch=4*C, ch=C, steps=3, sigma=0.02)

    def forward(self, x):
        # --- Stem ---
        f0 = self.stem(x)               # [B, C, H, W]

        # [OFF] Illumination modulation removed
        # L  = self.illum_head(f0)
        # f0 = f0 * (0.7 + 0.6 * L)

        # --- Encoder ---
        e1 = self.enc1(f0)              # [B, C, H, W]
        d1 = self.down1(e1)             # [B, 2C, H/2, W/2]

        e2 = self.enc2(d1)              # [B, 2C, H/2, W/2]
        d2 = self.down2(e2)             # [B, 4C, H/4, W/4]

        e3 = self.enc3(d2)              # [B, 4C, H/4, W/4]
        d3 = self.down3(e3)             # [B, 4C, H/8, W/8]

        # --- Bottleneck (plain conv, no CSWA) ---
        m  = self.mid1(d3)
        m  = self.mid2(m)

        # [OFF] Aux predictions removed
        aux_out = self.aux(m)           # returns {}

        # --- Decoder + skip connections ---
        u3 = self.up3(m)                # H/4
        u3 = u3 + e3
        u3 = self.dec3(u3)

        u2 = self.up2(u3)               # H/2
        u2 = u2 + e2
        u2 = self.dec2(u2)

        u1 = self.up1(u2)               # H
        u1 = u1 + e1
        u1 = self.dec1(u1)

        # [OFF] Pyramid predictions removed (no y1, y2)
        # y1 = torch.sigmoid(self.side1(u3))
        # y2 = torch.sigmoid(self.side2(u2))

        y_final = torch.sigmoid(self.out(u1))   # full-res output

        # [OFF] Diffusion refiner removed
        # y_final = self.refiner(y3, m)

        # aux_out is empty dict (no T, Mr, Ms, illum, pyr)
        return y_final, aux_out


# =========================================================
# [ON] Change 4: Multi-Objective Loss  <-- ACTIVE
# =========================================================
# This is the ONLY change active in this ablation variant.
#
# Combines:
#   - L1 reconstruction loss
#   - Perceptual / frequency loss  (FFT-based, no VGG needed)
#   - SSIM loss
# All weighted and summed into a single scalar.
# =========================================================

class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss (1 - SSIM), computed per-channel.
    Window-based, pure PyTorch — no external dependency.
    """
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.ws = window_size
        kernel = self._gaussian_kernel(window_size, sigma)
        # [1, 1, ws, ws]  — will be expanded per channel at runtime
        self.register_buffer('kernel', kernel)

    @staticmethod
    def _gaussian_kernel(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.outer(g)
        return kernel.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]

    def forward(self, pred, target):
        C = pred.shape[1]
        k = self.kernel.expand(C, 1, self.ws, self.ws)
        pad = self.ws // 2

        mu1 = F.conv2d(pred,   k, padding=pad, groups=C)
        mu2 = F.conv2d(target, k, padding=pad, groups=C)

        mu1_sq  = mu1 * mu1
        mu2_sq  = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sig1  = F.conv2d(pred   * pred,   k, padding=pad, groups=C) - mu1_sq
        sig2  = F.conv2d(target * target, k, padding=pad, groups=C) - mu2_sq
        sig12 = F.conv2d(pred   * target, k, padding=pad, groups=C) - mu1_mu2

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2*mu1_mu2 + c1) * (2*sig12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sig1 + sig2 + c2))
        return 1.0 - ssim_map.mean()


class FFTPerceptualLoss(nn.Module):
    """
    Frequency-domain perceptual loss.
    Penalises differences in amplitude spectrum — no VGG required.
    """
    def forward(self, pred, target):
        # pred / target: [B, C, H, W] in [0, 1]
        fft_pred   = torch.fft.rfft2(pred,   norm='ortho')
        fft_target = torch.fft.rfft2(target, norm='ortho')
        amp_pred   = torch.abs(fft_pred)
        amp_target = torch.abs(fft_target)
        return F.l1_loss(amp_pred, amp_target)


class MultiObjectiveLoss(nn.Module):
    """
    Multi-Objective Loss for unified image restoration.

    Components (all active in this ablation):
      - L1  reconstruction loss          (weight: w_l1)
      - FFT perceptual loss              (weight: w_fft)
      - SSIM loss  (1 - SSIM)            (weight: w_ssim)

    Usage:
        criterion = MultiObjectiveLoss()
        loss, breakdown = criterion(pred, target, aux_out)

    Returns:
        total  : scalar tensor (differentiable)
        info   : dict with individual loss values (for logging)
    """
    def __init__(self, w_l1=1.0, w_fft=0.1, w_ssim=0.5):
        super().__init__()
        self.w_l1   = w_l1
        self.w_fft  = w_fft
        self.w_ssim = w_ssim

        self.ssim_loss = SSIMLoss(window_size=11)
        self.fft_loss  = FFTPerceptualLoss()

    def forward(self, pred, target, aux_out=None):
        """
        Args:
            pred     : [B, C, H, W]  model output (sigmoid-activated, [0,1])
            target   : [B, C, H, W]  clean ground-truth
            aux_out  : dict from model.forward() — unused in this ablation
                       (no T/Mr/Ms/illum/pyr keys present)
        Returns:
            total    : scalar loss tensor
            info     : dict with per-component losses for logging
        """
        # --- Core reconstruction losses ---
        l1   = F.l1_loss(pred, target)
        fft  = self.fft_loss(pred, target)
        ssim = self.ssim_loss(pred, target)

        total = (self.w_l1  * l1 +
                 self.w_fft * fft +
                 self.w_ssim * ssim)

        info = {
            "loss/total": total.item(),
            "loss/l1":    l1.item(),
            "loss/fft":   fft.item(),
            "loss/ssim":  ssim.item(),
        }
        return total, info


# =========================================================
# Quick smoke-test  (remove before training)
# =========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model     = NightFormer(inp_channels=3, out_channels=3, dim=16).to(device)
    criterion = MultiObjectiveLoss().to(device)

    x      = torch.randn(2, 3, 256, 256).to(device)
    target = torch.rand(2, 3, 256, 256).to(device)

    pred, aux_out = model(x)
    loss, info    = criterion(pred, target, aux_out)

    print(f"Output shape : {pred.shape}")
    print(f"aux_out keys : {list(aux_out.keys())}")   # should be []
    for k, v in info.items():
        print(f"  {k}: {v:.4f}")
    print("Backward pass ... ", end="")
    loss.backward()
    print("OK")