import torch
import torch.nn as nn
import torch.nn.functional as F

# IGFE:
#1. FocusStructure → slicing + concat → conv
#2. ConvDownSampling ×2 → conv stride=2
#3. RESBLOCK ×4 → CNN blocks con residual connections

# --- Focus Structure ---
class FocusStructure(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(FocusStructure, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        # Slicing: [B, C, H, W] → concatenate quadranti
        patch_tl = x[..., ::2, ::2]  # top-left
        patch_tr = x[..., ::2, 1::2]  # top-right
        patch_bl = x[..., 1::2, ::2]  # bottom-left
        patch_br = x[..., 1::2, 1::2]  # bottom-right
        x = torch.cat([patch_tl, patch_tr, patch_bl, patch_br], dim=1)  # [B, 4C, H/2, W/2]
        return self.conv(x)

# --- ConvDownSampling ---
class ConvDownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDownSampling, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.block(x)

# --- Residual Block (RESBLOCK) ---
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.relu(x + self.block(x))

# --- IGFE ---
class IGFE(nn.Module):
    def __init__(self):
        super(IGFE, self).__init__()
        self.focus = FocusStructure(3, 64)         # From [3,48,144] to [64,24,72]
        self.down1 = ConvDownSampling(64, 128)     # [128,12,36]
        self.res1 = ResBlock(128)
        self.res2 = ResBlock(128)
        self.down2 = ConvDownSampling(128, 256)    # [256,6,18]
        self.res3 = ResBlock(256)
        self.res4 = ResBlock(256)
        self.final_conv = nn.Conv2d(256, 512, kernel_size=1)  # [512,6,18]

    def forward(self, x):
        x = self.focus(x)
        x = self.down1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.down2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.final_conv(x)
        return x  # [B, 512, 6, 18]
    


