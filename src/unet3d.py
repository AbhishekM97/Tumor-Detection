import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        print("In UNet3D init")
        super(UNet3D, self).__init__()
        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 1))  # No downsampling along depth
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.bottleneck = ConvBlock(128, 256)

        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.dec1 = ConvBlock(64, 32)

        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        print("in forward")
        if self.training:
            enc1 = checkpoint(self.enc1, x, use_reentrant=False)
            enc2 = checkpoint(self.enc2, self.pool1(enc1), use_reentrant=False)
            enc3 = checkpoint(self.enc3, self.pool2(enc2), use_reentrant=False)

            bottleneck = checkpoint(self.bottleneck, self.pool3(enc3), use_reentrant=False)

            up3 = self.up3(bottleneck)
            up3 = self._match_size(up3, enc3)
            dec3 = checkpoint(self.dec3, torch.cat([up3, enc3], dim=1), use_reentrant=False)

            up2 = self.up2(dec3)
            up2 = self._match_size(up2, enc2)
            dec2 = checkpoint(self.dec2, torch.cat([up2, enc2], dim=1), use_reentrant=False)

            up1 = self.up1(dec2)
            up1 = self._match_size(up1, enc1)
            dec1 = checkpoint(self.dec1, torch.cat([up1, enc1], dim=1), use_reentrant=False)
        else:
            enc1 = self.enc1(x)
            enc2 = self.enc2(self.pool1(enc1))
            enc3 = self.enc3(self.pool2(enc2))

            bottleneck = self.bottleneck(self.pool3(enc3))

            up3 = self.up3(bottleneck)
            up3 = self._match_size(up3, enc3)
            dec3 = self.dec3(torch.cat([up3, enc3], dim=1))

            up2 = self.up2(dec3)
            up2 = self._match_size(up2, enc2)
            dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

            up1 = self.up1(dec2)
            up1 = self._match_size(up1, enc1)
            dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        return self.final_conv(dec1)



    def _match_size(self, upsampled, encoder):
        # Dynamically pad or crop the depth axis to match
        diff = encoder.shape[-1] - upsampled.shape[-1]
        if diff > 0:
            padding = (0, diff)
            upsampled = F.pad(upsampled, padding)
        elif diff < 0:
            upsampled = upsampled[..., :encoder.shape[-1]]
        return upsampled
