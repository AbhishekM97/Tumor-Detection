'''Import the main torch vision library and nn which contains nerual network building blocks (layers, activations, etc.)'''
import torch
import torch.nn as nn

class UNET3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, init_features=32):
        super(UNet3D, self).__init__()

        #self.encoder1 learns spatial features from the input volume
        #self.pool1 -= havles the spatial resolution to learn hierarchial features (downsampling).
        features = init_features
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        #Second encoder block:
        #Doubles feature count (deeper layers learn more complex patterns).
        #Another max pooling to further reduce spatial size.
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        ### Deepest part of the U-Net )most abtstract feautres).
        #  Does not reduce size further, just transforms features.
        # Usually has the most filters. 
        # ###

        self.bottleneck = self.block(features * 2, features * 4)

        ###
        # ConvTranspose3d: performs upsampling (opposite of pooling).
        # After upsampling, we concatenate the corresponding encoder output (skip connection) -- hence features * 4 input to decoder2.
        # ###

        self.up2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 4, features * 2)

        ###
        # Continue upsampling and decoding.
        # Each decoder stage learns how to reconstruct finer spatial details using skip connections.
        # ###

        self.up1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)

        ###
        # FINAL OUTPUT LAYER
        # 1x1x1 convulution that maps from features to out_channels (e.g., 1 class --> binary mask
        # 
        # This is the segmentation output.###

        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)


    ###
    # FORWARD PASS
    # 
    # Pass input x through encoder blocks, storing outputs (for skip connections).
    # Go down into bottleneck after reducing spatial size.
    # ###

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(x)

        bottleneck = self.bottleneck(self.pool2(enc2))

        ###
        # Upsample from bottleneck.
        # Concatenate with the encoder's output (skip connection).
        # Pass through decoder block. 
        # ###

        dec2 = self.up2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        ###
        # Repeat for the next level up.
        # Final out passes through a 1x1x1 conv --> sigmoid --> binary mask (values in range [0, 1]).
        # ###

        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))
    

###
# _block METHOD
# Defines a reusable block of:
# 2 convolution layers.
# Batch normalization (for faster, stable training)
# reLU activation.
# These are common in U-Net to extract complex features.
# ###


def _block(self, in_channels, out_channels):

    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )


