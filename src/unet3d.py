'''Import the main torch vision library and nn which contains nerual network building blocks (layers, activations, etc.)'''
import torch
import torch.nn as nn

class UNET3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, init_features=32):
    