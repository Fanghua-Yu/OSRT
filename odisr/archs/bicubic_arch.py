import torch
import torch.nn as nn
import math
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class BICUBICUP(nn.Module):
    def __init__(self, upscale=2):
        super(BICUBICUP, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=upscale, mode='bicubic', align_corners=True)

    def forward(self, x):
        return self.up_sample(x)


if __name__ == '__main__':
    model = BICUBICUP(upscale=4)
    model(torch.rand([1, 3, 128, 256]))