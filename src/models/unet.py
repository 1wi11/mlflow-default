# src/models/unet.py
import torch.nn as nn
from src.core.registries import MODELS

def _conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(True),
        nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(True)
    )

@MODELS.register("unet")
class MiniUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=16, **_):
        super().__init__()
        self.e1 = _conv(in_channels, base_channels); self.p1 = nn.MaxPool2d(2)
        self.e2 = _conv(base_channels, base_channels*2); self.p2 = nn.MaxPool2d(2)
        self.b  = _conv(base_channels*2, base_channels*4)
        self.u2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, 2)
        self.d2 = _conv(base_channels*4, base_channels*2)
        self.u1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, 2)
        self.d1 = _conv(base_channels*2, base_channels)
        self.out = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.p1(e1)); b = self.b(self.p2(e2))
        d2 = self.d2(nn.functional.interpolate(self.u2(b), size=e2.shape[-2:]))
        d2 = self.d2(nn.functional.relu(nn.concat([d2, e2], 1))) if False else nn.functional.relu(d2)  # 단순화
        d1 = self.d1(nn.functional.interpolate(self.u1(d2), size=e1.shape[-2:]))
        return self.out(d1)
