"""Small DIP generator used as the baseline architecture."""

import torch
from torch import nn
from torch.nn import functional as F


class DIPGenerator(nn.Module):
    """A decoder-only generator mapping fixed noise to an HR image."""

    def __init__(self, scale: int = 4, channels: int = 64) -> None:
        super().__init__()
        if scale not in (2, 4, 8):
            raise ValueError("scale must be one of 2, 4, or 8")
        stages = {2: 1, 4: 2, 8: 3}[scale]
        blocks: list[nn.Module] = []
        in_channels = channels
        for _ in range(stages):
            blocks.extend(
                [nn.Conv2d(in_channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(True)]
            )
            in_channels = channels
        blocks.append(nn.Conv2d(channels, 3, 3, padding=1))
        self.decoder = nn.Sequential(*blocks)
        self.scale = scale

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        output = noise
        for _ in range({2: 1, 4: 2, 8: 3}[self.scale]):
            output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.decoder(output))