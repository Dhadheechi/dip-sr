"""Metrics used by the notebook and W&B runs."""

import math

import torch


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Return PSNR in dB for tensors whose values are in [0, 1]."""
    mse = torch.mean((prediction.detach().float() - target.detach().float()) ** 2).item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)