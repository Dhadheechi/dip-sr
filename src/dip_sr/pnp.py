"""Plug-and-play image-space refinement for DIP super-resolution outputs."""

from dataclasses import dataclass, field

import numpy as np
import torch
from skimage.restoration import denoise_tv_chambolle

from .data import ImagePair, downsample
from .metrics import psnr


@dataclass
class PnPConfig:
    """Configuration for continuation-based PnP refinement."""

    outer_steps: int = 4
    inner_steps: int = 8
    data_step: float = 0.8
    denoise_relaxation: float = 0.8
    tv_weights: tuple[float, ...] = (0.08, 0.05, 0.03, 0.015)
    experiment_name: str = "dip-pnp-tv"
    wandb_project: str | None = None
    wandb_mode: str = "online"


@dataclass
class PnPResult:
    """Refined image and measurement-only history."""

    output: torch.Tensor
    history: list[dict[str, float]] = field(default_factory=list)


def _tv_denoise(image: torch.Tensor, weight: float) -> torch.Tensor:
    """Apply channel-wise TV denoising without using the HR reference."""
    array = image.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    denoised = denoise_tv_chambolle(array, weight=weight, channel_axis=-1)
    return torch.from_numpy(np.asarray(denoised, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)


def run_pnp_refinement(
    pair: ImagePair,
    initialization: torch.Tensor,
    config: PnPConfig | None = None,
) -> PnPResult:
    """Refine a DIP output using PnP data steps and a decreasing TV prior.

    The degradation operator is the same antialiased bicubic operator used to
    create ``pair.lr``. The HR target is never used by this optimization.
    """
    config = config or PnPConfig()
    if initialization.shape != pair.hr.shape:
        raise ValueError("initialization must have the same shape as pair.hr")
    if config.outer_steps < 1 or config.inner_steps < 1:
        raise ValueError("outer_steps and inner_steps must be positive")
    if len(config.tv_weights) != config.outer_steps:
        raise ValueError("tv_weights must contain one value per outer step")

    device = initialization.device
    target = pair.lr.to(device)
    current = initialization.detach().clone().clamp(0, 1)
    history: list[dict[str, float]] = []
    run = None
    if config.wandb_project:
        import wandb

        run = wandb.init(
            project=config.wandb_project,
            name=config.experiment_name,
            config=vars(config),
            mode=config.wandb_mode,
        )
    try:
        for outer_step, tv_weight in enumerate(config.tv_weights, start=1):
            for _ in range(config.inner_steps):
                current.requires_grad_(True)
                residual = downsample(current, target.shape[-2:]) - target
                gradient = torch.autograd.grad(
                    (residual.square().mean()),
                    current,
                )[0]
                data_consistent = (current - config.data_step * gradient).detach().clamp(0, 1)
                denoised = _tv_denoise(data_consistent, tv_weight).to(device)
                current = (
                    (1 - config.denoise_relaxation) * data_consistent
                    + config.denoise_relaxation * denoised
                ).clamp(0, 1)
            measurement = downsample(current, target.shape[-2:])
            metrics = {
                "outer_step": float(outer_step),
                "tv_weight": tv_weight,
                "lr_psnr": psnr(measurement, target),
            }
            history.append(metrics)
            if run:
                run.log(metrics, step=outer_step)
    finally:
        if run:
            run.finish()
    return PnPResult(output=current.cpu(), history=history)
