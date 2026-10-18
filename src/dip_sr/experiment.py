"""Experiment orchestration with optional W&B tracking."""

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .data import ImagePair, downsample
from .metrics import psnr
from .model import DIPGenerator


@dataclass
class ExperimentConfig:
    scale: int = 4
    iterations: int = 2_000
    learning_rate: float = 1e-2
    channels: int = 64
    seed: int = 42
    device: str | None = None
    log_every: int = 100
    experiment_name: str = "dip-baseline"
    wandb_project: str | None = None
    wandb_mode: str = "online"


@dataclass
class ExperimentResult:
    output: torch.Tensor
    history: list[dict[str, float]] = field(default_factory=list)
    config: ExperimentConfig | None = None


def _device(config: ExperimentConfig) -> torch.device:
    return torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))


def run_experiment(pair: ImagePair, config: ExperimentConfig) -> ExperimentResult:
    """Run DIP optimization and return the best-by-observation output history."""
    if pair.scale != config.scale:
        raise ValueError("pair.scale and config.scale must match")
    torch.manual_seed(config.seed)
    device = _device(config)
    model = DIPGenerator(config.scale, config.channels).to(device)
    noise = torch.rand(
        1,
        config.channels,
        pair.lr.shape[-2],
        pair.lr.shape[-1],
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    target = pair.lr.to(device)
    hr_target = pair.hr.to(device)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_output = None
    run: Any = None
    if config.wandb_project:
        import wandb

        run = wandb.init(project=config.wandb_project, name=config.experiment_name, config=vars(config), mode=config.wandb_mode)
    try:
        for iteration in range(1, config.iterations + 1):
            optimizer.zero_grad(set_to_none=True)
            output = model(noise)
            prediction = downsample(output, target.shape[-2:])
            loss = nn.functional.mse_loss(prediction, target)
            loss.backward()
            optimizer.step()
            if iteration == 1 or iteration % config.log_every == 0 or iteration == config.iterations:
                with torch.no_grad():
                    output = model(noise)
                    prediction = downsample(output, target.shape[-2:])
                    metrics = {
                        "iteration": float(iteration),
                        "loss": float(loss.item()),
                        "lr_psnr": psnr(prediction, target),
                        "hr_psnr": psnr(output, hr_target),
                    }
                    history.append(metrics)
                    if run:
                        run.log(metrics, step=iteration)
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best_output = output.detach().cpu()
    finally:
        if run:
            run.finish()
    if best_output is None:
        raise RuntimeError("training produced no output; iterations must be positive")
    return ExperimentResult(output=best_output, history=history, config=config)