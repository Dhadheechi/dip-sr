"""Image preparation and differentiable degradation utilities."""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


@dataclass(frozen=True)
class ImagePair:
    """A high-resolution target and its low-resolution observation."""

    hr: torch.Tensor
    lr: torch.Tensor
    scale: int


def _to_tensor(image: Image.Image | np.ndarray) -> torch.Tensor:
    array = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    if array.ndim != 3 or array.shape[-1] not in (1, 3):
        raise ValueError("image must have shape (height, width, channels)")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    tensor = torch.from_numpy(array.copy()).float()
    if tensor.max() > 1:
        tensor = tensor / 255.0
    return tensor.permute(2, 0, 1).unsqueeze(0).contiguous()


def make_image_pair(image: Image.Image | np.ndarray, scale: int = 4) -> ImagePair:
    """Create an aligned HR/LR pair using antialiased bicubic degradation."""
    if scale < 2:
        raise ValueError("scale must be at least 2")
    hr = _to_tensor(image)
    height, width = hr.shape[-2:]
    height -= height % scale
    width -= width % scale
    if min(height, width) < scale:
        raise ValueError("scale is larger than the input image")
    hr = hr[..., :height, :width]
    lr_size = (height // scale, width // scale)
    lr = resize(hr, lr_size, InterpolationMode.BICUBIC, antialias=True).clamp(0, 1)
    return ImagePair(hr=hr, lr=lr, scale=scale)


def downsample(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Apply the same differentiable degradation used to create the observation."""
    return resize(image, size, InterpolationMode.BICUBIC, antialias=True).clamp(0, 1)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a BCHW/CHW tensor to an HWC uint8 array for plotting or logging."""
    image = tensor.detach().cpu().squeeze(0).clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)