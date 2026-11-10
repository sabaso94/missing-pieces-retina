"""Cached images in memory, and augmentation on the GPU.

A whole site fits in RAM as uint8 (EyePACS: 35,126 x 3 x 384 x 384 bytes = 15.5 GB), so images are decoded
once, in parallel, and batches are cut from one tensor. Augmentation runs on the GPU on whole batches, which
keeps the data path out of the way of training.
"""
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from retina_shift.preprocess import IMAGE_SIZE
from retina_shift.splits import ROOT

CACHE_DIR = ROOT / "data" / "cache"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _decode(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).transpose(2, 0, 1)


def load_images(site: str, image_ids, workers: int = 32) -> torch.Tensor:
    """(N, 3, 384, 384) uint8 tensor of the cached images, in the order of image_ids."""
    paths = [str(CACHE_DIR / site / f"{i}.jpg") for i in image_ids]
    out = np.empty((len(paths), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    with Pool(workers) as pool:
        for k, array in enumerate(pool.imap(_decode, paths, chunksize=64)):
            out[k] = array
    return torch.from_numpy(out)


def normalise(x: torch.Tensor) -> torch.Tensor:
    """uint8 images -> float, scaled with the ImageNet mean and std the pretrained backbone expects."""
    x = x.float() / 255.0
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def gpu_augment(x: torch.Tensor, generator: torch.Generator, max_degrees: float = 15.0,
                jitter: float = 0.1) -> torch.Tensor:
    """Random flips, a rotation of up to +-15 degrees, and +-10% brightness and contrast, per image.

    Fundus photos have no fixed orientation between left and right eyes, so flips are safe; small rotations
    and exposure changes mimic differences in how a photo is taken. Random numbers come from `generator`
    (on the CPU), so a seed fixes the augmentation whatever the device.
    """
    n = len(x)
    x = x.float() / 255.0
    flip_h = torch.rand(n, generator=generator) < 0.5
    flip_v = torch.rand(n, generator=generator) < 0.5
    angle = (torch.rand(n, generator=generator) * 2 - 1) * max_degrees * torch.pi / 180
    brightness = 1 + (torch.rand(n, generator=generator) * 2 - 1) * jitter
    contrast = 1 + (torch.rand(n, generator=generator) * 2 - 1) * jitter

    cos, sin = torch.cos(angle), torch.sin(angle)
    sx = torch.where(flip_h, -1.0, 1.0)
    sy = torch.where(flip_v, -1.0, 1.0)
    theta = torch.stack([torch.stack([cos * sx, -sin * sy, torch.zeros(n)], dim=1),
                         torch.stack([sin * sx, cos * sy, torch.zeros(n)], dim=1)], dim=1).to(x.device)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * contrast.view(n, 1, 1, 1).to(x.device) + mean
    x = (x * brightness.view(n, 1, 1, 1).to(x.device)).clamp(0, 1)
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
