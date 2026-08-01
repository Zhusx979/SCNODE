"""Deterministic, test-only image corruptions for synthetic robustness studies."""

from __future__ import annotations

import io
from typing import Callable
import zlib

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


CORRUPTION_NAMES = (
    "brightness", "contrast", "saturation", "hue", "gamma", "white_balance",
    "gaussian_blur", "gaussian_noise", "jpeg",
)


def _factor(severity: int, values: tuple[float, float, float]) -> float:
    if severity not in {1, 2, 3}:
        raise ValueError("severity must be an integer in [0, 3]")
    return values[severity - 1]


def build_corruption(name: str, severity: int, seed: int) -> Callable[[Image.Image], Image.Image]:
    """Build one deterministic PIL transform; severity zero is exactly identity."""
    if name not in CORRUPTION_NAMES:
        raise ValueError(f"unknown corruption {name!r}; choose from {CORRUPTION_NAMES}")
    if severity == 0:
        return lambda image: image.copy()
    def transform(image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if name == "brightness":
            return ImageEnhance.Brightness(image).enhance(_factor(severity, (0.8, 0.65, 0.5)))
        if name == "contrast":
            return ImageEnhance.Contrast(image).enhance(_factor(severity, (0.8, 0.65, 0.5)))
        if name == "saturation":
            return ImageEnhance.Color(image).enhance(_factor(severity, (0.8, 0.6, 0.4)))
        if name == "hue":
            shift = int(_factor(severity, (4, 8, 12)))
            hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
            hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 256
            return Image.fromarray(hsv, mode="HSV").convert("RGB")
        if name == "gamma":
            gamma = _factor(severity, (1.2, 1.5, 1.8))
            pixels = np.asarray(image, dtype=np.float32) / 255.0
            return Image.fromarray(np.uint8(np.clip(pixels ** gamma, 0, 1) * 255))
        if name == "white_balance":
            scales = ((1.10, 0.96, 0.90), (1.20, 0.92, 0.80), (1.30, 0.88, 0.70))[severity - 1]
            pixels = np.asarray(image, dtype=np.float32) * np.asarray(scales)
            return Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))
        if name == "gaussian_blur":
            return image.filter(ImageFilter.GaussianBlur(_factor(severity, (0.6, 1.2, 2.0))))
        if name == "gaussian_noise":
            sigma = _factor(severity, (5.0, 12.0, 24.0))
            pixels = np.asarray(image, dtype=np.float32)
            # Derive a stable per-image seed. This stays reproducible across
            # DataLoader workers while avoiding one identical noise field for
            # every test image.
            image_seed = (seed + zlib.crc32(pixels.tobytes())) % (2 ** 32)
            rng = np.random.default_rng(image_seed)
            noise = rng.normal(0.0, sigma, size=pixels.shape)
            return Image.fromarray(np.uint8(np.clip(pixels + noise, 0, 255)))
        quality = int(_factor(severity, (70, 45, 25)))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as restored:
            return restored.convert("RGB").copy()

    return transform
