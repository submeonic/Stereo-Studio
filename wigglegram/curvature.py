"""Fixed safe curvature framing, sampled before final output downsampling."""

import cv2
import numpy as np


def safe_scale(amount):
    # Largest symmetric domain whose distorted corners remain inside the image.
    lo, hi = 0.0, 1.0
    for _ in range(32):
        mid = (lo + hi) / 2
        if mid * (1 + 0.36 * amount * mid * mid) <= 1:
            lo = mid
        else:
            hi = mid
    return lo


def coordinates(nx, ny, amount):
    scale = safe_scale(amount)
    nx, ny = nx * scale, ny * scale
    distortion = 1 + 0.18 * amount * (nx * nx + ny * ny)
    return nx * distortion, ny * distortion


def curve(image, amount):
    if amount <= 0:
        return image
    h, w = image.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)
    nx, ny = coordinates(2 * x / max(1, w - 1) - 1, 2 * y / max(1, h - 1) - 1, amount)
    return cv2.remap(
        image,
        ((nx + 1) * (w - 1) / 2).astype(np.float32),
        ((ny + 1) * (h - 1) / 2).astype(np.float32),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
