"""Non-destructive, shared-eye photo adjustments and output crop coordinates."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PhotoEdits:
    exposure: float = 0.0  # stops, in linear light
    temperature: float = 0.0  # -100 cool .. +100 warm
    tint: float = 0.0  # -100 green .. +100 magenta
    saturation: float = 1.0
    contrast: float = 1.0
    sharpness: float = 0.0
    auto_color: bool = False
    crop: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    def validate(self):
        for name, low, high in [
            ("exposure", -4, 4),
            ("temperature", -100, 100),
            ("tint", -100, 100),
            ("saturation", 0, 2),
            ("contrast", 0.5, 1.5),
            ("sharpness", 0, 2),
        ]:
            value = getattr(self, name)
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}.")
        if len(self.crop) != 4 or not all(np.isfinite(self.crop)):
            raise ValueError("Crop needs four finite normalized coordinates.")
        x0, y0, x1, y1 = self.crop
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("Crop must be an ordered region within 0..1.")


def linear_rgb(image):
    color = image.astype(np.float32) / 255
    return np.where(color <= 0.04045, color / 12.92, ((color + 0.055) / 1.055) ** 2.4)


def srgb(linear):
    color = np.clip(linear, 0, 1)
    color = np.where(color <= 0.0031308, color * 12.92, 1.055 * color ** (1 / 2.4) - 0.055)
    return np.rint(color * 255).clip(0, 255).astype(np.uint8)


def balance_gains(reference):
    """Estimate once from the left eye, never independently per eye or animation frame."""
    small = cv2.resize(
        reference, (160, max(1, round(reference.shape[0] * 160 / reference.shape[1])))
    )
    rgb = linear_rgb(small)
    luminance = rgb.mean(axis=2)
    keep = (luminance > 0.02) & (luminance < 0.85)
    if keep.sum() < 20:
        return np.ones(3, np.float32)
    average = rgb[keep].mean(axis=0)
    return np.clip(average.mean() / np.maximum(average, 0.01), 0.6, 1.6)


def adjust_photo(image, edits, gains=None):
    edits.validate()
    if not any(
        (
            edits.exposure,
            edits.temperature,
            edits.tint,
            edits.sharpness,
            edits.saturation != 1,
            edits.contrast != 1,
            edits.auto_color,
        )
    ):
        return image.copy()
    color = linear_rgb(image)
    gains = np.ones(3, np.float32) if gains is None else gains
    wb = np.exp(
        np.array(
            [
                edits.temperature / 250 + edits.tint / 500,
                -edits.tint / 250,
                -edits.temperature / 250 + edits.tint / 500,
            ],
            np.float32,
        )
    )
    color *= wb * (gains if edits.auto_color else 1) * 2**edits.exposure
    luminance = color @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    color = luminance[..., None] + edits.saturation * (color - luminance[..., None])
    color = (color - 0.18) * edits.contrast + 0.18
    output = srgb(color)
    if edits.sharpness:
        base = output.astype(np.float32)
        detail = base - cv2.GaussianBlur(base, (0, 0), 1.0)
        detail[abs(detail) < 2] = 0
        output = np.rint(base + edits.sharpness * detail).clip(0, 255).astype(np.uint8)
    return output


def crop_pixels(shape, crop):
    h, w = shape[:2]
    x0, y0, x1, y1 = crop
    x0, y0 = int(round(x0 * w)), int(round(y0 * h))
    x1, y1 = min(w, int(round(x1 * w))), min(h, int(round(y1 * h)))
    # Stable even output sizes also work for H.264 yuv420p.
    x1 = x0 + (x1 - x0) // 2 * 2
    y1 = y0 + (y1 - y0) // 2 * 2
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise ValueError("Crop must retain at least 16 × 16 pixels.")
    return x0, y0, x1, y1


def ratio_crop(crop, shape, aspect):
    """Fit a centered ratio rectangle inside the current normalized crop."""
    if aspect is None:
        return tuple(crop)
    h, w = shape[:2]
    x0, y0, x1, y1 = crop
    cw = (x1 - x0) * w
    ch = (y1 - y0) * h
    if cw / ch > aspect:
        cw = ch * aspect
    else:
        ch = cw / aspect
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    return cx - cw / w / 2, cy - ch / h / 2, cx + cw / w / 2, cy + ch / h / 2
