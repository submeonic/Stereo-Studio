"""Extensible render styles, kept separate from photographic corrections."""

from dataclasses import dataclass

import cv2
import numpy as np

from .editing import linear_rgb, srgb


@dataclass
class Style:
    name: str = "clean"
    amount: float = 0.65
    separation: float = 0.5
    chromatic: float = 6.0  # pixels at working resolution
    tear: float = 0.2

    def validate(self):
        if self.name not in STYLE_RENDERERS:
            raise ValueError("Unknown processing style.")
        for key, high in [("amount", 1), ("separation", 1), ("chromatic", 40), ("tear", 1)]:
            value = getattr(self, key)
            if not np.isfinite(value) or not 0 <= value <= high:
                raise ValueError(f"Invalid style {key}.")


def stereo_echo(a, anchor, t, frame, style):
    from .render import splat

    sign = a.metadata.get("alignment", {}).get("depth_order_sign", 1)
    spread = style.separation * 0.35
    lt = max(0, t - spread)
    rt = min(1, t + spread)
    left, _, lv = splat(a.left, a.dl, a.support_l, lt, anchor.disparity, depth_sign=sign)
    right, _, rv = splat(a.right, a.dr, a.support_r, rt, anchor.disparity, True, sign)
    left[~lv] = frame[~lv]
    right[~rv] = frame[~rv]
    double = (left + right) * 0.5
    h, w = frame.shape[:2]
    yy, xx = np.indices((h, w), np.float32)
    offset = style.chromatic
    red = cv2.remap(
        left[:, :, 0], xx - offset, yy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE
    )
    blue = cv2.remap(
        right[:, :, 2], xx + offset, yy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE
    )
    echo = np.stack((red, double[:, :, 1], blue), axis=-1)
    # Emphasize actual geometry boundaries/uncertainty; keep quieter areas readable.
    edges = np.maximum(
        abs(cv2.Sobel(a.dl, cv2.CV_32F, 1, 0)), abs(cv2.Sobel(a.dl, cv2.CV_32F, 0, 1))
    )
    accent = np.clip(edges / 8 + (~a.valid_l) * 0.5, 0, 1)
    opacity = style.amount * (0.4 + 0.6 * accent)
    result = frame * (1 - opacity[..., None]) + echo * opacity[..., None]
    # Rows move as a function of viewpoint, not wall clock / previous frame: loop-safe.
    bands = np.floor(yy / max(4, h / 28))
    gate = np.sin(bands * 2.399) > 0.45
    shift = style.tear * w * 0.045 * np.sin(bands * 1.7 + t * 2 * np.pi) * gate
    result = cv2.remap(
        result.astype(np.float32),
        xx + shift.astype(np.float32),
        yy,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return np.rint(result).clip(0, 255).astype(np.uint8)


STYLE_RENDERERS = {"clean": None, "stereo-echo": stereo_echo}
STYLE_LABELS = {
    "clean": "Clean photography",
    "stereo-echo": "Stereo Echo · glitch / double exposure",
}


def apply_style(a, anchor, t, frame, style):
    style.validate()
    effect = STYLE_RENDERERS[style.name]
    return frame if effect is None or style.amount == 0 else effect(a, anchor, t, frame, style)


def anaglyph(left, right, method="dubois"):
    if method == "color":
        return np.dstack((left[:, :, 0], right[:, :, 1], right[:, :, 2]))
    if method == "gray":
        left_gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        r = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
        return np.dstack((left_gray, r, r))
    if method != "dubois":
        raise ValueError("Unknown anaglyph method.")
    # Dubois published red/cyan projection coefficients (CRT/plasma reference).
    # Combine in linear light. A generic preset, not calibration for a user's glasses/display.
    lm = np.float32([[0.437, 0.449, 0.164], [-0.062, -0.062, -0.024], [-0.048, -0.050, -0.017]])
    rm = np.float32([[-0.011, -0.032, -0.007], [0.377, 0.761, 0.009], [-0.026, -0.093, 1.234]])
    return srgb(linear_rgb(left) @ lm.T + linear_rgb(right) @ rm.T)
