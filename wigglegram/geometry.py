"""Conservative, reusable edge refinement and independent image/render grids."""

import hashlib
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .core import align_pair, read_pair


def reliability(a, eye="l"):
    d, other = (a.dl, a.dr) if eye == "l" else (a.dr, a.dl)
    valid = getattr(a, "valid_" + eye)
    y, x = np.indices(d.shape, dtype=np.float32)
    sampled = cv2.remap(other, x - d, y, cv2.INTER_LINEAR)
    # Reliability is evidence, never promoted by filling or smoothing.
    return (np.exp(-np.square(d + sampled) / 2) * valid).astype(np.float32)


def refine_edge(d, image, valid, confidence, radius=2):
    """Choose a nearby supported surface, rather than averaging across its boundary.

    Only uncertain pixels change. A color/spatial cost selects a reliable neighbor;
    high-confidence geometry is preserved, and validity is not inflated.
    """
    h, w = d.shape
    pad = radius
    colors = image.astype(np.float32) / 255
    dc = np.pad(d, pad, mode="edge")
    cc = np.pad(colors, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    vc = np.pad(valid, pad, constant_values=False)
    best = np.full((h, w), np.inf, np.float32)
    result = d.copy()
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            ys, xs = slice(pad + dy, pad + dy + h), slice(pad + dx, pad + dx + w)
            color_cost = np.sum((cc[ys, xs] - colors) ** 2, axis=2)
            # Weak preference for the original estimate, no smoothing of correct surfaces.
            score = color_cost + 0.003 * (dx * dx + dy * dy)
            score += 0.015 * np.minimum(abs(dc[ys, xs] - d), 3)
            score += 0.03 * (dx == 0 and dy == 0)
            use = vc[ys, xs] & (score < best) & (confidence < 0.65)
            result[use] = dc[ys, xs][use]
            best[use] = score[use]
    return result


def refine_analysis(a, enabled=True, cache_dir=".cache/wigglegram", cancelled=lambda: False):
    if not enabled or a.metadata.get("backend") == "sparse":
        return a
    key = a.metadata.get("cache_key")
    cache = Path(cache_dir) / f"{key}-edges-v1.npz" if cache_dir and key else None
    if cache and cache.exists():
        with np.load(cache) as z:
            dl, dr = z["dl"], z["dr"]
    else:
        dl = refine_edge(a.dl, a.left, a.valid_l, reliability(a))
        if cancelled():
            raise InterruptedError("Cancelled")
        dr = refine_edge(a.dr, a.right, a.valid_r, reliability(a, "r"))
        if cache:
            import os
            import tempfile

            cache.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(dir=cache.parent, suffix=".npz")
            os.close(fd)
            try:
                np.savez_compressed(name, dl=dl, dr=dr)
                os.replace(name, cache)
            finally:
                Path(name).unlink(missing_ok=True)
    return replace(
        a, dl=dl, dr=dr, metadata={**a.metadata, "edge_refinement": "surface-selection-v1"}
    )


def render_grid(a, anchor, source, right=None, swap=False, profile="auto", width=960):
    """Re-read original colors; reuse geometry without rerunning stereo matching."""
    left, right_image, info = read_pair(source, right, swap, width)
    h, w = left.shape[:2]
    ah, aw = a.dl.shape
    sx, sy = w / aw, h / ah
    alignment = dict(a.metadata["alignment"])
    if alignment["type"] == "nominal":
        matrix = np.vstack((alignment["right_affine"], [0, 0, 1]))
        scale = np.diag([sx, sy, 1])
        matrix = (scale @ matrix @ np.linalg.inv(scale))[:2].astype(np.float32)
        right_image = cv2.warpAffine(right_image, matrix, (w, h), flags=cv2.INTER_LINEAR)
        sl = np.ones((h, w), bool)
        sr = cv2.warpAffine(np.ones((h, w), np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST) > 0
        alignment["right_affine"] = matrix.tolist()
        if alignment.get("vertical_median_px") is not None:
            alignment["vertical_median_px"] *= sy
    else:
        left, right_image, sl, sr, alignment = align_pair(left, right_image, profile, swap)

    def resize(value):
        return cv2.resize(value.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

    dl, dr = resize(a.dl) * sx, resize(a.dr) * sx
    vl, vr = resize(a.valid_l) > 0.5, resize(a.valid_r) > 0.5
    if a.metadata.get("edge_refinement") and sx > 1:
        # Coarse colors provide the surface evidence; full-resolution colors locate the edge.
        dl, guided_vl = guided_upsample(a.dl, a.left, left, a.valid_l)
        dr, guided_vr = guided_upsample(a.dr, a.right, right_image, a.valid_r)
        dl *= sx
        dr *= sx
        vl &= guided_vl
        vr &= guided_vr
    meta = {
        **a.metadata,
        **info,
        "alignment": alignment,
        "geometry_size": [aw, ah],
        "render_scale": sx,
        "source_color_detail": True,
    }
    # The model cache key identifies analysis, not this derived grid.
    meta["render_grid_key"] = hashlib.sha256(
        f"{a.metadata.get('cache_key')}/{w}/{h}".encode()
    ).hexdigest()[:16]
    return replace(
        a,
        left=left,
        right=right_image,
        dl=dl,
        dr=dr,
        valid_l=vl,
        valid_r=vr,
        support_l=sl,
        support_r=sr,
        metadata=meta,
    ), replace(anchor, x=anchor.x * sx, y=anchor.y * sy, disparity=anchor.disparity * sx)


def guided_upsample(disparity, guide, target, valid=None):
    """Joint color-guided surface selection, with smooth interpolation inside a surface.

    Avoids blurring foreground disparity into background. This can align a coarse
    boundary with a visible color edge, but cannot recover absent stereo evidence.
    """
    h, w = target.shape[:2]
    gh, gw = disparity.shape
    y, x = np.indices((h, w), dtype=np.float32)
    x = (x + 0.5) * gw / w - 0.5
    y = (y + 0.5) * gh / h - 0.5
    ix, iy = np.rint(x).astype(int), np.rint(y).astype(int)
    target_color = target.astype(np.float32) / 255
    colors = guide.astype(np.float32) / 255
    nearest = cv2.resize(disparity, (w, h), interpolation=cv2.INTER_NEAREST)
    best = np.full((h, w), np.inf, np.float32)
    winner = nearest.copy()
    winner_valid = np.ones((h, w), bool)
    # Select the surface from a small local coarse neighborhood.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx, cy = np.clip(ix + dx, 0, gw - 1), np.clip(iy + dy, 0, gh - 1)
            cost = np.sum((colors[cy, cx] - target_color) ** 2, axis=2)
            cost += 0.025 * ((cx - x) ** 2 + (cy - y) ** 2)
            use = cost < best
            winner[use] = disparity[cy, cx][use]
            if valid is not None:
                winner_valid[use] = valid[cy, cx][use]
            best[use] = cost[use]
    # Interpolate subpixel disparity only within the selected surface.
    total = np.zeros((h, w), np.float32)
    weights = np.zeros_like(total)
    invalid_weights = np.zeros_like(total)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx, cy = np.clip(ix + dx, 0, gw - 1), np.clip(iy + dy, 0, gh - 1)
            d = disparity[cy, cx]
            color_error = np.sum((colors[cy, cx] - target_color) ** 2, axis=2)
            weight = np.exp(-color_error / 0.02 - ((cx - x) ** 2 + (cy - y) ** 2) / 2)
            weight *= abs(d - winner) < 0.75
            total += weight * d
            weights += weight
            if valid is not None:
                invalid_weights += weight * (~valid[cy, cx])
    result = np.where(weights > 1e-8, total / np.maximum(weights, 1e-8), winner).astype(np.float32)
    if valid is None:
        return result
    return result, winner_valid & (invalid_weights < 1e-6)
