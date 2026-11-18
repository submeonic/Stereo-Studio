from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np

from .core import Analysis, Anchor
from .editing import PhotoEdits, adjust_photo, balance_gains, crop_pixels
from .effects import Style, anaglyph, apply_style


@dataclass
class Motion:
    mode: str = "smooth"
    strength: float = 0.65
    views: int = 7
    fps: int = 20
    hold_ms: int = 180
    cycles: int = 4
    easing: bool = True
    rotation: int = 0
    antialias: bool = False
    edge_cleanup: bool = False
    anaglyph_method: str = "dubois"
    fusion: str = "symmetric"
    repair_edges: bool = True

    def validate(self):
        if self.fusion not in ("symmetric", "legacy", "left", "right"):
            raise ValueError("Unknown fusion method.")
        if self.mode not in ("smooth", "classic", "anaglyph"):
            raise ValueError("Unknown animation mode.")
        if not 0 <= self.strength <= 1:
            raise ValueError("Sweep strength must be between 0 and 1.")
        if not 2 <= self.views <= 121:
            raise ValueError("Use 2–121 views.")
        if not 1 <= self.fps <= 60:
            raise ValueError("FPS must be between 1 and 60.")
        if not 20 <= self.hold_ms <= 5000:
            raise ValueError("Hold must be between 20 and 5000 ms.")
        if not 1 <= self.cycles <= 100:
            raise ValueError("Cycles must be between 1 and 100.")
        if self.anaglyph_method not in ("dubois", "color", "gray"):
            raise ValueError("Unknown anaglyph method.")
        if self.rotation not in (0, 90, 180, 270):
            raise ValueError("Rotation must be 0, 90, 180, or 270.")


def view_positions(m: Motion):
    m.validate()
    if m.mode == "anaglyph":
        return np.array([0.5])
    if m.mode == "classic":
        return np.array([0.0, 1.0])
    t = np.linspace(0, 1, m.views)
    if m.easing:
        t = (1 - np.cos(np.pi * t)) / 2
    return 0.5 + (t - 0.5) * m.strength


def loop_indices(m: Motion):
    if m.mode == "anaglyph":
        return [0]
    if m.mode == "classic":
        return [0, 1]
    return list(range(m.views)) + list(range(m.views - 2, 0, -1))


def splat(
    image,
    disparity,
    support,
    t,
    anchor_d,
    from_right=False,
    depth_sign=1,
    reliable=None,
    depth_scale=1,
):
    """Forward splat with subpixel weights and a foreground z-buffer."""
    h, w = disparity.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    factor = (1 - t) if from_right else t
    # Both sources project to the same anchored intermediate coordinate system.
    dest = xx - factor * disparity + t * anchor_d
    depth = disparity * (-depth_sign if from_right else depth_sign)
    base = np.floor(dest).astype(np.int32)
    fraction = dest - base
    z = np.full(h * w, -np.inf, np.float32)
    records = []
    for offset, weight in ((0, 1 - fraction), (1, fraction)):
        target = base + offset
        valid = support & np.isfinite(disparity) & (target >= 0) & (target < w) & (weight > 1e-4)
        sy, sx = np.where(valid)
        index = sy * w + target[sy, sx]
        depths = depth[sy, sx]
        np.maximum.at(z, index, depths)
        records.append((sy, sx, index, depths, weight[sy, sx]))
    color = np.zeros((h * w, 3), np.float32)
    total = np.zeros(h * w, np.float32)
    confidence_sum = np.zeros(h * w, np.float32)
    for sy, sx, index, depths, weights in records:
        keep = depths >= z[index] - 0.75 * depth_scale
        idx = index[keep]
        weights = weights[keep].astype(np.float32)
        np.add.at(total, idx, weights)
        if reliable is not None:
            np.add.at(confidence_sum, idx, weights * reliable[sy[keep], sx[keep]])
        for c in range(3):
            np.add.at(color[:, c], idx, image[sy[keep], sx[keep], c] * weights)
    result = (
        (color / np.maximum(total[:, None], 1e-6)).reshape(h, w, 3),
        z.reshape(h, w),
        total.reshape(h, w) > 0,
    )
    if reliable is not None:
        return (*result, (confidence_sum / np.maximum(total, 1e-6)).reshape(h, w) > 0.5)
    return result


def fill_small_holes(image, valid, depth, max_gap=12):
    """Scanline background extension; no invented textures or random framewise fill."""
    h, w = valid.shape
    columns = np.broadcast_to(np.arange(w), (h, w))
    before = np.maximum.accumulate(np.where(valid, columns, -w), axis=1)
    after = np.minimum.accumulate(np.where(valid, columns, 2 * w)[:, ::-1], axis=1)[:, ::-1]
    rows = np.arange(h)[:, None]
    b = np.clip(before, 0, w - 1)
    e = np.clip(after, 0, w - 1)
    use_before = depth[rows, b] <= depth[rows, e]
    nearest = np.where(use_before, b, e)
    eligible = (~valid) & (before >= 0) & (after < w) & ((after - before - 1) <= max_gap)
    result = image.copy()
    result[eligible] = image[rows, nearest][eligible]
    return result, valid | eligible


def render_view(
    a: Analysis,
    anchor: Anchor,
    t: float,
    edge_cleanup=False,
    fusion="symmetric",
    repair_edges=True,
    diagnostics=False,
):
    if (t <= 0 or t >= 1) and not diagnostics:
        frame, mask = classic_frame(a, anchor, 0 if t <= 0 else 1)
        return frame, mask, mask
    sign = a.metadata.get("alignment", {}).get("depth_order_sign", 1)
    scale = a.metadata.get("render_scale", 1)
    left_color, zl, vl, cl = splat(
        a.left,
        a.dl,
        a.support_l,
        t,
        anchor.disparity,
        depth_sign=sign,
        reliable=a.valid_l,
        depth_scale=scale,
    )
    r, zr, vr, cr = splat(
        a.right,
        a.dr,
        a.support_r,
        t,
        anchor.disparity,
        True,
        sign,
        reliable=a.valid_r,
        depth_scale=scale,
    )
    both = vl & vr
    delta = abs(np.where(both, zl, 0) - np.where(both, zr, 0))
    compatible = both & cl & cr & (delta < 1.5 * scale)
    color_error = np.max(abs(left_color - r), axis=2)
    disagreement = both & (color_error > 24)
    if edge_cleanup:
        compatible &= ~disagreement
    if fusion == "legacy":
        take_left = vl & (~vr | (~cr) | (cl & (zl >= zr)))
        if edge_cleanup:
            take_left |= disagreement & cl & cr & (delta < 1.5 * scale)
        weight = np.where(take_left, 0.0, 1.0).astype(np.float32)
    else:
        # Reliable visibility wins. Unknown/unknown is resolved by viewpoint,
        # never by a permanent reference eye. Same-depth ties are symmetric.
        take_left = vl & (~vr | (cl & ~cr) | (cl & cr & (zl > zr)))
        weight = np.where(take_left, 0.0, 1.0).astype(np.float32)
        unknown = both & ~cl & ~cr
        weight[unknown] = t  # continuous fallback; no new midpoint source-switch pop
        if edge_cleanup:
            ties = both & cl & cr & disagreement & (delta < 1.5 * scale)
            weight[ties] = 0 if t < 0.5 else (1 if t > 0.5 else 0.5)
        if fusion in ("left", "right"):
            weight[unknown | (both & disagreement)] = float(fusion == "right")
    weight[compatible] = t
    if fusion in ("left", "right"):
        weight[both & (disagreement | (~cl & ~cr))] = float(fusion == "right")
    # No abrupt switch from fused pixels to a raw endpoint: ease the last
    # 15% toward the corresponding camera's own projected sample.
    if fusion != "legacy":
        endpoint = np.clip((abs(t - 0.5) - 0.35) / 0.15, 0, 1)
        endpoint = endpoint * endpoint * (3 - 2 * endpoint)
        if t < 0.5:
            weight[vl] *= 1 - endpoint
        else:
            weight[vr] += (1 - weight[vr]) * endpoint
    result = left_color * (1 - weight[..., None]) + r * weight[..., None]
    valid = vl | vr
    depth = np.where(weight < 0.5, zl, zr)
    trusted = ((weight < 0.01) & cl & vl) | ((weight > 0.99) & cr & vr) | (cl & cr & both)
    # Fill existing holes first, so a local repair cannot spread indirectly
    # through the hole filler into pixels outside its explicit repair mask.
    result, filled = fill_small_holes(
        result, valid, depth, max_gap=max(4, round(a.left.shape[1] * 0.012))
    )
    # Low confidence alone is not a repair trigger. Only conflicting,
    # unsupported edge pixels are candidates; sound image interiors are intact.
    suspicious = both & ~cl & ~cr & disagreement & (delta > 1.5 * scale)
    repaired = np.zeros(valid.shape, bool)
    if repair_edges and fusion != "legacy":
        result, repaired = repair_unsupported_edges(
            result, depth, trusted, suspicious, np.minimum(zl, zr), scale
        )
        # Preserve exact camera limits; repairs vanish smoothly at the endpoints.
        fade = 1 - np.clip((abs(t - 0.5) - 0.35) / 0.15, 0, 1)
        original = left_color * (1 - weight[..., None]) + r * weight[..., None]
        result[repaired] = (result * fade + original * (1 - fade))[repaired]
    result[~filled] = 0
    frame = np.clip(result, 0, 255).astype(np.uint8)
    if diagnostics:
        return (
            frame,
            valid,
            filled,
            {
                "left": left_color.clip(0, 255).astype(np.uint8),
                "right": r.clip(0, 255).astype(np.uint8),
                "right_weight": weight,
                "disagreement": color_error,
                "repair": repaired,
                "unsupported": suspicious,
            },
        )
    return frame, valid, filled


def repair_unsupported_edges(image, depth, trusted, suspicious, background, scale):
    """Copy a nearby supported background sample only into flagged edge errors.

    No global convolution, no random patch selection, and no recursive use of
    repaired pixels as evidence. A small radius limits silhouette erosion.
    """
    h, w = depth.shape
    if not suspicious.any():
        return image.copy(), np.zeros(depth.shape, bool)
    radius = max(2, min(6, round(3 * scale)))
    result = image.copy()
    best = np.full((h, w), np.inf, np.float32)
    repaired = np.zeros((h, w), bool)
    for distance in range(1, radius + 1):
        for dy, dx in ((0, distance), (0, -distance), (distance, 0), (-distance, 0)):
            ys, xs = slice(max(0, dy), min(h, h + dy)), slice(max(0, dx), min(w, w + dx))
            sy, sx = slice(max(0, -dy), min(h, h - dy)), slice(max(0, -dx), min(w, w - dx))
            target = background[ys, xs]
            finite = np.isfinite(depth[sy, sx]) & np.isfinite(target)
            error = np.full(target.shape, np.inf, np.float32)
            np.subtract(depth[sy, sx], target, out=error, where=finite)
            error = abs(error)
            score = error / max(scale, 0.01) + 0.2 * distance
            use = (
                suspicious[ys, xs]
                & trusted[sy, sx]
                & (error < 1.5 * scale)
                & (score < best[ys, xs])
            )
            result[ys, xs][use] = image[sy, sx][use]
            best[ys, xs][use] = score[use]
            repaired[ys, xs][use] = True
    return result, repaired


def largest_rectangle(mask):
    """Largest all-valid axis-aligned rectangle, O(width * height)."""
    h, w = mask.shape
    heights = np.zeros(w, np.int32)
    best = (0, 0, 0, 0)
    area = 0
    for y in range(h):
        heights = np.where(mask[y], heights + 1, 0)
        stack = []
        for x in range(w + 1):
            current = int(heights[x]) if x < w else 0
            start = x
            while stack and stack[-1][1] > current:
                pos, height = stack.pop()
                size = (x - pos) * height
                if size > area:
                    area = size
                    best = (pos, y - height + 1, x, y + 1)
                start = pos
            if not stack or stack[-1][1] < current:
                stack.append((start, current))
    return best


def classic_frame(a, anchor, eye):
    image = a.left if eye == 0 else a.right
    support = a.support_l if eye == 0 else a.support_r
    matrix = np.array([[1, 0, anchor.disparity * eye], [0, 1, 0]], np.float32)
    h, w = image.shape[:2]
    return cv2.warpAffine(image, matrix, (w, h)), cv2.warpAffine(
        support.astype(np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST
    ) > 0


def supersampled_analysis(a):
    h, w = a.dl.shape
    size = (w * 2, h * 2)
    values = {}
    for key in ("left", "right"):
        values[key] = cv2.resize(getattr(a, key), size, interpolation=cv2.INTER_LINEAR)
    for key in ("dl", "dr"):
        values[key] = cv2.resize(getattr(a, key), size, interpolation=cv2.INTER_NEAREST) * 2
    for key in ("valid_l", "valid_r", "support_l", "support_r"):
        values[key] = (
            cv2.resize(getattr(a, key).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        )
    values["metadata"] = {**a.metadata, "render_scale": a.metadata.get("render_scale", 1) * 2}
    return replace(a, **values)


class Sequence:
    """A reusable render; photo edits/crop are applied identically for preview and export."""

    def __init__(
        self,
        a,
        anchor,
        motion,
        directory,
        progress=lambda _: None,
        cancelled=lambda: False,
        edits=None,
        style=None,
        output_width=None,
    ):
        from pathlib import Path

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.curvature_amount = 0.0
        self.output_width = output_width
        self.motion = motion
        self.anchor = anchor
        self.style = style or Style()
        self.style.validate()
        self.positions = view_positions(motion)
        if motion.mode == "anaglyph" and self.style.name != "clean":
            raise ValueError("Anaglyph uses the clean style to preserve eye-channel separation.")
        self.color_gains = balance_gains(a.left)
        self.effective_aa = bool(
            motion.antialias and motion.mode == "smooth" and self.style.name == "clean"
        )
        render_a = supersampled_analysis(a) if self.effective_aa else a
        render_anchor = (
            replace(anchor, x=anchor.x * 2, y=anchor.y * 2, disparity=anchor.disparity * 2)
            if self.effective_aa
            else anchor
        )
        common = a.support_l & a.support_r
        self.raw_hole_fractions = []
        self.remaining_hole_fractions = []
        for i, t in enumerate(self.positions):
            if cancelled():
                raise InterruptedError("Cancelled")
            progress(f"Rendering view {i + 1}/{len(self.positions)}")
            if motion.mode == "anaglyph":
                left, ml = classic_frame(a, anchor, 0)
                right, mr = classic_frame(a, anchor, 1)
                np.save(self.directory / "anaglyph-left.npy", left)
                np.save(self.directory / "anaglyph-right.npy", right)
                frame = anaglyph(left, right, motion.anaglyph_method)
                mask = ml & mr
                raw = filled = mask
            elif motion.mode == "classic":
                frame, mask = classic_frame(a, anchor, i)
                raw = filled = mask
            else:
                frame, raw, filled = render_view(
                    render_a,
                    render_anchor,
                    float(t),
                    edge_cleanup=motion.edge_cleanup and self.style.name == "clean",
                    fusion=motion.fusion,
                    repair_edges=motion.repair_edges,
                )
                h, w = a.dl.shape
                if self.effective_aa:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                    raw = (
                        cv2.resize(raw.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
                        > 0.99
                    )
                    filled = (
                        cv2.resize(filled.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
                        > 0.99
                    )
                shift = t * anchor.disparity
                left_shifts = -t * a.dl[a.support_l] + shift
                right_shifts = -(1 - t) * a.dr[a.support_r] + shift
                margin_l = (
                    int(np.ceil(max(0, float(left_shifts.max()), float(right_shifts.max())))) + 2
                )
                margin_r = (
                    int(np.ceil(max(0, float(-left_shifts.min()), float(-right_shifts.min())))) + 2
                )
                mask = np.ones((h, w), bool)
                mask[:, :margin_l] = False
                if margin_r:
                    mask[:, max(0, w - margin_r) :] = False
                mask[:2] = False
                mask[-2:] = False
            if motion.mode != "anaglyph":
                frame = apply_style(a, anchor, float(t), frame, self.style)
            common &= mask
            self.raw_hole_fractions.append(float((~raw).mean()))
            self.remaining_hole_fractions.append(float((~filled).mean()))
            np.save(self.directory / f"{i:03d}.npy", frame)
            np.save(self.directory / f"{i:03d}-filled.npy", filled)
        x0, y0, x1, y1 = largest_rectangle(common)
        x1 = x0 + (x1 - x0) // 2 * 2
        y1 = y0 + (y1 - y0) // 2 * 2
        if x1 - x0 < 32 or y1 - y0 < 32:
            raise ValueError(
                "Not enough common image area. Try a smaller sweep or inspect alignment."
            )
        self.crop = (x0, y0, x1, y1)
        self.indices = loop_indices(motion)
        self.durations_ms = (
            [motion.hold_ms] * 2
            if motion.mode == "classic"
            else [1000 / motion.fps] * len(self.indices)
        )
        self.visible_holes = []
        for i in range(len(self.positions)):
            filled = np.load(self.directory / f"{i:03d}-filled.npy")
            self.visible_holes.append(float((~filled[y0:y1, x0:x1]).mean()))
        self.metadata = {
            "crop_xyxy": list(self.crop),
            "positions": self.positions.tolist(),
            "uncovered_before_fill_fraction": self.raw_hole_fractions,
            "uncovered_after_fill_fraction": self.remaining_hole_fractions,
            "visible_unfilled_fraction": self.visible_holes,
            "style": asdict(self.style),
            "antialias_effective": "2x" if self.effective_aa else "off",
            "color_balance_gains": self.color_gains.tolist(),
            "output_width": output_width,
            "render_grid": list(a.left.shape[1::-1]),
        }
        self.set_edits(edits or PhotoEdits())

    def set_edits(self, edits):
        edits.validate()
        self.edits = replace(edits)
        x0, y0, x1, y1 = self.crop
        shape = (y1 - y0, x1 - x0)
        if self.motion.rotation in (90, 270):
            shape = shape[::-1]
        self.output_crop = crop_pixels(shape, self.edits.crop)
        self.metadata["photo_edits"] = asdict(self.edits)
        self.metadata["photo_crop_xyxy"] = list(self.output_crop)
        self.metadata["photo_crop_grid"] = "rotated common animation crop"

    def full_frame(self, index):
        if self.motion.mode == "anaglyph":
            # Apply identical photographic edits BEFORE encoding colors for the glasses.
            left = adjust_photo(
                np.load(self.directory / "anaglyph-left.npy"), self.edits, self.color_gains
            )
            right = adjust_photo(
                np.load(self.directory / "anaglyph-right.npy"), self.edits, self.color_gains
            )
            frame = anaglyph(left, right, self.motion.anaglyph_method)
        else:
            frame = adjust_photo(
                np.load(self.directory / f"{index:03d}.npy"), self.edits, self.color_gains
            )
        x0, y0, x1, y1 = self.crop
        frame = frame[y0:y1, x0:x1]
        if self.motion.rotation:
            frame = np.rot90(frame, -self.motion.rotation // 90)
        return np.ascontiguousarray(frame)

    def frame(self, index):
        frame = self.full_frame(index)
        x0, y0, x1, y1 = self.output_crop
        frame = np.ascontiguousarray(frame[y0:y1, x0:x1])
        from .curvature import curve

        frame = curve(frame, self.curvature_amount)
        if self.output_width:
            w = int(self.output_width) // 2 * 2
            h = max(2, round(frame.shape[0] * w / frame.shape[1] / 2) * 2)
            frame = cv2.resize(
                frame,
                (w, h),
                interpolation=cv2.INTER_AREA if w < frame.shape[1] else cv2.INTER_LANCZOS4,
            )
        return frame
