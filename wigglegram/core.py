from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageOps

Progress = Callable[[str], None]
CACHE_VERSION = 3


@dataclass
class Analysis:
    left: np.ndarray
    right: np.ndarray
    dl: np.ndarray
    dr: np.ndarray
    valid_l: np.ndarray
    valid_r: np.ndarray
    support_l: np.ndarray
    support_r: np.ndarray
    metadata: dict = field(default_factory=dict)


@dataclass
class Anchor:
    x: float
    y: float
    disparity: float
    automatic: bool
    message: str


def read_pair(
    source: str | Path, right: str | Path | None = None, swap=False, width=960
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not 160 <= width <= 4096:
        raise ValueError("Working width must be between 160 and 4096 pixels.")
    try:
        with Image.open(source) as im:
            info = {
                "source": Path(source).name,
                "camera": str(im.getexif().get(272, "Unknown")),
                "source_size": list(im.size),
                "source_frames": getattr(im, "n_frames", 1),
            }
            orientation = im.getexif().get(274, 1)
            if right is None and getattr(im, "n_frames", 1) < 2:
                raise ValueError(
                    "This file has one image. Open an MPO or supply a right-eye image."
                )
            left_image = ImageOps.exif_transpose(im).convert("RGB").copy()
            if right is None:
                im.seek(1)
                # Some MPO writers omit orientation on the secondary frame.
                im.getexif()[274] = orientation
                right_image = ImageOps.exif_transpose(im).convert("RGB").copy()
        if right is not None:
            with Image.open(right) as im:
                right_image = ImageOps.exif_transpose(im).convert("RGB").copy()
            info["right_source"] = Path(right).name
    except (OSError, EOFError, SyntaxError) as exc:
        raise ValueError(f"Cannot decode stereo input: {exc}") from exc
    if left_image.size != right_image.size:
        raise ValueError("Left and right images must have the same dimensions.")
    if swap:
        left_image, right_image = right_image, left_image
    w, h = left_image.size
    scale = min(width / w, 1.0)
    size = (round(w * scale), round(h * scale))
    left = np.array(left_image.resize(size, Image.Resampling.LANCZOS))
    right_array = np.array(right_image.resize(size, Image.Resampling.LANCZOS))
    info.update(working_size=list(size), scale=scale, swap=bool(swap), orientation=orientation)
    return left, right_array, info


def correspondences(left, right):
    cv2.setRNGSeed(0)
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.025)
    kl, dl = sift.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_RGB2GRAY), None)
    kr, dr = sift.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_RGB2GRAY), None)
    if dl is None or dr is None:
        return np.empty((0, 2)), np.empty((0, 2))
    matches = cv2.BFMatcher().knnMatch(dl, dr, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.72 * m[1].distance]
    pl = np.float32([kl[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pr = np.float32([kr[m.trainIdx].pt for m in good]).reshape(-1, 2)
    if len(pl) >= 12:
        _, mask = cv2.findFundamentalMat(pl, pr, cv2.FM_RANSAC, 1.0, 0.999)
        if mask is not None and mask.sum() >= 12:
            pl, pr = pl[mask.ravel() != 0], pr[mask.ravel() != 0]
    return pl, pr


def align_pair(left, right, profile="auto", swap=False):
    h, w = left.shape[:2]
    mask = np.ones((h, w), np.uint8) * 255
    if profile not in ("auto", "w3-wide", "3ds"):
        cfg = json.loads(Path(profile).read_text())
        required = ("image_size", "K1", "K2", "D1", "D2", "R", "T")
        if any(k not in cfg for k in required):
            raise ValueError(
                "Measured profile requires image_size, K1, K2, D1, D2, R and T (meters)."
            )
        if swap:
            raise ValueError(
                "Measured calibration expects its original eye order; disable Swap eyes."
            )
        sw, sh = cfg["image_size"]
        if abs(w / h - sw / sh) > 0.01:
            raise ValueError("Profile aspect ratio does not match the input capture mode.")
        k1, k2 = (np.array(cfg[k], dtype=float) for k in ("K1", "K2"))
        for k in (k1, k2):
            k[0] *= w / sw
            k[1] *= h / sh
        d1, d2, r, t = (np.array(cfg[k], dtype=float) for k in ("D1", "D2", "R", "T"))
        if np.linalg.norm(t) <= 0:
            raise ValueError("Calibration baseline must be nonzero and expressed in meters.")
        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(k1, d1, k2, d2, (w, h), r, t, alpha=0)
        if abs(p2[1, 3]) > abs(p2[0, 3]):
            raise ValueError("This release requires a horizontal stereo baseline.")
        maps = [
            cv2.initUndistortRectifyMap(k, d, rr, p, (w, h), cv2.CV_32FC1)
            for k, d, rr, p in ((k1, d1, r1, p1), (k2, d2, r2, p2))
        ]
        left_value, rr = [
            cv2.remap(img, *m, cv2.INTER_LINEAR) for img, m in zip((left, right), maps)
        ]
        ml, mr = [cv2.remap(mask, *m, cv2.INTER_NEAREST) > 0 for m in maps]
        pl, pr = correspondences(left_value, rr)
        residual = float(np.median(abs(pl[:, 1] - pr[:, 1]))) if len(pl) else None
        return (
            left_value,
            rr,
            ml,
            mr,
            {
                "type": "measured",
                "profile": cfg,
                "Q": q.tolist(),
                "P1": p1.tolist(),
                "P2": p2.tolist(),
                "vertical_median_px": residual,
                "depth_order_sign": 1 if p2[0, 3] < 0 else -1,
            },
        )
    pl, pr = correspondences(left, right)
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
    notes = []
    residual = None
    if len(pl) >= 12:
        # Fit only vertical correction. Horizontal scene disparity stays untouched.
        x = np.column_stack((pr[:, 0] / w, pr[:, 1] / h, np.ones(len(pr))))
        target = pl[:, 1] - pr[:, 1]
        keep = abs(target - np.median(target)) < max(
            3.0, 4 * np.median(abs(target - np.median(target)))
        )
        for _ in range(5):
            if keep.sum() < 8:
                break
            coef = np.linalg.lstsq(x[keep], target[keep], rcond=None)[0]
            err = target - x @ coef
            keep = abs(err) < max(0.7, 3 * np.median(abs(err[keep])))
        if keep.sum() >= 8:
            correction = x @ coef
            if np.max(abs(correction)) < h * 0.06:
                matrix[1] = [coef[0] / w, 1 + coef[1] / h, coef[2]]
                residual = float(np.median(abs(target[keep] - correction[keep])))
            else:
                notes.append(
                    "Large vertical correction rejected; check eye order and camera profile."
                )
        if residual is not None and residual > 1.0:
            notes.append(
                "Residual vertical mismatch exceeds 1 px; measured calibration may improve edges."
            )
    else:
        notes.append("Too few reliable features for automatic alignment; using original alignment.")
    aligned = cv2.warpAffine(right, matrix, (w, h), flags=cv2.INTER_LINEAR)
    support = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST) > 0
    disparities = pl[:, 0] - pr[:, 0] if len(pl) else np.array([0.0, w * 0.06])
    return (
        left,
        aligned,
        mask > 0,
        support,
        {
            "type": "nominal",
            "profile": profile,
            "right_affine": matrix.tolist(),
            "matches": len(pl),
            "vertical_median_px": residual,
            "disparity_percentiles": np.percentile(disparities, [1, 50, 99]).tolist(),
            "depth_order_sign": -1 if swap else 1,
            "warnings": notes,
            "metric_depth_available": False,
        },
    )


def fill_disparity(disparity, valid):
    """Nearest reliable geometry is a fallback, never a new reliable match."""
    if not valid.any():
        raise ValueError(
            "No reliable stereo matches. Try the other backend or a more textured photo."
        )
    _, labels = cv2.distanceTransformWithLabels(
        (~valid).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    values = np.zeros(labels.max() + 1, np.float32)
    values[labels[valid]] = disparity[valid]
    return values[labels]


def consistency(dl, dr, sl, sr, tolerance=1.5):
    y, x = np.indices(dl.shape, dtype=np.float32)
    xr = (x - dl).astype(np.float32)
    sampled = cv2.remap(
        dr, xr, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1e6
    )
    support = cv2.remap(sr.astype(np.uint8), xr, y, cv2.INTER_NEAREST) > 0
    return sl & support & np.isfinite(dl) & (abs(dl + sampled) < tolerance)


def sgbm(left, right, support_l, support_r, ranges):
    lo, _, hi = ranges
    minimum = int(np.floor(lo)) - 12
    count = max(32, int(np.ceil((hi - minimum + 16) / 16)) * 16)
    count = min(count, ((left.shape[1] // 2) // 16) * 16)
    gray = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in (left, right)]
    matcher = cv2.StereoSGBM_create(
        minDisparity=minimum,
        numDisparities=count,
        blockSize=5,
        P1=8 * 25,
        P2=32 * 25,
        disp12MaxDiff=1,
        preFilterCap=31,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    rm = cv2.ximgproc.createRightMatcher(matcher)
    raw_l = matcher.compute(*gray)
    raw_r = rm.compute(gray[1], gray[0])
    vl = (raw_l > (minimum - 1) * 16) & support_l
    vr = (raw_r > (rm.getMinDisparity() - 1) * 16) & support_r
    valid_l = consistency(raw_l / 16.0, raw_r / 16.0, vl, vr)
    valid_r = consistency(raw_r / 16.0, raw_l / 16.0, vr, vl)
    filt = cv2.ximgproc.createDisparityWLSFilter(matcher)
    filt.setLambda(8000)
    filt.setSigmaColor(1.3)
    filtered_l = filt.filter(raw_l, left, disparity_map_right=raw_r).astype(np.float32) / 16
    # Right-eye edge guidance: mirror and swap, making its magnitude left-like.
    flipped_matcher = cv2.StereoSGBM_create(minDisparity=minimum, numDisparities=count, blockSize=5)
    right_filter = cv2.ximgproc.createDisparityWLSFilter(flipped_matcher)
    right_filter.setLambda(8000)
    right_filter.setSigmaColor(1.3)
    filtered_r = (
        -right_filter.filter(
            np.ascontiguousarray(-raw_r[:, ::-1]),
            np.ascontiguousarray(right[:, ::-1]),
            disparity_map_right=np.ascontiguousarray(-raw_l[:, ::-1]),
        )[:, ::-1].astype(np.float32)
        / 16
    )
    # WLS can extrapolate through textureless areas; reserve 'valid' for observed agreement.
    filtered_l = np.where(
        (filtered_l >= minimum) & (filtered_l <= minimum + count), filtered_l, raw_l / 16.0
    ).astype(np.float32)
    filtered_r = np.where(
        (filtered_r >= -minimum - count) & (filtered_r <= -minimum), filtered_r, raw_r / 16.0
    ).astype(np.float32)
    dl = fill_disparity(filtered_l, valid_l & np.isfinite(filtered_l))
    dr = fill_disparity(filtered_r, valid_r & np.isfinite(filtered_r))
    dl = cv2.medianBlur(dl, 3)
    dr = cv2.medianBlur(dr, 3)
    return dl, dr, valid_l, valid_r


def cache_key(source, right, width, swap, profile, backend):
    h = hashlib.sha256()
    for path in (source, right):
        if path:
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
    profile_content = (
        profile if profile in ("auto", "w3-wide", "3ds") else Path(profile).read_text()
    )
    h.update(json.dumps([CACHE_VERSION, width, swap, profile_content, backend]).encode())
    if backend == "neural":
        from .neural import model_identity

        h.update(json.dumps(model_identity(), sort_keys=True).encode())
    return h.hexdigest()


def analyze(
    source,
    right=None,
    width=1280,
    swap=False,
    profile="auto",
    backend="sgbm",
    cache_dir=".cache/wigglegram",
    progress: Progress = lambda _: None,
    cancelled=lambda: False,
):
    cv2.setNumThreads(min(8, os.cpu_count() or 1))
    if backend not in ("sgbm", "neural", "sparse"):
        raise ValueError("Backend must be sgbm or neural.")
    key = cache_key(source, right, width, swap, profile, backend)
    cache = Path(cache_dir) / f"{key}.npz" if cache_dir else None
    if cache and cache.exists():
        progress("Loading cached stereo analysis")
        try:
            with np.load(cache, allow_pickle=False) as data:
                return Analysis(
                    **{k: data[k] for k in Analysis.__dataclass_fields__ if k != "metadata"},
                    metadata=json.loads(str(data["metadata"])),
                )
        except (ValueError, OSError, KeyError):
            progress("Cache unreadable; rebuilding analysis")
    progress("Decoding stereo pair")
    left, right_image, info = read_pair(source, right, swap, width)
    progress("Aligning vertical geometry")
    left, right_image, sl, sr, alignment = align_pair(left, right_image, profile, swap)
    if cancelled():
        raise InterruptedError("Cancelled")
    progress("Estimating stereo geometry — " + backend)
    ranges = alignment.get("disparity_percentiles")
    if ranges is None:
        pl, pr = correspondences(left, right_image)
        ranges = (
            np.percentile(pl[:, 0] - pr[:, 0], [1, 50, 99]).tolist()
            if len(pl)
            else [0, 16, width * 0.15]
        )
    if backend == "sparse":
        pl, pr = correspondences(left, right_image)
        dl = np.zeros(left.shape[:2], np.float32)
        dr = dl.copy()
        vl = np.zeros(dl.shape, np.uint8)
        vr = vl.copy()
        good = abs(pl[:, 1] - pr[:, 1]) < 1.5
        for left_value, r in zip(pl[good], pr[good]):
            d = float(left_value[0] - r[0])
            lp = tuple(np.rint(left_value).astype(int))
            rp = tuple(np.rint(r).astype(int))
            cv2.circle(dl, lp, 3, d, -1)
            cv2.circle(dr, rp, 3, -d, -1)
            cv2.circle(vl, lp, 3, 1, -1)
            cv2.circle(vr, rp, 3, 1, -1)
        vl = vl.astype(bool) & sl
        vr = vr.astype(bool) & sr
        if vl.any() and vr.any():
            dl = fill_disparity(dl, vl)
            dr = fill_disparity(dr, vr)
        else:
            info["warnings"] = [
                "No reliable anchor matches; classic mode uses zero horizontal offset."
            ]
    elif backend == "neural":
        from .neural import estimate, model_identity

        dl, dr = estimate(left, right_image, ranges, alignment["depth_order_sign"], cancelled)
        vl = consistency(dl, dr, sl, sr)
        vr = consistency(dr, dl, sr, sl)
        # Suppress unconstrained extreme predictions at image borders.
        for disparity, valid in ((dl, vl), (dr, vr)):
            if valid.sum() < 20:
                raise ValueError(
                    "Neural stereo found too few consistent matches. Check eye order or use CPU/classic mode."
                )
            low, high = np.percentile(disparity[valid], [0.2, 99.8])
            implausible = (disparity < low - 8) | (disparity > high + 8)
            nearest = fill_disparity(disparity, valid)
            disparity[implausible] = nearest[implausible]
        info["model"] = model_identity()
    else:
        dl, dr, vl, vr = sgbm(left, right_image, sl, sr, ranges)
    if cancelled():
        raise InterruptedError("Cancelled")
    info.update(
        backend=backend,
        alignment=alignment,
        cache_key=key,
        reliable_fraction_left=float(vl.mean()),
        reliable_fraction_right=float(vr.mean()),
    )
    result = Analysis(left, right_image, dl, dr, vl, vr, sl, sr, info)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".npz", dir=cache.parent)
        os.close(fd)
        try:
            np.savez_compressed(
                tmp,
                **{k: getattr(result, k) for k in Analysis.__dataclass_fields__ if k != "metadata"},
                metadata=json.dumps(info),
            )
            os.replace(tmp, cache)
        finally:
            Path(tmp).unlink(missing_ok=True)
    return result


def local_anchor_disparity(a, point):
    """Sparse-mode fallback: verify a textured clicked patch by horizontal correlation."""
    h, w = a.dl.shape
    x, y = map(lambda value: int(round(value)), point)
    radius = max(8, round(w * 0.018))
    if not (radius <= x < w - radius and radius + 2 <= y < h - radius - 2):
        return None
    left = cv2.cvtColor(a.left, cv2.COLOR_RGB2GRAY)
    right = cv2.cvtColor(a.right, cv2.COLOR_RGB2GRAY)
    patch = left[y - radius : y + radius + 1, x - radius : x + radius + 1]
    if patch.std() < 4:
        return None
    ranges = a.metadata.get("alignment", {}).get("disparity_percentiles", [-w * 0.15, 0, w * 0.15])
    low = max(0, int(np.floor(x - ranges[2] - radius - 6)))
    high = min(w, int(np.ceil(x - ranges[0] + radius + 7)))
    search = right[y - radius - 2 : y + radius + 3, low:high]
    if search.shape[1] < patch.shape[1]:
        return None
    scores = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(scores)
    px, py = location
    other = scores.copy()
    other[:, max(0, px - 3) : px + 4] = -1
    if score < 0.82 or score - float(other.max()) < 0.02:
        return None
    # A three-sample parabolic peak refines the horizontal position below one pixel.
    fractional = 0.0
    if 0 < px < scores.shape[1] - 1:
        before, center, after = scores[py, px - 1 : px + 2]
        denominator = before - 2 * center + after
        if abs(denominator) > 1e-6:
            fractional = float(np.clip(0.5 * (before - after) / denominator, -0.5, 0.5))
    return float(x - (low + px + radius + fractional))


def choose_anchor(a: Analysis, point=None):
    h, w = a.dl.shape
    fallback = ""
    if point is not None:
        x, y = point
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("Anchor must be inside the aligned left image.")
        radius = max(4, round(w * 0.012))
        xs = slice(max(0, int(x) - radius), min(w, int(x) + radius + 1))
        ys = slice(max(0, int(y) - radius), min(h, int(y) + radius + 1))
        valid = a.valid_l[ys, xs]
        if valid.sum() >= 12:
            return Anchor(
                float(x), float(y), float(np.median(a.dl[ys, xs][valid])), False, "Clicked subject"
            )
        if a.metadata.get("backend") == "sparse":
            disparity = local_anchor_disparity(a, point)
            if disparity is not None:
                return Anchor(
                    float(x), float(y), disparity, False, "Clicked subject · verified local match"
                )
        fallback = "Clicked patch has too few reliable matches; using automatic anchor. "
    # Score spatially coherent depth components near the center, not a lone nearest pixel.
    sign = a.metadata.get("alignment", {}).get("depth_order_sign", 1)
    y, x = np.indices((h, w))
    central = (x > w * 0.18) & (x < w * 0.82) & (y > h * 0.15) & (y < h * 0.9) & a.valid_l
    if central.sum() < 20:
        mask = a.valid_l
        if not mask.any():
            return Anchor(
                w / 2,
                h / 2,
                0.0,
                True,
                fallback + "No reliable matches: zero-offset anchor; inspect classic preview.",
            )
        yy, xx = np.where(mask)
        idx = np.argmin((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
        ax, ay = xx[idx], yy[idx]
        return Anchor(
            float(ax),
            float(ay),
            float(a.dl[ay, ax]),
            True,
            fallback + "Low-confidence center fallback",
        )
    threshold = np.percentile(sign * a.dl[central], 65)
    foreground = (central & (sign * a.dl >= threshold)).astype(np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, centers = cv2.connectedComponentsWithStats(foreground)
    choices = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < max(30, w * h * 0.0005):
            continue
        cx, cy = centers[i]
        score = stats[i, cv2.CC_STAT_AREA] / (1 + 6 * ((cx / w - 0.5) ** 2 + (cy / h - 0.5) ** 2))
        choices.append((score, i))
    selected = (labels == max(choices)[1]) & a.valid_l if choices else central
    disparity = float(np.median(a.dl[selected]))
    yy, xx = np.where(selected)
    score = (
        ((xx - w * 0.5) / w) ** 2
        + ((yy - h * 0.5) / h) ** 2
        + abs(a.dl[yy, xx] - disparity) / max(w * 0.1, 1)
    )
    idx = int(np.argmin(score))
    return Anchor(
        float(xx[idx]), float(yy[idx]), disparity, True, fallback + "Automatic central foreground"
    )
