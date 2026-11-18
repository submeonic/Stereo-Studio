#!/usr/bin/env python3
"""Run the upstream serialized model in its own environment, without its demo UI."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    for key in ("repo", "model", "input", "output"):
        p.add_argument("--" + key, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to this process.")
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import core.foundation_stereo  # noqa: F401  Registers serialized-module aliases.
    from core.utils.utils import InputPadder

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Only the explicitly installed upstream checkpoint is loaded. Photo files are never unpickled.
    model = torch.load(args.model, map_location="cpu", weights_only=False)
    model.args.valid_iters = 8
    model.args.max_disp = 192
    model.cuda().eval()
    with np.load(args.input, allow_pickle=False) as data:
        left, right = data["left"], data["right"]
        ranges = data["ranges"]
        sign = int(data["depth_sign"])
    if sign < 0:
        left, right = right, left
        ranges = -ranges[::-1]
    # W3 parallax adjustment can shift the entire disparity range below zero.
    # Translate the target eye temporarily, then undo that offset in disparity units.
    offset = max(0, int(np.ceil(12 - float(ranges[0]))))
    h, w = left.shape[:2]
    if float(ranges[-1]) + offset > 180:
        raise RuntimeError("Disparity exceeds the model range. Reduce working width.")

    def infer(a, b):
        b = cv2.warpAffine(
            b, np.float32([[1, 0, -offset], [0, 1, 0]]), (w, h), borderMode=cv2.BORDER_REPLICATE
        )
        tensors = [
            torch.from_numpy(np.ascontiguousarray(im)).cuda().float()[None].permute(0, 3, 1, 2)
            for im in (a, b)
        ]
        padder = InputPadder(tensors[0].shape, divis_by=32, force_square=False)
        tensors = padder.pad(*tensors)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            disp = model.forward(
                *tensors, iters=8, test_mode=True, optimize_build_volume="pytorch1"
            )
        return padder.unpad(disp.float()).cpu().numpy().reshape(h, w) - offset

    dl = infer(left, right)
    # Horizontal mirror + eye reversal keeps the physical disparity direction positive.
    dr = -infer(right[:, ::-1], left[:, ::-1])[:, ::-1]
    if sign < 0:
        dl, dr = dr, dl
    if not np.isfinite(dl).all() or not np.isfinite(dr).all():
        raise RuntimeError("Model produced non-finite disparities.")
    np.savez(args.output, dl=dl.astype(np.float32), dr=dr.astype(np.float32))
    print("peak_cuda_bytes", torch.cuda.max_memory_allocated())


if __name__ == "__main__":
    main()
