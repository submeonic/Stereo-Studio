"""Isolated NVIDIA Fast-FoundationStereo adapter. Never downloads at runtime."""

import hashlib
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def paths():
    return (
        Path(
            os.environ.get(
                "WIGGLE_GPU_PYTHON",
                ROOT
                / (".venv-gpu/Scripts/python.exe" if os.name == "nt" else ".venv-gpu/bin/python"),
            )
        ),
        Path(os.environ.get("WIGGLE_FFS_REPO", ROOT / "third_party/Fast-FoundationStereo")),
        Path(
            os.environ.get(
                "WIGGLE_FFS_MODEL",
                ROOT / "models/fast-foundation-stereo/23-36-37/model_best_bp2_serialize.pth",
            )
        ),
    )


def available():
    python, repo, model = paths()
    return python.is_file() and (repo / "core").is_dir() and model.is_file()


@lru_cache(maxsize=4)
def _digest(path, size, mtime):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_identity():
    python, repo, model = paths()
    if not available():
        raise ValueError(
            "Neural backend is not installed. Run ./scripts/setup-gpu.sh or select CPU SGBM."
        )
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    stat = model.stat()
    return {
        "backend": "Fast-FoundationStereo",
        "checkpoint": model.parent.name,
        "sha256": _digest(str(model), stat.st_size, stat.st_mtime_ns),
        "upstream_revision": revision,
        "iterations": 8,
        "max_disparity": 192,
        "adapter_version": 1,
    }


def estimate(left, right, ranges, depth_sign, cancelled=lambda: False):
    python, repo, model = paths()
    model_identity()
    with tempfile.TemporaryDirectory(prefix="wiggle-stereo-") as temp:
        temp = Path(temp)
        np.savez(temp / "input.npz", left=left, right=right, ranges=ranges, depth_sign=depth_sign)
        with open(temp / "log.txt", "w+") as log:
            env = os.environ.copy()
            env.update(
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                TORCH_HOME=str(ROOT / ".cache/torch"),
                TRITON_CACHE_DIR=str(ROOT / ".cache/triton"),
            )
            proc = subprocess.Popen(
                [
                    str(python),
                    str(ROOT / "scripts/neural_worker.py"),
                    "--repo",
                    str(repo),
                    "--model",
                    str(model),
                    "--input",
                    str(temp / "input.npz"),
                    "--output",
                    str(temp / "output.npz"),
                ],
                stdout=log,
                stderr=log,
                env=env,
            )
            try:
                while True:
                    try:
                        code = proc.wait(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        if cancelled():
                            raise InterruptedError("Cancelled")
                if code:
                    log.seek(0)
                    message = log.read()[-5000:]
                    raise RuntimeError(
                        "GPU stereo failed. Try CPU SGBM or lower resolution.\n" + message
                    )
                with np.load(temp / "output.npz", allow_pickle=False) as data:
                    return data["dl"], data["dr"]
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
