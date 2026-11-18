from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .core import Analysis


def ffmpeg_available():
    if not shutil.which("ffmpeg"):
        raise ValueError(
            "FFmpeg is required for MP4 export. Install it with: sudo apt install ffmpeg"
        )
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True
    )
    if "libx264" not in result.stdout:
        raise ValueError("Your FFmpeg installation does not provide the libx264 H.264 encoder.")


def write_animation(
    sequence, output, overwrite=False, progress=lambda _: None, cancelled=lambda: False
):
    path = Path(output).expanduser().resolve()
    allowed = (".jpg", ".jpeg") if sequence.motion.mode == "anaglyph" else (".gif", ".mp4")
    if path.suffix.lower() not in allowed:
        raise ValueError("Anaglyph exports require .jpg; motion exports require .gif or .mp4.")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Choose a new name or use --overwrite.")
    if path.suffix.lower() == ".mp4":
        ffmpeg_available()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.stem + "-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    process = None
    try:
        m = sequence.motion
        if m.mode == "anaglyph":
            progress("Saving anaglyph JPEG")
            Image.fromarray(sequence.frame(0)).save(tmp, format="JPEG", quality=97, subsampling=0)
        elif path.suffix.lower() == ".gif":
            progress("Building shared GIF palette")
            # A single palette prevents colors changing as the camera moves.
            thumbs = []
            for index in sequence.indices[: min(24, len(sequence.indices))]:
                thumb = Image.fromarray(sequence.frame(index))
                thumb.thumbnail((256, 144))
                thumbs.append(thumb)
            atlas = Image.new("RGB", (256, 144 * len(thumbs)))
            for i, thumb in enumerate(thumbs):
                atlas.paste(thumb, (0, 144 * i))
            palette = atlas.quantize(colors=256)
            frames = []
            for index in sequence.indices:
                if cancelled():
                    raise InterruptedError("Cancelled")
                frames.append(
                    Image.fromarray(sequence.frame(index)).quantize(
                        palette=palette, dither=Image.Dither.NONE
                    )
                )
            # GIF time is centiseconds: distribute rounding over the complete cycle.
            boundaries = np.rint(np.cumsum([0] + sequence.durations_ms) / 10) * 10
            durations = np.maximum(10, np.diff(boundaries)).astype(int).tolist()
            frames[0].save(
                tmp,
                save_all=True,
                append_images=frames[1:],
                loop=0,
                duration=durations,
                disposal=2,
                optimize=False,
            )
        else:
            frame = sequence.frame(0)
            h, w = frame.shape[:2]
            command = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{w}x{h}",
                "-r",
                str(m.fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                tmp,
            ]
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
                if m.mode == "classic" and not getattr(sequence, "timeline_expanded", False):
                    holds = max(1, round(m.hold_ms * m.fps / 1000))
                    order = [i for i in sequence.indices for _ in range(holds)]
                else:
                    order = sequence.indices
                for cycle in range(m.cycles):
                    progress(f"Encoding MP4 cycle {cycle + 1}/{m.cycles}")
                    for index in order:
                        if cancelled():
                            raise InterruptedError("Cancelled")
                        process.stdin.write(sequence.frame(index).tobytes())
                process.stdin.close()
                code = process.wait()
                if code:
                    errors.seek(0)
                    raise RuntimeError("FFmpeg failed: " + errors.read().decode(errors="replace"))
        if cancelled():
            raise InterruptedError("Cancelled")
        if overwrite:
            os.replace(tmp, path)
        else:
            # Exclusive link avoids a race that could overwrite a newly-created user file.
            os.link(tmp, path)
            os.unlink(tmp)
        return path
    except BrokenPipeError as exc:
        raise RuntimeError(
            "FFmpeg stopped while encoding. Check disk space and encoder availability."
        ) from exc
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        Path(tmp).unlink(missing_ok=True)


def write_bundle(a: Analysis, sequence, destination):
    path = Path(destination).resolve()
    if path.exists():
        raise FileExistsError(f"Asset bundle already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".assets-", dir=path.parent))
    try:
        Image.fromarray(a.left).save(tmp / "left.png")
        Image.fromarray(a.right).save(tmp / "right.png")
        np.savez_compressed(
            tmp / "stereo.npz",
            disparity_left=a.dl,
            disparity_right=a.dr,
            reliable_left=a.valid_l,
            reliable_right=a.valid_r,
            support_left=a.support_l,
            support_right=a.support_r,
        )
        meta = {
            "schema": "w3-wigglegram.stereo",
            "version": 1,
            "grid": "full aligned working-resolution images",
            "disparity_units": "pixels",
            "left_convention": "x_left - x_right",
            "right_convention": "x_right - x_left",
            "origin": "top-left; x rightward, y downward",
            "anchor": asdict(sequence.anchor),
            "motion": asdict(sequence.motion),
            "render": sequence.metadata,
            "analysis": a.metadata,
            "metric_depth": False,
        }
        if "Q" in a.metadata["alignment"]:
            points = cv2.reprojectImageTo3D(a.dl, np.array(a.metadata["alignment"]["Q"]))
            depth = points[:, :, 2]
            depth[~a.valid_l | (depth <= 0)] = np.nan
            np.save(tmp / "depth_meters.npy", depth)
            meta["metric_depth"] = True
        (tmp / "manifest.json").write_text(json.dumps(meta, indent=2))
        os.rename(tmp, path)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return path
