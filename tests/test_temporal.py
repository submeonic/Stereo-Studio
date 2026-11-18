from dataclasses import replace

import cv2
import numpy as np
import pytest
from PIL import Image

from wigglegram.core import Analysis, Anchor
from wigglegram.editing import PhotoEdits, linear_rgb, srgb
from wigglegram.geometry import refine_analysis, refine_edge, render_grid
from wigglegram.render import Motion, Sequence
from wigglegram.temporal import SPECS, TimelineSequence, spatial_effects, validate


class Frames:
    def __init__(self, images, motion=None, order=None):
        self.images = images
        self.motion = motion or Motion(views=3, fps=30)
        self.indices = order or list(range(len(images)))
        self.positions = np.linspace(0, 1, len(images))
        self.anchor = Anchor(20, 20, 0, False, "")
        self.crop = (0, 0, images[0].shape[1], images[0].shape[0])
        self.edits = PhotoEdits()
        self.metadata = {}

    def frame(self, i):
        return self.images[i].copy()


def test_persistence_matches_warmed_circular_history(tmp_path):
    images = [np.full((16, 32, 3), v, np.uint8) for v in (20, 180, 60, 250)]
    base = Frames(images)
    p = {"amount": 1, "red_ms": 100, "green_ms": 300, "blue_ms": 40}
    seq = TimelineSequence(base, {"persistence": p}, tmp_path)
    decay = np.exp(-1000 / 30 / np.array([100, 300, 40]))
    state = np.zeros((16, 32, 3))
    for _ in range(300):
        for image in images:
            state = state * decay + linear_rgb(image) * (1 - decay)
    for i, image in enumerate(images):
        state = state * decay + linear_rgb(image) * (1 - decay)
        assert np.max(abs(seq.frame(i).astype(int) - srgb(state).astype(int))) <= 1
    # Starts with the previous loop's light; there is no black-history reset.
    assert seq.frame(0).mean() > images[0].mean()


def test_effects_advance_on_return_and_are_reproducible(tmp_path):
    rng = np.random.default_rng(7)
    images = [rng.integers(0, 256, (32, 64, 3), np.uint8)] * 3
    settings = {"tearing": {"bands": 7, "amount": 0.7}, "static": {"amount": 0.1}}
    base = Frames(images, order=[0, 1, 2, 1])
    a = TimelineSequence(base, settings, tmp_path / "a")
    b = TimelineSequence(base, settings, tmp_path / "b")
    assert not np.array_equal(a.frame(1), a.frame(3))
    assert all(np.array_equal(a.frame(i), b.frame(i)) for i in range(4))
    assert a.indices == [0, 1, 2, 3]
    assert sum(a.durations_ms) == pytest.approx(4000 / 30)
    classic = Frames(images[:2], Motion(mode="classic", fps=30, hold_ms=180))
    c = TimelineSequence(classic, settings, tmp_path / "c")
    assert len(c.indices) == 10  # 180 ms rounds to five 30 FPS frames per eye, no time compression


def test_effect_validation_and_all_modules(tmp_path):
    image = np.random.default_rng(8).integers(0, 256, (32, 64, 3), np.uint8)
    settings = validate({name: {} for name in SPECS})
    seq = TimelineSequence(Frames([image] * 4), settings, tmp_path / "all")
    assert seq.frame(0).shape == image.shape and seq.frame(0).dtype == np.uint8
    for bad in (
        {"unknown": {}},
        {"tearing": {"bands": 0}},
        {"tearing": {"bands": 2.5}},
        {"rgb": {"offset": float("nan")}},
    ):
        with pytest.raises(ValueError):
            validate(bad)
    with pytest.raises(ValueError, match="anaglyph"):
        TimelineSequence(Frames([image], Motion(mode="anaglyph")), {"echo": {}}, tmp_path / "bad")
    with pytest.raises(ValueError, match="cycles"):
        TimelineSequence(Frames([image] * 4), {"flicker": {"cycles": 3}}, tmp_path / "fast")
    # Periodic effects agree after a complete cycle, excluding intentionally random static.
    settings.pop("static")
    assert np.array_equal(
        spatial_effects(image, settings, 0, 0, 42), spatial_effects(image, settings, 1, 0, 42)
    )


def test_refinement_selects_surface_without_inventing_confidence():
    image = np.zeros((20, 30, 3), np.uint8)
    image[:, 15:] = 255
    d = np.full((20, 30), 2, np.float32)
    d[:, 15:] = 10
    d[:, 14] = 10  # wrong foreground spill into the dark background
    valid = np.ones(d.shape, bool)
    valid[:, 14] = False
    confidence = valid.astype(np.float32)
    refined = refine_edge(d, image, valid, confidence)
    assert np.all(refined[:, 14] == 2)
    assert np.array_equal(refined[valid], d[valid])
    assert set(np.unique(refined)) == {2, 10}  # no averaged intermediate surface
    a = Analysis(image, image, d, -d, valid, valid, valid, valid, {"backend": "synthetic"})
    b = refine_analysis(a, cache_dir=None)
    assert np.array_equal(b.valid_l, a.valid_l)
    assert np.array_equal(a.dl, d)


def test_export_grid_reads_original_colors_and_scales_geometry(tmp_path):
    image = np.random.default_rng(2).integers(0, 256, (160, 320, 3), np.uint8)
    path = tmp_path / "original.mpo"
    Image.fromarray(image).save(
        path, format="MPO", save_all=True, append_images=[Image.fromarray(image)]
    )
    low = cv2.resize(image, (160, 80))
    d = np.full((80, 160), 3, np.float32)
    mask = np.ones(d.shape, bool)
    a = Analysis(
        low,
        low,
        d,
        -d,
        mask,
        mask,
        mask,
        mask,
        {"alignment": {"type": "nominal", "right_affine": [[1, 0, 0], [0, 1, 0]]}},
    )
    grid, anchor = render_grid(a, Anchor(80, 40, 3, False, ""), path, width=320)
    assert grid.left.shape == image.shape and np.all(grid.dl == 6)
    assert (anchor.x, anchor.y, anchor.disparity) == (160, 80, 6)
    enlarged = cv2.resize(low, (320, 160))
    assert not np.array_equal(grid.left, enlarged)
    seq = Sequence(grid, anchor, Motion(mode="classic"), tmp_path / "frames", output_width=160)
    assert seq.frame(0).shape[1] == 160
    assert seq.frame(0).shape[0] % 2 == 0
    # Edits change final frames, never geometry/source colors.
    before = grid.left.copy()
    seq.set_edits(replace(seq.edits, exposure=1))
    assert np.array_equal(grid.left, before)


def test_guided_upsampling_tracks_color_edge_without_depth_average():
    from wigglegram.geometry import guided_upsample

    small = np.zeros((10, 16, 3), np.uint8)
    small[:, 8:] = 255
    d = np.full((10, 16), 2, np.float32)
    d[:, 8:] = 10
    target = np.zeros((20, 32, 3), np.uint8)
    target[:, 17:] = 255
    result = guided_upsample(d, small, target)
    assert np.allclose(result[:, :17], 2)
    assert np.allclose(result[:, 17:], 10)


def test_temporal_mp4_duration_is_not_expanded_twice(tmp_path):
    import json
    import shutil
    import subprocess

    from wigglegram.export import write_animation

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    images = [np.full((16, 32, 3), v, np.uint8) for v in (30, 210)]
    base = Frames(images, Motion(mode="classic", hold_ms=200, fps=30, cycles=2))
    seq = TimelineSequence(base, {"persistence": {}}, tmp_path / "frames")
    output = write_animation(seq, tmp_path / "loop.mp4")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,width,height,duration",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    assert int(stream["nb_frames"]) == 24
    assert float(stream["duration"]) == pytest.approx(0.8, abs=0.01)
    assert (stream["width"], stream["height"]) == (32, 16)


def test_upsampling_does_not_promote_inferred_geometry():
    from wigglegram.geometry import guided_upsample

    guide = np.full((8, 8, 3), 100, np.uint8)
    target = cv2.resize(guide, (16, 16))
    d = np.ones((8, 8), np.float32)
    valid = np.ones((8, 8), bool)
    valid[:, 4] = False
    _, reliable = guided_upsample(d, guide, target, valid)
    assert not reliable[:, 8:10].any()
    assert reliable[:, :3].all()
