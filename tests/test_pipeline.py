import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest
from PIL import Image

from wigglegram.core import (
    Analysis,
    align_pair,
    analyze,
    cache_key,
    choose_anchor,
    consistency,
    read_pair,
)
from wigglegram.export import write_animation, write_bundle
from wigglegram.render import (
    Motion,
    Sequence,
    largest_rectangle,
    loop_indices,
    render_view,
    splat,
    view_positions,
)


def scene(width=160, height=80):
    rng = np.random.default_rng(18)
    texture = rng.integers(30, 200, (height, width + 40, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (5, 5), 1.0)
    pairs = []
    maps = []
    for t in [0, 1]:
        im = texture[:, t * 2 : t * 2 + width].copy()
        d = np.full((height, width), 2, np.float32)
        start = 70 - t * 12
        im[20:65, start : start + 30] = [240, 20, 20]
        d[20:65, start : start + 30] = 12
        pairs.append(im)
        maps.append(d)
    mask = np.ones((height, width), bool)
    return Analysis(
        *pairs,
        maps[0],
        -maps[1],
        mask.copy(),
        mask.copy(),
        mask.copy(),
        mask.copy(),
        {"alignment": {"depth_order_sign": 1}, "backend": "synthetic"},
    )


def test_mpo_decode_swap_and_bad_input(tmp_path):
    left = Image.new("RGB", (200, 100), "red")
    right = Image.new("RGB", (200, 100), "blue")
    path = tmp_path / "sample.mpo"
    left.save(path, format="MPO", save_all=True, append_images=[right])
    left_array, r, info = read_pair(path, width=160)
    assert (
        left_array.shape == (80, 160, 3)
        and left_array[:, :, 0].mean() > 250
        and r[:, :, 2].mean() > 250
    )
    rl, rr, _ = read_pair(path, width=160, swap=True)
    assert np.array_equal(rl, r) and np.array_equal(rr, left_array)
    assert info["source_frames"] == 2
    left.save(tmp_path / "left.jpg")
    right.save(tmp_path / "right.jpg")
    read_pair(tmp_path / "left.jpg", tmp_path / "right.jpg", width=160)
    with pytest.raises(ValueError, match="one image"):
        read_pair(tmp_path / "left.jpg")
    bad = tmp_path / "bad.mpo"
    bad.write_text("not an image")
    with pytest.raises(ValueError, match="Cannot decode"):
        read_pair(bad)


def test_vertical_alignment_preserves_horizontal_disparity():
    rng = np.random.default_rng(7)
    left = rng.integers(0, 256, (240, 400, 3), dtype=np.uint8)
    left = cv2.GaussianBlur(left, (3, 3), 0.6)
    right = cv2.warpAffine(left, np.float32([[1, 0, -10], [0, 1, 4]]), (400, 240))
    _, corrected, _, _, info = align_pair(left, right)
    assert info["vertical_median_px"] < 0.25
    assert abs(info["right_affine"][1][2] + 4) < 0.3
    assert abs(info["disparity_percentiles"][1] - 10) < 0.2
    assert np.mean(abs(corrected[5:-5, :-10].astype(float) - left[5:-5, 10:])) < 1


def test_anchor_and_known_midpoint():
    a = scene()
    anchor = choose_anchor(a, (85, 40))
    assert anchor.disparity == 12 and not anchor.automatic
    for t in np.linspace(0, 1, 7):
        frame, valid, _ = render_view(a, anchor, float(t))
        foreground = (frame[:, :, 0] > 225) & (frame[:, :, 1] < 25)
        yy, xx = np.where(foreground)
        assert xx.min() == 70 and xx.max() == 99
        assert abs(xx.mean() - 84.5) < 0.1
    frame, _, _ = render_view(a, anchor, 0.5)
    # A background point shifts +5 pixels relative to anchored foreground.
    assert np.array_equal(frame[10, 40], a.left[10, 35])
    # Region revealed by the right eye retains the background texture.
    assert np.array_equal(frame[40, 102], a.right[40, 95])
    assert np.array_equal(render_view(a, anchor, 0)[0], a.left)


def test_visibility_prioritizes_near_surface_and_negative_offsets():
    image = np.zeros((1, 8, 3), np.uint8)
    image[0, 2] = [20, 200, 20]
    image[0, 4] = [240, 20, 20]
    d = np.zeros((1, 8), np.float32)
    d[0, 4] = 4
    support = np.zeros((1, 8), bool)
    support[0, [2, 4]] = True
    rendered, _, _ = splat(image, d, support, 0.5, 0)
    assert np.array_equal(rendered[0, 2], [240, 20, 20])
    # Subtracting a common offset must not invert foreground ordering.
    rendered, _, _ = splat(image, d - 10, support, 0.5, -10)
    assert np.array_equal(rendered[0, 2], [240, 20, 20])


def test_signed_consistency_and_invalid_anchor():
    a = scene()
    vl = consistency(a.dl, a.dr, a.support_l, a.support_r)
    assert vl[40, 80]
    a.valid_l = np.zeros_like(vl)
    a.valid_l[20:60, 110:130] = True
    anchor = choose_anchor(a, (85, 40))
    assert anchor.automatic and "too few" in anchor.message
    with pytest.raises(ValueError, match="inside"):
        choose_anchor(a, (-1, 10))


def test_sequence_zero_motion_crop_and_loop(tmp_path):
    a = scene()
    anchor = choose_anchor(a, (85, 40))
    m = Motion(strength=0, views=5)
    seq = Sequence(a, anchor, m, tmp_path)
    assert loop_indices(m) == [0, 1, 2, 3, 4, 3, 2, 1]
    assert np.all(view_positions(m) == 0.5)
    assert all(np.array_equal(seq.frame(0), seq.frame(i)) for i in range(5))
    h, w = seq.frame(0).shape[:2]
    assert h % 2 == 0 and w % 2 == 0
    assert max(seq.visible_holes) == 0
    mask = np.zeros((10, 20), bool)
    mask[2:8, 3:17] = True
    assert largest_rectangle(mask) == (3, 2, 17, 8)


def test_exports_and_bundle(tmp_path):
    a = scene()
    anchor = choose_anchor(a, (85, 40))
    m = Motion(mode="classic", hold_ms=200, fps=30, cycles=3)
    seq = Sequence(a, anchor, m, tmp_path / "frames")
    gif = write_animation(seq, tmp_path / "a.gif")
    with Image.open(gif) as im:
        assert im.info["loop"] == 0 and im.n_frames == 2 and im.info["duration"] == 200
    with pytest.raises(FileExistsError):
        write_animation(seq, gif)
    bundle = write_bundle(a, seq, tmp_path / "assets")
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["metric_depth"] is False and manifest["render"]["crop_xyxy"] == list(seq.crop)
    with np.load(bundle / "stereo.npz") as data:
        assert np.array_equal(data["disparity_left"], a.dl)
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        mp4 = write_animation(seq, tmp_path / "a.mp4")
        info = json.loads(
            subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(mp4)]
            )
        )
        stream = info["streams"][0]
        assert stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
        assert stream["r_frame_rate"] == "30/1"
        assert abs(float(info["format"]["duration"]) - 1.2) < 0.02
        raw = subprocess.check_output(
            ["ffmpeg", "-v", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        )
        frames = np.frombuffer(raw, np.uint8).reshape(-1, stream["height"], stream["width"], 3)
        assert len(frames) == 36
        assert np.mean(abs(frames[0].astype(float) - seq.frame(0))) < 12
        assert np.mean(abs(frames[6].astype(float) - seq.frame(1))) < 12
        blob = mp4.read_bytes()
        assert blob.index(b"moov") < blob.index(b"mdat")


def test_cancel_does_not_publish(tmp_path):
    a = scene()
    seq = Sequence(a, choose_anchor(a, (85, 40)), Motion(views=3), tmp_path / "frames")
    output = tmp_path / "cancel.gif"
    with pytest.raises(InterruptedError):
        write_animation(seq, output, cancelled=lambda: True)
    assert not output.exists()
    assert not list(tmp_path.glob(".cancel-*"))


def test_sparse_classic_survives_textureless_input(tmp_path):
    im = Image.new("RGB", (200, 100), "gray")
    path = tmp_path / "plain.mpo"
    im.save(path, format="MPO", save_all=True, append_images=[im])
    a = analyze(path, width=160, backend="sparse", cache_dir=tmp_path / "cache")
    anchor = choose_anchor(a)
    assert anchor.disparity == 0 and "No reliable" in anchor.message
    cached = analyze(path, width=160, backend="sparse", cache_dir=tmp_path / "cache")
    assert np.array_equal(a.left, cached.left)
    assert cache_key(path, None, 160, False, "auto", "sgbm") != cache_key(
        path, None, 160, True, "auto", "sgbm"
    )


def test_measured_depth_roundtrip(tmp_path):
    a = scene()
    h, w = a.dl.shape
    # f=100 px; baseline=.075 m; Q produces Z=f*B/disparity.
    a.metadata["alignment"]["Q"] = [
        [1, 0, 0, -w / 2],
        [0, 1, 0, -h / 2],
        [0, 0, 0, 100],
        [0, 0, 1 / 0.075, 0],
    ]
    seq = Sequence(a, choose_anchor(a, (85, 40)), Motion(views=3), tmp_path / "frames")
    bundle = write_bundle(a, seq, tmp_path / "calibrated")
    z = np.load(bundle / "depth_meters.npy")
    assert abs(z[40, 85] - 0.625) < 1e-6 and abs(z[10, 40] - 3.75) < 1e-6


def test_unreliable_second_eye_does_not_cover_reference():
    a = scene()
    a.right[:] = [10, 10, 240]
    a.valid_r[:] = False
    # A spurious near prediction in an obstructed second lens must not win by depth alone.
    a.dr[:] = -45
    frame, _, _ = render_view(a, choose_anchor(a, (85, 40)), 0.5)
    assert np.array_equal(frame[40, 80], [240, 20, 20])
    assert not np.any((frame[25:60, 75:95, 2] > 200) & (frame[25:60, 75:95, 0] < 30))


def test_measured_profile_rejects_wrong_capture_mode(tmp_path):
    profile = {
        "image_size": [200, 200],
        "K1": [[100, 0, 100], [0, 100, 100], [0, 0, 1]],
        "K2": [[100, 0, 100], [0, 100, 100], [0, 0, 1]],
        "D1": [0, 0, 0, 0, 0],
        "D2": [0, 0, 0, 0, 0],
        "R": np.eye(3).tolist(),
        "T": [-0.075, 0, 0],
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(profile))
    image = np.zeros((100, 200, 3), np.uint8)
    with pytest.raises(ValueError, match="aspect ratio"):
        align_pair(image, image, str(path))


def test_smooth_gif_loop_timing_and_frame_order(tmp_path):
    a = scene()
    m = Motion(views=5, fps=30, strength=0.8)
    seq = Sequence(a, choose_anchor(a, (85, 40)), m, tmp_path / "frames")
    path = write_animation(seq, tmp_path / "smooth.gif")
    with Image.open(path) as im:
        total = 0
        decoded = []
        for i in range(im.n_frames):
            im.seek(i)
            total += im.info["duration"]
            decoded.append(np.array(im.convert("RGB")))
        assert im.n_frames == 8 and abs(total - 8000 / 30) <= 10
    assert np.array_equal(decoded[1], decoded[-1])
    assert np.array_equal(decoded[2], decoded[-2])
    assert not np.array_equal(decoded[0], decoded[-1])
