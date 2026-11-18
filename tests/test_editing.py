import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from wigglegram.core import Analysis, Anchor
from wigglegram.editing import PhotoEdits, adjust_photo, crop_pixels, linear_rgb, ratio_crop
from wigglegram.effects import Style, anaglyph
from wigglegram.export import write_animation, write_bundle
from wigglegram.render import Motion, Sequence, render_view


def pair():
    yy, xx = np.indices((80, 160))
    image = np.stack((30 + xx, 50 + yy, 80 + xx // 2), axis=-1).astype(np.uint8)
    image[20:60, 60:95] = [240, 20, 20]
    d = np.full((80, 160), 4, np.float32)
    mask = np.ones(d.shape, bool)
    return Analysis(
        image,
        np.roll(image, -4, axis=1),
        d,
        -d,
        mask.copy(),
        mask.copy(),
        mask.copy(),
        mask.copy(),
        {"alignment": {"depth_order_sign": 1}, "backend": "synthetic"},
    )


def test_adjustments_identity_linear_exposure_and_monochrome():
    image = np.full((30, 40, 3), 80, np.uint8)
    assert np.array_equal(adjust_photo(image, PhotoEdits()), image)
    result = adjust_photo(image, PhotoEdits(exposure=1))
    assert abs(float(linear_rgb(result).mean() / linear_rgb(image).mean()) - 2) < 0.03
    colored = np.full((30, 40, 3), [150, 90, 30], np.uint8)
    gray = adjust_photo(colored, PhotoEdits(saturation=0))
    assert np.array_equal(gray[:, :, 0], gray[:, :, 1]) and np.array_equal(
        gray[:, :, 1], gray[:, :, 2]
    )
    with pytest.raises(ValueError):
        PhotoEdits(exposure=float("nan")).validate()
    with pytest.raises(ValueError):
        PhotoEdits(crop=(0.8, 0, 0.2, 1)).validate()


def test_crop_rotation_export_and_manifest_match(tmp_path):
    a = pair()
    edits = PhotoEdits(exposure=0.2, crop=(0.1, 0.2, 0.9, 0.8))
    sequence = Sequence(
        a,
        Anchor(80, 40, 4, False, ""),
        Motion(mode="classic", rotation=90),
        tmp_path / "frames",
        edits=edits,
    )
    full = sequence.full_frame(0)
    x0, y0, x1, y1 = crop_pixels(full.shape, edits.crop)
    assert np.array_equal(sequence.frame(0), full[y0:y1, x0:x1])
    crop = ratio_crop((0, 0, 1, 1), full.shape, 1)
    sequence.set_edits(replace(edits, crop=crop))
    h, w = sequence.frame(0).shape[:2]
    assert abs(h - w) <= 2
    bundle = write_bundle(a, sequence, tmp_path / "bundle")
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["render"]["photo_edits"]["exposure"] == 0.2
    assert manifest["render"]["photo_crop_xyxy"] == list(sequence.output_crop)
    # Geometry cache/source arrays remain unedited.
    assert np.array_equal(a.left, pair().left)


def test_anaglyph_eye_assignment_and_still_export(tmp_path):
    left = np.full((10, 20, 3), [100, 200, 230], np.uint8)
    right = np.full((10, 20, 3), [240, 80, 60], np.uint8)
    assert np.array_equal(anaglyph(left, right, "color")[0, 0], [100, 80, 60])
    a = pair()
    m = Motion(mode="anaglyph", anaglyph_method="color")
    seq = Sequence(
        a, Anchor(80, 40, 4, False, ""), m, tmp_path / "frames", edits=PhotoEdits(saturation=0)
    )
    assert seq.indices == [0] and len(seq.positions) == 1
    output = write_animation(seq, tmp_path / "stereo.jpg")
    with Image.open(output) as im:
        assert im.format == "JPEG" and getattr(im, "n_frames", 1) == 1
        assert np.mean(abs(np.asarray(im).astype(float) - seq.frame(0))) < 3
    with pytest.raises(ValueError, match="Anaglyph"):
        write_animation(seq, tmp_path / "bad.mp4")
    with pytest.raises(ValueError):
        write_animation(
            Sequence(a, Anchor(80, 40, 4, False, ""), Motion(views=3), tmp_path / "motion"),
            tmp_path / "motion.jpg",
        )
    with pytest.raises(ValueError):
        Sequence(
            a, Anchor(80, 40, 4, False, ""), m, tmp_path / "bad", style=Style(name="stereo-echo")
        )


def test_glitch_deterministic_identity_and_aa_override(tmp_path):
    a = pair()
    anchor = Anchor(80, 40, 4, False, "")
    motion = Motion(views=3, antialias=True)
    art = Sequence(a, anchor, motion, tmp_path / "one", style=Style(name="stereo-echo"))
    again = Sequence(a, anchor, motion, tmp_path / "two", style=Style(name="stereo-echo"))
    assert not art.effective_aa
    assert all(np.array_equal(art.frame(i), again.frame(i)) for i in range(3))
    off = Sequence(
        a, anchor, replace(motion, antialias=False, edge_cleanup=False), tmp_path / "off"
    )
    zero = Sequence(a, anchor, motion, tmp_path / "zero", style=Style(name="stereo-echo", amount=0))
    assert np.array_equal(zero.frame(1), off.frame(1))
    assert not np.array_equal(art.frame(1), off.frame(1))


def test_aa_dimensions_edge_coverage_and_clean_fusion(tmp_path):
    a = pair()
    anchor = Anchor(80, 40, 4, False, "")
    aa = Sequence(a, anchor, Motion(views=3, antialias=True), tmp_path / "aa")
    off = Sequence(a, anchor, Motion(views=3, antialias=False), tmp_path / "off")
    assert aa.effective_aa and aa.frame(1).shape == off.frame(1).shape
    # The flat interior is preserved while the hard silhouette gains subpixel coverage.
    assert np.array_equal(aa.frame(1)[35, 75], off.frame(1)[35, 75])
    assert not np.array_equal(aa.frame(1), off.frame(1))
    a.right[:] = [20, 220, 20]
    clean = render_view(a, anchor, 0.5, edge_cleanup=True, fusion="legacy")[0]
    blend = render_view(a, anchor, 0.5, edge_cleanup=False)[0]
    assert np.array_equal(clean[40, 75], [240, 20, 20])
    assert not np.array_equal(clean[40, 75], blend[40, 75])


def test_sparse_clicked_patch_fallback_is_verified():
    from wigglegram.core import choose_anchor

    a = pair()
    rng = np.random.default_rng(71)
    a.left = rng.integers(0, 256, a.left.shape, np.uint8)
    a.right = np.roll(a.left, -7, axis=1)
    a.valid_l[:] = False
    a.valid_r[:] = False
    a.metadata["backend"] = "sparse"
    anchor = choose_anchor(a, (80, 40))
    assert not anchor.automatic and abs(anchor.disparity - 7) < 0.1
    a.left[:] = 100
    a.right[:] = 100
    fallback = choose_anchor(a, (80, 40))
    assert fallback.automatic
