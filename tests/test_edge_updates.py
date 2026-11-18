import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtWidgets

from wigglegram.core import Analysis, Anchor
from wigglegram.curvature import coordinates, curve
from wigglegram.editing import PhotoEdits
from wigglegram.gui import Window
from wigglegram.render import Motion, Sequence, render_view, repair_unsupported_edges
from wigglegram.temporal import TimelineSequence, leak_schedule, validate


def uncertain_pair():
    h, w = 40, 100
    left = np.full((h, w, 3), [220, 20, 20], np.uint8)
    right = np.full((h, w, 3), [20, 20, 220], np.uint8)
    d = np.zeros((h, w), np.float32)
    valid = np.zeros((h, w), bool)
    support = ~valid
    return Analysis(
        left, right, d, d, valid, valid, support, support, {"alignment": {"depth_order_sign": 1}}
    )


def test_symmetric_endpoint_convergence_and_eye_exchange():
    a = uncertain_pair()
    anchor = Anchor(50, 20, 0, False, "")
    for t in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        swapped = replace(
            a, left=a.right, right=a.left, dl=a.dr, dr=a.dl, valid_l=a.valid_r, valid_r=a.valid_l
        )
        result = render_view(a, anchor, t)[0]
        reverse = render_view(swapped, anchor, 1 - t)[0]
        assert np.max(abs(result.astype(int) - reverse.astype(int))) <= 1
    before_mid = render_view(a, anchor, 0.499)[0]
    after_mid = render_view(a, anchor, 0.501)[0]
    assert np.max(abs(before_mid.astype(int) - after_mid.astype(int))) <= 1
    near = render_view(a, anchor, 0.999)[0]
    exact = render_view(a, anchor, 1)[0]
    assert np.max(abs(near.astype(int) - exact.astype(int))) <= 1
    legacy = render_view(a, anchor, 0.999, fusion="legacy")[0]
    assert abs(legacy.astype(float) - exact).mean() > 100


def test_local_repair_never_changes_sound_pixels_or_blurs_texture():
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, (20, 30, 3), np.uint8).astype(np.float32)
    depth = np.full((20, 30), 2, np.float32)
    trusted = np.ones(depth.shape, bool)
    bad = np.zeros(depth.shape, bool)
    bad[10, 15] = True
    trusted[bad] = False
    depth[bad] = 12
    background = np.full(depth.shape, 2, np.float32)
    result, mask = repair_unsupported_edges(image, depth, trusted, bad, background, 1)
    assert mask.sum() == 1 and mask[10, 15]
    assert np.array_equal(result[~bad], image[~bad])
    assert any(
        np.array_equal(result[10, 15], image[y, x])
        for y, x in [(10, 14), (10, 16), (9, 15), (11, 15)]
    )
    # Foreground-only neighbors cannot fill a background hole.
    depth[trusted] = 12
    result, mask = repair_unsupported_edges(image, depth, trusted, bad, background, 1)
    assert not mask.any() and np.array_equal(result, image)


def test_geometry_canvas_actually_visible_and_maps_switch():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = Window()
    win.analysis = uncertain_pair()
    win.refined_analysis = win.analysis
    win.analysis.dl[:] = np.arange(100)[None, :]
    win.analysis.dr[:] = -np.arange(100)[None, :]
    win.analysis.valid_l[:] = True
    win.analysis.valid_r[:] = True
    win.show()
    win.tabs.setCurrentIndex(2)
    win.refresh_diagnostic()
    app.processEvents()
    assert win.depth.isVisible() and win.depth.image is not None
    assert win.depth.width() > 100 and win.depth.height() > 100
    raw = win.depth.image.copy()
    win.diagnostic.setCurrentIndex(2)
    app.processEvents()
    assert win.depth.image != raw
    for i in range(4, 9):
        win.diagnostic.setCurrentIndex(i)
        app.processEvents()
        assert win.depth.isVisible() and win.depth.image is not None
    win.close()
    app.processEvents()


def test_curvature_bounds_and_original_detail_before_resize(tmp_path):
    for amount in [0, 0.1, 0.5, 1]:
        image = np.full((41, 80, 3), 180, np.uint8)
        assert np.all(curve(image, amount) == 180)  # no black/replicated external area
        y, x = np.indices((101, 101), dtype=np.float32)
        xx, yy = coordinates(x / 50 - 1, y / 50 - 1, amount)
        assert abs(xx).max() <= 1.000001 and abs(yy).max() <= 1.000001
    a = uncertain_pair()
    anchor = Anchor(50, 20, 0, False, "")
    base = Sequence(a, anchor, Motion(views=3), tmp_path / "base", output_width=64)
    seq = TimelineSequence(base, {"curvature": {"amount": 1}}, tmp_path / "fx")
    assert base.curvature_amount == 1
    assert seq.frame(0).shape[1] == 64
    assert "before output resize" in seq.metadata["curvature_stage"]


def test_fixed_leaks_exact_count_bursts_and_seed(tmp_path):
    settings = validate({"leak": {"frames": 3, "burst": 2, "amount": 0.8}})
    selected = leak_schedule(8, settings["leak"])
    assert len(selected) == 3 and selected == leak_schedule(8, settings["leak"])
    assert len(leak_schedule(2, settings["leak"])) == 2
    a = uncertain_pair()
    a.left[:] = 80
    a.right[:] = 80
    base = Sequence(
        a, Anchor(50, 20, 0, False, ""), Motion(views=5), tmp_path / "base", edits=PhotoEdits()
    )
    seq = TimelineSequence(base, settings, tmp_path / "fx")
    affected = [
        i for i in seq.indices if not np.array_equal(seq.frame(i), base.frame(base.indices[i]))
    ]
    assert set(affected) == selected
    assert all(np.array_equal(seq.frame(affected[0]), seq.frame(i)) for i in affected)
