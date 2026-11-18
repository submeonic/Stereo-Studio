import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import time

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")
from PySide6 import QtCore, QtTest, QtWidgets

from wigglegram.gui import Window


def wait_idle(app, window, timeout=15):
    deadline = time.monotonic() + timeout
    while window.busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert not window.busy


def test_desktop_preview_anchor_and_stale_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(3)
    pixels = rng.integers(0, 256, (160, 320, 3), np.uint8)
    path = tmp_path / "test.mpo"
    Image.fromarray(pixels).save(
        path,
        format="MPO",
        save_all=True,
        append_images=[Image.fromarray(np.roll(pixels, -6, axis=1))],
    )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = Window()
    assert win.tabs.widget(3) is win.preview
    win.backend.setCurrentIndex(0)
    win.width.setCurrentIndex(0)
    win.mode.setCurrentIndex(1)
    win.load_file(str(path))
    win.show()
    app.processEvents()
    # Anchor placement and its indication work BEFORE any stereo preview exists.
    assert win.analysis is None
    win.set_anchor(160, 80)
    assert win.point == (160, 80) and win.left.anchor is not None
    assert "depth" in win.hint.text().lower()
    assert win.width.toolTip() and win.aa.toolTip()
    win.build()
    wait_idle(app, win)
    assert win.sequence is not None and win.export_button.isEnabled()
    assert win.analysis.metadata["backend"] == "sparse"
    win.tabs.setCurrentIndex(0)
    app.processEvents()
    clicks = []
    win.left.clicked.connect(lambda x, y: clicks.append((x, y)))
    r = win.left.image_rect()
    p = QtCore.QPoint(round(r.center().x()), round(r.center().y()))
    QtTest.QTest.mouseClick(win.left, QtCore.Qt.LeftButton, pos=p)
    assert clicks and abs(clicks[0][0] - 160) < 1 and abs(clicks[0][1] - 80) < 1
    assert not win.export_button.isEnabled()
    win.build()
    wait_idle(app, win)
    assert win.export_button.isEnabled()
    win.swap.setChecked(True)
    assert win.analysis is None and not win.export_button.isEnabled()
    win.close()
    app.processEvents()


def test_photo_crop_controls_and_anaglyph_ui(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(23)
    pixels = rng.integers(0, 256, (160, 320, 3), np.uint8)
    path = tmp_path / "photo.mpo"
    Image.fromarray(pixels).save(
        path,
        format="MPO",
        save_all=True,
        append_images=[Image.fromarray(np.roll(pixels, -5, axis=1))],
    )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = Window()
    assert win.tabs.widget(3) is win.preview
    win.backend.setCurrentIndex(0)
    win.mode.setCurrentIndex(2)
    win.load_file(str(path))
    win.show()
    app.processEvents()
    win.set_anchor(160, 80)
    win.build()
    wait_idle(app, win)
    assert win.sequence.motion.mode == "anaglyph" and not win.timer.isActive()
    assert not win.fps.isEnabled() and not win.aa.isEnabled()
    assert "JPEG" in win.export_button.text()
    win.tabs.setCurrentWidget(win.photo)
    app.processEvents()
    win.photo.fields["exposure"].setValue(0.7)
    win.photo.aspect.setCurrentIndex(2)
    win.refresh_photo()
    assert win.sequence.edits.exposure == 0.7
    h, w = win.sequence.frame(0).shape[:2]
    assert abs(h - w) <= 2
    canvas = win.photo.canvas
    r = canvas.image_rect()
    start = QtCore.QPoint(round(r.left() + r.width() * 0.15), round(r.top() + r.height() * 0.15))
    end = QtCore.QPoint(round(r.left() + r.width() * 0.75), round(r.top() + r.height() * 0.85))
    QtTest.QTest.mousePress(canvas, QtCore.Qt.LeftButton, pos=start)
    QtTest.QTest.mouseMove(canvas, end)
    QtTest.QTest.mouseRelease(canvas, QtCore.Qt.LeftButton, pos=end)
    win.refresh_photo()
    assert 0 <= canvas.crop[0] < canvas.crop[2] <= 1
    assert (
        abs(
            (canvas.crop[2] - canvas.crop[0]) * canvas.image.width()
            - (canvas.crop[3] - canvas.crop[1]) * canvas.image.height()
        )
        < 2
    )
    win.mode.setCurrentIndex(0)
    assert win.effects.isEnabled()
    assert win.aa.isChecked() and not win.cleanup.isChecked()
    win.effects.groups["tearing"].setChecked(True)
    win.effects.fields["tearing"]["bands"].setValue(12)
    assert win.effects.settings()["tearing"]["bands"] == 12
    assert win.render_dirty
    win.effects.bypass.setChecked(True)
    assert win.effects.settings() == {}
    win.width.setCurrentIndex(0)
    assert win.photo.fields["exposure"].value() == 0.7
    win.close()
    app.processEvents()


def test_temporal_preview_photo_rebuild_and_independent_export(tmp_path, monkeypatch):
    import shutil

    from wigglegram.temporal import TimelineSequence

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    monkeypatch.chdir(tmp_path)
    pixels = np.random.default_rng(4).integers(0, 256, (160, 320, 3), np.uint8)
    path = tmp_path / "effects.mpo"
    Image.fromarray(pixels).save(
        path,
        format="MPO",
        save_all=True,
        append_images=[Image.fromarray(np.roll(pixels, -5, axis=1))],
    )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: errors.append(args[-1]))
    win = Window()
    win.mode.setCurrentIndex(1)
    win.preview_width.setCurrentIndex(0)
    win.effects.groups["persistence"].setChecked(True)
    win.load_file(str(path))
    win.build()
    wait_idle(app, win)
    assert not errors
    assert isinstance(win.sequence, TimelineSequence)
    cached = win.analysis
    win.photo.fields["exposure"].setValue(0.3)
    assert win.render_dirty and not win.export_button.isEnabled()
    win.build()
    wait_idle(app, win)
    assert not errors and win.analysis is cached
    assert win.sequence.edits.exposure == 0.3
    output = tmp_path / "effects.mp4"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(output), "MP4")
    )
    # Preview remains 640; independently export 960 from original image data.
    win.export_width.setCurrentIndex(1)
    win.export_scale.setCurrentIndex(1)
    win.export()
    wait_idle(app, win)
    assert not errors and output.exists()
    import cv2

    cap = cv2.VideoCapture(str(output))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[1] == 960
    assert win.sequence.frame(0).shape[1] == 640
    win.close()
    app.processEvents()
