import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest
from PIL import Image
from PySide6 import QtWidgets

from wigglegram.core import analyze, cache_key
from wigglegram.editing import PhotoEdits
from wigglegram.gui import Window
from wigglegram.user_presets import load_preset, photo_state, save_preset


def test_defaults_and_main_label_tooltips():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = Window()
    assert win.width.currentData() == 1280
    assert win.views.value() == 7 and win.fps.value() == 20
    assert win.preview_width.currentData() == 960
    assert win.export_width.currentData() == 1280
    assert win.export_scale.currentData() == 2 and win.aa.isChecked()
    assert "experimental" not in win.aa.text().lower()
    for row in range(win.parameter_form.rowCount()):
        label = win.parameter_form.itemAt(row, QtWidgets.QFormLayout.LabelRole)
        field = win.parameter_form.itemAt(row, QtWidgets.QFormLayout.FieldRole)
        assert label.widget().toolTip() and label.widget().toolTip() == field.widget().toolTip()
    win.effects.groups["rgb"].setChecked(True)
    assert win.aa.isEnabled() and win.motion().antialias
    win.close()
    app.processEvents()


def test_photo_preset_validation_and_atomic_roundtrip(tmp_path):
    path = tmp_path / "photo.json"
    state = photo_state(
        PhotoEdits(exposure=0.7, saturation=0.4, crop=(0.1, 0.2, 0.9, 0.8)), 90, 1.5
    )
    save_preset(path, "photo", state)
    restored = load_preset(path, "photo")
    assert restored["rotation"] == 90 and restored["adjustments"]["exposure"] == 0.7
    assert restored["adjustments"]["crop"] == [0.1, 0.2, 0.9, 0.8]
    with pytest.raises(ValueError, match="effects panel"):
        load_preset(path, "effects")
    original = path.read_bytes()
    state["adjustments"]["exposure"] = float("nan")
    with pytest.raises(ValueError):
        save_preset(path, "photo", state)
    assert path.read_bytes() == original
    data = json.loads(original)
    data["version"] = 99
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_preset(path, "photo")


def test_gui_save_load_preserves_disabled_effects_and_photo(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = Window()
    errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: errors.append(args[-1]))
    paths = {"effects": tmp_path / "effects.json", "photo": tmp_path / "photo.json"}
    current = ["effects"]
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(paths[current[0]]), "JSON")
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", lambda *args: (str(paths[current[0]]), "JSON")
    )
    win.effects.fields["rgb"]["offset"].setValue(9)
    win.effects.groups["persistence"].setChecked(True)
    win.effects.bypass.setChecked(True)
    win.effects.findChild(QtWidgets.QPushButton, "effects_save").click()
    win.effects.reset()
    win.effects.findChild(QtWidgets.QPushButton, "effects_load").click()
    assert win.effects.fields["rgb"]["offset"].value() == 9
    assert not win.effects.groups["rgb"].isChecked()
    assert win.effects.groups["persistence"].isChecked() and win.effects.bypass.isChecked()
    current[0] = "photo"
    win.photo.fields["exposure"].setValue(1.2)
    win.rotation.setCurrentIndex(1)
    win.photo.canvas.crop = (0.1, 0.1, 0.8, 0.9)
    win.photo.findChild(QtWidgets.QPushButton, "photo_save").click()
    win.photo.reset()
    win.rotation.setCurrentIndex(0)
    win.load_user_preset("photo")
    assert win.photo.fields["exposure"].value() == 1.2 and win.rotation.currentData() == 90
    assert win.photo.canvas.crop == (0.1, 0.1, 0.8, 0.9)
    assert not errors
    # Invalid data must leave all controls untouched.
    paths["photo"].write_text('{"schema":"bad"}')
    before = win.photo.edits()
    win.load_user_preset("photo")
    assert errors and win.photo.edits() == before
    win.close()
    app.processEvents()


def test_nominal_3ds_mpo_path_and_cache_identity(tmp_path):
    image = np.random.default_rng(31).integers(0, 256, (240, 320, 3), np.uint8)
    path = tmp_path / "stereo.mpo"
    Image.fromarray(image).save(
        path,
        format="MPO",
        save_all=True,
        append_images=[Image.fromarray(np.roll(image, -4, axis=1))],
    )
    a = analyze(path, width=960, profile="3ds", backend="sparse", cache_dir=None)
    assert a.left.shape[:2] == (240, 320)  # no analysis upscaling
    assert a.metadata["alignment"]["profile"] == "3ds"
    assert a.metadata["alignment"]["type"] == "nominal" and "Q" not in a.metadata["alignment"]
    assert cache_key(path, None, 960, False, "3ds", "sparse") != cache_key(
        path, None, 960, False, "auto", "sparse"
    )


def test_cli_accepts_user_effect_preset(tmp_path, monkeypatch, capsys):
    from wigglegram.cli import main, parser
    from wigglegram.temporal import SPECS, validate

    monkeypatch.chdir(tmp_path)
    image = np.random.default_rng(4).integers(0, 256, (100, 160, 3), np.uint8)
    source = tmp_path / "input.mpo"
    Image.fromarray(image).save(
        source,
        format="MPO",
        save_all=True,
        append_images=[Image.fromarray(np.roll(image, -3, axis=1))],
    )
    preset = tmp_path / "effects.json"
    save_preset(
        preset,
        "effects",
        dict(parameters=validate({name: {} for name in SPECS}), enabled=["rgb"], bypass=False),
    )
    output = tmp_path / "output.gif"
    assert (
        main(
            [
                "render",
                str(source),
                "--mode",
                "classic",
                "--width",
                "160",
                "--output-width",
                "160",
                "--supersample",
                "1",
                "--effects-json",
                str(preset),
                "-o",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert set(report["render"]["effects"]) == {"rgb"}
    assert output.exists()
    defaults = parser().parse_args(["render", "input.mpo", "-o", "out.mp4"])
    assert defaults.output_width == 1280 and defaults.supersample == 2
    assert (defaults.width, defaults.views, defaults.fps) == (1280, 7, 20)
