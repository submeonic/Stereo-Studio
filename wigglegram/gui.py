from __future__ import annotations

import sys
import tempfile
import threading
import traceback
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore
from PySide6 import QtWidgets as Q

from .core import Anchor, align_pair, analyze, choose_anchor, read_pair
from .editing import adjust_photo, balance_gains
from .export import write_animation, write_bundle
from .geometry import refine_analysis, reliability, render_grid
from .neural import available
from .render import Motion, Sequence, render_view
from .temporal import TimelineSequence
from .user_presets import load_preset, photo_state, save_preset
from .widgets import Canvas, EffectsPanel, PhotoPanel


class Worker(QtCore.QThread):
    progress = QtCore.Signal(str)
    result = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.cancel = threading.Event()

    def run(self):
        try:
            self.result.emit(self.fn(self.progress.emit, self.cancel.is_set))
        except InterruptedError:
            self.failed.emit("Cancelled. No partial export was kept.")
        except Exception as exc:
            self.failed.emit(str(exc))
            traceback.print_exc()


class Window(Q.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stereo Studio")
        self.resize(1280, 840)
        self.source = None
        self.right_path = None
        self.point = None
        self.analysis = None
        self.loaded_left = None
        self.render_dirty = True
        self.sequence = None
        self.temp = None
        self.worker = None
        self.frame_index = 0
        self.busy = False
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.advance)
        root = Q.QWidget()
        self.setCentralWidget(root)
        outer = Q.QVBoxLayout(root)
        outer.setContentsMargins(24, 20, 24, 16)
        title = Q.QLabel("STEREO STUDIO")
        title.setStyleSheet("font-size:24px;font-weight:700;letter-spacing:3px")
        subtitle = Q.QLabel("Give your photographs a little room to move.")
        subtitle.setStyleSheet("color:#9caabd;font-size:14px")
        outer.addWidget(title)
        outer.addWidget(subtitle)
        layout = Q.QHBoxLayout()
        outer.addLayout(layout, 1)
        self.controls = Q.QWidget()
        self.controls.setFixedWidth(330)
        side = Q.QVBoxLayout(self.controls)
        side.setContentsMargins(0, 12, 14, 0)
        open_button = Q.QPushButton("Open MPO…")
        open_button.clicked.connect(self.open_mpo)
        pair_button = Q.QPushButton("Open left / right JPEGs…")
        pair_button.clicked.connect(self.open_pair)
        side.addWidget(open_button)
        side.addWidget(pair_button)
        self.file_label = Q.QLabel("No photo selected")
        self.file_label.setWordWrap(True)
        self.file_label.setToolTip("Currently loaded MPO or left/right image pair.")
        side.addWidget(self.file_label)
        self.swap = Q.QCheckBox("Swap eyes")
        side.addWidget(self.swap)
        form = Q.QFormLayout()
        side.addLayout(form)
        self.profile = Q.QComboBox()
        self.profile.addItem("Auto · image alignment", "auto")
        self.profile.addItem("W3 Wide · nominal", "w3-wide")
        self.profile.addItem("Nintendo 3DS · nominal", "3ds")
        form.addRow("Camera", self.profile)
        load_profile = Q.QPushButton("Load measured calibration…")
        load_profile.clicked.connect(self.load_profile)
        side.addWidget(load_profile)
        form2 = Q.QFormLayout()
        self.parameter_form = form2
        side.addLayout(form2)
        self.backend = Q.QComboBox()
        self.backend.addItem("CPU · SGBM", "sgbm")
        self.backend.addItem("GPU · Fast-FoundationStereo", "neural")
        if available():
            self.backend.setCurrentIndex(1)
        form2.addRow("Stereo", self.backend)
        self.width = Q.QComboBox()
        for width in [640, 960, 1280, 1920]:
            self.width.addItem(f"{width} px", width)
        self.width.setCurrentIndex(1)
        form2.addRow("Geometry width", self.width)
        self.preview_width = Q.QComboBox()
        self.export_width = Q.QComboBox()
        for value in [640, 960, 1280, 1920]:
            self.preview_width.addItem(f"{value} px", value)
            self.export_width.addItem(f"{value} px", value)
        self.preview_width.setCurrentIndex(1)
        self.export_width.setCurrentIndex(2)
        self.export_scale = Q.QComboBox()
        for value in [1, 2]:
            self.export_scale.addItem(f"{value}×", value)
        self.export_scale.setCurrentIndex(1)
        self.refinement = Q.QCheckBox("Refine uncertain geometry edges")
        self.refinement.setChecked(True)
        form2.addRow("Preview width", self.preview_width)
        form2.addRow("Export width", self.export_width)
        form2.addRow("Export supersampling", self.export_scale)
        side.addWidget(self.refinement)
        self.mode = Q.QComboBox()
        self.mode.addItem("Smooth 3D motion", "smooth")
        self.mode.addItem("Classic two-frame", "classic")
        self.mode.addItem("Anaglyph 3D · still JPEG", "anaglyph")
        form2.addRow("Mode", self.mode)
        self.sweep = Q.QDoubleSpinBox()
        self.sweep.setRange(0, 1)
        self.sweep.setSingleStep(0.05)
        self.sweep.setValue(0.65)
        form2.addRow("Sweep", self.sweep)
        self.views = Q.QSpinBox()
        self.views.setRange(2, 121)
        self.views.setValue(25)
        form2.addRow("Views", self.views)
        self.fps = Q.QSpinBox()
        self.fps.setRange(1, 60)
        self.fps.setValue(30)
        form2.addRow("FPS", self.fps)
        self.hold = Q.QSpinBox()
        self.hold.setRange(20, 5000)
        self.hold.setValue(180)
        self.hold.setSuffix(" ms")
        form2.addRow("Classic hold", self.hold)
        self.cycles = Q.QSpinBox()
        self.cycles.setRange(1, 100)
        self.cycles.setValue(4)
        form2.addRow("MP4 cycles", self.cycles)
        self.rotation = Q.QComboBox()
        for angle in [0, 90, 180, 270]:
            self.rotation.addItem(f"{angle}° clockwise", angle)

        self.easing = Q.QCheckBox("Ease at turnarounds")
        self.easing.setChecked(True)
        side.addWidget(self.easing)
        self.anaglyph_method = Q.QComboBox()
        for label, value in [
            ("Dubois · color optimized", "dubois"),
            ("Color · red / cyan", "color"),
            ("Monochrome · red / cyan", "gray"),
        ]:
            self.anaglyph_method.addItem(label, value)
        form2.addRow("Anaglyph", self.anaglyph_method)
        self.aa = Q.QCheckBox("Preview frame AA · 2×")
        self.aa.setChecked(True)
        side.addWidget(self.aa)
        self.cleanup = Q.QCheckBox("Compare: mismatched-edge blending rejection")
        self.cleanup.setChecked(False)
        auto = Q.QPushButton("Use automatic anchor")
        auto.clicked.connect(self.auto_anchor)
        side.addWidget(auto)
        self.generate = Q.QPushButton("Build preview")
        self.generate.setObjectName("primary")
        self.generate.clicked.connect(self.build)

        self.assets = Q.QCheckBox("Also save stereo assets")

        self.export_button = Q.QPushButton("Export GIF / MP4…")
        self.export_button.clicked.connect(self.export)
        self.export_button.setEnabled(False)

        side.addStretch()
        scroll = Q.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(Q.QFrame.NoFrame)
        scroll.setWidget(self.controls)
        scroll.setMinimumWidth(355)
        scroll.setMaximumWidth(375)
        sidebar = Q.QVBoxLayout()
        sidebar.addWidget(scroll, 1)
        self.actions = Q.QWidget()
        actions_layout = Q.QVBoxLayout(self.actions)
        actions_layout.setContentsMargins(0, 4, 12, 0)
        for widget in (self.generate, self.assets, self.export_button):
            actions_layout.addWidget(widget)
        sidebar.addWidget(self.actions)
        layout.addLayout(sidebar)
        main = Q.QVBoxLayout()
        layout.addLayout(main, 1)
        self.tabs = Q.QTabWidget()
        self.left = Canvas(True)
        self.right = Canvas()
        self.depth = Canvas()
        self.depth.empty_text = "Build a preview to calculate depth / disparity."
        self.preview = Canvas()
        for name, canvas in [
            ("Anchor · left eye", self.left),
            ("Right eye", self.right),
            ("Animation", self.preview),
        ]:
            self.tabs.addTab(canvas, name)
        geometry_tab = Q.QWidget()
        geometry_layout = Q.QVBoxLayout(geometry_tab)
        self.diagnostic = Q.QComboBox()
        self.diagnostic.addItems(
            [
                "Depth / disparity · raw",
                "Depth / disparity · refined",
                "Match reliability",
                "Refinement difference",
                "Left projection",
                "Right projection",
                "Source contribution · red = left, blue = right",
                "Projection disagreement",
                "Edge repair mask",
            ]
        )
        self.diagnostic.setCurrentIndex(1)
        self.diagnostic.currentIndexChanged.connect(self.refresh_diagnostic)
        self.diagnostic.setToolTip("Display only: selecting a map does not change rendering.")
        geometry_layout.addWidget(Q.QLabel("Display diagnostic (does not change rendering)"))
        geometry_layout.addWidget(self.diagnostic)
        self.diagnostic_eye = Q.QComboBox()
        self.diagnostic_eye.addItems(["Left eye", "Right eye"])
        self.diagnostic_eye.currentIndexChanged.connect(self.refresh_diagnostic)
        geometry_layout.addWidget(self.diagnostic_eye)
        self.diagnostic_t = Q.QDoubleSpinBox()
        self.diagnostic_t.setRange(0, 1)
        self.diagnostic_t.setPrefix("Viewpoint: ")
        self.diagnostic_t.setFixedWidth(150)
        self.diagnostic_t.setSingleStep(0.05)
        self.diagnostic_t.setValue(0.5)
        self.diagnostic_t.setToolTip(
            "Virtual viewpoint for projection diagnostics: 0 left, 1 right."
        )
        self.diagnostic_t.valueChanged.connect(self.refresh_diagnostic)
        geometry_layout.addWidget(self.diagnostic_t)
        self.fusion = Q.QComboBox()
        for label, value in [
            ("Symmetric source selection", "symmetric"),
            ("Legacy left bias · comparison", "legacy"),
            ("Left priority · comparison", "left"),
            ("Right priority · comparison", "right"),
        ]:
            self.fusion.addItem(label, value)
        self.fusion.currentIndexChanged.connect(self.motion_changed)
        geometry_layout.addWidget(Q.QLabel("Rendering controls (rebuild preview to apply)"))
        geometry_layout.addWidget(self.fusion)
        self.edge_repair = Q.QCheckBox("Repair unsupported edge conflicts only")
        self.edge_repair.setChecked(True)
        self.edge_repair.setToolTip(
            "Use nearby reliable samples from the background surface only at unsupported conflicting edges. No whole-image blur."
        )
        self.edge_repair.toggled.connect(self.motion_changed)
        geometry_layout.addWidget(self.edge_repair)
        geometry_layout.addWidget(self.cleanup)
        geometry_layout.addWidget(self.depth, 1)
        self.depth.show()
        self.tabs.insertTab(2, geometry_tab, "Geometry")
        self.photo = PhotoPanel(self.rotation)
        self.tabs.addTab(self.photo, "Photo edit")
        self.photo.changed.connect(self.photo_changed)
        self.effects = EffectsPanel()
        self.tabs.addTab(self.effects, "Effects")
        self.effects.changed.connect(self.mode_changed)
        self.photo.save_requested.connect(lambda: self.save_user_preset("photo"))
        self.photo.load_requested.connect(lambda: self.load_user_preset("photo"))
        self.effects.save_requested.connect(lambda: self.save_user_preset("effects"))
        self.effects.load_requested.connect(lambda: self.load_user_preset("effects"))
        self.edit_timer = QtCore.QTimer(self)
        self.edit_timer.setSingleShot(True)
        self.edit_timer.timeout.connect(self.refresh_photo)
        self.left.clicked.connect(self.set_anchor)
        main.addWidget(self.tabs, 1)
        self.hint = Q.QLabel(
            "Click the subject in the left eye, then build a preview to keep it still."
        )
        self.hint.setWordWrap(True)
        main.addWidget(self.hint)
        self.details = Q.QLabel("Local processing • Original photographs stay untouched")
        self.details.setWordWrap(True)
        self.details.setStyleSheet("color:#93a4b9")
        main.addWidget(self.details)
        playback = Q.QHBoxLayout()
        self.play = Q.QPushButton("Pause")
        self.play.clicked.connect(self.toggle_play)
        playback.addWidget(self.play)
        playback.addStretch()
        main.addLayout(playback)
        status = Q.QHBoxLayout()
        self.status = Q.QLabel("Ready")
        self.status.setWordWrap(True)
        status.addWidget(self.status, 1)
        self.cancel = Q.QPushButton("Cancel")
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.cancel_job)
        status.addWidget(self.cancel)
        outer.addLayout(status)
        for widget in [self.profile, self.backend, self.width, self.swap]:
            signal = (
                widget.toggled if isinstance(widget, Q.QCheckBox) else widget.currentIndexChanged
            )
            signal.connect(self.geometry_changed)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.rotation.currentIndexChanged.connect(self.rotation_changed)
        self.preview_width.currentIndexChanged.connect(self.motion_changed)
        self.refinement.toggled.connect(self.motion_changed)
        self.anaglyph_method.currentIndexChanged.connect(self.motion_changed)
        self.aa.toggled.connect(self.motion_changed)
        self.cleanup.toggled.connect(self.motion_changed)
        for widget in [self.sweep, self.views, self.fps, self.hold, self.cycles]:
            widget.valueChanged.connect(self.motion_changed)
        self.easing.toggled.connect(self.motion_changed)
        self.assets.toggled.connect(self.motion_changed)
        tips = {
            open_button: "Open one W3 or Nintendo 3DS MPO containing both eyes. No separate JPEG is needed.",
            pair_button: "Choose the left image first, then the matching right image.",
            self.swap: "Reverse the two eye images. Use when the apparent depth direction is reversed.",
            self.profile: "Auto, W3 Wide, and Nintendo 3DS use image-based alignment without measured calibration. A measured calibration can correct lens distortion and support metric depth.",
            load_profile: "Load an OpenCV stereo calibration JSON for this camera and capture mode.",
            self.backend: "CPU SGBM is faster to start. The GPU neural matcher often gives denser geometry. Analysis is cached.",
            self.width: "Stereo analysis width, cached independently of preview and export. Higher is slower and not always more accurate.",
            self.preview_width: "Final preview width after crop. Does not change stereo analysis quality.",
            self.export_width: "Final export width after crop. Original image colors are reloaded at render resolution.",
            self.export_scale: "Export only: render at up to twice the chosen width before cropping and downsampling. Limited by source resolution. Screen effects are applied after resizing.",
            self.refinement: "Select nearby color-compatible reliable geometry at uncertain edges. Cached separately; never promotes filled pixels to reliable matches.",
            self.mode: "Smooth synthesizes intermediate views; Classic alternates originals; Anaglyph creates one red/cyan JPEG for 3D glasses.",
            self.sweep: "0 is stationary; 1 spans the captured viewpoints. Smaller sweeps make residual edge errors less noticeable.",
            self.views: "Distinct views in one direction. The return journey reuses them without duplicating the turnarounds.",
            self.fps: "Video frames per second. More views at the same FPS make a slower camera sweep.",
            self.hold: "Time each original eye is displayed in Classic mode. Video timing is rounded to whole frames.",
            self.cycles: "Number of complete loops inside the MP4 file. GIF repeats indefinitely.",
            self.rotation: "Rotate the finished output clockwise, after stereo alignment. Crop is selected in the rotated image.",
            self.easing: "Slow down smoothly near each end of the camera move.",
            auto: "Clear your clicked anchor and select a coherent foreground region near the center.",
            self.generate: "Analyze stereo if needed and render the current settings. You can select an anchor before this step.",
            self.assets: "Also export aligned eyes, disparity, reliability masks, and a JSON settings manifest. Requires dense stereo even for still/classic output.",
            self.export_button: "Render at the export width with photo edits, crop, and effects, then save GIF, MP4, or anaglyph JPEG.",
            self.anaglyph_method: "Red filter over the LEFT eye, cyan over RIGHT. Dubois uses a generic display/glasses color projection; color and monochrome are simpler alternatives.",
            self.aa: "Preview-only 2× frame sampling reduces jagged sampling edges; enabled by default. It does not fix wrong depth. Export has independent supersampling.",
            self.cleanup: "Experimental comparison only; reject blending when colors disagree. Off by default while geometry refinement is evaluated.",
            self.play: "Pause or resume the rendered motion preview.",
            self.cancel: "Cancel the current job after the current CPU stage or render frame; GPU inference is terminated.",
            self.left: "Click any subject before or after the first preview. The crosshair marks your choice; disparity is resolved during analysis.",
            self.right: "Aligned right-eye image for checking stereo order and obstructions.",
            self.depth: "Separate raw/refined disparity, continuous match reliability, and magnitude of refinement changes. Reliability is not depth.",
            self.preview: "Rendered output, including photo adjustments and crop.",
        }
        for widget, tip in tips.items():
            widget.setToolTip(tip)
        # Labels and controls expose the same explanation, including hidden mode rows.
        for settings_form in (form, form2):
            for row in range(settings_form.rowCount()):
                label_item = settings_form.itemAt(row, Q.QFormLayout.LabelRole)
                field_item = settings_form.itemAt(row, Q.QFormLayout.FieldRole)
                if label_item and field_item and label_item.widget() and field_item.widget():
                    label_item.widget().setToolTip(field_item.widget().toolTip())
        self.sync_controls()
        self.setStyleSheet("""QWidget { background:#1b222c; color:#edf2f7; font-size:13px; }
          QPushButton { background:#2c3746; border:1px solid #3e4b5e; padding:8px; border-radius:6px; }
          QPushButton:hover { background:#3b4b5f; } QPushButton:disabled { color:#68788b; }
          QPushButton#primary { background:#3aaf9a; color:#102b28; font-weight:700; }
          QComboBox,QSpinBox,QDoubleSpinBox { background:#111923; padding:5px; border:1px solid #3e4b5e; border-radius:4px; }
          QTabBar::tab { padding:10px; background:#252f3e; } QTabBar::tab:selected { color:#75e1ce; background:#11161c; }
          QTabWidget::pane { border:1px solid #354153; } QToolTip { background:#111923;color:white; }""")

    def save_user_preset(self, kind):
        path, _ = Q.QFileDialog.getSaveFileName(
            self, f"Save {kind} preset", f"user-presets/{kind}.json", "JSON preset (*.json)"
        )
        if not path:
            return
        if not Path(path).suffix:
            path += ".json"
        state = (
            photo_state(
                self.photo.edits(), self.rotation.currentData(), self.photo.aspect.currentData()
            )
            if kind == "photo"
            else self.effects.preset_state()
        )
        try:
            save_preset(path, kind, state)
            self.status.setText(f"Saved {kind} preset: {Path(path).name}")
        except (OSError, ValueError) as exc:
            Q.QMessageBox.warning(self, "Could not save preset", str(exc))

    def load_user_preset(self, kind):
        path, _ = Q.QFileDialog.getOpenFileName(
            self, f"Load {kind} preset", "user-presets", "JSON preset (*.json)"
        )
        if not path:
            return
        try:
            state = load_preset(path, kind)  # Validate completely before changing controls.
            if kind == "photo":
                blockers = [
                    QtCore.QSignalBlocker(widget)
                    for widget in (self.photo, self.rotation, self.photo.aspect)
                ]
                for key, spin in self.photo.fields.items():
                    spin.setValue(state["adjustments"][key])
                self.photo.auto.setChecked(state["adjustments"]["auto_color"])
                self.rotation.setCurrentIndex(self.rotation.findData(state["rotation"]))
                self.photo.aspect.setCurrentIndex(self.photo.aspect.findData(state["aspect"]))
                self.photo.canvas.crop = tuple(state["adjustments"]["crop"])
                self.photo.canvas.aspect = (
                    None if state["aspect"] == "original" else state["aspect"]
                )
                del blockers
                self.invalidate()
                self.refresh_photo()
            else:
                self.effects.apply_state(state)
            self.status.setText(f"Loaded {kind} preset. Build preview to apply.")
        except (OSError, ValueError) as exc:
            Q.QMessageBox.warning(self, "Could not load preset", str(exc))

    def load_file(self, path, right=None, reset_edits=True):
        try:
            left_image, right_image, _ = read_pair(
                path, right, self.swap.isChecked(), self.width.currentData()
            )
            left_image, right_image, _, _, _ = align_pair(
                left_image, right_image, self.profile.currentData(), self.swap.isChecked()
            )
        except Exception as exc:
            Q.QMessageBox.warning(self, "Cannot open photo", str(exc))
            return
        self.source = path
        self.right_path = right
        self.point = None
        self.analysis = None
        self.sequence = None
        if self.temp:
            self.temp.cleanup()
            self.temp = None
        self.loaded_left = left_image
        self.depth.image = None
        self.depth.update()
        self.invalidate()
        if reset_edits:
            self.photo.reset()
        self.file_label.setText(Path(path).name + (" + " + Path(right).name if right else ""))
        self.left.set_image(left_image)
        self.right.set_image(right_image)
        self.tabs.setCurrentIndex(0)
        self.refresh_photo()
        self.status.setText("Click the subject to place an anchor, then build a preview.")

    def open_mpo(self):
        path, _ = Q.QFileDialog.getOpenFileName(
            self,
            "Open stereo photo",
            QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.PicturesLocation)
            or str(Path.home()),
            "Stereo photo (*.MPO *.mpo)",
        )
        if path:
            self.load_file(path)

    def open_pair(self):
        left, _ = Q.QFileDialog.getOpenFileName(
            self, "Select LEFT eye", "", "Images (*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG)"
        )
        if not left:
            return
        right, _ = Q.QFileDialog.getOpenFileName(
            self,
            "Select RIGHT eye",
            str(Path(left).parent),
            "Images (*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG)",
        )
        if right:
            self.load_file(left, right)

    def load_profile(self):
        path, _ = Q.QFileDialog.getOpenFileName(
            self, "Load measured calibration", "", "JSON (*.json)"
        )
        if path:
            self.profile.addItem(Path(path).stem, path)
            self.profile.setCurrentIndex(self.profile.count() - 1)

    def invalidate(self):
        self.render_dirty = True
        self.timer.stop()
        self.export_button.setEnabled(False)
        self.status.setText("Settings changed. Build preview to apply.")

    def geometry_changed(self):
        if self.source:
            old = self.point
            size = self.loaded_left.shape[:2] if self.loaded_left is not None else None
            preserve = self.sender() not in (self.swap, self.profile)
            self.load_file(self.source, self.right_path, reset_edits=False)
            if preserve and old is not None and size is not None:
                h, w = self.loaded_left.shape[:2]
                self.set_anchor(old[0] * w / size[1], old[1] * h / size[0])
        else:
            self.point = None
            self.analysis = None
            self.invalidate()

    def motion_changed(self):
        self.invalidate()

    def auto_anchor(self):
        self.point = None
        if self.analysis:
            self.update_anchor()
        elif self.loaded_left is not None:
            self.left.set_image(self.loaded_left)
        self.invalidate()

    def set_anchor(self, x, y):
        if self.busy:
            return
        if self.loaded_left is None:
            return
        self.point = (x, y)
        if self.analysis is not None:
            self.update_anchor()
        else:
            self.left.set_image(self.loaded_left, Anchor(x, y, 0.0, False, "Pending anchor"))
            self.hint.setText(
                f"Anchor selected at ({x:.0f}, {y:.0f}); depth will be resolved when you build."
            )
        self.invalidate()
        self.status.setText(self.hint.text() + " · Build preview to apply.")

    def update_anchor(self):
        anchor = choose_anchor(self.analysis, self.point)
        self.left.set_image(self.analysis.left, anchor)
        self.hint.setText(anchor.message + f" · disparity {anchor.disparity:.2f} px")

    def motion(self):
        return Motion(
            self.mode.currentData(),
            self.sweep.value(),
            self.views.value(),
            self.fps.value(),
            self.hold.value(),
            self.cycles.value(),
            self.easing.isChecked(),
            self.rotation.currentData(),
            antialias=self.aa.isChecked(),
            edge_cleanup=self.cleanup.isChecked(),
            anaglyph_method=self.anaglyph_method.currentData(),
            fusion=self.fusion.currentData(),
            repair_edges=self.edge_repair.isChecked(),
        )

    def sync_controls(self):
        still = self.mode.currentData() == "anaglyph"
        smooth = self.mode.currentData() == "smooth"
        self.effects.setEnabled(not still and not self.busy)
        for widget in (self.sweep, self.views, self.easing):
            widget.setEnabled(smooth)
        for widget in (self.fps, self.cycles, self.play):
            widget.setEnabled(not still)
        self.hold.setEnabled(self.mode.currentData() == "classic")
        self.anaglyph_method.setEnabled(still)
        self.aa.setEnabled(smooth)
        self.cleanup.setEnabled(smooth and not self.busy)
        self.fusion.setEnabled(smooth and not self.busy)
        self.edge_repair.setEnabled(smooth and not self.busy)
        for widget, visible in [
            (self.sweep, smooth),
            (self.views, smooth),
            (self.fps, not still),
            (self.cycles, not still),
            (self.hold, self.mode.currentData() == "classic"),
            (self.anaglyph_method, still),
        ]:
            self.parameter_form.setRowVisible(widget, visible)

    def mode_changed(self):
        self.sync_controls()
        self.invalidate()
        self.refresh_photo()

    def refresh_diagnostic(self):
        if self.analysis is None:
            return
        raw = self.analysis
        refined = getattr(self, "refined_analysis", raw)
        mode = self.diagnostic.currentIndex()
        eye = "l" if self.diagnostic_eye.currentIndex() == 0 else "r"
        if mode >= 4:
            _, _, _, maps = render_view(
                refined,
                choose_anchor(refined, self.point),
                self.diagnostic_t.value(),
                fusion=self.fusion.currentData(),
                repair_edges=self.edge_repair.isChecked(),
                diagnostics=True,
            )
            if mode in (4, 5):
                image = maps["left" if mode == 4 else "right"]
            elif mode == 6:
                weight = maps["right_weight"]
                image = np.stack((1 - weight, np.zeros_like(weight), weight), axis=-1)
                image = np.uint8(image * 255)
            else:
                values = maps["disagreement"] if mode == 7 else maps["repair"] * 255
                image = np.repeat(np.uint8(values.clip(0, 255))[:, :, None], 3, axis=2)
        elif mode == 2:
            values = np.uint8(reliability(raw, eye) * 255)
            image = np.repeat(values[:, :, None], 3, axis=2)
        else:
            rd, fd = getattr(raw, "d" + eye), getattr(refined, "d" + eye)
            d = abs(fd - rd) if mode == 3 else (rd if mode == 0 else fd)
            v = rd[getattr(raw, "valid_" + eye)]
            low, high = (
                (0, max(1, float(d.max())))
                if mode == 3
                else (np.percentile(v, [2, 98]) if v.size else (0, 1))
            )
            values = np.uint8(np.clip((d - low) / max(high - low, 0.01), 0, 1) * 255)
            image = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)[:, :, ::-1]
        self.depth.set_image(image)
        self.depth.show()

    def rotation_changed(self):
        self.photo.canvas.crop = (0.0, 0.0, 1.0, 1.0)
        self.invalidate()
        self.refresh_photo()
        self.photo.aspect_changed()

    def photo_changed(self):
        if self.busy:
            return
        if isinstance(self.sequence, TimelineSequence):
            self.invalidate()
        self.edit_timer.start(100)

    def refresh_photo(self):
        if self.loaded_left is None:
            return
        edits = self.photo.edits()
        self.photo.canvas.curvature_amount = (
            self.effects.settings().get("curvature", {}).get("amount", 0)
            if self.mode.currentData() != "anaglyph"
            else 0
        )
        try:
            if self.sequence is not None and not self.render_dirty:
                self.sequence.set_edits(edits)
                image = self.sequence.full_frame(len(self.sequence.positions) // 2)
                self.preview.set_image(
                    self.sequence.frame(
                        self.sequence.indices[(self.frame_index - 1) % len(self.sequence.indices)]
                    )
                )
                self.photo.note.setText(
                    "Exact output area before your crop. Drag corners to resize; drag inside to move."
                )
            else:
                image = adjust_photo(self.loaded_left, edits, balance_gains(self.loaded_left))
                if self.rotation.currentData():
                    image = np.rot90(image, -self.rotation.currentData() // 90)
                self.photo.note.setText(
                    "Provisional framing on the left eye. Build a preview to refine crop on the exact shared output area."
                )
            if self.photo.canvas.curvature_amount:
                self.photo.note.setText(
                    self.photo.note.text()
                    + " Gold outline: sampled region after safe curvature. Animation shows final framing."
                )
            self.photo.canvas.set_image(image)
            aspect = self.photo.aspect.currentData()
            if aspect is not None:
                value = image.shape[1] / image.shape[0] if aspect == "original" else aspect
                self.photo.canvas.aspect = value
                from .editing import ratio_crop

                self.photo.canvas.crop = ratio_crop(self.photo.canvas.crop, image.shape, value)
                if self.sequence is not None and not self.render_dirty:
                    self.sequence.set_edits(self.photo.edits())
            self.photo.canvas.update()
        except ValueError as exc:
            self.status.setText(str(exc))

    def start(self, fn, done):
        if self.busy:
            return
        self.busy = True
        self.controls.setEnabled(False)
        self.actions.setEnabled(False)
        self.photo.setEnabled(False)
        self.effects.setEnabled(False)
        self.cleanup.setEnabled(False)
        self.fusion.setEnabled(False)
        self.edge_repair.setEnabled(False)
        self.cancel.setEnabled(True)
        self.timer.stop()
        self.worker = Worker(fn, self)
        self.worker.progress.connect(self.status.setText)
        self.worker.result.connect(done)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.finished)
        self.worker.start()

    def finished(self):
        self.busy = False
        self.controls.setEnabled(True)
        self.actions.setEnabled(True)
        self.photo.setEnabled(True)
        self.sync_controls()
        self.cancel.setEnabled(False)
        self.worker.deleteLater()
        self.worker = None

    def failed(self, message):
        self.status.setText(message)
        if not message.startswith("Cancelled"):
            Q.QMessageBox.warning(self, "Processing could not finish", message)

    def cancel_job(self):
        if self.worker:
            self.worker.cancel.set()
            self.status.setText("Cancelling after the current processing step…")

    def build(self):
        if not self.source:
            self.open_mpo()
        if not self.source:
            return
        motion = self.motion()
        point = self.point
        edits = self.photo.edits()
        settings = self.effects.settings() if motion.mode != "anaglyph" else {}
        preview_width = self.preview_width.currentData()
        refine = self.refinement.isChecked()
        opts = dict(
            source=self.source,
            right=self.right_path,
            width=self.width.currentData(),
            swap=self.swap.isChecked(),
            profile=self.profile.currentData(),
            backend=self.backend.currentData(),
        )
        if motion.mode in ("classic", "anaglyph") and not self.assets.isChecked():
            opts["backend"] = "sparse"
        cached = self.analysis
        if cached is not None and cached.metadata["backend"] != opts["backend"]:
            cached = None

        def job(progress, cancelled):
            a = (
                cached
                if cached is not None
                else analyze(**opts, progress=progress, cancelled=cancelled)
            )
            refined = refine_analysis(a, refine, cancelled=cancelled)
            anchor = choose_anchor(refined, point)
            grid, scaled_anchor = render_grid(
                refined,
                anchor,
                self.source,
                self.right_path,
                opts["swap"],
                opts["profile"],
                preview_width,
            )
            temp = tempfile.TemporaryDirectory(prefix="wiggle-preview-")
            try:
                seq = Sequence(
                    grid,
                    scaled_anchor,
                    motion,
                    temp.name,
                    progress,
                    cancelled,
                    edits=edits,
                    output_width=preview_width,
                )
                if settings:
                    seq = TimelineSequence(
                        seq, settings, Path(temp.name) / "effects", progress, cancelled
                    )
            except Exception:
                temp.cleanup()
                raise
            return a, refined, anchor, temp, seq

        self.start(job, self.built)

    def built(self, result):
        if self.temp:
            self.temp.cleanup()
        self.analysis, self.refined_analysis, anchor, self.temp, self.sequence = result
        self.render_dirty = False
        self.loaded_left = self.analysis.left
        self.left.set_image(self.analysis.left, anchor)
        self.right.set_image(self.analysis.right)
        a = self.analysis
        self.refresh_diagnostic()
        self.hint.setText(anchor.message + " · Click in the left-eye tab to change the anchor.")
        residual = a.metadata["alignment"].get("vertical_median_px")
        residual_text = f"{residual:.2f} px" if residual is not None else "unverified"
        holes = max(self.sequence.metadata.get("visible_unfilled_fraction", [0]))
        self.details.setText(
            f"{a.metadata['backend']} · reliable matches {a.valid_l.mean():.0%} · vertical residual {residual_text} · crop {self.sequence.crop} · unfilled {holes:.2%}"
        )
        warnings = a.metadata["alignment"].get("warnings", [])
        self.status.setText(
            "Preview ready."
            + (" " + " ".join(warnings) if warnings else "")
            + (" Try a smaller sweep if gaps are visible." if holes > 0.002 else "")
        )
        self.export_button.setEnabled(True)
        self.frame_index = 0
        self.tabs.setCurrentIndex(3)
        self.play.setText("Pause")
        self.refresh_photo()
        self.export_button.setText(
            "Export JPEG…" if self.sequence.motion.mode == "anaglyph" else "Export GIF / MP4…"
        )
        self.advance()

    def advance(self):
        if not self.sequence:
            return
        seq = self.sequence
        i = self.frame_index % len(seq.indices)
        self.preview.set_image(seq.frame(seq.indices[i]))
        self.frame_index += 1
        if seq.motion.mode != "anaglyph":
            self.timer.start(max(10, round(seq.durations_ms[i])))

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play.setText("Play")
        else:
            self.advance()
            self.play.setText("Pause")

    def export(self):
        if not self.sequence or self.render_dirty:
            return
        self.edit_timer.stop()
        self.sequence.set_edits(self.photo.edits())
        still = self.sequence.motion.mode == "anaglyph"
        extension = "jpg" if still else "mp4"
        default = str(
            Path.cwd()
            / "outputs"
            / f"{Path(self.source).stem}-{self.sequence.motion.mode}.{extension}"
        )
        path, selected = Q.QFileDialog.getSaveFileName(
            self,
            "Export image" if still else "Export animation",
            default,
            "JPEG image (*.jpg *.jpeg)" if still else "H.264 MP4 (*.mp4);;Looping GIF (*.gif)",
        )
        if not path:
            return
        if not Path(path).suffix:
            path += ".jpg" if still else (".gif" if "GIF" in selected else ".mp4")
        bundle = (
            Path(path).with_suffix("").with_name(Path(path).stem + "-assets")
            if self.assets.isChecked()
            else None
        )
        if bundle and bundle.exists():
            Q.QMessageBox.warning(
                self,
                "Assets already exist",
                "Choose a different output name to preserve the existing asset bundle.",
            )
            return
        a = self.refined_analysis
        anchor = choose_anchor(a, self.point)
        motion = replace(self.motion(), antialias=False)
        edits = self.photo.edits()
        settings = self.effects.settings() if motion.mode != "anaglyph" else {}
        output_width = self.export_width.currentData()
        render_width = output_width * self.export_scale.currentData()
        source, right = self.source, self.right_path
        swap, profile = self.swap.isChecked(), self.profile.currentData()

        def job(progress, cancelled):
            grid, scaled_anchor = render_grid(a, anchor, source, right, swap, profile, render_width)
            with tempfile.TemporaryDirectory(prefix="wiggle-export-") as directory:
                seq = Sequence(
                    grid,
                    scaled_anchor,
                    motion,
                    directory,
                    progress,
                    cancelled,
                    edits=edits,
                    output_width=output_width,
                )
                seq.metadata["export_supersampling_requested"] = render_width / output_width
                if settings:
                    seq = TimelineSequence(
                        seq, settings, Path(directory) / "effects", progress, cancelled
                    )
                result = write_animation(
                    seq, path, overwrite=True, progress=progress, cancelled=cancelled
                )
                if bundle:
                    write_bundle(grid, seq, bundle)
                return result

        self.start(job, lambda result: self.status.setText("Saved " + str(result)))

    def closeEvent(self, event):
        if self.busy:
            self.cancel_job()
            event.ignore()
            return
        self.timer.stop()
        self.edit_timer.stop()
        if self.temp:
            self.temp.cleanup()
        event.accept()


def main():
    app = Q.QApplication.instance() or Q.QApplication(sys.argv)
    app.setApplicationName("Stereo Studio")
    window = Window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
