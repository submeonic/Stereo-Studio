from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui
from PySide6 import QtWidgets as Q

from .editing import PhotoEdits, ratio_crop


class Canvas(Q.QWidget):
    clicked = QtCore.Signal(float, float)

    def __init__(self, clickable=False):
        super().__init__()
        self.image = None
        self.anchor = None
        self.empty_text = "Open a stereo photo to begin"
        self.clickable = clickable
        self.setMinimumSize(400, 260)
        if clickable:
            self.setCursor(QtCore.Qt.CrossCursor)

    def set_image(self, array, anchor=None):
        array = np.ascontiguousarray(array)
        h, w = array.shape[:2]
        self.image = QtGui.QImage(
            array.data, w, h, array.strides[0], QtGui.QImage.Format_RGB888
        ).copy()
        self.anchor = anchor
        self.update()

    def image_rect(self):
        if self.image is None:
            return QtCore.QRectF()
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        w = self.image.width() * scale
        h = self.image.height() * scale
        return QtCore.QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#11161c"))
        if self.image is None:
            p.setPen(QtGui.QColor("#8491a2"))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, self.empty_text)
            return
        r = self.image_rect()
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        p.drawImage(r, self.image)
        if self.anchor:
            x = r.left() + self.anchor.x / self.image.width() * r.width()
            y = r.top() + self.anchor.y / self.image.height() * r.height()
            p.setPen(QtGui.QPen(QtGui.QColor("#65e3cb"), 2))
            p.drawEllipse(QtCore.QPointF(x, y), 9, 9)
            p.drawLine(QtCore.QPointF(x - 15, y), QtCore.QPointF(x + 15, y))
            p.drawLine(QtCore.QPointF(x, y - 15), QtCore.QPointF(x, y + 15))

    def mousePressEvent(self, event):
        if self.clickable and self.image:
            r = self.image_rect()
            p = event.position()
            if r.contains(p):
                self.clicked.emit(
                    (p.x() - r.x()) / r.width() * self.image.width(),
                    (p.y() - r.y()) / r.height() * self.image.height(),
                )


class CropCanvas(Canvas):
    selected = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        self.crop = (0.0, 0.0, 1.0, 1.0)
        self.aspect = None
        self.drag = None
        self.curvature_amount = 0.0
        self.setCursor(QtCore.Qt.CrossCursor)

    def point_normalized(self, point):
        r = self.image_rect()
        return np.clip((point.x() - r.x()) / r.width(), 0, 1), np.clip(
            (point.y() - r.y()) / r.height(), 0, 1
        )

    def crop_rect(self):
        r = self.image_rect()
        x0, y0, x1, y1 = self.crop
        return QtCore.QRectF(
            r.x() + x0 * r.width(),
            r.y() + y0 * r.height(),
            (x1 - x0) * r.width(),
            (y1 - y0) * r.height(),
        )

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.image is None:
            return
        painter = QtGui.QPainter(self)
        r = self.image_rect()
        c = self.crop_rect()
        shade = QtGui.QPainterPath()
        shade.addRect(r)
        cut = QtGui.QPainterPath()
        cut.addRect(c)
        painter.fillPath(shade.subtracted(cut), QtGui.QColor(0, 0, 0, 135))
        if self.curvature_amount:
            from .curvature import coordinates

            edge = np.linspace(-1, 1, 40)
            nx = np.concatenate((edge, np.ones(40), edge[::-1], -np.ones(40)))
            ny = np.concatenate((-np.ones(40), edge, np.ones(40), edge[::-1]))
            nx, ny = coordinates(nx, ny, self.curvature_amount)
            polygon = QtGui.QPolygonF(
                [
                    QtCore.QPointF(
                        c.left() + (x + 1) * c.width() / 2, c.top() + (y + 1) * c.height() / 2
                    )
                    for x, y in zip(nx, ny)
                ]
            )
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffcc66"), 2, QtCore.Qt.DashLine))
            painter.drawPolygon(polygon)
        painter.setPen(QtGui.QPen(QtGui.QColor("#65e3cb"), 2))
        painter.drawRect(c)
        for fraction in (1 / 3, 2 / 3):
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 90), 1))
            painter.drawLine(
                QtCore.QPointF(c.left() + fraction * c.width(), c.top()),
                QtCore.QPointF(c.left() + fraction * c.width(), c.bottom()),
            )
            painter.drawLine(
                QtCore.QPointF(c.left(), c.top() + fraction * c.height()),
                QtCore.QPointF(c.right(), c.top() + fraction * c.height()),
            )
        painter.setBrush(QtGui.QColor("#65e3cb"))
        for corner in (c.topLeft(), c.topRight(), c.bottomLeft(), c.bottomRight()):
            painter.drawRect(QtCore.QRectF(corner.x() - 4, corner.y() - 4, 8, 8))

    def mousePressEvent(self, event):
        if not self.image or not self.image_rect().contains(event.position()):
            return
        p = self.point_normalized(event.position())
        c = self.crop_rect()
        old = self.crop
        corners = [c.topLeft(), c.topRight(), c.bottomLeft(), c.bottomRight()]
        for i, corner in enumerate(corners):
            if (corner - event.position()).manhattanLength() < 18:
                opposite = [(old[2], old[3]), (old[0], old[3]), (old[2], old[1]), (old[0], old[1])][
                    i
                ]
                self.drag = ("resize", opposite, old)
                return
        if c.contains(event.position()) and old != (0.0, 0.0, 1.0, 1.0):
            self.drag = ("move", p, old)
        else:
            self.drag = ("resize", p, old)

    def mouseMoveEvent(self, event):
        if self.drag is None:
            return
        kind, start, old = self.drag
        x, y = self.point_normalized(event.position())
        if kind == "move":
            dx = np.clip(x - start[0], -old[0], 1 - old[2])
            dy = np.clip(y - start[1], -old[1], 1 - old[3])
            self.crop = (old[0] + dx, old[1] + dy, old[2] + dx, old[3] + dy)
        else:
            dx = x - start[0]
            dy = y - start[1]
            if self.aspect:
                ratio = self.aspect * self.image.height() / self.image.width()
                if abs(dx) > abs(dy) * ratio:
                    dx = np.sign(dx) * abs(dy) * ratio
                else:
                    dy = np.sign(dy) * abs(dx) / ratio
            self.crop = (
                min(start[0], start[0] + dx),
                min(start[1], start[1] + dy),
                max(start[0], start[0] + dx),
                max(start[1], start[1] + dy),
            )
        self.update()

    def mouseReleaseEvent(self, event):
        if self.drag is None:
            return
        self.mouseMoveEvent(event)
        old = self.drag[2]
        self.drag = None
        if (self.crop[2] - self.crop[0]) * self.image.width() < 16 or (
            self.crop[3] - self.crop[1]
        ) * self.image.height() < 16:
            self.crop = old
        self.update()
        self.selected.emit(tuple(float(x) for x in self.crop))

    def set_aspect(self, aspect):
        self.aspect = aspect
        if self.image:
            self.crop = ratio_crop(self.crop, (self.image.height(), self.image.width()), aspect)
            self.update()
            self.selected.emit(self.crop)

    def reset(self):
        self.crop = (0.0, 0.0, 1.0, 1.0)
        self.set_aspect(self.aspect)
        self.update()


class PhotoPanel(Q.QWidget):
    changed = QtCore.Signal()
    save_requested = QtCore.Signal()
    load_requested = QtCore.Signal()

    def __init__(self, rotation):
        super().__init__()
        layout = Q.QVBoxLayout(self)
        preset_buttons(self, layout, "photo")
        grid = Q.QGridLayout()
        layout.addLayout(grid)
        self.fields = {}
        specs = [
            (
                "exposure",
                "Exposure (EV)",
                -4,
                4,
                0,
                0.1,
                "Linear-light exposure in stops. +1 doubles the light; -1 halves it.",
            ),
            (
                "temperature",
                "Warmth",
                -100,
                100,
                0,
                5,
                "Cooler blue to warmer amber. The same correction is applied to both eyes.",
            ),
            (
                "tint",
                "Tint",
                -100,
                100,
                0,
                5,
                "Green to magenta color correction, shared across both eyes.",
            ),
            (
                "saturation",
                "Saturation",
                0,
                2,
                1,
                0.1,
                "0 is monochrome; 1 keeps original saturation; 2 doubles it.",
            ),
            (
                "contrast",
                "Contrast",
                0.5,
                1.5,
                1,
                0.05,
                "Contrast around a midtone pivot. 1 preserves the original.",
            ),
            (
                "sharpness",
                "Sharpness",
                0,
                2,
                0,
                0.1,
                "Unsharp-mask detail enhancement. High values can emphasize jaggies or halos.",
            ),
        ]
        for i, (key, label, low, high, value, step, tip) in enumerate(specs):
            spin = Q.QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setSingleStep(step)
            spin.setValue(value)
            spin.setToolTip(tip)
            self.fields[key] = spin
            row = i // 2
            col = (i % 2) * 2
            grid.addWidget(Q.QLabel(label), row, col)
            grid.addWidget(SliderField(spin), row, col + 1)
            spin.valueChanged.connect(self.changed)
        self.auto = Q.QCheckBox("Auto color balance")
        self.auto.setToolTip(
            "Estimate gray-world white balance once from the left image and use it for both eyes and every frame."
        )
        self.auto.toggled.connect(self.changed)
        grid.addWidget(self.auto, 3, 0, 1, 2)
        grid.addWidget(Q.QLabel("Rotation"), 3, 2)
        grid.addWidget(rotation, 3, 3)
        self.aspect = Q.QComboBox()
        for label, value in [
            ("Free crop", None),
            ("Original ratio", "original"),
            ("Square · 1:1", 1.0),
            ("Landscape · 4:3", 4 / 3),
            ("Photo · 3:2", 1.5),
            ("Wide · 16:9", 16 / 9),
            ("Portrait · 9:16", 9 / 16),
        ]:
            self.aspect.addItem(label, value)
        self.aspect.setToolTip(
            "Constrain the crop rectangle. Drag its corners to resize; drag inside to reposition. The same crop applies to every frame."
        )
        grid.addWidget(Q.QLabel("Crop ratio"), 4, 0)
        grid.addWidget(self.aspect, 4, 1)
        buttons = Q.QHBoxLayout()
        layout.addLayout(buttons)
        clear = Q.QPushButton("Reset crop")
        clear.setToolTip("Restore the full image, constrained to the selected aspect ratio.")
        buttons.addWidget(clear)
        reset = Q.QPushButton("Reset photo edits")
        reset.setToolTip("Reset color adjustments and crop. Rotation is kept.")
        buttons.addWidget(reset)
        buttons.addStretch()
        self.note = Q.QLabel(
            "Drag a crop rectangle; drag inside to move it; drag corners to resize."
        )
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.canvas = CropCanvas()
        self.canvas.setToolTip(
            "Crop selection: drag to draw, drag inside to move, or drag a corner to resize. Darkened areas are excluded from export."
        )
        layout.addWidget(self.canvas, 1)
        self.canvas.selected.connect(lambda _: self.changed.emit())
        clear.clicked.connect(self.canvas.reset)
        reset.clicked.connect(self.reset)
        self.aspect.currentIndexChanged.connect(self.aspect_changed)

    def aspect_changed(self):
        value = self.aspect.currentData()
        if value == "original":
            if self.canvas.image is None:
                return
            value = self.canvas.image.width() / self.canvas.image.height()
        self.canvas.set_aspect(value)

    def edits(self):
        return PhotoEdits(
            **{key: spin.value() for key, spin in self.fields.items()},
            auto_color=self.auto.isChecked(),
            crop=self.canvas.crop,
        )

    def reset(self):
        self.blockSignals(True)
        for key, spin in self.fields.items():
            spin.setValue(1 if key in ("saturation", "contrast") else 0)
        self.auto.setChecked(False)
        self.aspect.setCurrentIndex(0)
        self.canvas.crop = (0.0, 0.0, 1.0, 1.0)
        self.canvas.aspect = None
        self.blockSignals(False)
        self.changed.emit()


class SliderField(Q.QWidget):
    """A close label, draggable slider, and compact exact-value field."""

    def __init__(self, spin, parent=None):
        super().__init__(parent)
        layout = Q.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.slider = Q.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setToolTip(spin.toolTip())
        spin.setFixedWidth(82)
        span = spin.maximum() - spin.minimum()

        def sync(value):
            blocker = QtCore.QSignalBlocker(self.slider)
            self.slider.setValue(round((value - spin.minimum()) / span * 1000))
            del blocker

        def move(value):
            raw = spin.minimum() + value / 1000 * span
            step = spin.singleStep()
            spin.setValue(round(raw / step) * step)

        self.slider.valueChanged.connect(move)
        spin.valueChanged.connect(sync)
        sync(spin.value())
        layout.addWidget(self.slider, 1)
        layout.addWidget(spin)


class EffectsPanel(Q.QWidget):
    changed = QtCore.Signal()
    save_requested = QtCore.Signal()
    load_requested = QtCore.Signal()

    def __init__(self):
        super().__init__()
        from .temporal import SPECS

        layout = Q.QVBoxLayout(self)
        self.bypass = Q.QCheckBox("Bypass all effects")
        self.bypass.setToolTip(
            "Compare the clean render without losing individual effect settings."
        )
        layout.addWidget(self.bypass)
        self.bypass.toggled.connect(self.changed)
        note = Q.QLabel(
            "Effects follow playback time, after photo edits. Cycles are per full loop.\nPixel controls use final output pixels. Build preview to apply changes."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        preset_buttons(self, layout, "effects")
        scroll = Q.QScrollArea()
        scroll.setWidgetResizable(True)
        content = Q.QWidget()
        cards = Q.QVBoxLayout(content)
        self.groups, self.fields = {}, {}
        for name, (label, specs) in SPECS.items():
            group = Q.QGroupBox(label)
            group.setCheckable(True)
            group.setChecked(False)
            group_layout = Q.QVBoxLayout(group)
            body = Q.QWidget()
            form = Q.QFormLayout(body)
            group_layout.addWidget(body)
            body.setVisible(False)
            group.toggled.connect(body.setVisible)
            form.setFieldGrowthPolicy(Q.QFormLayout.AllNonFixedFieldsGrow)
            self.groups[name] = group
            self.fields[name] = {}
            for key, (title, low, high, default, step, tip) in specs.items():
                spin = Q.QDoubleSpinBox()
                spin.setDecimals(0 if step == 1 else 2)
                spin.setRange(low, high)
                spin.setSingleStep(step)
                spin.setValue(default)
                spin.setToolTip(tip)
                form.addRow(title, SliderField(spin))
                self.fields[name][key] = spin
                spin.valueChanged.connect(self.changed)
            group.toggled.connect(self.changed)
            cards.addWidget(group)
        cards.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        reset = Q.QPushButton("Reset all effects")
        reset.clicked.connect(self.reset)
        layout.addWidget(reset)

    def settings(self):
        if self.bypass.isChecked():
            return {}
        return {
            name: {key: spin.value() for key, spin in self.fields[name].items()}
            for name, group in self.groups.items()
            if group.isChecked()
        }

    def reset(self):
        from .temporal import SPECS

        self.blockSignals(True)
        self.bypass.setChecked(False)
        for name, group in self.groups.items():
            group.setChecked(False)
            for key, spin in self.fields[name].items():
                spin.setValue(SPECS[name][1][key][3])
        self.blockSignals(False)
        self.changed.emit()

    def preset_state(self):
        return {
            "parameters": {
                name: {key: spin.value() for key, spin in fields.items()}
                for name, fields in self.fields.items()
            },
            "enabled": [name for name, group in self.groups.items() if group.isChecked()],
            "bypass": self.bypass.isChecked(),
        }

    def apply_state(self, state):
        blocker = QtCore.QSignalBlocker(self)
        for name, fields in state["parameters"].items():
            for key, value in fields.items():
                self.fields[name][key].setValue(value)
            self.groups[name].setChecked(name in state["enabled"])
        self.bypass.setChecked(state["bypass"])
        del blocker
        self.changed.emit()


def preset_buttons(panel, layout, kind):
    row = Q.QHBoxLayout()
    for label, signal, tip in [
        (
            "Save preset…",
            panel.save_requested,
            f"Save your {kind} settings to a portable JSON file.",
        ),
        (
            "Load preset…",
            panel.load_requested,
            f"Replace {kind} settings with a saved user preset. No image files are included.",
        ),
    ]:
        button = Q.QPushButton(label)
        button.setToolTip(tip)
        button.setObjectName(f"{kind}_{'save' if label.startswith('Save') else 'load'}")
        button.clicked.connect(lambda checked=False, requested=signal: requested.emit())
        row.addWidget(button)
    row.addStretch()
    layout.addLayout(row)
