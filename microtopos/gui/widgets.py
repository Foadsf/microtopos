"""Reusable GUI pieces: status chips, toast notifications, banners and a physical-units image panel."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, QTimer
from PySide6.QtWidgets import QGraphicsOpacityEffect, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .theme import RADIUS, hex_rgb


class Chip(QLabel):
    """A small rounded status label. kind: success | warning | danger | info | neutral."""

    def __init__(self, text: str = "", kind: str = "neutral", tokens: dict | None = None):
        super().__init__(text)
        self.kind = kind
        self.tokens = tokens or {}
        self.setAlignment(Qt.AlignCenter)
        self._apply()

    def set(self, text: str, kind: str | None = None, tip: str | None = None):
        self.setText(text)
        if kind:
            self.kind = kind
        if tip is not None:
            self.setToolTip(tip)
        self._apply()

    def retheme(self, tokens: dict):
        self.tokens = tokens
        self._apply()

    def _apply(self):
        t = self.tokens
        if not t:
            return
        col = {"success": t["success"], "warning": t["warning"], "danger": t["danger"], "info": t["accent"],
               "neutral": t["muted"]}.get(self.kind, t["muted"])
        r, g, b = hex_rgb(col)
        self.setStyleSheet(f"background: rgba({r},{g},{b},0.16); color: {col}; border: 1px solid rgba({r},{g},{b},0.5);"
                           f" border-radius: {RADIUS + 4}px; padding: 3px 10px; font-weight: 700; font-size: 12px;")


class Toast(QLabel):
    """Non-blocking notification that fades out at the bottom of its parent."""

    def __init__(self, parent: QWidget, tokens: dict):
        super().__init__(parent)
        self.tokens = tokens
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignCenter)
        self.hide()
        self._fx = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._fx)
        self._anim = QPropertyAnimation(self._fx, b"opacity", self)
        self._timer = QTimer(self, singleShot=True, timeout=self._fade)
        self._anim.finished.connect(self._done)

    def show_message(self, text: str, kind: str = "info", ms: int = 3500):
        t = self.tokens
        col = {"success": t["success"], "warning": t["warning"], "danger": t["danger"]}.get(kind, t["accent"])
        self.setStyleSheet(f"background: {t['elevated']}; color: {t['text']}; border: 1px solid {col};"
                           f" border-left: 4px solid {col}; border-radius: {RADIUS}px; padding: 10px 16px;")
        self.setText(text)
        p = self.parentWidget()
        w = min(560, p.width() - 40)
        self.setFixedWidth(w)
        self.adjustSize()
        self.move((p.width() - w) // 2, p.height() - self.height() - 28)
        self._anim.stop()
        self._fx.setOpacity(1.0)
        self.show()
        self.raise_()
        self._timer.start(ms)

    def _fade(self):
        self._anim.setDuration(600)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.start()

    def _done(self):
        if self._fx.opacity() < 0.05:
            self.hide()


class Banner(QWidget):
    """Inline, non-modal prompt with action buttons (e.g. objective change detected)."""

    def __init__(self, tokens: dict):
        super().__init__()
        self.tokens = tokens
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        self.label = QLabel()
        self.label.setWordWrap(True)
        lay.addWidget(self.label, 1)
        self.buttons_box = QHBoxLayout()
        lay.addLayout(self.buttons_box)
        self.hide()

    def ask(self, text: str, actions: list[tuple[str, callable]], kind: str = "warning"):
        t = self.tokens
        col = t.get(kind, t["warning"])
        self.setStyleSheet(f"background: {t['elevated']}; border: 1px solid {col}; border-radius: {RADIUS}px;")
        self.label.setText(text)
        while self.buttons_box.count():
            w = self.buttons_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        for label, fn in actions:
            b = QPushButton(label)
            b.setFocusPolicy(Qt.TabFocus)
            b.clicked.connect(lambda _=False, f=fn: (self.hide(), f()))
            self.buttons_box.addWidget(b)
        self.show()


class ImagePanel(pg.GraphicsLayoutWidget):
    """Image in physical units (um axes), optional colour bar, NaN shown as the neutral 'nan' token,
    cursor readout and an empty-state message."""

    def __init__(self, tokens: dict, empty_text: str, colorbar_label: str | None = None, cmap: str = "viridis"):
        super().__init__()
        self.tokens = tokens
        self.plot = self.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.plot.setLabel("bottom", "x", units="µm")
        self.plot.setLabel("left", "y", units="µm")
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        # image, overlay and annotations live in one group so the whole view can be rotated (a camera that is not
        # square to the stage axes); item coordinates stay in µm of the unrotated image
        self.group = pg.ItemGroup()
        self.plot.addItem(self.group)
        self.rotation = 0.0
        self.img = pg.ImageItem()
        self.img.setParentItem(self.group)
        self.overlay = pg.ImageItem()
        self.overlay.setZValue(5)
        self.overlay.setParentItem(self.group)
        self.bar = None
        if colorbar_label is not None:
            self.bar = pg.ColorBarItem(colorMap=pg.colormap.get(cmap), label=colorbar_label, interactive=False, width=14)
            self.bar.setImageItem(self.img, insert_in=self.plot)
        self.empty = pg.TextItem(empty_text, anchor=(0.5, 0.5))
        self.plot.addItem(self.empty)
        self.readout = pg.TextItem("", anchor=(0, 0))
        self.readout.setZValue(10)
        self.plot.addItem(self.readout)
        self.data = None
        self.extra = None  # optional second layer for readout (e.g. confidence)
        self.um_per_px = 1.0
        self.unit = ""
        self.scene().sigMouseMoved.connect(self._moved)
        self.retheme(tokens)
        self._show_empty(True)

    def retheme(self, t: dict):
        self.tokens = t
        self.setBackground(t["surface"])
        self.plot.getViewBox().setBackgroundColor(t["nan"])
        for ax in ("bottom", "left"):
            a = self.plot.getAxis(ax)
            a.setPen(pg.mkPen(t["border"]))
            a.setTextPen(pg.mkPen(t["muted"]))
        self.empty.setColor(t["muted"])
        self.readout.setColor(t["text"])
        self.readout.fill = pg.mkBrush(*hex_rgb(t["elevated"]), 220)

    def _show_empty(self, on: bool):
        self.empty.setVisible(on)
        if on:
            self.img.clear()
            self.overlay.clear()
            self.plot.setRange(xRange=(-1, 1), yRange=(-1, 1), padding=0)
            self.empty.setPos(0, 0)

    def set_image(self, data: np.ndarray | None, um_per_px: float, levels=None, unit: str = "", rgb: bool = False,
                  autorange: bool = False):
        if data is None:
            self.data = None
            self._show_empty(True)
            return
        first = self.data is None or self.data.shape[:2] != data.shape[:2] or abs(um_per_px - self.um_per_px) > 1e-12
        self.data, self.um_per_px, self.unit = data, um_per_px, unit
        self._show_empty(False)
        if rgb:
            self.img.setImage(data, autoLevels=False, levels=(0, 255))
        else:
            finite = data[np.isfinite(data)]
            if levels is None:
                levels = (float(np.percentile(finite, 1)), float(np.percentile(finite, 99))) if finite.size else (0, 1)
                if levels[1] <= levels[0]:
                    levels = (levels[0], levels[0] + 1e-6)
            if self.bar is not None:
                self.bar.setLevels(levels)
            self.img.setImage(data, autoLevels=False, levels=levels)
        h, w = data.shape[:2]
        rect = QRectF(0, 0, w * um_per_px, h * um_per_px)
        self.img.setRect(rect)
        self.overlay.setRect(rect)
        if first or autorange:
            self.plot.setRange(self.group.mapRectToParent(rect), padding=0.02)

    def add_item(self, item):
        """Add an annotation in image µm coordinates; it rotates with the image."""
        item.setParentItem(self.group)

    def set_rotation(self, deg: float):
        """Rotate the displayed image about its origin (clockwise on screen, degrees)."""
        self.rotation = float(deg)
        self.group.setRotation(self.rotation)
        lab = "" if abs(self.rotation) < 1e-6 else f"  (view rotated {self.rotation:+.1f}°)"
        self.plot.setLabel("bottom", "x" + lab, units="µm")
        if self.data is not None:
            self.plot.setRange(self.group.mapRectToParent(self.img.mapRectToParent(self.img.boundingRect())),
                               padding=0.02)

    def set_overlay(self, rgba: np.ndarray | None, rect: QRectF | None = None):
        """RGBA overlay; by default it covers the image, or `rect` (in um) when given."""
        if rgba is None:
            self.overlay.clear()
        else:
            self.overlay.setImage(rgba, autoLevels=False, levels=(0, 255))
            if rect is not None:
                self.overlay.setRect(rect)
            elif self.data is not None:
                h, w = self.data.shape[:2]
                self.overlay.setRect(QRectF(0, 0, w * self.um_per_px, h * self.um_per_px))

    def _moved(self, pos):
        if self.data is None or not self.plot.sceneBoundingRect().contains(pos):
            self.readout.setText("")
            return
        p = self.group.mapFromScene(pos)  # µm in the unrotated image
        x, y = p.x() / self.um_per_px, p.y() / self.um_per_px
        h, w = self.data.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            v = self.data[int(y), int(x)]
            val = "—" if (np.ndim(v) == 0 and not np.isfinite(v)) else (f"{float(v):.4g} {self.unit}" if np.ndim(v) == 0 else "")
            extra = ""
            if self.extra is not None and self.extra.shape[:2] == (h, w):
                extra = f"  conf {float(self.extra[int(y), int(x)]):.2f}"
            self.readout.setText(f"x {p.x():.1f} µm  y {p.y():.1f} µm  {val}{extra}")
            vr = self.plot.vb.viewRect()
            self.readout.setPos(vr.left(), vr.top())
        else:
            self.readout.setText("")


def section(title: str, *widgets, hint: str | None = None) -> QWidget:
    """Titled vertical group used in the side panels."""
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 4, 0, 4)
    lay.setSpacing(6)
    t = QLabel(title)
    t.setObjectName("title")
    lay.addWidget(t)
    if hint:
        h = QLabel(hint)
        h.setObjectName("muted")
        h.setWordWrap(True)
        lay.addWidget(h)
    for w in widgets:
        if isinstance(w, QWidget):
            lay.addWidget(w)
        else:
            lay.addLayout(w)
    lay.addStretch(1)
    return box
