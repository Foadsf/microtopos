"""MicroTopos main window: guided session checklist, workspace tabs, telemetry, workers and exports."""

from __future__ import annotations

import html
import logging
import math
import time
import os
from pathlib import Path

logger = logging.getLogger("microtopos.gui")

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QSettings, QSize, QStandardPaths, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QMainWindow, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSlider, QSpinBox,
                               QStackedWidget, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from .. import __version__, export, metrology, zstack
from ..calibration import CalibrationStore, CalStatus, radiometric_key, today
from ..camera import CameraSource, ReplaySource, SimulatorSource, list_cameras, opencv_threads
from ..focus import MEASURES
from ..kinematics import Override, State, guess_objective
from ..live import LiveEngine, SweepCapture
from ..optics import OBJECTIVE_NAMES, OBJECTIVES
from ..superres import dwell_superres
from .theme import DARK, LIGHT, hex_rgb, stylesheet
from .widgets import Banner, Chip, ImagePanel, Toast, section

OPTICS_NOTES = """
<h3>What software can and cannot fix</h3>
<p><b>Lateral scale.</b> The 3.0 µm pixel pitch and "no optics in the camera" are unverified assumptions, and old
microscopes differ in tube length (160 mm, 170 mm or infinity). Until you calibrate with a stage micrometer, all
lateral lengths are <i>nominal</i>.</p>
<p><b>Colour sensor and MJPEG.</b> A Bayer colour sensor behind 8-bit MJPEG compression resolves less than its pixel
pitch suggests. In-camera sharpening is switched off here because it distorts focus measures. Your camera delivers
about 12.5 frames per second at 1920×1080, although it advertises 25.</p>
<p><b>No Z encoder.</b> Turning the focus knob by hand gives no height scale and an uneven speed. Heights are shown in
<i>focus frames</i> until you calibrate Z (knob travel, step standard) or use the experimental DOF ruler. A frame index
is monotonic with height, so shapes are right, but uneven knob speed distorts the scale.</p>
<p><b>Textureless or mirror-polished surfaces.</b> Shape from focus needs local contrast. Pixels without a clear focus
peak are left blank (NaN) and excluded from roughness. Remedies: a light etch, oblique or dark-field lighting, or a
pattern projected through the field diaphragm.</p>
<p><b>40×/0.85 and 100×/1.25 oil objectives</b> are usually corrected for a 0.17 mm cover glass and transmitted light.
On bare opaque metal without epi-illumination they lose contrast to spherical aberration and lack of light; the
100× also needs oil on the specimen. Deconvolution cannot replace missing illumination.</p>
<p><b>Focus breathing</b> (a small magnification change with focus) is not corrected: it could not be measured
reliably from defocused frames. Its effect is limited to the far-defocused frames at the image edges.</p>
"""

STEPS = ["Camera", "Objective", "Dark && flat", "Lateral scale", "Z scale", "Inspect"]


class Bridge(QObject):
    sweep = Signal(object)
    objective_hint = Signal(object)


class Job(QThread):
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kw):
        super().__init__()
        self.fn, self.args, self.kw = fn, args, kw
        self.cancel = False

    def run(self):
        try:
            res = self.fn(*self.args, progress=lambda f, m: self.progress.emit(f, m),
                          cancelled=lambda: self.cancel, **self.kw)
            self.done.emit(res)
        except zstack.Cancelled:
            self.failed.emit("Cancelled.")
        except Exception as e:  # surfaced to the user, never silently dropped
            self.failed.emit(f"{type(e).__name__}: {e}")


def _sr_job(frames, psf, progress=None, cancelled=None):
    if progress:
        progress(0.1, "Registering dwell frames and drizzling...")
    return dwell_superres(frames, psf)


class MainWindow(QMainWindow):
    def __init__(
        self,
        simulator_only: bool = False,
        replay: str | None = None,
        verbose: bool = False,
        fusion: str = "sharpest",
        crop: tuple | None = None,
        reset_geometry: bool = False,
    ):
        super().__init__()
        opencv_threads()
        self.settings = QSettings("MicroTopos", "MicroTopos")
        self.dark = self.settings.value("dark", True, type=bool)
        self.t = DARK if self.dark else LIGHT
        data_dir = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation) or Path.home() / ".microtopos")
        self.store = CalibrationStore(data_dir)
        self.obj = OBJECTIVES[self.settings.value("objective", "10x / 0.25")] \
            if self.settings.value("objective", "10x / 0.25") in OBJECTIVES else OBJECTIVES["10x / 0.25"]
        self.engine = LiveEngine(self.store, self.obj)
        self.fusion = fusion
        self.crop = crop
        if hasattr(self.engine, "mosaic"):
            self.engine.mosaic.fusion = fusion
        self.bridge = Bridge()
        self.engine.verbose = verbose
        self.engine.on_sweep = self.bridge.sweep.emit
        self.engine.on_objective_hint = self.bridge.objective_hint.emit
        self.bridge.sweep.connect(self._on_sweep)
        self.bridge.objective_hint.connect(self._on_objective_hint)
        self.source = None
        self.sim_source: SimulatorSource | None = None
        self.result: dict | None = None
        self.prev_result: dict | None = None
        self.repeat_armed = False
        self.job: Job | None = None
        self.sr_result = None
        self.last_mosaic_version = -1
        self.history: list[dict] = []
        self._result_t = -1e9
        self.loaded_mosaic = None
        self.current_hmosaic = None
        self.simulator_only = simulator_only
        self.replay = replay

        self.setWindowTitle(f"MicroTopos {__version__}")
        self.resize(1560, 940)
        self._build()
        self._shortcuts()
        self.engine.start()
        self.cameras = [] if (replay or simulator_only) else list_cameras()
        self._fill_sources()
        self._set_rotation(self.sp_rot.value())
        self._apply_theme()

        if reset_geometry:
            primary = QGuiApplication.primaryScreen()
            if primary:
                avail = primary.availableGeometry()
                w = min(1560, avail.width() - 40)
                h = min(940, avail.height() - 40)
                self.resize(w, h)
                self.move(avail.center().x() - w // 2, avail.center().y() - h // 2)
        else:
            g = self.settings.value("geometry")
            if g is not None:
                self.restoreGeometry(g)
                # Ensure window is visible on an active connected screen
                geo = self.geometry()
                screens = QGuiApplication.screens()
                intersects = any(s.geometry().intersects(geo) for s in screens)
                if not intersects or QGuiApplication.screenAt(geo.center()) is None:
                    primary = QGuiApplication.primaryScreen()
                    if primary:
                        avail = primary.availableGeometry()
                        w = min(geo.width(), avail.width() - 40)
                        h = min(geo.height(), avail.height() - 40)
                        geo.setSize(QSize(max(800, w), max(600, h)))
                        geo.moveCenter(avail.center())
                        self.setGeometry(geo)
        st = self.settings.value("state")
        if st is not None:
            self.restoreState(st)
        self.timer = QTimer(self, interval=66, timeout=self._tick)  # 15 Hz: above the camera's 12.5 fps
        self.timer.start()
        self._refresh_checklist()

    # ================================================================== layout
    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)
        # header
        head = QHBoxLayout()
        title = QLabel("MicroTopos")
        title.setObjectName("title")
        head.addWidget(title)
        self.chip_state = Chip("DWELL", "success")
        self.chip_state.setToolTip("Motion state detected from the video: DWELL (still), PAN (X-Y moving), "
                                   "Z_SWEEP (focus moving), OBJECTIVE_CHANGE")
        self.chip_xy = Chip()
        self.chip_z = Chip()
        self.chip_rad = Chip()
        self.chip_src = Chip()
        self.chip_fps = Chip("", "neutral")
        self.chip_sat = Chip("CLIPPING", "danger")
        self.chip_sat.setToolTip("More than 1% of pixels are saturated: lower the exposure, specular metal clips easily")
        self.chip_sat.hide()
        self.chip_rec = Chip("● REC", "danger")
        self.chip_rec.setToolTip("Raw camera frames are being saved; press Stop recording under the Live view")
        self.chip_rec.hide()
        for w in (self.chip_rec, self.chip_state, self.chip_src, self.chip_xy, self.chip_z, self.chip_rad, self.chip_sat):
            head.addWidget(w)
        head.addStretch(1)
        head.addWidget(self.chip_fps)
        self.btn_theme = QPushButton()
        self.btn_theme.clicked.connect(self._toggle_theme)
        head.addWidget(self.btn_theme)
        v.addLayout(head)
        self.banner = Banner(self.t)
        v.addWidget(self.banner)
        self.guide = QLabel()
        self.guide.setObjectName("guide")
        self.guide.setWordWrap(True)
        self.guide.setTextFormat(Qt.RichText)
        v.addWidget(self.guide)
        self.last_ignored = None
        # tabs
        self.tabs = QTabWidget()
        v.addWidget(self.tabs, 1)
        t = self.t
        self.p_live = ImagePanel(t, "No camera frames yet.\nChoose a source in step 1 (Camera).")
        self.lat_line = pg.LineSegmentROI([[20, 20], [120, 20]], pen=pg.mkPen(t["overlay_line"], width=2))
        self.lat_line.hide()
        self.p_live.add_item(self.lat_line)
        self.p_mosaic = ImagePanel(t, "The mosaic grows as you move the X-Y stage.")
        from PySide6.QtWidgets import QGraphicsRectItem
        self.cam_box = QGraphicsRectItem()
        self.cam_box.setZValue(20)
        self.p_mosaic.add_item(self.cam_box)
        self.mosaic_origin = (0, 0)
        self.p_edf = ImagePanel(t, "No Z-stack yet.\nTurn the fine focus slowly through the sample, or press Space.")
        self.p_height = ImagePanel(t, "No height map yet.\nTurn the fine focus slowly through the sample.", "height")
        self.prof_roi = pg.LineSegmentROI([[0, 0], [1, 1]], pen=pg.mkPen(t["overlay_profile"], width=2))
        self.prof_roi.hide()
        self.p_height.add_item(self.prof_roi)
        self.prof_roi.sigRegionChanged.connect(self._update_profile)
        self.prof_plot = pg.PlotWidget()
        self.prof_plot.setMaximumHeight(180)
        self.prof_curve = self.prof_plot.plot()
        self.prof_label = QLabel("Drag the orange line on the height map to see a profile.")
        self.prof_label.setObjectName("muted")

        self.tabs.addTab(self._wrap(self.p_live, self._live_bar()), "Live")
        self.tabs.addTab(self._wrap(self.p_mosaic, self._mosaic_bar()), "Mosaic")
        self.tabs.addTab(self._wrap(self.p_edf, self._edf_bar()), "All-in-focus")
        self.tabs.addTab(self._wrap(self.p_height, self._height_bar(), self.prof_plot, self.prof_label), "Height")
        self.tab3d = QWidget()
        self._build_3d()
        self.tabs.addTab(self.tab3d, "3D")
        self.tabs.currentChanged.connect(lambda i: QTimer.singleShot(400, self._check_gl) if i == 4 else None)
        self.report = QTextBrowser()
        self.report.setOpenExternalLinks(False)
        self.tabs.addTab(self.report, "Report")
        self._render_report()

        self._build_left()
        self._build_right()
        self.toast = Toast(central, self.t)
        self.setStatusBar(None)

    def _wrap(self, *ws) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        # tool bars (layouts) above the image, everything else in the given order
        ws = [x for x in ws if not isinstance(x, QWidget)] + [x for x in ws if isinstance(x, QWidget)]
        for x in ws:
            if isinstance(x, QWidget):
                lay.addWidget(x, 1 if isinstance(x, ImagePanel) else 0)
            else:
                lay.addLayout(x)
        return w

    def _btn(self, text, fn, tip="", primary=False) -> QPushButton:
        b = QPushButton(text)
        b.setFocusPolicy(Qt.TabFocus)  # a mouse click does not take focus, so Space never re-triggers it
        b.clicked.connect(fn)
        if tip:
            b.setToolTip(tip)
        if primary:
            b.setObjectName("primary")
        return b

    def _live_bar(self):
        h = QHBoxLayout()
        self.chk_focus = QCheckBox("Outline in-focus regions")
        self.chk_focus.setToolTip("During a focus sweep, outlines the areas that are currently at (or near) their sharpest")
        self.chk_focus.setChecked(False)
        h.addWidget(self.chk_focus)
        self.chk_zebra = QCheckBox("Show clipped pixels")
        self.chk_zebra.setToolTip("Magenta stripes where the image is saturated (≥ 250): lower the exposure there")
        self.chk_zebra.setChecked(True)
        h.addWidget(self.chk_zebra)
        self.btn_rec = QPushButton("● Record raw stream")
        self.btn_rec.setCheckable(True)
        self.btn_rec.setFocusPolicy(Qt.TabFocus)
        self.btn_rec.setToolTip("Save the camera's exact MJPEG frames to the 'recordings' folder for offline replay "
                                "(uv run microtopos --replay FOLDER)")
        self.btn_rec.toggled.connect(self._toggle_record)
        h.addWidget(self.btn_rec)
        self.lbl_live = QLabel()
        self.lbl_live.setObjectName("muted")
        h.addStretch(1)
        h.addWidget(self.lbl_live)
        return h

    def _mosaic_bar(self):
        h = QHBoxLayout()
        self.cmb_mview = QComboBox()
        self.cmb_mview.addItems(["Image", "Image + height overlay", "Height only"])
        self.cmb_mview.setCurrentIndex(1)
        self.cmb_mview.setToolTip("Every height map stays pinned to its place in the mosaic; overlapping maps are "
                                  "levelled to each other (least squares) and blended")
        self.cmb_mview.currentIndexChanged.connect(lambda: setattr(self, "last_mosaic_version", -1))
        h.addWidget(self.cmb_mview)
        h.addWidget(QLabel("Opacity"))
        self.sl_hop = QSlider(Qt.Horizontal)
        self.sl_hop.setRange(10, 100)
        self.sl_hop.setValue(55)
        self.sl_hop.setFixedWidth(110)
        self.sl_hop.valueChanged.connect(lambda: setattr(self, "last_mosaic_version", -1))
        h.addWidget(self.sl_hop)
        self.lbl_hlegend = QLabel()
        self.lbl_hlegend.setObjectName("mono")
        h.addWidget(self.lbl_hlegend)
        h.addStretch(1)
        h2 = QHBoxLayout()  # second row: actions and status, so nothing overlaps on narrow windows
        h2.addWidget(self._btn("Export height mosaic", self._export_height_mosaic,
                               "Stitched height mosaic as a Gwyddion .gsf"))
        h2.addWidget(self._btn("Reset mosaic  (R)", self._reset_mosaic, "Start a new mosaic"))
        self.lbl_mosaic = QLabel()
        self.lbl_mosaic.setObjectName("muted")
        h2.addStretch(1)
        h2.addWidget(self.lbl_mosaic)
        v = QVBoxLayout()
        v.addLayout(h)
        v.addLayout(h2)
        return v

    def _edf_bar(self):
        h = QHBoxLayout()
        self.chk_grains = QCheckBox("Overlay grain boundaries")
        self.chk_grains.toggled.connect(self._show_result)
        h.addWidget(self.chk_grains)
        self.cmb_edf = QComboBox()
        self.cmb_edf.addItems(["All-in-focus", "Super-resolution (experimental)"])
        self.cmb_edf.currentIndexChanged.connect(self._show_result)
        h.addWidget(self.cmb_edf)
        h.addStretch(1)
        self.lbl_edf = QLabel()
        self.lbl_edf.setObjectName("muted")
        h.addWidget(self.lbl_edf)
        return h

    def _height_bar(self):
        h = QHBoxLayout()
        self.cmb_height = QComboBox()
        self.cmb_height.addItems(["Height (form removed, S/L filtered)", "Height (raw fit)", "Confidence"])
        self.cmb_height.currentIndexChanged.connect(self._show_result)
        h.addWidget(self.cmb_height)
        self.cmb_hist = QComboBox()
        self.cmb_hist.setToolTip("Every height map of this session (most recent first)")
        self.cmb_hist.currentIndexChanged.connect(self._pick_history)
        h.addWidget(self.cmb_hist)
        h.addStretch(1)
        self.lbl_height = QLabel()
        self.lbl_height.setObjectName("muted")
        h.addWidget(self.lbl_height)
        return h

    def _build_3d(self):
        lay = QVBoxLayout(self.tab3d)
        lay.setContentsMargins(8, 8, 8, 8)
        self.gl = None
        try:
            import pyqtgraph.opengl as gl

            class SafeGL(gl.GLViewWidget):
                """GL view that records initialisation/paint failures instead of raising inside Qt."""
                failed = False

                def initializeGL(self):
                    try:
                        super().initializeGL()
                    except Exception as exc:
                        logger.debug("OpenGL SafeGL initializeGL failed: %s", exc)
                        self.failed = True

                def paintGL(self, *a, **k):
                    if not self.failed:
                        try:
                            super().paintGL(*a, **k)
                        except Exception as exc:
                            logger.debug("OpenGL SafeGL paintGL failed: %s", exc)
                            self.failed = True

            self.gl = SafeGL()
            self.gl_grid = gl.GLGridItem()
            self.gl.addItem(self.gl_grid)
            self.gl_surf = gl.GLSurfacePlotItem(shader="shaded", smooth=True, computeNormals=True)
            self.gl_surf.setVisible(False)  # an empty surface has no mesh: drawing it raises in pyqtgraph
            self.gl.addItem(self.gl_surf)
            lay.addWidget(self.gl, 1)
        except Exception as exc:
            logger.warning("OpenGL 3D view unavailable; falling back to 2D shaded relief: %s", exc)
            self.gl = None
        self.relief = ImagePanel(self.t, "No height map yet.")
        self.relief.setVisible(self.gl is None)
        lay.addWidget(self.relief, 1)
        self.lbl_3d = QLabel("3D view needs OpenGL; showing a shaded relief instead." if self.gl is None else "")
        self.lbl_3d.setObjectName("muted")
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Z exaggeration"))
        self.sl_exag = QSlider(Qt.Horizontal)
        self.sl_exag.setRange(1, 100)
        self.sl_exag.setValue(10)
        self.sl_exag.valueChanged.connect(self._update_3d)
        bar.addWidget(self.sl_exag, 1)
        bar.addWidget(QLabel("Colour map"))
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["viridis", "cividis", "inferno", "plasma", "magma", "turbo"])
        self.cmb_cmap.currentIndexChanged.connect(self._update_3d)
        bar.addWidget(self.cmb_cmap)
        bar.addWidget(self.lbl_3d)
        lay.addLayout(bar)


    def _check_gl(self):
        """Fall back to a shaded relief if the GL widget could not create a valid context."""
        if self.gl is not None and self.tabs.currentWidget() is self.tab3d and (self.gl.failed or not self.gl.isValid()):
            self.gl.hide()
            self.gl = None
            self.relief.show()
            self.lbl_3d.setText("OpenGL context unavailable; showing a shaded relief instead.")
            self._update_3d()

    # ------------------------------------------------------------------ left: guided session
    def _build_left(self):
        dock = QDockWidget("Session", self)
        dock.setObjectName("session")
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        self.step_btns, self.step_chips = [], []
        for i, name in enumerate(STEPS):
            row = QHBoxLayout()
            b = QPushButton(f"{i + 1}. {name}")
            b.setObjectName("step")
            b.setCheckable(True)
            b.setFocusPolicy(Qt.TabFocus)
            b.clicked.connect(lambda _=False, k=i: self._open_step(k))
            chip = Chip("", "neutral")
            chip.setFixedWidth(96)
            row.addWidget(b, 1)
            row.addWidget(chip)
            lay.addLayout(row)
            self.step_btns.append(b)
            self.step_chips.append(chip)
        self.stack = QStackedWidget()
        for page in (self._page_camera(), self._page_objective(), self._page_radiometric(), self._page_lateral(),
                     self._page_z(), self._page_inspect()):
            sc = QScrollArea()
            sc.setWidgetResizable(True)
            sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            sc.setWidget(page)
            self.stack.addWidget(sc)
        lay.addSpacing(8)
        line = QLabel()
        line.setFixedHeight(1)
        line.setObjectName("rule")
        lay.addWidget(line)
        lay.addWidget(self.stack, 1)
        dock.setWidget(w)
        dock.setMinimumWidth(360)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self._open_step(0)

    def _open_step(self, k: int):
        for i, b in enumerate(self.step_btns):
            b.setChecked(i == k)
        self.stack.setCurrentIndex(k)
        if k == 3:
            self.tabs.setCurrentIndex(0)
            self._show_lateral_line(True)
        else:
            self._show_lateral_line(False)

    def _page_camera(self):
        self.cmb_source = QComboBox()
        self.cmb_source.currentIndexChanged.connect(self._on_source)
        self.cmb_res = QComboBox()
        # 16:9 modes only: lateral calibration is scaled between modes, which assumes the same field of view
        self.cmb_res.addItems(["1920x1080", "1280x720"])
        self.cmb_res.setToolTip("Capture size. Full resolution is used for Z-stacks; live analysis uses a 640 px preview")
        self.cmb_res.currentIndexChanged.connect(lambda: self._on_source(self.cmb_source.currentIndex()))
        self.chk_autoexp = QCheckBox("Automatic exposure")
        self.chk_autoexp.setToolTip("Turn off for measurements: automatic exposure changes brightness between frames")
        self.sl_exp = QSlider(Qt.Horizontal)
        self.sl_exp.setRange(78, 738)
        self.sl_exp.setValue(int(self.settings.value("exposure", 300)))
        self.lbl_exp = QLabel()
        self.lbl_exp.setObjectName("mono")
        self.chk_autoexp.toggled.connect(self._apply_exposure)
        self._exp_timer = QTimer(self)
        self._exp_timer.setSingleShot(True)
        self._exp_timer.setInterval(250)  # debounce: keyboard, wheel and drags all apply, at most 4 times a second
        self._exp_timer.timeout.connect(self._apply_exposure)
        self.sl_exp.valueChanged.connect(lambda v: (self.lbl_exp.setText(f"{v / 10:.1f} ms"), self._exp_timer.start()))
        self.btn_autoexp_once = self._btn("Set once", self._auto_exposure_once,
                                          "Adjust the manual exposure once so the brightest areas are just below "
                                          "saturation, then keep it locked")
        self.lbl_exp.setText(f"{self.sl_exp.value() / 10:.1f} ms")
        self.chk_linear = QCheckBox("Linearise gamma (radiometric luminance)")
        self.chk_linear.setToolTip("Off: Rec.601 luma of the camera's gamma-encoded output. On: inverts gamma 2.2 first.")
        self.chk_linear.toggled.connect(lambda c: setattr(self.engine, "linearize", c))
        self.lbl_readback = QLabel("—")
        self.lbl_readback.setObjectName("muted")
        self.lbl_readback.setWordWrap(True)
        exp = QHBoxLayout()
        exp.addWidget(self.sl_exp, 1)
        exp.addWidget(self.lbl_exp)
        exp.addWidget(self.btn_autoexp_once)
        self.sp_rot = QDoubleSpinBox()
        self.sp_rot.setRange(-180.0, 180.0)
        self.sp_rot.setSingleStep(0.5)
        self.sp_rot.setDecimals(1)
        self.sp_rot.setSuffix(" °")
        self.sp_rot.setToolTip("Rotate the Live, Mosaic, All-in-focus and Height views, e.g. when the camera sits "
                               "rotated in the eyepiece tube. Display only: measurements are unaffected.")
        self.sp_rot.setValue(float(self.settings.value("view_rotation", 0.0)))
        self.sp_rot.valueChanged.connect(self._set_rotation)
        self.btn_rot_measure = self._btn("Measure from X pan", self._measure_rotation,
                                         "Click, move only the stage's X knob for a few field widths, click again: "
                                         "the views are rotated so stage X is horizontal")
        self.btn_rot_measure.setCheckable(True)
        rot = QHBoxLayout()
        rot.addWidget(QLabel("View rotation"))
        rot.addWidget(self.sp_rot, 1)
        rot.addWidget(self.btn_rot_measure)
        demo = QGridLayout()
        demo.addWidget(self._btn("Pan around", lambda: self._demo("pan"), "Simulator: move the stage in a loop"), 0, 0)
        demo.addWidget(self._btn("Focus sweep", lambda: self._demo("sweep"), "Simulator: hand-like focus sweep"), 0, 1)
        demo.addWidget(self._btn("Sweep + reversal", lambda: self._demo("rev"),
                                 "Simulator: sweep that turns back through focus"), 1, 0, 1, 2)
        self.demo_box = section("Demo (simulator only)", demo,
                                hint="Synthetic specimen. Its heights are simulation truth and are labelled SIMULATED.")
        return section("1. Camera", self.cmb_source, self.cmb_res, self.chk_autoexp, exp, rot, self.chk_linear,
                       self.lbl_readback, self.demo_box,
                       hint="Pick the microscope camera. Exposure is locked and in-camera sharpening is switched off, "
                            "because both would corrupt focus measurements.")

    def _page_objective(self):
        self.cmb_obj = QComboBox()
        self.cmb_obj.addItems(OBJECTIVE_NAMES)
        self.cmb_obj.setCurrentText(self.obj.name)
        self.cmb_obj.currentTextChanged.connect(self._on_objective)
        self.lbl_optics = QLabel()
        self.lbl_optics.setObjectName("mono")
        self.lbl_optics.setWordWrap(True)
        return section("2. Objective  (keys 1–4)", self.cmb_obj, self.lbl_optics,
                       hint="Sets the numerical aperture used for focus windows, depth of field and the roughness "
                            "S-filter. Change it whenever you turn the nosepiece.")

    def _page_radiometric(self):
        self.btn_dark = self._btn("Record dark  (lamp off)", lambda: self._record("dark"),
                                  "Average 24 frames with the light blocked: removes sensor offset")
        self.btn_flat = self._btn("Record flat  (blank, defocused)", lambda: self._record("flat"),
                                  "Average 24 frames of a featureless, defocused area (a mirror or polished blank): "
                                  "removes vignetting and dust")
        self.btn_clear_rad = self._btn("Clear for this objective", self._clear_radiometric)
        self.lbl_rad = QLabel()
        self.lbl_rad.setObjectName("muted")
        self.lbl_rad.setWordWrap(True)
        return section("3. Dark & flat", self.btn_dark, self.btn_flat, self.btn_clear_rad, self.lbl_rad,
                       hint="Stored per objective, capture size and exposure. Recommended before measuring roughness; "
                            "optional for viewing.")

    def _page_lateral(self):
        self.btn_lat = self._btn("Use the blue line on Live…", self._calibrate_lateral,
                                 "Place the line over a known stage-micrometer interval, then enter its length", True)
        self.btn_lat_clear = self._btn("Clear (use nominal)", self._clear_lateral)
        self.lbl_lat = QLabel()
        self.lbl_lat.setObjectName("muted")
        self.lbl_lat.setWordWrap(True)
        return section("4. Lateral scale", self.btn_lat, self.btn_lat_clear, self.lbl_lat,
                       hint="Put a stage micrometer under the objective. Drag the blue line's ends onto two graduation "
                            "marks on the Live view, then enter their distance in µm.")

    def _page_z(self):
        self.btn_z_travel = self._btn("From last sweep's knob travel…", self._calibrate_z_travel,
                                      "Turn the fine-focus knob from one mark to another during a sweep, then enter "
                                      "the travel in µm (knob divisions × µm per division). Assumes even speed.", True)
        self.btn_z_ruler = self._btn("Apply DOF ruler estimate (experimental)", self._calibrate_z_ruler,
                                     "Estimates µm per frame by comparing the measured focus-curve width with the "
                                     "optical model. Simulator tests: errors up to ±35%.")
        self.btn_z_clear = self._btn("Clear (focus-frame units)", self._clear_z)
        self.lbl_z = QLabel()
        self.lbl_z.setObjectName("muted")
        self.lbl_z.setWordWrap(True)
        return section("5. Z scale", self.btn_z_travel, self.btn_z_ruler, self.btn_z_clear, self.lbl_z,
                       hint="The focus knob has no encoder, so heights are in focus frames until calibrated.")

    def _page_inspect(self):
        self.cmb_override = QComboBox()
        self.cmb_override.addItems([o.value for o in Override])
        self.cmb_override.setToolTip("Auto: motion is detected from the video. The others force a mode.")
        self.cmb_override.currentTextChanged.connect(lambda s: self.engine.set_override(Override(s)))
        self.btn_manual = self._btn("Start Z capture  (Space)", self._toggle_manual, "Capture a focus sweep manually")
        self.btn_repeat = self._btn("Repeat sweep here", self._arm_repeat,
                                    "Repeat a sweep at the same spot to measure repeatability (σ_inst)")
        self.btn_sr = self._btn("Super-resolve current view (experimental)", self._run_sr,
                                "Drizzle + Richardson-Lucy on the last still frames, 512×512 centre crop. Gains only "
                                "where the camera undersamples the optics (low magnification).")
        self.cmb_measure = QComboBox()
        self.cmb_measure.addItems(MEASURES)
        self.cmb_measure.setToolTip("Focus measure for shape from focus. DoG band-pass is the most robust default.")
        self.sp_form = QSpinBox()
        self.sp_form.setRange(0, 3)
        self.sp_form.setValue(1)
        self.sp_form.setToolTip("F-operator: polynomial order removed before roughness (1 = plane)")
        self.sp_ls = QDoubleSpinBox()
        self.sp_ls.setRange(0, 100)
        self.sp_ls.setDecimals(2)
        self.sp_ls.setSuffix(" µm")
        self.sp_ls.setToolTip("S-filter nesting index λs (removes shorter wavelengths). Default: optical cutoff λ/2NA")
        self.sp_ls.setValue(self.obj.lateral_cutoff_um)
        self.sp_lc = QDoubleSpinBox()
        self.sp_lc.setRange(0, 5000)
        self.sp_lc.setDecimals(1)
        self.sp_lc.setSuffix(" µm")
        self.sp_lc.setSpecialValueText("off")
        self.sp_lc.setToolTip("L-filter nesting index λc (removes longer wavelengths, i.e. waviness). 0 = off")
        f = QFormLayout()
        f.addRow("Mode", self.cmb_override)
        f.addRow("Focus measure", self.cmb_measure)
        f.addRow("Form removal order", self.sp_form)
        f.addRow("S-filter λs", self.sp_ls)
        f.addRow("L-filter λc", self.sp_lc)
        self.btn_recompute = self._btn("Recompute metrology", self._recompute)
        return section("6. Inspect", f, self.btn_manual, self.btn_repeat, self.btn_recompute, self.btn_sr,
                       hint="Move X-Y to grow the mosaic. Turn the fine focus slowly through the surface to build a "
                            "height map; it is reconstructed automatically when the focus stops.")

    # ------------------------------------------------------------------ right: telemetry
    def _build_right(self):
        dock = QDockWidget("Telemetry", self)
        dock.setObjectName("telemetry")
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        self.spark = pg.PlotWidget()
        self.spark.setFixedHeight(110)
        self.spark.hideAxis("bottom")
        self.spark.setLabel("left", "sharpness")
        self.spark_curve = self.spark.plot()
        self.coverage = ImagePanel(self.t, "No sweep yet", "focus", "inferno")
        self.coverage.setFixedHeight(190)
        self.coverage.plot.hideAxis("bottom")
        self.coverage.plot.hideAxis("left")
        self.lbl_motion = QLabel()
        self.lbl_motion.setObjectName("mono")
        self.lbl_motion.setWordWrap(True)
        self.prog = QProgressBar()
        self.prog.setRange(0, 1000)
        self.prog.setValue(0)
        self.btn_cancel = self._btn("Cancel", self._cancel)
        self.btn_cancel.setEnabled(False)
        pr = QHBoxLayout()
        pr.addWidget(self.prog, 1)
        pr.addWidget(self.btn_cancel)
        self.chk_autorec = QCheckBox("Reconstruct sweeps automatically")
        self.chk_autorec.setToolTip("Off: focus sweeps are only kept; press 'Reconstruct last sweep' to build a map")
        self.chk_autorec.setChecked(self.settings.value("auto_reconstruct", True, type=bool))
        self.chk_autorec.toggled.connect(lambda c: self.settings.setValue("auto_reconstruct", c))
        self.btn_lastsweep = self._btn("Reconstruct last sweep", self._reconstruct_last,
                                       "Build a height map from the last captured focus sweep, even an incomplete one")
        self.btn_lastsweep.setEnabled(False)
        pr2 = QVBoxLayout()  # two rows: the telemetry dock is narrow
        pr2.addWidget(self.chk_autorec)
        pr2.addWidget(self.btn_lastsweep)
        self.lbl_job = QLabel("Idle.")
        self.lbl_job.setObjectName("muted")
        self.lbl_job.setWordWrap(True)
        ex = QGridLayout()
        ex.addWidget(self._btn(".gsf", self._export_gsf, "Height map in SI units for Gwyddion"), 0, 0)
        ex.addWidget(self._btn("TIFF", self._export_tiff, "32-bit float height map with resolution tags"), 0, 1)
        ex.addWidget(self._btn("STL", self._export_stl, "Binary STL surface mesh in µm"), 0, 2)
        ex.addWidget(self._btn("Report", self._export_report, "JSON + CSV + PNG bundle  (Ctrl+E)"), 1, 0)
        ex.addWidget(self._btn("Save", self._save_session, "Save the session  (Ctrl+S)"), 1, 1)
        ex.addWidget(self._btn("Open", self._load_session, "Open a saved session  (Ctrl+O)"), 1, 2)
        self.diag = QPlainTextEdit()
        self.diag.setReadOnly(True)
        self.diag.setMaximumHeight(200)
        self.btn_diag = QPushButton("Diagnostics ▾")
        self.btn_diag.setCheckable(True)
        self.btn_diag.setChecked(False)
        self.btn_diag.setFocusPolicy(Qt.TabFocus)
        self.btn_diag.toggled.connect(lambda c: (self.diag.setVisible(c),
                                                 self.btn_diag.setText("Diagnostics ▴" if c else "Diagnostics ▾")))
        self.diag.hide()
        for x in (section("Sharpness", self.spark), section("Focus sweep coverage", self.coverage), self.lbl_motion,
                  section("Reconstruction", pr, pr2, self.lbl_job), section("Export", ex), self.btn_diag, self.diag):
            lay.addWidget(x) if isinstance(x, QWidget) else lay.addLayout(x)
        lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setWidget(w)
        dock.setWidget(sc)
        dock.setMinimumWidth(390)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    # ================================================================== theme & shortcuts
    def _apply_theme(self):
        t = self.t = DARK if self.dark else LIGHT
        QApplication.instance().setStyleSheet(stylesheet(t))
        self.btn_theme.setText("Light theme" if self.dark else "Dark theme")
        for c in (self.chip_rec, self.chip_state, self.chip_xy, self.chip_z, self.chip_rad, self.chip_src, self.chip_fps, self.chip_sat,
                  *self.step_chips):
            c.retheme(t)
        for p in (self.p_live, self.p_mosaic, self.p_edf, self.p_height, self.relief, self.coverage):
            p.retheme(t)
        for pw in (self.prof_plot, self.spark):
            pw.setBackground(t["surface"])
            for ax in ("left", "bottom"):
                pw.getAxis(ax).setPen(pg.mkPen(t["border"]))
                pw.getAxis(ax).setTextPen(pg.mkPen(t["muted"]))
        self.spark_curve.setPen(pg.mkPen(t["accent"], width=2))
        self.prof_curve.setPen(pg.mkPen(t["overlay_profile"], width=2))
        self.lat_line.setPen(pg.mkPen(t["overlay_line"], width=2))
        self.prof_roi.setPen(pg.mkPen(t["overlay_profile"], width=2))
        self.toast.tokens = t
        self.banner.tokens = t
        if self.gl is not None:
            self.gl.setBackgroundColor(t["surface"])
            self.gl_grid.setColor(pg.mkColor(t["grid"]))
        self._refresh_checklist()
        self._show_result()

    def _toggle_theme(self):
        self.dark = not self.dark
        self.settings.setValue("dark", self.dark)
        self._apply_theme()

    def _shortcuts(self):
        def add(key, fn):
            a = QAction(self)
            a.setShortcut(QKeySequence(key))
            a.setShortcutContext(Qt.WindowShortcut)
            a.triggered.connect(lambda: None if self._typing() else fn())
            self.addAction(a)

        for i, name in enumerate(OBJECTIVE_NAMES):
            add(str(i + 1), lambda n=name: self.cmb_obj.setCurrentText(n))
        add(Qt.Key_Space, self._toggle_manual)
        add("R", self._reset_mosaic)
        add("Ctrl+S", self._save_session)
        add("Ctrl+O", self._load_session)
        add("Ctrl+E", self._export_report)
        add("F1", lambda: self.tabs.setCurrentIndex(5))

    @staticmethod
    def _typing() -> bool:
        w = QApplication.focusWidget()
        return isinstance(w, (QLineEdit, QAbstractSpinBox, QComboBox, QPlainTextEdit))

    def notify(self, text: str, kind: str = "info"):
        self.toast.show_message(text, kind)

    # ================================================================== sources & controls
    def _fill_sources(self):
        self.cmb_source.blockSignals(True)
        self.cmb_source.clear()
        self.cmb_source.addItem("Simulator (synthetic specimen)", None)
        if self.replay:
            self.cmb_source.addItem(f"Recording: {self.replay}", "replay")
        for c in self.cameras:
            self.cmb_source.addItem(f"{c.name}  ({c.path or c.index})", c)
        # prefer a microscope camera over a laptop webcam
        pick = 1 if self.replay else 0
        for i, c in enumerate(self.cameras, 1):
            if not self.simulator_only and ("okular" in c.name.lower() or "microscope" in c.name.lower()):
                pick = i
        self.cmb_source.setCurrentIndex(pick)
        self.cmb_source.blockSignals(False)
        self._on_source(pick)

    def _on_source(self, idx: int):
        info = self.cmb_source.itemData(idx)
        if self.source is not None:
            self.engine.set_source(None)
            self.source.stop()
        if info is None:
            self.sim_source = SimulatorSource(7, self.obj)
            self.source = self.sim_source
        elif info == "replay":
            self.sim_source = None
            self.source = ReplaySource(self.replay)
        else:
            w, h = (int(v) for v in self.cmb_res.currentText().split("x"))
            self.sim_source = None
            self.source = CameraSource(info, w, h, 25)
        self.source.start()
        self.engine.set_source(self.source)
        if isinstance(self.source, CameraSource):
            rng = self.source.exposure_range()
            if rng:  # use the device's own limits (the defaults are the Bresser's)
                v = self.sl_exp.value()
                self.sl_exp.blockSignals(True)
                self.sl_exp.setRange(*rng)
                self.sl_exp.setValue(int(np.clip(v, *rng)))
                self.sl_exp.blockSignals(False)
                self.lbl_exp.setText(f"{self.sl_exp.value() / 10:.1f} ms")
        self.btn_autoexp_once.setEnabled(isinstance(self.source, CameraSource))
        self.demo_box.setVisible(self.sim_source is not None)
        self.cmb_res.setVisible(self.sim_source is None)
        self._apply_exposure()
        QTimer.singleShot(1500, self._show_readback)
        self._refresh_checklist()

    def _rot_panels(self):
        return (self.p_live, self.p_mosaic, self.p_edf, self.p_height, self.coverage)

    def _set_rotation(self, deg: float):
        self.settings.setValue("view_rotation", float(deg))
        for p in self._rot_panels():
            p.set_rotation(deg)

    def _measure_rotation(self, on: bool):
        s = self.engine.snapshot()
        if on:
            self._rot_start = s.mosaic_xy
            self.btn_rot_measure.setText("Finish (after an X-only move)")
            self.notify("Move only the stage's X knob for at least a field width, then click 'Finish'.")
            return
        self.btn_rot_measure.setText("Measure from X pan")
        x0, y0 = getattr(self, "_rot_start", (0.0, 0.0))
        dx, dy = s.mosaic_xy[0] - x0, s.mosaic_xy[1] - y0
        if math.hypot(dx, dy) < 40 or s.mosaic_lost:
            self.notify("Not enough tracked X movement to measure the angle; move further while the mosaic tracks.",
                        "warning")
            return
        a = math.degrees(math.atan2(dy, dx))
        a = (a + 90.0) % 180.0 - 90.0  # direction of the axis, not of travel
        self.sp_rot.setValue(round(-a, 1))
        self.notify(f"Stage X runs at {a:+.1f}° in the camera image; views rotated by {-a:+.1f}°.", "success")

    def _auto_exposure_once(self):
        """Software one-shot exposure: scale the manual exposure until the 99.5th percentile is ~220/255."""
        if not isinstance(self.source, CameraSource):
            self.notify("One-shot exposure needs a real camera.", "warning")
            return
        self.chk_autoexp.setChecked(False)
        self._ae_steps = 0
        self._ae_seq = -1
        self.btn_autoexp_once.setEnabled(False)
        QTimer.singleShot(600, self._auto_exposure_step)

    def _auto_exposure_step(self):
        s = self.engine.snapshot()
        if s.preview is None or s.seq == self._ae_seq:
            QTimer.singleShot(300, self._auto_exposure_step)
            return
        self._ae_seq = s.seq
        p = float(np.percentile(s.preview, 99.5))
        lo, hi = self.sl_exp.minimum(), self.sl_exp.maximum()
        cur = self.sl_exp.value()
        new = int(np.clip(round(cur * 220.0 / max(p, 1.0)) if p < 250 else cur * 0.6, lo, hi))
        self._ae_steps += 1
        if abs(new - cur) <= max(2, 0.03 * cur) or self._ae_steps >= 8 or new == cur:
            self.btn_autoexp_once.setEnabled(True)
            note = "" if 150 < p < 250 else " The lamp may be too dim or too bright for the exposure range."
            self.notify(f"Exposure set to {cur / 10:.1f} ms (brightest areas {p:.0f}/255).{note}",
                        "success" if not note else "warning")
            return
        self.sl_exp.setValue(new)
        self._apply_exposure()
        QTimer.singleShot(700, self._auto_exposure_step)  # let a few frames with the new exposure arrive

    def _apply_exposure(self):
        auto = self.chk_autoexp.isChecked()
        self.sl_exp.setEnabled(not auto)
        self.settings.setValue("exposure", self.sl_exp.value())
        self.engine.exposure = None if auto else float(self.sl_exp.value())
        if isinstance(self.source, CameraSource):
            got = self.source.set_manual_exposure(None if auto else self.sl_exp.value())
            self.cam_controls = got
            self._show_readback()
        self.engine._corrector_key = None

    def _show_readback(self):
        if isinstance(self.source, CameraSource):
            rb = dict(self.source.readback)
            rb.update(getattr(self, "cam_controls", {}) or {})
            err = f"  Error: {self.source.error}" if self.source.error else ""
            self.lbl_readback.setText("Device reports: " + ", ".join(f"{k}={v}" for k, v in rb.items()) + err)
        else:
            self.lbl_readback.setText("Simulator: 640×360, randomised noise, MJPEG, tremor and focus breathing.")

    def _on_objective(self, name: str):
        self.obj = OBJECTIVES[name]
        self.settings.setValue("objective", name)
        self.engine.set_objective(self.obj)
        if self.sim_source is not None:
            self.sim_source.set_objective(self.obj)
        self.sp_ls.setValue(self.obj.lateral_cutoff_um)
        self.engine.reset_mosaic()
        self._refresh_checklist()
        self.notify(f"Objective {name}. Mosaic restarted for the new magnification.")

    def _on_objective_hint(self, ratio):
        if ratio is not None and abs(math.log(ratio)) < math.log(1.6):
            self.notify("The view changed abruptly but the magnification looks the same (stage jump or occlusion?). "
                        "The mosaic re-locates when you return to a mapped area.", "warning")
            return
        mags = [OBJECTIVES[n].magnification for n in OBJECTIVE_NAMES]
        g = guess_objective(self.obj.magnification, ratio, mags)
        guess = OBJECTIVE_NAMES[g] if g is not None else None
        text = "Objective change suspected (the image changed abruptly)."
        if guess and guess != self.obj.name:
            text += f" Texture scale suggests <b>{guess}</b> (a guess)."
        acts = [(n, lambda n=n: self.cmb_obj.setCurrentText(n)) for n in OBJECTIVE_NAMES]
        acts.append(("Dismiss", lambda: None))
        self.banner.ask(text, acts)

    def _demo(self, kind):
        if self.sim_source is None:
            return
        if kind == "pan":
            self.sim_source.demo_pan()
        else:
            self.sim_source.demo_sweep(reversal=(kind == "rev"))
        self.tabs.setCurrentIndex(1 if kind == "pan" else 0)

    def _toggle_manual(self):
        if self.engine.sm.override == Override.Z:
            self.cmb_override.setCurrentText(Override.AUTO.value)
            self.btn_manual.setText("Start Z capture  (Space)")
        else:
            self.cmb_override.setCurrentText(Override.Z.value)
            self.btn_manual.setText("Finish Z capture  (Space)")
            self.notify("Capturing: turn the fine focus through the surface, then press Space.")

    def _toggle_record(self, on: bool):
        if not isinstance(self.source, CameraSource):
            if on:
                self.btn_rec.setChecked(False)
                self.notify("Recording needs a camera source.", "warning")
            return
        if on:
            import datetime
            folder = Path(self.settings.value("rec_dir", str(Path.cwd() / "recordings"))) / \
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.source.start_recording(folder)
            self.rec_folder = folder
            self.btn_rec.setText("■ Stop recording")
            self.notify(f"Recording raw frames to {folder}")
        else:
            n = self.source.stop_recording()
            self.btn_rec.setText("● Record raw stream")
            self.notify(f"Saved {n} frames to {self.rec_folder}", "success")

    def _reset_mosaic(self):
        self.engine.reset_mosaic()
        self.notify("Mosaic cleared.")

    # ================================================================== calibration actions
    def _record(self, kind):
        self.engine.request_calibration(kind, 24)
        self.notify(f"Recording {kind} frame: keep still…")

    def _clear_radiometric(self):
        c = self.store.get(self.obj)
        c.darks.clear()
        c.flats.clear()
        self.store.save()
        self.engine._corrector_key = None
        self._refresh_checklist()

    def _show_lateral_line(self, on: bool):
        self.lat_line.setVisible(on)
        if on and self.p_live.data is not None:
            h, w = self.p_live.data.shape[:2]
            um = self.p_live.um_per_px
            if not getattr(self, "_lat_placed", False):
                self.lat_line.setPos(0, 0)
                hs = self.lat_line.getHandles()
                hs[0].setPos(w * um * 0.3, h * um * 0.5)
                hs[1].setPos(w * um * 0.7, h * um * 0.5)
                self._lat_placed = True

    def _calibrate_lateral(self):
        if self.p_live.data is None:
            self.notify("No live image yet.", "warning")
            return
        h0, h1 = (self.lat_line.mapToParent(hd.pos()) for hd in self.lat_line.getHandles())
        um_now = self.p_live.um_per_px
        px = math.hypot(h1.x() - h0.x(), h1.y() - h0.y()) / um_now  # length in preview pixels
        if px < 10:
            self.notify("Drag the blue line's ends onto two micrometer marks first.", "warning")
            return
        val, ok = QInputDialog.getDouble(self, "Lateral calibration",
                                         f"The line spans {px:.1f} preview pixels.\nTrue distance (µm):", 100.0, 0.1, 1e5, 2)
        if not ok:
            return
        prev_w = self.p_live.data.shape[1]
        full = (val / px) * prev_w / 1920.0
        c = self.store.get(self.obj)
        c.lateral_um_per_px_full = full
        c.lateral_source = f"stage micrometer: {val:g} µm over {px:.1f} px"
        c.lateral_date = today()
        self.store.save()
        self._refresh_checklist()
        nominal = self.obj.nominal_um_per_px(1920)
        self.notify(f"Lateral scale {full:.4f} µm/px at 1920 px ({full / nominal:.2f}× the nominal value).", "success")

    def _clear_lateral(self):
        c = self.store.get(self.obj)
        c.lateral_um_per_px_full = None
        self.store.save()
        self._refresh_checklist()

    def _calibrate_z_travel(self):
        r = self.result
        if r is None or r.get("z_status") == CalStatus.SIMULATED.value:
            self.notify("Make a real focus sweep first (turn the knob between two marks).", "warning")
            return
        val, ok = QInputDialog.getDouble(self, "Z calibration from knob travel",
                                         f"The last sweep used {r['frames_used']} frames.\n"
                                         "Knob travel during that sweep (µm):", 10.0, 0.01, 1e4, 3)
        if ok:
            c = self.store.get(self.obj)
            c.z_um_per_frame = val / max(1, r["frames_used"] - 1)
            c.z_source = f"knob travel {val:g} µm over {r['frames_used']} frames (even speed assumed)"
            c.z_uncertainty_pct = None
            c.z_date = today()
            self.store.save()
            self._refresh_checklist()
            self.notify(f"Z scale {c.z_um_per_frame:.4f} µm per frame.", "success")

    def _calibrate_z_ruler(self):
        r = self.result
        est = r.get("dof_ruler_um_per_frame") if r else None
        if not est:
            self.notify("Needs a hand sweep reconstructed in focus-frame units first.", "warning")
            return
        c = self.store.get(self.obj)
        c.z_um_per_frame = float(est)
        c.z_source = "DOF ruler (experimental, simulator error up to ±35%)"
        c.z_uncertainty_pct = 35.0
        c.z_date = today()
        self.store.save()
        self._refresh_checklist()
        self.notify(f"Applied experimental Z scale {est:.4f} µm per frame.", "warning")

    def _clear_z(self):
        c = self.store.get(self.obj)
        c.z_um_per_frame = None
        self.store.save()
        self._refresh_checklist()

    def _arm_repeat(self):
        if self.result is None:
            self.notify("Make a first sweep, then press this and sweep again without moving X-Y.", "warning")
            return
        self.repeat_armed = True
        self.notify("Repeat armed: sweep again at the same spot without moving the stage.")

    # ================================================================== checklist & chips
    def _refresh_checklist(self):
        c = self.store.get(self.obj)
        live = self.engine.snapshot()
        cam_ok = self.source is not None and (live.seq > 0)
        sim = self.sim_source is not None
        rkey = radiometric_key(live.frame_shape, self.engine.exposure) if live.frame_shape[0] else None
        rad = rkey is not None and rkey in c.darks and rkey in c.flats
        states = [
            ("Simulator" if sim else ("Live" if cam_ok else "No frames"), "info" if sim else ("success" if cam_ok else "danger")),
            (self.obj.name.split(" /")[0], "success"),
            ("Done" if rad else ("Partial" if rkey in c.darks or rkey in c.flats else "Optional"),
             "success" if rad else "warning"),
            ("Calibrated" if c.lateral_um_per_px_full else "Nominal", "success" if c.lateral_um_per_px_full else "warning"),
            ("Calibrated" if c.z_um_per_frame else "Frames", "success" if c.z_um_per_frame else "warning"),
            ("Ready" if cam_ok else "Waiting", "success" if cam_ok else "neutral"),
        ]
        for chip, (txt, kind) in zip(self.step_chips, states):
            chip.set(txt, kind)
        nxt = next((i for i, (_, k) in enumerate(states) if k in ("danger",)), None)
        for i, b in enumerate(self.step_btns):
            b.setToolTip("Next step" if i == nxt else "")
        um_full = c.um_per_px(1920)
        self.chip_xy.set(f"XY {'CAL' if c.lateral_um_per_px_full else 'NOMINAL'} {um_full:.3f} µm/px",
                         "success" if c.lateral_um_per_px_full else "warning",
                         c.lateral_source or "Nameplate optics only; calibrate with a stage micrometer (step 4)")
        self.chip_z.set(f"Z {'CAL ' + format(c.z_um_per_frame, '.3f') + ' µm/f' if c.z_um_per_frame else 'FRAMES'}",
                        "success" if c.z_um_per_frame else "warning", c.z_source or "Heights in focus-frame units")
        self.chip_rad.set("DARK/FLAT ✓" if rad else "DARK/FLAT —", "success" if rad else "neutral")
        self.chip_src.set("SIMULATOR" if sim else "CAMERA", "info" if sim else "success")
        o = self.obj
        self.lbl_optics.setText(
            f"NA {o.na:.2f}{'  oil n=1.515' if o.is_oil else ''}\nRayleigh {o.rayleigh_um:.2f} µm\n"
            f"Optical cutoff λ/2NA {o.lateral_cutoff_um:.2f} µm\nDepth of field {o.dof_um:.2f} µm\n"
            f"Pixel {um_full:.3f} µm at 1920 px ({c.lateral_status.value.lower()})")
        lines = [f"{k}: dark {'✓' if k in c.darks else '—'}, flat {'✓' if k in c.flats else '—'}"
                 for k in sorted(set(c.darks) | set(c.flats))]
        self.lbl_rad.setText("\n".join(lines) or "Nothing recorded for this objective.")
        self.lbl_lat.setText(f"{c.lateral_status.value}: {c.lateral_source or 'nominal'}"
                             + (f" ({c.lateral_date})" if c.lateral_date else ""))
        self.lbl_z.setText(f"{c.z_status.value}: {c.z_source or 'focus-frame units'}" + (f" ({c.z_date})" if c.z_date else ""))

    # ================================================================== live tick
    def _tick(self):
        s = self.engine.snapshot()
        if s.preview is None:
            return
        cal = self.store.get(self.obj)
        # the preview covers the full field of view; the simulator's truth scale is the nominal one
        pw = s.preview.shape[1]
        um_prev = self.obj.nominal_um_per_px(pw) if self.sim_source is not None else cal.um_per_px(pw)
        self.um_preview = um_prev
        self.p_live.set_image(s.preview, um_prev, levels=(0, 255), unit="DN")
        ov = None
        if self.chk_focus.isChecked() and s.in_focus is not None:
            r, g, b = hex_rgb(self.t["overlay_focus"])
            m = s.in_focus.astype(np.uint8)
            edge = cv2.morphologyEx(cv2.medianBlur(m * 255, 5), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
            ov = np.zeros(s.in_focus.shape + (4,), np.uint8)
            ov[edge] = (r, g, b, 200)
        if self.chk_zebra.isChecked():
            clip = s.preview >= 250  # saturated: no focus contrast there, and heights there are unreliable
            if clip.any():
                hh, ww = clip.shape
                yy, xx = np.ogrid[:hh, :ww]
                stripes = clip & (((xx + yy + (s.seq % 8)) // 4) % 2 == 0)
                ov = np.zeros(clip.shape + (4,), np.uint8) if ov is None else ov
                ov[stripes] = (255, 40, 200, 230)
        self.p_live.set_overlay(ov)
        if s.coverage is not None:
            self.coverage.set_image(np.log1p(s.coverage), um_prev)
        self._update_guide(s)
        kinds = {State.DWELL: "success", State.PAN: "info", State.Z_SWEEP: "warning", State.OBJECTIVE_CHANGE: "danger"}
        label = s.state.value + (f"  {s.sweep_frames} frames" if s.state == State.Z_SWEEP else "")
        self.chip_state.set(label, kinds[s.state])
        self.chip_fps.set(f"{s.fps_in:.1f} fps in · {s.fps_proc:.1f} processed", "neutral")
        self.chip_sat.setVisible(s.saturation > 0.01)
        item = self.source.latest() if self.source is not None else None
        stale = item is not None and time.perf_counter() - item[1] > 2.0 and not getattr(self.source, "done", False)
        if stale or getattr(self.source, "error", None):
            self.chip_src.set("NO FRAMES", "danger", getattr(self.source, "error", None) or "No new frames for 2 s")
            self.guide.setText("<b>The camera stopped sending frames.</b> Check the USB cable and the lamp; "
                               "MicroTopos reconnects automatically.")
            self._stale = True
        elif getattr(self, "_stale", False):
            self._stale = False
            self._refresh_checklist()
        rec = getattr(self.source, "recording", False)
        self.chip_rec.setVisible(rec)
        if rec:
            secs = int(time.perf_counter() - self.source.rec_t0)
            self.chip_rec.set(f"● REC {secs // 60}:{secs % 60:02d}", "danger")
        self.lbl_live.setText((f"● REC {len(self.source.rec_stamps)} frames · " if rec else "") +
                              f"{s.frame_shape[1]}×{s.frame_shape[0]} capture · preview {um_prev:.3f} µm/px · "
                              f"{'averaging ' + str(min(s.dwell_frames, 8)) + ' frames' if s.state == State.DWELL else ''}")
        self.lbl_motion.setText(f"shift {s.shift_px:5.2f} px · corr {s.response:4.2f} · sharpness z {s.z_score:+5.1f}"
                                + ("\nMosaic tracking lost: move back over a mapped area" if s.mosaic_lost else ""))
        if s.brenner_hist:
            self.spark_curve.setData(np.asarray(s.brenner_hist, float))
        cp = self.engine.calibration_progress()
        if cp:
            self.lbl_job.setText(f"Recording {cp[0]}: {cp[1]}/{cp[2]} frames")
        elif getattr(self, "_cal_was", None):
            self.notify(f"{self._cal_was[0].capitalize()} frame recorded and saved.", "success")
            self._refresh_checklist()
        self._cal_was = cp
        if self.tabs.currentIndex() == 1:
            self._draw_camera_box(s, um_prev)
        if s.mosaic_version != self.last_mosaic_version and self.tabs.currentIndex() == 1:
            self.last_mosaic_version = s.mosaic_version
            (img, origin), hm, nkf, npatch = self.engine.mosaic_image()
            if nkf <= 1 and self.loaded_mosaic is not None:  # nothing mapped yet: show the opened session's mosaic
                (img, origin), hm = self.loaded_mosaic
            self.current_hmosaic = hm
            mode = self.cmb_mview.currentIndex()
            if mode == 2 and hm is not None:
                self.p_mosaic.set_image(hm[0], um_prev, unit=self._z_unit_label())
                self.p_mosaic.set_overlay(None)
                self.mosaic_origin = hm[1]
            else:
                # display copy: at most 1600 px wide, 8-bit RGBA (empty = transparent); cheap to repaint
                f = max(1.0, img.shape[1] / 1600.0)
                small = cv2.resize(np.nan_to_num(img, nan=0.0), (int(img.shape[1] / f), int(img.shape[0] / f)),
                                   interpolation=cv2.INTER_AREA) if f > 1 else np.nan_to_num(img, nan=0.0)
                valid = cv2.resize(np.isfinite(img).astype(np.uint8), small.shape[::-1], interpolation=cv2.INTER_NEAREST)
                g = np.clip(small, 0, 255).astype(np.uint8)
                rgba = np.dstack([g, g, g, valid * 255])
                self.p_mosaic.set_image(rgba, um_prev * img.shape[1] / small.shape[1], rgb=True)
                self.p_mosaic.data_full = img
                self.mosaic_origin = origin
                self._draw_height_overlay(hm if mode == 1 else None, origin, um_prev)
            self.lbl_mosaic.setText(f"{nkf} keyframes · {npatch} height patches · "
                                    f"{img.shape[1] * um_prev:.0f} × {img.shape[0] * um_prev:.0f} µm")
        if s.seq % 20 == 0:
            self._refresh_checklist()
            if self.diag.isVisible() and self.result is None:
                self.diag.setPlainText("\n".join(list(self.engine.events)[-40:]))

    def _draw_height_overlay(self, hm, origin, um):
        """Colour-mapped stitched height mosaic over the image mosaic, at its true position."""
        if hm is None or not np.isfinite(hm[0]).any():
            self.p_mosaic.set_overlay(None)
            self.lbl_hlegend.setText("")
            return
        z, (hx, hy) = hm
        lo, hi = np.nanpercentile(z, [2, 98])
        norm = np.clip((np.nan_to_num(z, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)
        rgba = pg.colormap.get("viridis").map(norm.ravel(), mode="byte").reshape(z.shape + (4,)).copy()
        rgba[..., 3] = np.where(np.isfinite(z), int(255 * self.sl_hop.value() / 100), 0)
        from PySide6.QtCore import QRectF
        rect = QRectF((hx - origin[0]) * um, (hy - origin[1]) * um, z.shape[1] * um, z.shape[0] * um)
        self.p_mosaic.set_overlay(rgba, rect)
        self.lbl_hlegend.setText(f"height {lo:.3g}–{hi:.3g} {self._z_unit_label()} (viridis)")

    def _export_height_mosaic(self):
        hm = getattr(self, "current_hmosaic", None)
        if hm is None:
            self.notify("No height maps in the mosaic yet.", "warning")
            return
        p = self._save_path("Export stitched height mosaic", "height_mosaic.gsf", "GSF (*.gsf)")
        if p:
            export.write_gwyddion_gsf(p, hm[0], self.um_preview, self._z_unit_label() == "µm", "MicroTopos height mosaic")
            self.notify(f"Saved {p}", "success")

    def _draw_camera_box(self, s, um):
        """Live outline of the camera's field of view on the mosaic (red when tracking is lost)."""
        if self.p_mosaic.data is None or not s.preview_shape[0]:
            self.cam_box.hide()
            return
        ph, pw = s.preview_shape
        x = (s.mosaic_xy[0] - self.mosaic_origin[0]) * um
        y = (s.mosaic_xy[1] - self.mosaic_origin[1]) * um
        key = (round(x / um), round(y / um), s.mosaic_lost)
        if key == getattr(self, "_box_key", None):
            return
        self._box_key = key
        self.cam_box.setRect(x, y, pw * um, ph * um)
        pen = pg.mkPen(self.t["danger"] if s.mosaic_lost else self.t["overlay_line"], width=2)
        pen.setCosmetic(True)
        self.cam_box.setPen(pen)
        self.cam_box.show()

    def _update_guide(self, s):
        """One-line coach above the image: what is happening and what to do next."""
        busy = self.job is not None and self.job.isRunning()
        if s.state == State.Z_SWEEP:
            pct = s.sweep_completeness
            txt = (f"<b>Focus sweep in progress</b> — keep turning slowly <i>past</i> sharp until the image is blurry "
                   f"again, then stop.  {s.sweep_frames} frames · {pct:.0%} of the field has gone sharp and blurry again"
                   + (" (enough)" if pct >= 0.5 else " (needs 50%)"))
        elif s.state == State.OBJECTIVE_CHANGE:
            txt = "<b>The view changed abruptly</b> — if you turned the nosepiece, pick the objective in the banner."
        elif s.state == State.PAN and s.shift_px > 0.18 * self.engine.sm.analysis_width and not s.mosaic_lost:
            txt = ("<b>Move slower</b> — the view moves more than a fifth of its width per frame; the mosaic may lose "
                   "track and the frames are blurred.")
        elif s.state == State.PAN:
            txt = ("<b>Mapping</b> — move slowly and pause briefly now and then: sharp, steady views become mosaic "
                   "tiles. The blue box on the Mosaic tab is the camera."
                   if not s.mosaic_lost else "<b>Mosaic tracking lost</b> — move back over an area already in the mosaic.")
        elif busy:
            txt = "<b>Building the height map…</b> You can keep working; it opens in the Height tab when ready."
        elif self.last_ignored is not None:
            txt = (f"<b>That focus movement was not used</b> (only {self.last_ignored:.0%} of the field went sharp and "
                   "blurry again). For a height map: start blurry, turn slowly through sharp, stop when blurry again.")
        elif self.result is None:
            txt = ("<b>Next: a height map.</b> Keep the stage still. Turn the fine focus until the image is blurry, then "
                   "turn slowly the other way — through sharp — until it is blurry again, and stop.  "
                   "<b>Or a mosaic:</b> focus, then move the X-Y stage slowly.")
        else:
            txt = (f"<b>Height map ready</b> ({self.result['valid_fraction']:.0%} reliable) — see the Height, 3D and "
                   "Report tabs. Sweep again anywhere for another; calibrate X-Y and Z (steps 4–5) for real µm.")
        if txt != self.guide.text():
            self.guide.setText(txt)

    # ================================================================== reconstruction
    def _z_unit_label(self):
        return (self.result or {}).get("z_unit", "frames")

    def _on_sweep(self, cap: SweepCapture):
        self.last_cap = cap
        self.btn_lastsweep.setEnabled(True)
        self.btn_lastsweep.setText(f"Reconstruct last sweep ({len(cap.frames)} frames)")
        if not self.chk_autorec.isChecked():
            self.notify(f"Focus sweep kept ({len(cap.frames)} frames, {cap.completeness:.0%} of the field through "
                        "focus). Press 'Reconstruct last sweep' to build its height map.")
            return
        self._reconstruct(cap)

    def _reconstruct_last(self):
        cap = getattr(self, "last_cap", None)
        if cap is not None:
            self._reconstruct(cap, force=True)

    def _reconstruct(self, cap: SweepCapture, force: bool = False):
        if cap.completeness < 0.5 and self.engine.sm.override != Override.Z and not force:
            # a quick refocus, or refocusing right after a successful sweep: not a failed attempt, stay quiet
            if len(cap.frames) < 45 or time.perf_counter() - self._result_t < 15.0:
                return
            self.last_ignored = cap.completeness
            self.notify(f"Focus movement ignored: only {cap.completeness:.0%} of the field passed through focus. "
                        "For a height map, sweep from clearly below to clearly above the surface.", "warning")
            return
        if self.job is not None and self.job.isRunning():
            self.notify("Still reconstructing the previous sweep; this one was skipped.", "warning")
            return
        obj = OBJECTIVES[cap.objective]
        cal = self.store.get(obj)
        w = cap.shape[1]
        um = obj.nominal_um_per_px(w) if cap.simulated else cal.um_per_px(w)
        if cap.z is not None and cap.simulated:
            z, unit, zstat, zcal = cap.z, "µm", CalStatus.SIMULATED.value, None
        elif cal.z_um_per_frame:
            z, unit, zstat, zcal = None, "µm", CalStatus.CALIBRATED.value, cal.z_um_per_frame
        else:
            z, unit, zstat, zcal = None, "frames", CalStatus.NOMINAL.value, None
        st = zstack.ZSettings(measure=self.cmb_measure.currentText(), form_order=self.sp_form.value(),
                              lam_s_um=self.sp_ls.value() or None, lam_c_um=self.sp_lc.value(),
                              workers=max(1, (os.cpu_count() or 2) - 1))
        meta = {"z_status": zstat, "lateral_status": cal.lateral_status.value if not cap.simulated else "SIMULATED",
                "pose": cap.pose, "capture_shape": cap.shape}
        self.job = Job(zstack.reconstruct, cap.frames, z, obj, um, unit, zcal, st)
        self.job.progress.connect(self._on_progress)
        self.job.done.connect(lambda r: self._on_result({**r, **meta}))
        self.job.failed.connect(self._on_failed)
        self.btn_cancel.setEnabled(True)
        self.job.start()
        self.notify(f"Focus sweep captured ({len(cap.frames)} frames). Reconstructing…")

    def _on_progress(self, f, msg):
        self.prog.setValue(int(f * 1000))
        self.lbl_job.setText(msg)

    def _on_failed(self, msg):
        self.btn_cancel.setEnabled(False)
        self.prog.setValue(0)
        self.lbl_job.setText(msg)
        self.notify(msg, "warning")

    def _cancel(self):
        if self.job is not None:
            self.job.cancel = True

    def _on_result(self, r: dict):
        self.btn_cancel.setEnabled(False)
        if self.repeat_armed and self.result is not None:
            self.repeat_armed = False
            a, b = self.result, r
            same = (a["objective"] == b["objective"] and a["height"].shape == b["height"].shape
                    and math.hypot(a["pose"][0] - b["pose"][0], a["pose"][1] - b["pose"][1]) <= 2.0)
            if same:
                sig = metrology.repeatability(a["height"], b["height"])
                r["sigma_inst"] = sig
                if sig is not None and r.get("Sq"):
                    r["sigma_inst_lt_Sq_over_10"] = bool(sig < r["Sq"] / 10)
            else:
                r["warnings"] = r.get("warnings", []) + ["Repeatability not computed: the stage moved, the objective "
                                                         "changed or the size differs between the two sweeps."]
        self.prev_result, self.result = self.result, r
        self._result_t = time.perf_counter()
        self._remember(r)
        self.last_ignored = None
        # pin into the mosaic at preview resolution
        s = self.engine.snapshot()
        if s.preview is not None:
            ph, pw = s.preview.shape
            edf_p = cv2.resize(r["edf"], (pw, ph), interpolation=cv2.INTER_AREA)
            h_p = cv2.resize(r["height"], (pw, ph), interpolation=cv2.INTER_NEAREST)
            self.engine.add_height_patch(r["pose"], edf_p, h_p, r["z_unit"])
            self.last_mosaic_version = -1
        self._show_result()  # stay on the current tab: the user may be mapping; the toast says where to look
        w = r.get("warnings") or []
        self.notify(f"Height map ready (Height tab): {r['valid_fraction']:.0%} of pixels reliable, "
                    f"{r['timings']['total']:.1f} s." + (f"  {len(w)} warning(s) in Report." if w else ""),
                    "warning" if w else "success")

    HISTORY = 10

    def _remember(self, r):
        self._n_results = getattr(self, "_n_results", 0) + 1
        r.setdefault("label", f"#{self._n_results}  {str(r.get('timestamp', ''))[11:19]}  "
                              f"{r['objective'].split(' /')[0]} · {r['valid_fraction']:.0%} reliable")
        self.history.insert(0, r)
        del self.history[self.HISTORY:]
        self.cmb_hist.blockSignals(True)
        self.cmb_hist.clear()
        self.cmb_hist.addItems([h["label"] for h in self.history])
        self.cmb_hist.setCurrentIndex(0)
        self.cmb_hist.blockSignals(False)

    def _pick_history(self, i):
        if 0 <= i < len(self.history) and self.history[i] is not self.result:
            self.result = self.history[i]
            self._show_result()

    def _recompute(self):
        r = self.result
        if r is None:
            return
        m = metrology.analyze(r["height"], r["edf_proc"], r["um_per_px"], r["z_unit"],
                              OBJECTIVES[r["objective"]].lateral_cutoff_um, self.sp_form.value(),
                              self.sp_ls.value() or None, self.sp_lc.value())
        r["warnings"] = [x for x in r.get("warnings", []) if "reliable depth" not in x] + m.pop("warnings")
        for k in ("Sa", "Sq", "Sp", "Sv", "Sz", "Ssk", "Sku", "Sdr_pct", "PSD_beta", "fractal_dimension", "Sk", "Spk",
                  "Svk", "Smr1_pct", "Smr2_pct", "Sal_um", "Str", "Std_deg"):
            r.pop(k, None)
        r.update(m)
        self._show_result()
        self.notify("Metrology recomputed with the new filter settings.", "success")

    def _run_sr(self):
        frames = self.engine.dwell_frames()
        if len(frames) < 8:
            self.notify("Hold still for a moment (needs 8 still frames).", "warning")
            return
        h, w = frames[0].shape
        c = min(512, h, w)
        y0, x0 = (h - c) // 2, (w - c) // 2
        crops = [f[y0:y0 + c, x0:x0 + c].astype(np.float32) for f in frames]
        cal = self.store.get(self.obj)
        um = cal.um_per_px(w)
        self.sr_job = Job(_sr_job, crops, self.obj.psf_sigma_px(um))
        self.sr_job.progress.connect(self._on_progress)
        self.sr_job.done.connect(lambda r: self._on_sr({**r, "um_per_px": um / 2, "input": crops[0]}))
        self.sr_job.failed.connect(self._on_failed)
        self.sr_job.start()

    def _on_sr(self, r):
        self.sr_result = r
        self.prog.setValue(1000)
        self.lbl_job.setText(f"Super-resolution: {len(r['shifts'])} frames, RL {r['rl_iterations']} iterations, "
                             f"PSF σ {r['psf_sigma_lr']:.2f} px ({r['psf_source']})")
        self.cmb_edf.setCurrentIndex(1)
        self.tabs.setCurrentIndex(2)
        self._show_result()

    # ================================================================== result views
    def _show_result(self):
        r = self.result
        if self.cmb_edf.currentIndex() == 1 and self.sr_result is not None:
            sr = self.sr_result
            self.p_edf.set_image(np.clip(sr["image"], 0, 255), sr["um_per_px"], levels=(0, 255), unit="DN", autorange=True)
            self.p_edf.set_overlay(None)
            self.lbl_edf.setText("Experimental: 2× drizzle + Richardson-Lucy of the centre crop")
        elif r is not None:
            um_cap = r["um_per_px_capture"]
            self.p_edf.set_image(r["edf"], um_cap, levels=(0, 255), unit="DN")
            if self.chk_grains.isChecked() and "watershed_labels" in r:
                lab = r["watershed_labels"]
                edges = (np.diff(lab, axis=0, prepend=lab[:1]) != 0) | (np.diff(lab, axis=1, prepend=lab[:, :1]) != 0)
                edges = cv2.resize(edges.astype(np.uint8), (r["edf"].shape[1], r["edf"].shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                ov = np.zeros(edges.shape + (4,), np.uint8)
                ov[edges] = (*hex_rgb(self.t["overlay_focus"]), 200)
                self.p_edf.set_overlay(ov)
            else:
                self.p_edf.set_overlay(None)
            self.lbl_edf.setText(f"{r['frames_used']} of {r['frames_in']} frames fused · {r.get('grain_count', 0)} grains")
        if r is None:
            self._render_report()
            return
        mode = self.cmb_height.currentIndex()
        um = r["um_per_px"]
        if mode == 0:
            data, unit = r["sl_surface"], r["z_unit"]
        elif mode == 1:
            data, unit = r["height"], r["z_unit"]
        else:
            data, unit = r["confidence"], ""
        self.p_height.bar.setColorMap(pg.colormap.get("cividis" if mode == 2 else "viridis"))
        self.p_height.set_image(data, um, levels=(0, 1) if mode == 2 else None, unit=unit)
        self.p_height.extra = r["confidence"]
        if not self.prof_roi.isVisible():
            h, w = data.shape
            self.prof_roi.setPos(0, 0)
            hs = self.prof_roi.getHandles()
            hs[0].setPos(w * um * 0.1, h * um * 0.5)
            hs[1].setPos(w * um * 0.9, h * um * 0.5)
            self.prof_roi.show()
        zs = r["z_status"]
        self.lbl_height.setText(f"{r['valid_fraction']:.0%} reliable · {um:.3f} µm/px · z in {r['z_unit']} ({zs})")
        self._update_profile()
        self._update_3d()
        self._render_report()

    def _update_profile(self):
        r = self.result
        if r is None:
            return
        mode = self.cmb_height.currentIndex()
        data = r["sl_surface"] if mode == 0 else (r["height"] if mode == 1 else r["confidence"])
        um = r["um_per_px"]
        h0, h1 = (self.prof_roi.mapToParent(hd.pos()) for hd in self.prof_roi.getHandles())
        x0, y0, x1, y1 = h0.x() / um, h0.y() / um, h1.x() / um, h1.y() / um
        n = int(max(2, math.hypot(x1 - x0, y1 - y0)))
        xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
        ok = (xs >= 0) & (xs < data.shape[1] - 1) & (ys >= 0) & (ys < data.shape[0] - 1)
        prof = np.full(n, np.nan, np.float32)
        if not ok.any():
            self.prof_curve.setData([], [])
            return
        prof[ok] = cv2.remap(data.astype(np.float32), xs[ok].astype(np.float32).reshape(1, -1),
                             ys[ok].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).ravel()
        dist = np.hypot(xs - x0, ys - y0) * um
        self.prof_curve.setData(dist, prof, connect="finite")
        self.prof_plot.setLabel("bottom", "distance", units="µm")
        self.prof_plot.getAxis("bottom").enableAutoSIPrefix(False)
        v = prof[np.isfinite(prof)]
        if v.size > 4 and mode != 2:
            v = v - v.mean()
            self.prof_label.setText(f"Profile over {dist[-1]:.1f} µm:  Ra {np.mean(np.abs(v)):.4g} {r['z_unit']}  ·  "
                                    f"Rq {np.sqrt(np.mean(v * v)):.4g}  ·  Rz(max-min) {np.ptp(v):.4g}  "
                                    "(no profile filtering; for orientation only)")

    def _update_3d(self):
        r = self.result
        if r is None:
            return
        z = r["sl_surface"]
        um = r["um_per_px"]
        step = max(1, int(max(z.shape) / 200))
        zs = z[::step, ::step]
        finite = np.isfinite(zs)
        if not finite.any():
            return
        fill = float(np.nanmedian(zs))
        zf = np.where(finite, zs, fill)
        lo, hi = np.percentile(zf[finite], [1, 99])
        cm = pg.colormap.get(self.cmb_cmap.currentText())
        norm = np.clip((zf - lo) / max(hi - lo, 1e-9), 0, 1)
        if self.gl is not None:
            exag = self.sl_exag.value() / 10.0
            zscale = exag * (1.0 if r["z_unit"] == "µm" else um)
            cols = cm.map(norm.ravel(), mode="float").reshape(zs.shape + (4,))
            cols[~finite, 3] = 0.15
            xs = np.arange(zs.shape[1]) * um * step
            ys = np.arange(zs.shape[0]) * um * step
            self.gl_surf.setData(x=xs - xs.mean(), y=ys - ys.mean(), z=(zf.T - fill) * zscale,
                                 colors=np.transpose(cols, (1, 0, 2)))
            self.gl_surf.setVisible(True)
            span = max(xs.max(), ys.max())
            self.gl_grid.setSize(span * 1.2, span * 1.2)
            self.gl_grid.setSpacing(span / 10, span / 10)
            self.gl.setCameraPosition(distance=span * 1.4)
        else:
            gy, gx = np.gradient(zf * (self.sl_exag.value() / 10.0))
            shade = np.clip(0.6 + 0.4 * (-gx - gy) / (np.hypot(gx, gy) + 1e-3 * (hi - lo) + 1e-12), 0, 1)
            rgb = (cm.map(norm.ravel(), mode="float").reshape(zs.shape + (4,))[..., :3] * shade[..., None] * 255)
            rgb[~finite] = hex_rgb(self.t["nan"])
            self.relief.set_image(rgb.astype(np.uint8), um * step, rgb=True)

    def _render_report(self):
        t = self.t
        r = self.result
        css = (f"body{{color:{t['text']};font-size:13px}} h2,h3{{margin:6px 0}} table{{border-collapse:collapse}}"
               f" td,th{{padding:4px 10px;border-bottom:1px solid {t['border']}}} th{{text-align:left;color:{t['muted']}}}"
               f" .w{{color:{t['warning']}}} .m{{color:{t['muted']}}}")
        if r is None:
            self.report.setHtml(f"<style>{css}</style><h2>Report</h2><p class='m'>No measurement yet. Turn the fine "
                                f"focus through the sample to create a height map.</p>{OPTICS_NOTES}")
            return
        zu = r["z_unit"]

        def row(name, key, unit, desc, fmt="{:.4g}"):
            v = r.get(key)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return f"<tr><td>{name}</td><td class='m'>—</td><td>{unit}</td><td class='m'>{desc}</td></tr>"
            return f"<tr><td>{name}</td><td><b>{fmt.format(v)}</b></td><td>{unit}</td><td class='m'>{desc}</td></tr>"

        warn = "".join(f"<p class='w'>⚠ {html.escape(w)}</p>" for w in r.get("warnings", []))
        if r.get("z_status") == CalStatus.SIMULATED.value:
            warn += "<p class='w'>⚠ Simulated specimen: heights are simulation truth units, not a measurement.</p>"
        elif zu == "frames":
            warn += "<p class='w'>⚠ Z is not calibrated: height parameters are in focus-frame units.</p>"
        if r.get("lateral_status") == "NOMINAL":
            warn += "<p class='w'>⚠ Lateral scale is nominal (uncalibrated): µm lengths and Sdr are indicative.</p>"
        rows = [
            ("Sa", "Sa", zu, "Arithmetic mean height"), ("Sq", "Sq", zu, "Root-mean-square height"),
            ("Sp", "Sp", zu, "Maximum peak height"), ("Sv", "Sv", zu, "Maximum pit depth"),
            ("Sz", "Sz", zu, "Maximum height (Sp + Sv)"), ("Ssk", "Ssk", "", "Skewness"), ("Sku", "Sku", "", "Kurtosis"),
            ("Sdr", "Sdr_pct", "%", "Developed interfacial area ratio (needs z in µm)"),
            ("Sk", "Sk", zu, "Core height (40% secant of the material ratio curve)"),
            ("Spk", "Spk", zu, "Reduced peak height"), ("Svk", "Svk", zu, "Reduced valley depth"),
            ("Smr1", "Smr1_pct", "%", "Material ratio at the top of the core"),
            ("Smr2", "Smr2_pct", "%", "Material ratio at the bottom of the core"),
            ("Sal", "Sal_um", "µm", "Fastest autocorrelation decay length (to 0.2)"),
            ("Str", "Str", "", "Texture aspect ratio: ~1 isotropic, ~0 strongly directional"),
            ("Std", "Std_deg", "°", "Texture direction (lay), from the angular spectrum"),
            ("σ_inst", "sigma_inst", zu, "Repeatability from two sweeps, std(Δz)/√2"),
            ("PSD slope β", "PSD_beta", "", "Log-log slope of the radial power spectrum"),
            ("Fractal dimension", "fractal_dimension", "", "(8−β)/2, only when 2 < β < 4"),
            ("GLCM contrast", "GLCM_contrast", "", "Haralick texture, 64 grey levels"),
            ("GLCM homogeneity", "GLCM_homogeneity", "", ""), ("GLCM correlation", "GLCM_correlation", "", ""),
            ("Grains", "grain_count", "", "Watershed regions not touching the border", "{:d}"),
            ("Mean grain area", "mean_grain_area_um2", "µm²", ""), ("Mean max Feret", "mean_max_feret_um", "µm", ""),
            ("Mean circularity", "mean_circularity", "", "4πA/P²"),
            ("Mean lineal intercept", "mean_intercept_um", "µm", "Heyn intercept on the grain boundaries"),
            ("ASTM E112 G", "ASTM_E112_G", "", "Grain size number from the mean intercept (calibrated X-Y needed)"),
        ]
        body = "".join(row(*x) for x in rows)
        prov = [("Objective", r["objective"]), ("Frames used", f"{r['frames_used']} of {r['frames_in']}"),
                ("Valid pixels", f"{r['valid_fraction']:.1%}"), ("Pixel size", f"{r['um_per_px']:.4f} µm"),
                ("λs / λc", f"{r['lambda_s_um']:.3g} µm / {r['lambda_c_um'] or 'off'}"),
                ("Form removal order", r["form_order"]), ("Focus measure", r["measure"]),
                ("Z status", r.get("z_status")), ("Lateral status", r.get("lateral_status")),
                ("Registration", f"ECC mean {r['ecc_mean']:.3f}, fallbacks {r['registration_fallbacks']}"
                 if r.get("ecc_mean") is not None else "—"),
                ("DOF ruler (experimental)", f"{r['dof_ruler_um_per_frame']:.4f} µm/frame" if r.get("dof_ruler_um_per_frame") else "—"),
                ("Time", f"{r['timings']['total']:.1f} s")]
        pv = "".join(f"<tr><td>{k}</td><td>{html.escape(str(v))}</td></tr>" for k, v in prov)
        self.report.setHtml(f"<style>{css}</style><h2>Surface report</h2>{warn}"
                            f"<p class='m'>ISO 25178-style parameters (not a certified implementation): form removal, "
                            f"Gaussian S-filter, optional L-filter, valid pixels only.</p>"
                            f"<table><tr><th>Parameter</th><th>Value</th><th>Unit</th><th>Meaning</th></tr>{body}</table>"
                            f"<h3>Provenance</h3><table>{pv}</table>{OPTICS_NOTES}")
        self.diag.setPlainText("stage timings (s)\n" + "\n".join(f"  {k}: {v}" for k, v in r["timings"].items())
                               + f"\nframes used: {r['frames_used']} of {r['frames_in']}"
                               + f"\nreversal at: {r.get('reversal_at')}\nfocus window: {r.get('focus_window_px')} px")

    # ================================================================== export / session
    def _need_result(self) -> bool:
        if self.result is None:
            self.notify("Nothing to export yet.", "warning")
            return False
        return True

    def _save_path(self, title, name, filt):
        d = self.settings.value("export_dir", str(Path.home()))
        p, _ = QFileDialog.getSaveFileName(self, title, str(Path(d) / name), filt)
        if p:
            self.settings.setValue("export_dir", str(Path(p).parent))
        return p

    def _export_gsf(self):
        if self._need_result() and (p := self._save_path("Export Gwyddion field", "height.gsf", "GSF (*.gsf)")):
            r = self.result
            export.write_gwyddion_gsf(p, r["sl_surface"], r["um_per_px"], r["z_unit"] == "µm",
                                      f"MicroTopos {r['objective']} {r.get('z_status')}")
            self.notify(f"Saved {p}", "success")

    def _export_tiff(self):
        if self._need_result() and (p := self._save_path("Export float TIFF", "height.tif", "TIFF (*.tif)")):
            export.write_float32_tiff(p, self.result["sl_surface"], self.result["um_per_px"])
            self.notify(f"Saved {p}", "success")

    def _export_stl(self):
        if self._need_result() and (p := self._save_path("Export STL", "surface.stl", "STL (*.stl)")):
            r = self.result
            z = r["sl_surface"] if r["z_unit"] == "µm" else r["sl_surface"] * r["um_per_px"]
            n, _ = export.write_binary_stl(p, z, r["um_per_px"], max_grid_dim=400)
            self.notify(f"Saved {p} ({n} triangles)" + ("" if r["z_unit"] == "µm" else "; z in frames × pixel size"),
                        "success")

    def _export_report(self):
        if not self._need_result():
            return
        d = QFileDialog.getExistingDirectory(self, "Folder for the report bundle",
                                             self.settings.value("export_dir", str(Path.home())))
        if d:
            import scipy
            import skimage
            files = export.write_report_bundle(d, self.result, {"microtopos": __version__, "numpy": np.__version__,
                                                                "opencv": cv2.__version__, "scipy": scipy.__version__,
                                                                "scikit-image": skimage.__version__})
            self.notify(f"Saved {len(files)} files to {d}", "success")

    def _save_session(self):
        if self._need_result() and (p := self._save_path("Save session", "session.npz", "Session (*.npz)")):
            (img, origin), hm, _, _ = self.engine.mosaic_image()
            extra = {"mosaic_image": img, "mosaic_origin": list(origin), "um_preview": getattr(self, "um_preview", None)}
            if hm is not None:
                extra.update(height_mosaic=hm[0], height_mosaic_origin=list(hm[1]))
            export.save_session(p, {**self.result, **extra},
                                {"objective": self.obj.name, "measure": self.cmb_measure.currentText()})
            self.notify(f"Saved {p}", "success")

    def _load_session(self):
        d = self.settings.value("export_dir", str(Path.home()))
        p, _ = QFileDialog.getOpenFileName(self, "Open session", d, "Session (*.npz)")
        if not p:
            return
        try:
            r, s = export.load_session(p)
            if "edf_proc" not in r:
                r["edf_proc"] = cv2.resize(r["edf"], (r["height"].shape[1], r["height"].shape[0]),
                                           interpolation=cv2.INTER_AREA)
            m = metrology.analyze(r["height"], r["edf_proc"], r["um_per_px"], r["z_unit"],
                                  OBJECTIVES[r["objective"]].lateral_cutoff_um, r.get("form_order", 1),
                                  r.get("lambda_s_um"), r.get("lambda_c_um", 0.0))
            r["warnings"] = m.pop("warnings")
            r.update(m)
            self.result = r
            self._remember(r)
            if isinstance(r.get("mosaic_image"), np.ndarray):
                hm = (r["height_mosaic"], tuple(r["height_mosaic_origin"])) if isinstance(
                    r.get("height_mosaic"), np.ndarray) else None
                self.loaded_mosaic = ((r["mosaic_image"], tuple(r["mosaic_origin"])), hm)
                self.last_mosaic_version = -1
            self._show_result()
            self.tabs.setCurrentIndex(3)
            self.notify(f"Opened {p}", "success")
        except Exception as e:
            self.notify(f"Could not open session: {e}", "danger")

    # ================================================================== shutdown
    def closeEvent(self, ev):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("state", self.saveState())
        self.timer.stop()
        if isinstance(self.source, CameraSource) and getattr(self.source, "recording", False):
            self.source.stop_recording()
        self.engine.stop()
        if self.source is not None:
            self.source.stop()
        super().closeEvent(ev)
