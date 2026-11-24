"""MicroTopos main window: guided session checklist, workspace tabs, telemetry, workers and exports."""

from __future__ import annotations

import html
import math
import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QSettings, QStandardPaths, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .. import __version__, export, metrology, units, zstack
from .. import objectives as objlib
from ..calibration import CalibrationStore, CalStatus, radiometric_key, today
from ..camera import CameraSource, ReplaySource, SimulatorSource, list_cameras, opencv_threads
from ..focus import MEASURES
from ..kinematics import Override, State, guess_objective
from ..live import LiveEngine, SweepCapture
from ..objectives import resolve, resolve_result
from ..stage import GrblStageDriver, SimulatedStageDriver, StageDriver, capture_z_sweep, list_serial_ports
from ..superres import dwell_superres
from .objectives_dialog import ObjectivesDialog
from .theme import DARK, LIGHT, hex_rgb, stylesheet
from .widgets import DARK_P99_DN, Banner, Chip, FlowWidget, ImagePanel, Toast, frame_percentile, section, stretch_levels

OPTICS_NOTES = """
<h3>What software can and cannot fix</h3>
<p><b>Lateral scale.</b> The 3.0 µm pixel pitch and "no optics in the camera" are unverified assumptions, and old
microscopes differ in tube length (160 mm, 170 mm or infinity). Until you calibrate with a stage micrometer, all
lateral lengths are <i>nominal</i>.</p>
<p><b>Colour sensor and MJPEG.</b> A Bayer colour sensor behind 8-bit MJPEG compression resolves less than its pixel
pitch suggests. In-camera sharpening is switched off here because it distorts focus measures. Your camera delivers
about 12.5 frames per second at 1920×1080, although it advertises 25.</p>
<p><b>No Z encoder.</b> Turning the focus knob by hand gives no height scale, no direction and an uneven speed. A hand
sweep's heights are relative focus progression in camera frames, assuming an even speed while the knob moves; uneven
speed distorts them, and without a stated direction the relief may be inverted. State a sweep's knob travel and
direction (Set Z…) for signed µm for that sweep; only a measured Z drive removes the speed assumption.</p>
<p><b>Textureless or mirror-polished surfaces.</b> Shape from focus needs local contrast. Pixels without a clear focus
peak are left blank (NaN) and excluded from roughness. Remedies: a light etch, oblique or dark-field lighting, or a
pattern projected through the field diaphragm.</p>
<p><b>Cover glass.</b> Objectives engraved 0.17 (for example many 40×/0.85 and 100×/1.25 oil) are corrected for a
0.17 mm cover glass and transmitted light. On bare opaque metal without epi-illumination they lose contrast to
spherical aberration and lack of light; an oil objective also needs oil on the specimen. Objectives engraved 0 or
∞/0 are made for uncovered specimens in reflected light; "-" means either. Deconvolution cannot replace missing
illumination.</p>
<p><b>Focus breathing</b> (a small magnification change with focus) is not corrected: it could not be measured
reliably from defocused frames. Its effect is limited to the far-defocused frames at the image edges.</p>
"""

STEPS = ["Camera", "Objective", "Dark && flat", "Lateral scale", "Z scale", "Inspect"]


class Bridge(QObject):
    sweep = Signal(object)
    objective_hint = Signal(object)


class Job(QThread):
    """A worker with one owner (the window), a cancellation flag the work polls, and a label for its progress."""
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, label: str = "Job", **kw):
        super().__init__()
        self.fn, self.args, self.kw = fn, args, kw
        self.cancel = False
        self.label = label

    def run(self):
        try:
            res = self.fn(*self.args, progress=lambda f, m: self.progress.emit(f, m),
                          cancelled=lambda: self.cancel, **self.kw)
            self.done.emit(res)
        except zstack.Cancelled:
            self.failed.emit("Cancelled.")
        except Exception as e:  # surfaced to the user, never silently dropped  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")


def _sr_job(frames, psf, progress=None, cancelled=None, scale=2, pixfrac=0.7, deconvolve=True):
    if progress:
        progress(0.1, "Registering dwell frames and drizzling...")
    return dwell_superres(frames, psf, scale=scale, pixfrac=pixfrac, deconvolve=deconvolve)


def _metrology_job(r, form_order, lam_s, lam_c, progress=None, cancelled=None):
    if progress:
        progress(0.2, "Recomputing metrology...")
    return metrology.analyze(r["height"], r["edf_proc"], r["um_per_px"], r["z_unit"],
                             resolve_result(r).lateral_cutoff_um, form_order, lam_s, lam_c)


def _rescale_job(r, factor, unit, progress=None, cancelled=None):
    if progress:
        progress(0.2, "Rescaling heights and recomputing metrology...")
    return zstack.rescale_heights(r, factor, unit)


DIRECTIONS = {"Unknown": None, "Up (focal plane toward the objective)": 1, "Down (focal plane away from the objective)": -1}


def z_key(r: dict) -> str:
    """Which height maps are the same quantity and may be levelled and blended together in the mosaic."""
    unit, orient, stat = units.canon(r["z_unit"]), r.get("z_orientation"), r.get("z_status")
    if unit == units.UM and orient in ("known", "stated") and stat in ("SIMULATED", "CALIBRATED"):
        return "um"
    if unit == units.UM and orient in ("known", "stated") and stat == "ESTIMATED":
        return f"um~{r['objective']}@{r.get('calibration_revision', 0)}"  # same stored estimate, same assumption
    return f"{unit}#{r.get('result_id', '')}"  # its own scale (or unknown sign): never mixed with others


def z_status_text(r: dict) -> str:
    unit, orient = units.label(r["z_unit"]), r.get("z_orientation", "known")
    o = {"known": "", "stated": ", direction stated", "unknown": ", direction unknown"}.get(orient, "")
    return f"Z {r.get('z_status', '?')} ({unit}{o})"


class MainWindow(QMainWindow):
    def __init__(self, simulator_only: bool = False, replay: str | None = None, verbose: bool = False):
        super().__init__()
        opencv_threads()
        self.settings = QSettings("MicroTopos", "MicroTopos")
        self.dark = self.settings.value("dark", True, type=bool)
        self.t = DARK if self.dark else LIGHT
        data_dir = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation) or Path.home() / ".microtopos")
        self.data_dir = data_dir
        self.store = CalibrationStore(data_dir)
        self.objset = objlib.load_set(data_dir)  # the user's objectives; the built-in set if none was saved
        objlib.set_active(self.objset)
        self.obj = (self.objset.get(str(self.settings.value("objective", ""))) or self.objset.get("10x / 0.25")
                    or self.objset.objectives[0])
        self.engine = LiveEngine(self.store, self.obj)
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
        self.job: Job | None = None  # the reconstruction / metrology job (one at a time)
        self.sr_job: Job | None = None
        self.jobs: list[Job] = []  # every job still owned by the window (joined on close)
        self.sr_scale = 2          # super-resolution linear scale (SR control strip)
        self.sr_pixfrac = 0.7      # drizzle drop size (fraction of an input pixel)
        self.sr_deconvolve = True  # Richardson-Lucy deconvolution after the drizzle step
        self.color_mode = False          # colour (RGB) preview through the engine
        self.mosaic_rgb = False          # render the mosaic tab in colour from tiles_rgb
        self.track_rotation = False      # mosaic tracks small in-plane rotations between keyframes
        self.stage_driver: StageDriver | None = None
        self.stage_job: Job | None = None  # the running stage operation (serial I/O never blocks the GUI thread)
        self.stage_jog_buttons: list = []
        self.level_picking = False       # 3-point level mode armed on the height view
        self.level_pts: list[tuple[float, float]] = []
        self.zst = zstack.ZSettings()    # 3D reconstruction settings edited from the 3D page
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
        self.cameras = [] if replay else list_cameras()  # --simulator still lists cameras, it only starts on the sim
        self._fill_sources()
        self._set_rotation(self.sp_rot.value())
        self._apply_theme()
        self._default_state = self.saveState()  # for "Reset layout"
        # dock layout first, window geometry second: restoring the docks into a window that restoreGeometry() has
        # already marked maximised (but not yet shown) leaves stale dock-separator widgets from the unmaximised
        # size, each drawing a 1-px line across the middle panel
        st = self.settings.value("state")
        if st is not None:
            self.restoreState(st)
        g = self.settings.value("geometry")
        if g is not None:
            self.restoreGeometry(g)
        self.timer = QTimer(self, interval=66, timeout=self._tick)  # 15 Hz: above the camera's 12.5 fps
        self.timer.start()
        self._refresh_checklist()
        if self.objset.load_error:
            self.notify(self.objset.load_error, "danger")

    # ================================================================== layout
    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)
        # header: the chips wrap onto a second row on narrow windows instead of forcing a wide minimum; the buttons
        # sit in a fixed group on the right, so a chip whose text changes every frame can never push them between
        # rows (the "Light theme" button used to flicker between the first and second row)
        head = FlowWidget()
        title = QLabel("MicroTopos")
        title.setObjectName("title")
        head.addWidget(title)
        ctx = QLabel("CAPTURE")
        ctx.setObjectName("ctx")
        ctx.setToolTip("The chips on this row describe the live capture: source, motion, and the calibration of the "
                       "objective selected now. The selected result has its own line below.")
        head.addWidget(ctx)
        self.chip_state = Chip("DWELL", "success", sticky=True)
        self.chip_state.reserve("OBJECTIVE_CHANGE", "Z_SWEEP  888 frames")
        self.chip_state.setToolTip("Motion state detected from the video: DWELL (still), PAN (X-Y moving), "
                                   "Z_SWEEP (focus moving), OBJECTIVE_CHANGE")
        # every header chip reserves its widest text, so no later text (NO FRAMES, XY CAL, Z EST ...) re-wraps the row
        self.chip_xy = Chip(sticky=True)
        self.chip_xy.reserve("XY NOMINAL 88.888 µm/px")
        self.chip_z = Chip(sticky=True)
        self.chip_z.reserve("Z EST 88.888 µm/f")
        self.chip_rad = Chip(sticky=True)
        self.chip_rad.reserve("DARK/FLAT ✓", "DARK/FLAT —")
        self.chip_src = Chip(sticky=True)
        self.chip_src.reserve("SIMULATOR", "NO FRAMES", "CAMERA")
        self.chip_fps = Chip("", "neutral", sticky=True)
        self.chip_fps.reserve("88.8 fps in · 88.8 processed")
        self.chip_sat = Chip("CLIPPING", "danger", sticky=True)
        self.chip_sat.setToolTip("More than 1% of pixels are saturated: lower the exposure, specular metal clips easily")
        self.chip_rec = Chip("● REC", "danger", sticky=True)
        self.chip_rec.reserve("● REC 88:88")
        self.chip_rec.setToolTip("Raw camera frames are being saved; press Stop recording under the Live view")
        for c in (self.chip_sat, self.chip_rec):  # toggling chips keep their row reserved: the header height is stable
            sp = c.sizePolicy()
            sp.setRetainSizeWhenHidden(True)
            c.setSizePolicy(sp)
            c.hide()
        # CLIPPING toggles with every exposure change, so it comes last: showing it moves no other chip
        for w in (self.chip_rec, self.chip_state, self.chip_src, self.chip_xy, self.chip_z, self.chip_rad, self.chip_fps,
                  self.chip_sat):
            head.addWidget(w)
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.btn_left = self._btn("◧", lambda c: self.dock_left.setVisible(c), "Show or hide the session checklist")
        self.btn_right = self._btn("◨", lambda c: self.dock_right.setVisible(c), "Show or hide the telemetry panel")
        for b in (self.btn_left, self.btn_right):
            b.setCheckable(True)
            b.setChecked(True)
            b.setAccessibleName(b.toolTip())
            buttons.addWidget(b)
        buttons.addWidget(self._btn("Reset layout", self._reset_layout, "Restore the default panel layout"))
        self.btn_theme = QPushButton()
        self.btn_theme.clicked.connect(self._toggle_theme)
        # sized for the longer of its two labels, so switching theme does not resize the group
        self.btn_theme.setMinimumWidth(max(self.btn_theme.fontMetrics().horizontalAdvance(s)
                                           for s in ("Light theme", "Dark theme")) + 28)
        buttons.addWidget(self.btn_theme)
        header = QHBoxLayout()
        header.setSpacing(12)
        header.addWidget(head, 1)
        header.addLayout(buttons)
        header.setAlignment(buttons, Qt.AlignTop)
        v.addLayout(header)
        self.banner = Banner(self.t)
        v.addWidget(self.banner)
        self.guide = QLabel()
        self.guide.setObjectName("guide")
        self.guide.setWordWrap(True)
        self.guide.setTextFormat(Qt.RichText)
        v.addWidget(self.guide)
        self.last_ignored = None
        # selected result: its own provenance, independent of the live capture
        self.card = QFrame()
        self.card.setObjectName("resultcard")
        cl = QHBoxLayout(self.card)
        cl.setContentsMargins(10, 4, 6, 4)
        ctx2 = QLabel("RESULT")
        ctx2.setObjectName("ctx")
        cl.addWidget(ctx2)
        self.lbl_card = QLabel("No height map yet.")
        self.lbl_card.setWordWrap(True)
        self.lbl_card.setTextFormat(Qt.RichText)
        cl.addWidget(self.lbl_card, 1)
        self.btn_view_height = self._btn("View height", lambda: self.tabs.setCurrentIndex(3))
        self.btn_card_z = self._btn("Set Z…", self._set_result_z, "State the knob travel and direction of THIS sweep")
        cl.addWidget(self.btn_card_z)
        cl.addWidget(self.btn_view_height)
        self.card.hide()
        v.addWidget(self.card)
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
        self.p_height.clicked_px.connect(self._on_height_clicked)
        self.mask_roi = pg.RectROI([0, 0], [60, 60], pen=pg.mkPen("#ff1493", width=2, style=Qt.DashLine))
        self.mask_roi.hide()
        self.p_height.add_item(self.mask_roi)
        self.step_roi_a = pg.RectROI([0, 0], [60, 22], pen=pg.mkPen("#ffa500", width=2, style=Qt.DashLine))
        self.step_roi_a.hide()
        self.p_height.add_item(self.step_roi_a)
        self.step_roi_b = pg.RectROI([0, 0], [60, 22], pen=pg.mkPen("#ffa500", width=2, style=Qt.DashLine))
        self.step_roi_b.hide()
        self.p_height.add_item(self.step_roi_b)
        self.prof_plot = pg.PlotWidget()
        self.prof_plot.setMaximumHeight(180)
        self.prof_curve = self.prof_plot.plot()
        self.prof_label = QLabel("Drag the orange line on the height map to see a profile.")
        self.prof_label.setObjectName("muted")

        self.replay_widget = QWidget()
        self.replay_widget.setLayout(self._replay_bar())
        self.replay_widget.setVisible(False)
        self.tabs.addTab(self._wrap(self.p_live, self._live_bar(), self.replay_widget), "Live")
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
        # status labels in the tool bars wrap instead of setting the window's minimum width
        from PySide6.QtWidgets import QSizePolicy
        for lab in (self.lbl_live, self.lbl_mosaic, self.lbl_hlegend, self.lbl_edf, self.lbl_height, self.lbl_3d,
                    self.prof_label):
            lab.setWordWrap(True)
            lab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            lab.setMinimumWidth(160)
            if lab is not self.prof_label:
                lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        for cmb in (self.cmb_height, self.cmb_hist, self.cmb_edf, self.cmb_mview):
            cmb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cmb.setMinimumContentsLength(14)
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
        self.chk_stretch = QCheckBox("Stretch brightness")
        self.chk_stretch.setToolTip("Display only: spreads a dim image's own brightness range over the full grey scale "
                                    "so you can see it. The recorded frames, measurements and the cursor readout keep "
                                    "the camera's real values; a dim image still gives noisy heights, so fix the light")
        self.chk_stretch.setChecked(self.settings.value("live_stretch", False, type=bool))
        self.chk_stretch.toggled.connect(lambda on: self.settings.setValue("live_stretch", on))
        self._stretch = None
        h.addWidget(self.chk_stretch)
        self.chk_rgb = QCheckBox("Colour preview (RGB)")
        self.chk_rgb.setToolTip("Render the camera's colour frame instead of the luma channel (needs a colour camera)")
        self.chk_rgb.toggled.connect(self._set_color_mode)
        h.addWidget(self.chk_rgb)
        self.chk_hbd = QCheckBox("High bit depth")
        self.chk_hbd.setToolTip("Request 12-/14-bit output from the camera when the driver reports it")
        h.addWidget(self.chk_hbd)
        h.addWidget(QLabel("SR scale"))
        self.cmb_sr_scale = QComboBox()
        self.cmb_sr_scale.addItems(["2×", "3×", "4×"])
        self.cmb_sr_scale.setCurrentText("2×")
        self.cmb_sr_scale.setFixedWidth(52)
        self.cmb_sr_scale.currentTextChanged.connect(
            lambda t: setattr(self, "sr_scale", int(t.replace("×", ""))))
        h.addWidget(self.cmb_sr_scale)
        h.addWidget(QLabel("pixfrac"))
        self.sl_sr_pixfrac = QSlider(Qt.Horizontal)
        self.sl_sr_pixfrac.setRange(3, 10)
        self.sl_sr_pixfrac.setValue(7)
        self.sl_sr_pixfrac.setFixedWidth(70)
        self.sl_sr_pixfrac.valueChanged.connect(lambda v: setattr(self, "sr_pixfrac", v / 10))
        h.addWidget(self.sl_sr_pixfrac)
        self.chk_sr_deconv = QCheckBox("SR deconvolve")
        self.chk_sr_deconv.setChecked(True)
        self.chk_sr_deconv.toggled.connect(lambda on: setattr(self, "sr_deconvolve", on))
        h.addWidget(self.chk_sr_deconv)
        self.btn_rec = QPushButton("● Record raw stream")
        self.btn_rec.setCheckable(True)
        self.btn_rec.setFocusPolicy(Qt.TabFocus)
        self.btn_rec.setToolTip("Save the camera's exact MJPEG frames to the 'recordings' folder for offline replay "
                                "(uv run microtopos --replay FOLDER)")
        self.btn_rec.toggled.connect(self._toggle_record)
        h.addWidget(self.btn_rec)
        self.lbl_live = QLabel()
        self.lbl_live.setObjectName("muted")
        h.addWidget(self.lbl_live, 1)
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
        self.cmb_mblend = QComboBox()
        self.cmb_mblend.addItems(["feather", "multiband"])
        self.cmb_mblend.setCurrentText("feather")
        self.cmb_mblend.currentTextChanged.connect(self._mosaic_blend)
        h.addWidget(QLabel("Blend"))
        h.addWidget(self.cmb_mblend)
        self.chk_rot_track = QCheckBox("Track rotation")
        self.chk_rot_track.toggled.connect(self._mosaic_track_rotation)
        h.addWidget(self.chk_rot_track)
        self.chk_mosaic_rgb = QCheckBox("Colour mosaic")
        self.chk_mosaic_rgb.toggled.connect(self._mosaic_rgb_mode)
        h.addWidget(self.chk_mosaic_rgb)
        self.lbl_hlegend = QLabel()
        self.lbl_hlegend.setObjectName("mono")
        h.addWidget(self.lbl_hlegend, 1)
        h2 = QHBoxLayout()  # second row: actions and status, so nothing overlaps on narrow windows
        h2.addWidget(self._btn("Export height mosaic", self._export_height_mosaic,
                               "Stitched height mosaic as a Gwyddion .gsf"))
        self.btn_mosaic_inset = self._btn("Add inset (current view)", self._mosaic_add_inset,
                                          "Pin the current live view as a mosaic inset: a second objective "
                                          "mapping the same area at higher magnification")
        h2.addWidget(self.btn_mosaic_inset)
        self.chk_keep_insets = QCheckBox("Keep insets on reset")
        self.chk_keep_insets.setChecked(True)
        h2.addWidget(self.chk_keep_insets)
        h2.addWidget(self._btn("Reset mosaic  (R)", self._reset_mosaic, "Start a new mosaic"))
        self.lbl_mosaic = QLabel()
        self.lbl_mosaic.setObjectName("muted")
        h2.addWidget(self.lbl_mosaic, 1)
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
        self.lbl_edf = QLabel()
        self.lbl_edf.setObjectName("muted")
        h.addWidget(self.lbl_edf, 1)
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
        self.chk_lock = QCheckBox("Lock colour range")
        self.chk_lock.setToolTip("Keep the current colour scale when another map is shown, to compare repeat scans")
        self.chk_lock.toggled.connect(lambda c: (setattr(self, "_locked_levels", None), self._show_result()))
        h.addWidget(self.chk_lock)
        self.lbl_height = QLabel()
        self.lbl_height.setObjectName("muted")
        h.addWidget(self.lbl_height, 1)
        self.btn_level3 = QPushButton("Level: pick 3 points")
        self.btn_level3.setCheckable(True)
        self.btn_level3.toggled.connect(self._level_mode)
        h.addWidget(self.btn_level3)
        self.btn_mask = QPushButton("Exclude ROI")
        self.btn_mask.setToolTip("Apply apply_exclusion_mask to the current height inside the mask ROI")
        self.btn_mask.clicked.connect(self._apply_mask_roi)
        h.addWidget(self.btn_mask)
        self.btn_step = QPushButton("Step height A→B")
        self.btn_step.clicked.connect(self._measure_step_roi)
        h.addWidget(self.btn_step)
        self.chk_topo_grains = QCheckBox("Topographic grains")
        self.chk_topo_grains.setToolTip("Summit/count parameters from metrology.grains_topographic on the surface")
        self.chk_topo_grains.toggled.connect(self._show_result)
        h.addWidget(self.chk_topo_grains)
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
                    except Exception:  # noqa: BLE001  # any OpenGL/pyqtgraph failure degrades to relief view
                        self.failed = True

                def paintGL(self, *a, **k):
                    if not self.failed:
                        try:
                            super().paintGL(*a, **k)
                        except Exception:  # noqa: BLE001  # any OpenGL/pyqtgraph failure degrades to relief view
                            self.failed = True

            self.gl = SafeGL()
            self.gl_grid = gl.GLGridItem()
            self.gl.addItem(self.gl_grid)
            self.gl_surf = gl.GLSurfacePlotItem(shader="shaded", smooth=True, computeNormals=True)
            self.gl_surf.setVisible(False)  # an empty surface has no mesh: drawing it raises in pyqtgraph
            self.gl.addItem(self.gl_surf)
            lay.addWidget(self.gl, 1)
        except Exception:  # noqa: BLE001  # OpenGL unavailable: fall back to a shaded relief view
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
        bar.addWidget(QLabel("Backend"))
        self.cmb_3d_backend = QComboBox()
        self.cmb_3d_backend.addItems(["CPU", "OpenCL"])
        self.cmb_3d_backend.currentTextChanged.connect(lambda t: setattr(self.zst, "backend", t.lower()))
        bar.addWidget(self.cmb_3d_backend)
        self.chk_3d_breathing = QCheckBox("Correct focus breathing")
        self.chk_3d_breathing.toggled.connect(lambda on: setattr(self.zst, "correct_breathing", on))
        bar.addWidget(self.chk_3d_breathing)
        self.chk_3d_tiling = QCheckBox("Allow tiling")
        self.chk_3d_tiling.setChecked(True)
        self.chk_3d_tiling.toggled.connect(lambda on: setattr(self.zst, "allow_tiling", on))
        bar.addWidget(self.chk_3d_tiling)
        bar.addWidget(self.lbl_3d, 1)
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
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
                         | QDockWidget.DockWidgetClosable)
        self.dock_left = dock
        dock.visibilityChanged.connect(lambda vis: self.btn_left.setChecked(vis))
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
        # a dropdown sizes itself to its longest entry (a camera name, a recording path): let it shrink and elide,
        # with the full text in the tooltip, instead of widening the page past the panel
        for cmb in self.stack.findChildren(QComboBox):
            cmb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cmb.setMinimumContentsLength(14)
            cmb.currentTextChanged.connect(lambda txt, c=cmb: c.setToolTip(txt) if len(txt) > 14 else None)
        self._fit_left_dock()
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self._open_step(0)

    def _fit_left_dock(self):
        """The pages scroll vertically only, so the panel must be at least as wide as the widest page; otherwise its
        right part is cut off under the middle panel (seen on the Camera page: 369 px of content in 344 px). Run after
        building and after every theme change, since the style sheet changes fonts and padding."""
        need = max(self.stack.widget(i).widget().minimumSizeHint().width() for i in range(self.stack.count()))
        sb = self.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        m = self.dock_left.widget().layout().contentsMargins()
        self.dock_left.setMinimumWidth(max(300, need + sb + m.left() + m.right() + 4))

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
        self.cmb_res.addItems(["640x480", "1080x1080", "1280x720", "1920x1080"])
        self.cmb_res.setToolTip("Capture size. Full resolution is used for Z-stacks; live analysis uses a 640 px preview")
        self.cmb_res.currentIndexChanged.connect(lambda: self._on_source(self.cmb_source.currentIndex()))
        self.chk_autoexp = QCheckBox("Automatic exposure")

        self.sl_exp = QSlider(Qt.Horizontal)
        self.sl_exp.setMinimumWidth(60)  # the style default made this row 348 px wide
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
        # two rows: side by side, label + spin box + "Measure from X pan" (or "Finish (after an X-only move)" while
        # measuring) were wider than the panel
        rot = QGridLayout()
        rot.addWidget(QLabel("View rotation"), 0, 0)
        rot.addWidget(self.sp_rot, 0, 1)
        rot.addWidget(self.btn_rot_measure, 1, 0, 1, 2)
        rot.setColumnStretch(1, 1)
        demo = QGridLayout()
        demo.addWidget(self._btn("Pan around", lambda: self._demo("pan"), "Simulator: move the stage in a loop"), 0, 0)
        demo.addWidget(self._btn("Focus sweep", lambda: self._demo("sweep"), "Simulator: hand-like focus sweep"), 0, 1)
        demo.addWidget(self._btn("Sweep + reversal", lambda: self._demo("rev"),
                                 "Simulator: sweep that turns back through focus"), 1, 0, 1, 2)
        self.demo_box = section("Demo (simulator only)", demo,
                                hint="Synthetic specimen. Its heights are simulation truth and are labelled SIMULATED.")
        stg = QGridLayout()
        stg.addWidget(QLabel("Stage driver"), 0, 0)
        self.cmb_stg = QComboBox()
        self.cmb_stg.addItems(["Simulated stage", "GRBL serial port", "None"])
        self.cmb_stg.currentTextChanged.connect(self._stage_driver_changed)
        stg.addWidget(self.cmb_stg, 0, 1)
        stg.addWidget(QLabel("Serial port"), 1, 0)
        self.cmb_stg_port = QComboBox()
        self.cmb_stg_port.addItems(list_serial_ports())
        self.cmb_stg_port.setEnabled(False)
        stg.addWidget(self.cmb_stg_port, 1, 1)
        self.btn_stg_connect = self._btn("Connect", self._stage_connect)
        stg.addWidget(self.btn_stg_connect, 2, 0)
        self.lbl_stg_pose = QLabel("stage: disconnected")
        self.lbl_stg_pose.setObjectName("muted")
        stg.addWidget(self.lbl_stg_pose, 2, 1)
        jog = QHBoxLayout()
        for name, dx, dy in (("X −10 µm", -10.0, 0.0), ("X +10 µm", 10.0, 0.0),
                             ("Y −10 µm", 0.0, -10.0), ("Y +10 µm", 0.0, 10.0)):
            b = self._btn(name, lambda dx=dx, dy=dy: self._stage_jog(dx, dy))
            b.setEnabled(False)
            self.stage_jog_buttons.append(b)
            jog.addWidget(b)
        self.btn_stg_zero = self._btn("Zero X/Y/Z", self._stage_zero)
        self.btn_stg_zero.setEnabled(False)
        jog.addWidget(self.btn_stg_zero)
        stg.addLayout(jog, 3, 0, 1, 2)
        sweep = QHBoxLayout()
        self.sp_stg_z0, self.sp_stg_z1 = QDoubleSpinBox(), QDoubleSpinBox()
        for sp, tip in ((self.sp_stg_z0, "Z where the sweep starts (µm, stage frame)"),
                        (self.sp_stg_z1, "Z where the sweep ends (µm, stage frame)")):
            sp.setRange(-50000.0, 50000.0)
            sp.setDecimals(2)
            sp.setSingleStep(1.0)
            sp.setSuffix(" µm")
            sp.setToolTip(tip)
            sweep.addWidget(sp)
        self.sp_stg_z0.setValue(-5.0)
        self.sp_stg_z1.setValue(5.0)
        self.sp_stg_n = QSpinBox()
        self.sp_stg_n.setRange(16, 240)
        self.sp_stg_n.setValue(48)
        self.sp_stg_n.setSuffix(" steps")
        self.sp_stg_n.setToolTip("Stage positions in the sweep: one processed frame is captured at each")
        sweep.addWidget(self.sp_stg_n)
        self.btn_stg_zsweep = self._btn("Z sweep", self._stage_zsweep,
                                        "Drive the stage Z from the first to the second height while the live engine "
                                        "captures the sweep; every frame is labelled with the stage's Z (µm) and the "
                                        "height map is reconstructed with those labels")
        self.btn_stg_zsweep.setEnabled(False)
        sweep.addWidget(self.btn_stg_zsweep)
        stg.addLayout(sweep, 4, 0, 1, 2)
        self.stage_box = section("Stage", stg,
                                 hint="Move a real (GRBL) or simulated stage. Jog moves are relative to the last "
                                      "reported pose; Zero re-defines the origin. Z sweep: the stage runs through the "
                                      "Z range while the camera frames, labelled with the stage's Z, are captured "
                                      "and reconstructed (the simulated stage drives the simulator's focus; the GRBL "
                                      "path is verified against a simulated controller only).")
        return section("1. Camera", self.cmb_source, self.cmb_res, self.chk_autoexp, exp, rot, self.chk_linear,
                       self.lbl_readback, self.demo_box, self.stage_box,
                       hint="Pick the microscope camera. Exposure is locked and in-camera sharpening is switched off, "
                            "because both would corrupt focus measurements.")

    def _stage_driver_changed(self, text: str):
        self.cmb_stg_port.setEnabled(text.startswith("GRBL"))
        if text.startswith("None") and self.stage_driver is not None:
            self._stage_disconnect()

    def _stage_async(self, label: str, fn, on_done) -> bool:
        """Run stage I/O (serial, waiting for motion) off the GUI thread. One stage operation at a time."""
        if self.stage_job is not None and self.stage_job.isRunning():
            self.notify("The stage is still busy; try again when it has finished.", "warning")
            return False
        job = Job(fn, label=label)
        self.stage_job = job
        self._stage_enable()
        job.finished.connect(self._stage_enable)
        return self._start_job(job, on_done)

    def _stage_enable(self):
        """Stage controls follow the connection and whether an operation is running."""
        busy = self.stage_job is not None and self.stage_job.isRunning()
        ready = self.stage_driver is not None and not busy
        self.btn_stg_connect.setEnabled(not busy)
        self.btn_stg_zero.setEnabled(ready)
        self.btn_stg_zsweep.setEnabled(self.stage_driver is not None)  # doubles as Cancel while a sweep runs
        for b in self.stage_jog_buttons:
            b.setEnabled(ready)

    def _stage_attach(self):
        """Label the active source's frames with the stage's pose (None detaches)."""
        if self.source is not None:
            self.source.stage_driver = self.stage_driver

    def _stage_connect(self):
        if self.stage_driver is not None:
            self._stage_disconnect()
            return
        text = self.cmb_stg.currentText()
        if text.startswith("Simulated"):
            if self.sim_source is None:
                self.notify("The stage simulator needs the simulator source (pick it in Camera).", "warning")
                return
            drv: StageDriver = SimulatedStageDriver(self.sim_source)
        elif text.startswith("GRBL"):
            drv = GrblStageDriver(self.cmb_stg_port.currentText())
        else:
            return
        # connect() can raise StageError (no answer, alarm), OSError (port busy or missing) or ImportError
        # (pyserial not installed: its message says to pip install microtopos[serial]); the job reports any of
        # them to the user through _on_failed instead of crashing the slot
        self._stage_async("Stage connect", lambda **_: (drv.connect(), drv.pose())[1],
                          lambda pose, d=drv: self._stage_connected(d, pose))

    def _stage_connected(self, drv: StageDriver, pose):
        self.stage_driver = drv
        self._stage_attach()
        self.btn_stg_connect.setText("Disconnect")
        if isinstance(drv, SimulatedStageDriver) and self.sim_source is not None:
            lo, hi = self.sim_source.relief_range_um()  # a sweep that passes the whole visible relief through focus
            self.sp_stg_z0.setValue(math.floor(lo * 100) / 100)
            self.sp_stg_z1.setValue(math.ceil(hi * 100) / 100)
        else:
            self.sp_stg_z0.setValue(round(pose.z_um - 5.0, 2))
            self.sp_stg_z1.setValue(round(pose.z_um + 5.0, 2))
        self._stage_show(pose)
        self._stage_enable()

    def _stage_disconnect(self):
        if self.stage_job is not None and self.stage_job.isRunning():
            self.stage_job.cancel = True  # a running sweep stops at its next position; disconnect afterwards
            self.notify("The stage is still busy; try again when it has finished.", "warning")
            return
        drv, self.stage_driver = self.stage_driver, None
        self._stage_attach()
        if drv is not None:
            self._stage_async("Stage disconnect", lambda **_: drv.close(), lambda _r: None)
        self.btn_stg_connect.setText("Connect")
        self.lbl_stg_pose.setText("stage: disconnected")
        self._stage_enable()

    def _stage_show(self, p):
        self.lbl_stg_pose.setText(f"stage: X {p.x_um:+.2f} µm  Y {p.y_um:+.2f} µm  Z {p.z_um:+.2f} µm  "
                                  f"({'busy' if p.moving else 'idle'})")

    def _stage_jog(self, dx_um: float, dy_um: float):
        drv = self.stage_driver
        if drv is not None:
            self._stage_async("Stage jog", lambda **_: drv.move_rel(dx_um=dx_um, dy_um=dy_um), self._stage_show)

    def _stage_zero(self):
        drv = self.stage_driver
        if drv is not None:
            self._stage_async("Stage zero", lambda **_: drv.zero(), self._stage_show)

    def _stage_zsweep(self):
        drv = self.stage_driver
        if drv is None:
            return
        if self.stage_job is not None and self.stage_job.isRunning():
            if getattr(self.stage_job, "is_sweep", False):
                self.stage_job.cancel = True  # the second click on 'Cancel sweep'
            return
        if self.source is None or self.source.stage_driver is not drv:
            self.notify("Attach the stage to the active source first (disconnect and connect again).", "warning")
            return
        zs = [float(z) for z in np.linspace(self.sp_stg_z0.value(), self.sp_stg_z1.value(), self.sp_stg_n.value())]
        src = self.source

        def run(progress=None, cancelled=None):
            return capture_z_sweep(drv, src, self.engine, zs, cancelled=cancelled, progress=progress)

        def done(poses):
            self.btn_stg_zsweep.setText("Z sweep")
            if poses:
                self._stage_show(poses[-1])
            self.notify(f"Z sweep done: {len(poses)} stage positions, {poses[0].z_um:+.2f} to "
                        f"{poses[-1].z_um:+.2f} µm. The capture is reconstructed when the engine has finished it.")

        if self._stage_async("Stage Z sweep", run, done):
            self.stage_job.is_sweep = True
            self.btn_stg_zsweep.setText("Cancel sweep")
            self.stage_job.finished.connect(lambda: self.btn_stg_zsweep.setText("Z sweep"))
    def _page_objective(self):
        self.cmb_obj = QComboBox()
        self.cmb_obj.addItems(self.objset.names)
        self.cmb_obj.setCurrentText(self.obj.name)
        self.cmb_obj.currentTextChanged.connect(self._on_objective)
        self.btn_edit_obj = self._btn("Edit objectives…", self._edit_objectives,
                                      "Add, edit or remove objectives, or load a preset such as Leitz NPL. "
                                      "Calibrations stay with the objective's name.")
        self.lbl_optics = QLabel()
        self.lbl_optics.setObjectName("mono")
        self.lbl_optics.setWordWrap(True)
        box = section(self._objective_title(), self.cmb_obj, self.btn_edit_obj, self.lbl_optics,
                      hint="Sets the numerical aperture used for focus windows, depth of field and the roughness "
                           "S-filter. Change it whenever you turn the nosepiece. Enter the objectives on your "
                           "nosepiece as engraved (Edit objectives…); the lateral scale still needs a stage "
                           "micrometer per objective (step 4).")
        self.lbl_obj_title = box.findChild(QLabel, "title")
        return box

    def _objective_title(self) -> str:
        n = min(len(self.objset.objectives), objlib.KEYED)
        return "2. Objective  (key 1)" if n == 1 else f"2. Objective  (keys 1–{n})"

    def _select_objective_index(self, i: int):
        if 0 <= i < min(len(self.objset.objectives), objlib.KEYED):
            self.cmb_obj.setCurrentText(self.objset.names[i])

    def _edit_objectives(self) -> ObjectivesDialog:
        """Open the objective editor (window-modal, non-blocking); returns it so tests can drive it."""
        dlg = ObjectivesDialog(self, self.objset, self.store, self.data_dir)
        dlg.committed.connect(self._objectives_changed)
        dlg.open()
        return dlg

    def _objectives_changed(self, new_set, notes, renames):
        self.objset = new_set
        current = renames.get(self.obj.name, self.obj.name)
        nxt = new_set.get(current) or new_set.objectives[0]
        self.cmb_obj.blockSignals(True)
        self.cmb_obj.clear()
        self.cmb_obj.addItems(new_set.names)
        self.cmb_obj.setCurrentText(nxt.name)
        self.cmb_obj.blockSignals(False)
        self.lbl_obj_title.setText(self._objective_title())
        if nxt != self.obj:
            self._on_objective(nxt.name)
        else:
            self._refresh_checklist()
        msg = f"Objective list saved ({len(new_set.objectives)} objectives, {new_set.source})."
        self.notify(" ".join([msg] + notes), "warning" if any("cleared" in n for n in notes) else "success")

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
        self.cmb_dir = QComboBox()
        self.cmb_dir.addItems(list(DIRECTIONS))
        self.cmb_dir.setCurrentText(self.settings.value("sweep_direction", "Unknown"))
        self.cmb_dir.setToolTip("Which way you turn the fine focus during a sweep. With 'Unknown' the relief may be "
                                "inverted and the result says so.")
        self.cmb_dir.currentTextChanged.connect(lambda t: self.settings.setValue("sweep_direction", t))
        self.cmb_dir.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_dir.setMinimumContentsLength(10)
        self.btn_z_travel = self._btn("Set Z of the selected result…", self._set_result_z,
                                      "State the knob travel (µm) and direction of THAT sweep. Calibrates that result "
                                      "only; optionally keeps the scale as an estimate for later sweeps.", True)
        self.btn_z_ruler = self._btn("Apply DOF ruler estimate (experimental)", self._calibrate_z_ruler,
                                     "Estimates µm per frame by comparing the measured focus-curve width with the "
                                     "optical model. Simulator tests: errors up to ±35%.")
        self.btn_z_clear = self._btn("Clear the stored estimate", self._clear_z)
        self.lbl_z = QLabel()
        self.lbl_z.setObjectName("muted")
        self.lbl_z.setWordWrap(True)
        f = QFormLayout()
        f.addRow("Sweep direction", self.cmb_dir)
        return section("5. Z scale", f, self.btn_z_travel, self.btn_z_ruler, self.btn_z_clear, self.lbl_z,
                       hint="The focus knob has no encoder. A hand sweep's heights are relative focus progression in "
                            "camera frames, assuming an even knob speed. For µm, state the knob travel of a sweep; a "
                            "stored µm-per-frame value is only an estimate for later sweeps.")

    def _page_inspect(self):
        self.cmb_override = QComboBox()
        self.cmb_override.addItems([o.value for o in Override])
        self.cmb_override.setToolTip("Auto: motion is detected from the video. The others force a mode.")
        self.cmb_override.currentTextChanged.connect(self._on_override_changed)
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
        self.sp_form.setToolTip("F-operator before roughness: 0 = mean only, 1 = plane, 2 = quadratic, 3 = cubic")
        self.sp_ls = QDoubleSpinBox()
        self.sp_ls.setRange(0, 100)
        self.sp_ls.setDecimals(2)
        self.sp_ls.setSuffix(" µm")
        self.sp_ls.setSpecialValueText("off")
        self.sp_ls.setToolTip("S-filter nesting index λs (removes shorter wavelengths). Default: optical cutoff λ/2NA; 0 = off")
        self.sp_ls.setValue(self.obj.lateral_cutoff_um)
        self._ls_user_set = False
        self.sp_ls.valueChanged.connect(lambda _v: setattr(self, "_ls_user_set", True))
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
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
                         | QDockWidget.DockWidgetClosable)
        self.dock_right = dock
        dock.visibilityChanged.connect(lambda vis: self.btn_right.setChecked(vis))
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
        dock.setMinimumWidth(300)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    # ================================================================== theme & shortcuts
    def _apply_theme(self):
        t = self.t = DARK if self.dark else LIGHT
        QApplication.instance().setStyleSheet(stylesheet(t))
        self.btn_theme.setText("Light theme" if self.dark else "Dark theme")
        if hasattr(self, "dock_left"):
            QTimer.singleShot(0, self._fit_left_dock)  # after the new style sheet has re-polished the pages
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

    def _reset_layout(self):
        self.restoreState(self._default_state)
        self.dock_left.show()
        self.dock_right.show()

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

        for i in range(objlib.KEYED):  # keys follow the objective list, whatever it holds
            add(str(i + 1), lambda i=i: self._select_objective_index(i))
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
        self._ae_active = False
        self._cancel_rotation_measure()
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
        if isinstance(self.stage_driver, SimulatedStageDriver) and self.stage_driver.sim_source is not self.sim_source:
            self._stage_disconnect()  # the simulated stage moves the simulator it was created for
        self._stage_attach()
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
        self.replay_widget.setVisible(isinstance(self.source, ReplaySource))
        if isinstance(self.source, ReplaySource):
            self.replay_seek.setRange(0, max(0, self.source.total_frames - 1))
            self._set_replay_transport_enabled(self.source.transport_supported)
        self._apply_exposure()
        QTimer.singleShot(1500, self._show_readback)
        self._refresh_checklist()

    def _rot_panels(self):
        return (self.p_live, self.p_mosaic, self.p_edf, self.p_height, self.coverage)

    def _set_rotation(self, deg: float):
        self.settings.setValue("view_rotation", float(deg))
        for p in self._rot_panels():
            p.set_rotation(deg)

    def _cancel_rotation_measure(self):
        if hasattr(self, "btn_rot_measure") and self.btn_rot_measure.isChecked():
            self.btn_rot_measure.blockSignals(True)
            self.btn_rot_measure.setChecked(False)
            self.btn_rot_measure.setText("Measure from X pan")
            self.btn_rot_measure.blockSignals(False)
        self._rot_start = None

    def _measure_rotation(self, on: bool):
        s = self.engine.snapshot()
        if on:
            self._rot_start = (s.mosaic_xy, s.mosaic_generation)
            self.btn_rot_measure.setText("Finish (after an X-only move)")
            self.notify("Move only the stage's X knob for at least a field width, then click 'Finish'.")
            return
        self.btn_rot_measure.setText("Measure from X pan")
        start = getattr(self, "_rot_start", None)
        self._rot_start = None
        if not start or start[1] != s.mosaic_generation:
            self.notify("Rotation measurement was reset because the mosaic changed.", "warning")
            return
        x0, y0 = start[0]
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
        self._ae_active = True
        self._ae_gen = self.engine.source_generation
        self._ae_steps = 0
        self._ae_stalls = 0
        self._ae_seq = self.engine.snapshot().seq
        self.btn_autoexp_once.setEnabled(False)
        QTimer.singleShot(600, self._auto_exposure_step)

    def _auto_exposure_step(self):
        if (not getattr(self, "_ae_active", False) or not isinstance(self.source, CameraSource)
                or self.engine.source_generation != getattr(self, "_ae_gen", -1)):
            self._ae_active = False
            self.btn_autoexp_once.setEnabled(isinstance(self.source, CameraSource))
            return
        s = self.engine.snapshot()
        if s.preview is None or s.seq == self._ae_seq or s.source_generation != self._ae_gen:
            self._ae_stalls = getattr(self, "_ae_stalls", 0) + 1
            if self._ae_stalls >= 15:
                self._ae_active = False
                self.btn_autoexp_once.setEnabled(True)
                self.notify("One-shot exposure timed out: no new frames from the camera.", "warning")
                return
            QTimer.singleShot(300, self._auto_exposure_step)
            return
        self._ae_stalls = 0
        self._ae_seq = s.seq
        p = s.raw_p995 if s.raw_p995 is not None else float(np.percentile(s.preview, 99.5))
        lo, hi = self.sl_exp.minimum(), self.sl_exp.maximum()
        cur = self.sl_exp.value()
        new = int(np.clip(round(cur * 220.0 / max(p, 1.0)) if p < 250 else cur * 0.6, lo, hi))
        self._ae_steps += 1
        if abs(new - cur) <= max(2, 0.03 * cur) or self._ae_steps >= 8 or new == cur:
            self._ae_active = False
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
        obj = self.objset.get(name)
        if obj is None:  # the combo is being refilled
            return
        self.obj = obj
        self.settings.setValue("objective", name)
        self.engine.set_objective(self.obj)
        if self.sim_source is not None:
            self.sim_source.set_objective(self.obj)
        self.sp_ls.blockSignals(True)
        self.sp_ls.setValue(self.obj.lateral_cutoff_um)
        self.sp_ls.blockSignals(False)
        self._ls_user_set = False
        self._cancel_rotation_measure()
        self.engine.reset_mosaic()
        self._refresh_checklist()
        self.notify(f"Objective {name}. Mosaic restarted for the new magnification.")

    def _on_objective_hint(self, ratio):
        if ratio is not None and abs(math.log(ratio)) < math.log(1.6):
            self.notify("The view changed abruptly but the magnification looks the same (stage jump or occlusion?). "
                        "The mosaic re-locates when you return to a mapped area.", "warning")
            return
        names = self.objset.names
        g = guess_objective(self.obj.magnification, ratio, self.objset.magnifications)
        guess = names[g] if g is not None else None
        text = "Objective change suspected (the image changed abruptly)."
        if guess and guess != self.obj.name:
            text += f" Texture scale suggests <b>{guess}</b> (a guess)."
        acts = [(n, lambda n=n: self.cmb_obj.setCurrentText(n)) for n in names]
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

    def _on_override_changed(self, text: str):
        mode = Override(text)
        self.engine.set_override(mode)
        self.btn_manual.setText("Finish Z capture  (Space)" if mode == Override.Z else "Start Z capture  (Space)")

    def _toggle_manual(self):
        if self.engine.sm.override == Override.Z:
            self.cmb_override.setCurrentText(Override.AUTO.value)
        else:
            self.cmb_override.setCurrentText(Override.Z.value)
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
            try:
                self.source.start_recording(folder)
            except OSError as e:  # includes a folder that is not empty
                self.btn_rec.blockSignals(True)
                self.btn_rec.setChecked(False)
                self.btn_rec.blockSignals(False)
                self.notify(f"Recording not started: {e}", "danger")
                return
            self.rec_folder = folder
            self._rec_err_shown = None
            self.btn_rec.setText("■ Stop recording")
            self.notify(f"Recording to {folder}")
        else:
            n = self.source.stop_recording()
            self.btn_rec.setText("● Record raw stream")
            mode = (getattr(self.source, "readback", {}) or {}).get("delivery", "")
            err = getattr(self.source, "rec_error", None)
            if err:
                self.notify(f"Recording stopped with an error after {n} frames: {err}", "danger")
            elif n == 0:
                self.notify("No frames were recorded.", "warning")
            else:
                self.notify(f"Saved {n} frames to {self.rec_folder}"
                            + (" (lossless PNG: this camera delivers decoded frames)" if "decoded" in mode else ""),
                            "success")

    def _reset_mosaic(self):
        self._cancel_rotation_measure()
        self.engine.reset_mosaic(keep_insets=self.chk_keep_insets.isChecked())
        self.notify("Mosaic cleared.")

    # ================================================================== calibration actions
    def _reject_simulator_cal(self) -> bool:
        if self.sim_source is not None:
            self.notify("Simulator active: calibration is only for real cameras (disk calibrations untouched).",
                        "warning")
            return True
        return False

    def _record(self, kind):
        if self._reject_simulator_cal():
            return
        self.engine.request_calibration(kind, 24)
        self.notify(f"Recording {kind} frame: keep still…")

    def _clear_radiometric(self):
        if self._reject_simulator_cal():
            return
        c = self.store.get(self.obj)
        c.darks.clear()
        c.flats.clear()
        self.store.touch(self.obj)
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
        if self._reject_simulator_cal():
            return
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
        c.lateral_magnification = self.obj.magnification
        self.store.touch(self.obj)
        self.store.save()
        self._refresh_checklist()
        nominal = self.obj.nominal_um_per_px(1920)
        self.notify(f"Lateral scale {full:.4f} µm/px at 1920 px ({full / nominal:.2f}× the nominal value).", "success")

    def _clear_lateral(self):
        if self._reject_simulator_cal():
            return
        c = self.store.get(self.obj)
        c.lateral_um_per_px_full = None
        c.lateral_magnification = None
        self.store.touch(self.obj)
        self.store.save()
        self._refresh_checklist()

    def _set_result_z(self):
        """Bind knob travel and direction to the SELECTED result (its own sweep), not to the live objective."""
        r = self.result
        if r is None or r.get("z_status") == CalStatus.SIMULATED.value or r.get("z_orientation") == "known":
            self.notify("Select a hand-sweep result first: simulated or motor-measured heights already have Z.",
                        "warning")
            return
        if self.job is not None and self.job.isRunning():
            self.notify("Wait for the running job to finish.", "warning")
            return
        # camera frames of knob movement in the part used (pauses excluded), whatever the current scale
        span = max(1.0, abs(r["z_last"] - r["z_first"]) / (r.get("z_um_per_frame_used") or 1.0))
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Z of {r.get('label', 'this result')}")
        f = QFormLayout(dlg)
        travel = QDoubleSpinBox()
        travel.setRange(0.0, 1e4)
        travel.setDecimals(3)
        travel.setSuffix(" µm")
        travel.setSpecialValueText("unknown (keep relative)")
        cur = r.get("z_um_per_frame_used")
        travel.setValue(abs(cur) * span if cur else 0.0)
        direction = QComboBox()
        direction.addItems(list(DIRECTIONS))
        sign = r.get("z_sign")
        direction.setCurrentText(next((k for k, v in DIRECTIONS.items() if v == sign), "Unknown"))
        keep = QCheckBox(f"Also keep µm per frame as an estimate for later {r['objective']} sweeps")
        f.addRow(QLabel(f"The part of this sweep that was used spans {span:.0f} camera frames of knob movement "
                        f"({r['frames_used']} frames kept)."))
        f.addRow("Knob travel during the sweep", travel)
        f.addRow("Focal plane moved", direction)
        f.addRow(keep)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        f.addRow(bb)
        if dlg.exec() != QDialog.Accepted:
            return
        new_sign = DIRECTIONS[direction.currentText()]
        old_sign = r.get("z_sign") or 1
        old_scale = r.get("z_um_per_frame_used") or 1.0  # current µm (or frames) per source frame
        if travel.value() > 0:
            new_scale, unit, stat = travel.value() / span, units.UM, CalStatus.CALIBRATED.value
        else:
            new_scale, unit, stat = 1.0, units.FRAMES, CalStatus.NOMINAL.value
        factor = (new_scale / old_scale) * ((new_sign or 1) / old_sign)
        meta = {"z_status": stat, "z_sign": new_sign, "z_orientation": "stated" if new_sign else "unknown",
                "z_um_per_frame_used": new_scale if unit == units.UM else None,
                "z_source": (f"knob travel {travel.value():g} µm over {span:.0f} frames (this sweep, even speed "
                             f"assumed)" if unit == units.UM else "relative focus progression")}
        if keep.isChecked() and unit == units.UM:
            c = self.store.get(r["objective"])  # the result's objective, whatever is selected now
            c.z_um_per_frame = new_scale
            c.z_source = meta["z_source"].replace("this sweep", "from one sweep; an estimate for others")
            c.z_uncertainty_pct = None
            c.z_date = today()
            self.store.touch(r["objective"])
            self.store.save()
        old = r
        self._start_job(Job(_rescale_job, r, factor, unit, label="Z"),
                        lambda res: self._replace_result(old, {**res, **meta}), main=True)

    def _replace_result(self, old: dict, new: dict):
        """Swap a result for its rescaled version everywhere (history, mosaic patch, views)."""
        new["warnings"] = [w for w in new.get("warnings", []) if "Sweep direction unknown" not in w] + (
            ["Sweep direction unknown: heights are relative focus progression and the relief may be inverted."]
            if new.get("z_orientation") == "unknown" else [])
        for i, h in enumerate(self.history):
            if h is old:
                self.history[i] = new
        if self.result is old:
            self.result = new
        rid = new.get("result_id")
        if rid and new.get("mosaic_generation") == self.engine.mosaic_generation and new.get("preview_shape"):
            self.engine.remove_height_patch(rid)
            self._pin(new)
        self._refresh_history()
        self._refresh_checklist()
        self._show_result()
        self.notify(f"Z of the selected result set: {z_status_text(new)}.", "success")

    def _calibrate_z_ruler(self):
        if self._reject_simulator_cal():
            return
        r = self.result
        est = r.get("dof_ruler_um_per_frame") if r else None
        if not est or r.get("z_status") == CalStatus.SIMULATED.value:
            self.notify("Needs a hand sweep reconstructed in focus-frame units first.", "warning")
            return
        c = self.store.get(r["objective"])  # the objective the sweep was made with
        c.z_um_per_frame = float(est)
        c.z_source = "DOF ruler (experimental model estimate, simulator error up to ±35%)"
        c.z_uncertainty_pct = 35.0
        c.z_date = today()
        self.store.touch(r["objective"])
        self.store.save()
        self._refresh_checklist()
        self.notify(f"Stored an experimental Z estimate of {est:.4f} µm per frame for {r['objective']}; "
                    "new sweeps use it and are labelled ESTIMATED.", "warning")

    def _clear_z(self):
        if self._reject_simulator_cal():
            return
        c = self.store.get(self.obj)
        c.z_um_per_frame = None
        self.store.touch(self.obj)
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
            (f"{self.obj.magnification:g}×", "success"),
            ("Done" if rad else ("Partial" if rkey in c.darks or rkey in c.flats else "Optional"),
             "success" if rad else "warning"),
            ("Calibrated" if c.lateral_um_per_px_full else "Nominal", "success" if c.lateral_um_per_px_full else "warning"),
            ("Estimate" if c.z_um_per_frame else "Frames", "warning"),
            ("Ready" if cam_ok else "Waiting", "success" if cam_ok else "neutral"),
        ]
        for chip, (txt, kind) in zip(self.step_chips, states):
            chip.set(txt, kind)
        nxt = next((i for i, (_, k) in enumerate(states) if k in ("danger",)), None)
        for i, b in enumerate(self.step_btns):
            b.setToolTip("Next step" if i == nxt else "")
        um_full = c.um_per_px(1920, self.obj)
        self.chip_xy.set(f"XY {'CAL' if c.lateral_um_per_px_full else 'NOMINAL'} {um_full:.3f} µm/px",
                         "success" if c.lateral_um_per_px_full else "warning",
                         c.lateral_source or "Nameplate optics only; calibrate with a stage micrometer (step 4)")
        self.chip_z.set(f"Z {'EST ' + format(c.z_um_per_frame, '.3f') + ' µm/f' if c.z_um_per_frame else 'FRAMES'}",
                        "warning", (c.z_source + " — an estimate for new sweeps (same knob speed assumed); state "
                                    "each sweep's travel for CALIBRATED heights") if c.z_um_per_frame
                        else "New hand sweeps: relative focus progression in camera frames")
        self.chip_rad.set("DARK/FLAT ✓" if rad else "DARK/FLAT —", "success" if rad else "neutral")
        self.chip_src.set("SIMULATOR" if sim else "CAMERA", "info" if sim else "success")
        o = self.obj
        self.lbl_optics.setText(
            (f"{o.maker} {o.series}".strip() + "\n" if o.maker or o.series else "")
            + f"{o.magnification:g}×  NA {o.na:.2f}{f'  immersion n={o.immersion_n:g}' if o.is_oil else '  dry'}"
            + (f"  cover {o.cover}" if o.cover else "") + f"\nRayleigh {o.rayleigh_um:.2f} µm\n"
            f"Optical cutoff λ/2NA {o.lateral_cutoff_um:.2f} µm\nDepth of field {o.dof_um:.2f} µm\n"
            f"Pixel {um_full:.3f} µm at 1920 px ({c.lateral_status.value.lower()})")
        lines = [f"{k}: dark {'✓' if k in c.darks else '—'}, flat {'✓' if k in c.flats else '—'}"
                 for k in sorted(set(c.darks) | set(c.flats))]
        self.lbl_rad.setText("\n".join(lines) or "Nothing recorded for this objective.")
        self.lbl_lat.setText(f"{c.lateral_status.value}: {c.lateral_source or 'nominal'}"
                             + (f" ({c.lateral_date})" if c.lateral_date else ""))
        self.lbl_z.setText(f"Stored for {self.obj.name}: {c.z_status.value}"
                           + (f", {c.z_um_per_frame:.4f} µm per frame — {c.z_source}" if c.z_um_per_frame else
                              " (new sweeps in camera frames)") + (f" ({c.z_date})" if c.z_date else ""))

    # ================================================================== live tick
    def _brightness_note(self, preview: np.ndarray) -> str:
        """Live-label suffix: says when the display is stretched, and warns when the camera image is very dark."""
        if self._stretch is not None:
            return f" · display stretched {self._stretch[0]:.0f}–{self._stretch[1]:.0f} DN"
        p99 = frame_percentile(preview, 99)
        if p99 < DARK_P99_DN:
            return f" · very dark image (99 % of pixels ≤ {p99:.0f} DN): check the lamp, or tick Stretch brightness"
        return ""

    def _tick(self):
        s = self.engine.snapshot()
        if s.preview is None:
            return
        cal = self.store.get(self.obj)
        # the preview covers the full field of view; the simulator's truth scale is the nominal one
        pw = s.preview.shape[1]
        um_prev = self.obj.nominal_um_per_px(pw) if self.sim_source is not None else cal.um_per_px(pw)
        self.um_preview = um_prev
        if self.chk_stretch.isChecked():
            self._stretch = stretch_levels(s.preview, self._stretch)
            levels = self._stretch
        else:
            self._stretch = None
            levels = (0, 255)
        self.p_live.set_image(s.preview, um_prev, levels=levels, unit="DN")
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
        rec_err = getattr(self.source, "rec_error", None)
        if rec_err and rec_err != getattr(self, "_rec_err_shown", None):
            self._rec_err_shown = rec_err
            self.btn_rec.blockSignals(True)
            self.btn_rec.setChecked(False)
            self.btn_rec.setText("● Record raw stream")
            self.btn_rec.blockSignals(False)
            self.notify(f"Recording stopped: {rec_err}", "danger")
        if s.engine_error and s.engine_error != getattr(self, "_eng_err_shown", None):
            self._eng_err_shown = s.engine_error
            self.notify(f"Processing error (the engine keeps running; see Diagnostics): {s.engine_error}", "danger")
        if self.engine.calib_error and self.engine.calib_error != getattr(self, "_cal_err_shown", None):
            self._cal_err_shown = self.engine.calib_error
            self.notify(self.engine.calib_error, "warning")
        self.lbl_live.setText((f"● REC {self.source.rec_frames} frames · " if rec else "") +
                              f"{s.frame_shape[1]}×{s.frame_shape[0]} capture · preview {um_prev:.3f} µm/px · "
                              f"{'averaging ' + str(min(s.dwell_frames, 8)) + ' frames' if s.state == State.DWELL else ''}"
                              + self._brightness_note(s.preview))
        if isinstance(self.source, ReplaySource):
            paused = self.btn_replay_pause.isChecked()
            self.btn_replay_pause.setText("▶ Play" if paused else "‖ Pause")
            if not paused:
                self.replay_seek.blockSignals(True)
                self.replay_seek.setValue(min(self.replay_seek.maximum(), self.source.position))
                self.replay_seek.blockSignals(False)
        self.lbl_motion.setText(f"shift {s.shift_px:5.2f} px · corr {s.response:4.2f} · sharpness z {s.z_score:+5.1f}"
                                + ("\nMosaic tracking lost: move back over a mapped area" if s.mosaic_lost else ""))
        if s.brenner_hist:
            self.spark_curve.setData(np.asarray(s.brenner_hist, float))
        cp = self.engine.calibration_progress()
        if cp:
            self.lbl_job.setText(f"Recording {cp[0]}: {cp[1]}/{cp[2]} frames")
        elif getattr(self, "_cal_was", None):
            if not self.engine.calib_error:
                self.notify(f"{self._cal_was[0].capitalize()} frame recorded and saved.", "success")
            self._refresh_checklist()
        self._cal_was = cp
        if s.mosaic_version != self.last_mosaic_version and self.tabs.currentIndex() == 1:
            self.last_mosaic_version = s.mosaic_version
            (img, origin, d), parts, nkf, npatch = self.engine.mosaic_image()
            showing_loaded = nkf <= 1 and self.loaded_mosaic is not None
            if showing_loaded:  # nothing mapped yet: show the opened session's mosaic
                (img, origin, d), parts, um_loaded = self.loaded_mosaic
                m_um = um_loaded or um_prev
            else:
                m_um = um_prev
            self.mosaic_um = m_um
            self.current_parts = parts
            mode = self.cmb_mview.currentIndex()
            if mode == 2 and parts:
                key, hz, horig, _ = parts[0]
                self.p_mosaic.set_image(hz, m_um, unit=self._key_unit(key))
                self.p_mosaic.set_overlay(None)
                self.mosaic_origin = horig
                self.lbl_hlegend.setText(self._legend(parts, hz))
            else:
                valid = np.isfinite(img).astype(np.uint8)
                g = np.clip(np.nan_to_num(img, nan=0.0), 0, 255).astype(np.uint8)
                rgba = np.dstack([g, g, g, valid * 255])  # empty = transparent; at most ~1600 px: cheap to repaint
                if self.mosaic_rgb:
                    rgb = self.engine.mosaic.image_rgb(d)
                    if rgb is not None:
                        rgba = rgb
                self.p_mosaic.set_image(rgba, m_um * d, rgb=True)
                self.mosaic_origin = origin
                self._draw_height_overlay(parts if mode == 1 else [], origin, m_um)
            span = (img.shape[1] * d * m_um, img.shape[0] * d * m_um)
            self.lbl_mosaic.setText(f"{nkf} keyframes · {npatch} height patches · {span[0]:.0f} × {span[1]:.0f} µm"
                                    + (f" · shown at 1/{d}" if d > 1 else ""))
        if self.tabs.currentIndex() == 1:
            showing_loaded = self.loaded_mosaic is not None and self.engine.mosaic_image()[2] <= 1
            if showing_loaded:
                self.cam_box.hide()
            else:
                self._draw_camera_box(s, um_prev)
        if s.seq % 20 == 0:
            self._refresh_checklist()
            if self.diag.isVisible() and self.result is None:
                self.diag.setPlainText("\n".join(list(self.engine.events)[-40:]))

    @staticmethod
    def _key_unit(key: str) -> str:
        return "µm" if key.startswith("um") else "frames"

    def _legend(self, parts, z) -> str:
        key, _, _, n = parts[0]
        lo, hi = np.nanpercentile(z, [2, 98]) if np.isfinite(z).any() else (0, 0)
        kind = {"um": "calibrated µm"}.get(key, "estimated µm (stored knob scale)" if key.startswith("um~")
                                           else "relative, own scale")
        more = f" · {len(parts) - 1} other group(s) not blended" if len(parts) > 1 else ""
        return f"height {lo:.3g}–{hi:.3g} {self._key_unit(key)} ({kind}, {n} map(s)){more}"

    def _draw_height_overlay(self, parts, origin, um):
        """Colour-mapped heights over the image mosaic at their true positions. Each group of comparable maps is
        normalised on its own: maps in different quantities are never put on one colour scale."""
        parts = [p for p in parts if np.isfinite(p[1]).any()]
        if not parts:
            self.p_mosaic.set_overlay(None)
            self.lbl_hlegend.setText("")
            return
        x0 = min(p[2][0] for p in parts)
        y0 = min(p[2][1] for p in parts)
        x1 = max(p[2][0] + p[1].shape[1] for p in parts)
        y1 = max(p[2][1] + p[1].shape[0] for p in parts)
        rgba = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
        cmap = pg.colormap.get("viridis")
        alpha = int(255 * self.sl_hop.value() / 100)
        for _key, z, (hx, hy), _n in reversed(parts):  # the most recent group is drawn last (on top)
            lo, hi = np.nanpercentile(z, [2, 98])
            norm = np.clip((np.nan_to_num(z, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)
            col = cmap.map(norm.ravel(), mode="byte").reshape(z.shape + (4,))
            ok = np.isfinite(z)
            sub = rgba[hy - y0:hy - y0 + z.shape[0], hx - x0:hx - x0 + z.shape[1]]
            sub[ok, :3] = col[ok, :3]
            sub[ok, 3] = alpha
        from PySide6.QtCore import QRectF
        rect = QRectF((x0 - origin[0]) * um, (y0 - origin[1]) * um, (x1 - x0) * um, (y1 - y0) * um)
        self.p_mosaic.set_overlay(rgba, rect)
        self.lbl_hlegend.setText(self._legend(parts, parts[0][1]))

    def _export_height_mosaic(self):
        parts = getattr(self, "current_parts", None)
        if not parts:
            self.notify("No height maps in the mosaic yet.", "warning")
            return
        key, hz, origin, n = parts[0]
        p = self._save_path("Export stitched height mosaic", "height_mosaic.gsf", "GSF (*.gsf)")
        if p and self._confirm_export(p, f"stitched height mosaic ({n} map(s), group '{key}')", self._key_unit(key),
                                      "median-filled in the .gsf; the mask file records them"):
            try:
                export.export_height(p, "gsf", hz, getattr(self, "mosaic_um", getattr(self, "um_preview", 1.0)),
                                     "um" if key.startswith("um") else "frames",
                                     {"kind": "height mosaic", "group": key, "maps": n, "origin_px": list(origin)},
                                     "MicroTopos height mosaic")
                self.notify(f"Saved {p} (+ mask and provenance)", "success")
            except OSError as e:
                self.notify(f"Export failed: {e}", "danger")

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
            txt = (f"<b>{html.escape(self.job.label)} running…</b> You can keep working; the result line below the "
                   "coach says when it is ready.")
        elif self.last_ignored is not None:
            txt = (f"<b>That focus movement was not used</b> (only {self.last_ignored:.0%} of the field went sharp and "
                   "blurry again). For a height map: start blurry, turn slowly through sharp, stop when blurry again.")
        elif self.result is None:
            txt = ("<b>Next: a height map.</b> Keep the stage still. Turn the fine focus until the image is blurry, then "
                   "turn slowly the other way — through sharp — until it is blurry again, and stop.  "
                   "<b>Or a mosaic:</b> focus, then move the X-Y stage slowly.")
        else:
            r = self.result
            nxt = (" State this sweep's knob travel and direction (Set Z…) for signed µm."
                   if r.get("z_orientation") == "unknown" or units.canon(r["z_unit"]) == units.FRAMES else "")
            txt = (f"<b>Height map ready</b> ({r['valid_fraction']:.0%} reliable) — see the Height, 3D and Report "
                   f"tabs. Sweep again anywhere for another.{nxt}")
        if txt != self.guide.text():
            self.guide.setText(txt)

    # ================================================================== reconstruction
    def _z_unit_label(self):
        return units.label((self.result or {}).get("z_unit", "frames"))

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

    def _lam_s(self, obj) -> float | None:
        """λs from the spin box; 0.0 when explicitly set to off, None when at the objective's optical default."""
        v = self.sp_ls.value()
        if v == 0:
            return 0.0
        if not getattr(self, "_ls_user_set", False):
            return None
        return None if abs(v - round(obj.lateral_cutoff_um, 2)) < 1e-9 else v

    def _start_job(self, job: Job, on_done, main: bool = False) -> bool:
        """Start a job the window owns. main=True: the reconstruction slot (one at a time)."""
        if main and self.job is not None and self.job.isRunning():
            self.notify(f"{self.job.label} is still running; try again when it has finished.", "warning")
            return False
        job.progress.connect(lambda f, m, j=job: self._on_progress(j, f, m))
        job.done.connect(on_done)
        job.failed.connect(lambda msg, j=job: self._on_failed(j, msg))
        job.finished.connect(lambda j=job: self.jobs.remove(j) if j in self.jobs else None)
        self.jobs.append(job)
        if main:
            self.job = job
            self.btn_cancel.setEnabled(True)
            job.finished.connect(lambda: self.btn_cancel.setEnabled(False))
        job.start()
        return True

    def _reconstruct(self, cap: SweepCapture, force: bool = False):
        if cap.completeness < 0.5 and not cap.manual and self.engine.sm.override != Override.Z and not force:
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
        obj = cap.optics or resolve(cap.objective)  # the optics at capture, even if the list was edited since
        cal = self.store.get(obj)
        w = cap.shape[1]
        um = obj.nominal_um_per_px(w) if cap.simulated else cal.um_per_px(w, obj)
        sign = DIRECTIONS.get(self.cmb_dir.currentText())
        zsrc = ""
        if cap.z is not None and cap.simulated:
            z, unit, zstat, zcal = cap.z, units.UM, CalStatus.SIMULATED.value, None
        elif cap.z is not None:  # labelled by a motorised stage (capture_z_sweep): um, but only as good as its steps/mm
            z, unit, zstat, zcal = cap.z, units.UM, CalStatus.ESTIMATED.value, None
            zsrc = "stage driver readout (not checked against a reference)"
        elif cal.z_um_per_frame:
            z, unit, zstat, zcal = None, units.UM, CalStatus.ESTIMATED.value, cal.z_um_per_frame
        else:
            z, unit, zstat, zcal = None, units.FRAMES, CalStatus.NOMINAL.value, None
        st = zstack.ZSettings(measure=self.cmb_measure.currentText(), form_order=self.sp_form.value(),
                              lam_s_um=self._lam_s(obj), lam_c_um=self.sp_lc.value(),
                              workers=max(1, (os.cpu_count() or 2) - 1),
                              backend=self.zst.backend, correct_breathing=self.zst.correct_breathing,
                              allow_tiling=self.zst.allow_tiling)
        meta = {"z_status": zstat, "lateral_status": cal.lateral_status.value if not cap.simulated else "SIMULATED",
                "pose": cap.pose, "capture_shape": cap.shape, "result_id": uuid.uuid4().hex[:12],
                "source_id": cap.source_id, "source_generation": cap.source_generation,
                "mosaic_generation": cap.mosaic_generation, "calibration_revision": cal.revision,
                "preview_shape": list(cap.preview_shape), "exposure": cap.exposure, "linearize": cap.linearize,
                "z_source": zsrc or (cal.z_source if zstat == CalStatus.ESTIMATED.value else ""),
                "sweep_seq_range": [cap.seqs[0], cap.seqs[-1]] if cap.seqs else None}
        sample_pos = np.asarray(cap.seqs, float) if cap.seqs and z is None else None
        job = Job(zstack.reconstruct, cap.frames, z, obj, um, unit, zcal, st, label="Reconstruction",
                  sample_pos=sample_pos, z_sign=sign if z is None else None)
        if self._start_job(job, lambda r: self._on_result({**r, **meta}), main=True):
            self.notify(f"Focus sweep captured ({len(cap.frames)} frames). Reconstructing…")

    def _on_progress(self, job, f, msg):
        if job is not self.job and self.job is not None and self.job.isRunning():
            return
        self.prog.setValue(int(f * 1000))
        self.lbl_job.setText(f"{job.label}: {msg}")

    def _on_failed(self, job, msg):
        if job is self.job:
            self.btn_cancel.setEnabled(False)
        if job is not self.job and self.job is not None and self.job.isRunning():
            self.notify(f"{job.label}: {msg}", "warning")
            return
        self.prog.setValue(0)
        self.lbl_job.setText(f"{job.label}: {msg}")
        self.notify(f"{job.label}: {msg}", "warning")

    def _cancel(self):
        if self.job is not None:
            self.job.cancel = True

    def _repeat_check(self, a: dict, b: dict) -> str | None:
        """Why two results cannot be compared for repeatability, or None when they can."""
        if a["objective"] != b["objective"]:
            return "the objective changed"
        if a["height"].shape != b["height"].shape:
            return "the image size differs"
        if math.hypot(a["pose"][0] - b["pose"][0], a["pose"][1] - b["pose"][1]) > 2.0:
            return "the stage moved"
        if z_key(a) != z_key(b) or z_key(a).startswith("frames"):
            return "the two maps do not share a calibrated, signed height scale (unit, direction or calibration)"
        return None

    def _on_result(self, r: dict):
        if self.repeat_armed and self.result is not None:
            self.repeat_armed = False
            why = self._repeat_check(self.result, r)
            if why is None:
                rep = metrology.repeatability(self.result["height"], r["height"])
                if rep is not None:
                    r["sigma_inst"] = rep["sigma_inst"]
                    r["repeat_offset"] = rep["mean_offset"]
                    r["repeat_overlap"] = rep["overlap_fraction"]
                    if r.get("Sq"):
                        r["sigma_inst_lt_Sq_over_10"] = bool(rep["sigma_inst"] < r["Sq"] / 10)
            else:
                r["warnings"] = r.get("warnings", []) + [f"Repeatability not computed: {why}."]
        self.prev_result, self.result = self.result, r
        self._result_t = time.perf_counter()
        self._remember(r)
        self.last_ignored = None
        self._pin(r)
        self._show_result()  # stay on the current tab: the user may be mapping; the result card says where to look
        w = r.get("warnings") or []
        self.notify(f"Height map ready: {r['valid_fraction']:.0%} of pixels reliable, "
                    f"{r['timings']['total']:.1f} s." + (f"  {len(w)} warning(s) in Report." if w else ""),
                    "warning" if w else "success")

    def _pin(self, r: dict):
        """Pin a result into the mosaic it was captured in, at the preview size of ITS capture. The engine drops it
        if that mosaic has been restarted since (reset, new source or objective)."""
        ps = r.get("preview_shape")
        if not ps or not ps[0]:
            return
        ph, pw = int(ps[0]), int(ps[1])
        edf_p = cv2.resize(r["edf"], (pw, ph), interpolation=cv2.INTER_AREA)
        h_p = cv2.resize(r["height"], (pw, ph), interpolation=cv2.INTER_NEAREST)
        self.engine.add_height_patch(r["pose"], edf_p, h_p, r["z_unit"], z_key(r), r.get("result_id", ""),
                                     r.get("mosaic_generation"))
        self.last_mosaic_version = -1

    HISTORY = 10

    def _remember(self, r):
        self._n_results = getattr(self, "_n_results", 0) + 1
        r.setdefault("label", f"#{self._n_results}  {str(r.get('timestamp', ''))[11:19]}  "
                              f"{r['objective'].split(' /')[0]}")
        self.history.insert(0, r)
        del self.history[self.HISTORY:]
        self._refresh_history()

    def _refresh_history(self):
        self.cmb_hist.blockSignals(True)
        self.cmb_hist.clear()
        self.cmb_hist.addItems([f"{h['label']} · {h['valid_fraction']:.0%} · {units.label(h['z_unit'])}"
                                for h in self.history])
        self.cmb_hist.setCurrentIndex(next((i for i, h in enumerate(self.history) if h is self.result), 0))
        self.cmb_hist.blockSignals(False)

    def _pick_history(self, i):
        if 0 <= i < len(self.history) and self.history[i] is not self.result:
            self.result = self.history[i]
            self._show_result()

    def _recompute(self):
        r = self.result
        if r is None:
            return
        job = Job(_metrology_job, r, self.sp_form.value(), self._lam_s(resolve_result(r)),
                  self.sp_lc.value(), label="Metrology")
        self._start_job(job, lambda m, r=r: self._on_recomputed(r, m), main=True)

    def _on_recomputed(self, r, m):
        new = dict(r)
        for k in metrology.PARAMETER_KEYS:
            new.pop(k, None)
        new["warnings"] = [x for x in r.get("warnings", []) if not metrology.is_metrology_warning(x)] + m.pop("warnings")
        new.update(m)
        for i, h in enumerate(self.history):
            if h is r:
                self.history[i] = new
        if self.result is r:
            self.result = new
        self.prog.setValue(1000)
        self.lbl_job.setText("Metrology recomputed.")
        self._show_result()
        self.notify("Metrology recomputed with the new filter settings.", "success")

    def _run_sr(self):
        if self.sr_job is not None and self.sr_job.isRunning():
            self.notify("Super-resolution is already running.", "warning")
            return
        frames = self.engine.dwell_frames()
        if len(frames) < 8:
            self.notify("Hold still for a moment (needs 8 still frames).", "warning")
            return
        h, w = frames[0].shape
        c = min(512, h, w)
        y0, x0 = (h - c) // 2, (w - c) // 2
        crops = [f[y0:y0 + c, x0:x0 + c].astype(np.float32) for f in frames]
        cal = self.store.get(self.obj)
        um = self.obj.nominal_um_per_px(w) if self.sim_source is not None else cal.um_per_px(w)
        job = Job(_sr_job, crops, self.obj.psf_sigma_px(um),
                  scale=self.sr_scale, pixfrac=self.sr_pixfrac, deconvolve=self.sr_deconvolve,
                  label="Super-resolution")
        self.sr_job = job
        self._start_job(job, lambda r: self._on_sr({**r, "um_per_px": um / 2, "input": crops[0]}))

    def _on_sr(self, r):
        self.sr_result = r
        self.sr_count = getattr(self, "sr_count", 0) + 1
        if self.job is None or not self.job.isRunning():
            self.prog.setValue(1000)
            self.lbl_job.setText(f"Super-resolution (2× linear, 4× pixels): {r['frames_used']} frames used, "
                                 f"{r['frames_rejected']} rejected, {r['subpixel_bins']} sub-pixel phases, "
                                 f"RL {r['rl_iterations']} iterations, PSF σ {r['psf_sigma_lr']:.2f} px ({r['psf_source']})")
        self.cmb_edf.setCurrentIndex(1)
        self.tabs.setCurrentIndex(2)
        self._show_result()

    # ================================================================== result views
    def _update_card(self):
        r = self.result
        if r is None:
            self.card.hide()
            return
        lat = r.get("lateral_status", "?")
        vf = r["valid_fraction"]
        bits = [f"<b>{html.escape(r.get('label', ''))}</b>", html.escape(r["objective"]),
                f"XY {lat}", html.escape(z_status_text(r)), f"{vf:.0%} valid"]
        if r.get("support_fraction") is not None and r["support_fraction"] < 0.98:
            bits.append(f"{r['support_fraction']:.0%} of the field observed in every frame")
        warn = len(r.get("warnings", []))
        if warn:
            bits.append(f"<span style='color:{self.t['warning']}'>⚠ {warn} warning(s) in Report</span>")
        self.lbl_card.setText(" · ".join(bits))
        self.btn_card_z.setVisible(r.get("z_orientation") != "known" and r.get("z_status") != "SIMULATED")
        self.card.show()

    def _show_result(self):
        r = self.result
        self._update_card()
        if self.cmb_edf.currentIndex() == 1 and self.sr_result is not None:
            sr = self.sr_result
            self.p_edf.set_image(np.clip(sr["image"], 0, 255), sr["um_per_px"], levels=(0, 255), unit="DN", autorange=True)
            self.p_edf.set_overlay(None)
            self.lbl_edf.setText("Experimental: 2× (linear) drizzle + Richardson-Lucy of the centre crop; a display "
                                 "image, not a measured resolution gain")
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
            self.lbl_edf.setText(f"{r['frames_used']} of {r['frames_in']} frames fused · {r.get('grain_count', 0)} "
                                 "watershed regions (image-based)")
        if r is None:
            self._render_report()
            return
        mode = self.cmb_height.currentIndex()
        um = r["um_per_px"]
        zl = units.label(r["z_unit"])
        if mode == 0:
            data, unit = r["sl_surface"], zl
        elif mode == 1:
            data, unit = r["height"], zl
        else:
            data, unit = r["confidence"], ""
        levels = (0, 1) if mode == 2 else None
        if self.chk_lock.isChecked() and mode != 2:
            lk = getattr(self, "_locked_levels", None)
            if lk is None or lk[0] != (mode, units.canon(r["z_unit"])):
                fin = data[np.isfinite(data)]
                lk = ((mode, units.canon(r["z_unit"])), (float(np.percentile(fin, 1)), float(np.percentile(fin, 99)))
                      if fin.size else (0.0, 1.0))
                self._locked_levels = lk
            levels = lk[1]
        self.p_height.bar.setColorMap(pg.colormap.get("cividis" if mode == 2 else "viridis"))
        self.p_height.set_image(data, um, levels=levels, unit=unit)
        self.p_height.extra = r["confidence"]
        if not self.prof_roi.isVisible():
            h, w = data.shape
            self.prof_roi.setPos(0, 0)
            hs = self.prof_roi.getHandles()
            hs[0].setPos(w * um * 0.1, h * um * 0.5)
            hs[1].setPos(w * um * 0.9, h * um * 0.5)
            self.prof_roi.show()
        what = ["filtered height", "raw fitted height", "confidence (0–1)"][mode]
        self.lbl_height.setText(f"{what} · {r['valid_fraction']:.0%} reliable · {um:.3f} µm/px · {z_status_text(r)}")
        if self.chk_topo_grains.isChecked() and mode == 0:
            try:
                gt = metrology.grains_topographic(r["sl_surface"], um)
                r["topo_grain_count"] = int(gt.get("topo_grain_count", 0))
                r["topo_mean_grain_area_um2"] = float(gt.get("topo_mean_grain_area_um2", 0.0))
            except Exception:  # noqa: BLE001 - a bad grains estimate must not redraw the view
                r.pop("topo_grain_count", None)
                r.pop("topo_mean_grain_area_um2", None)
        self._update_profile()
        self._update_3d()
        self._render_report()

    def _update_profile(self):
        r = self.result
        if r is None:
            return
        mode = self.cmb_height.currentIndex()
        data = r["sl_surface"] if mode == 0 else (r["height"] if mode == 1 else r["confidence"])
        zl = units.label(r["z_unit"])
        um = r["um_per_px"]
        self.prof_plot.setLabel("left", "confidence" if mode == 2 else "height", units="" if mode == 2 else zl)
        self.prof_plot.getAxis("left").enableAutoSIPrefix(False)
        self.prof_plot.setLabel("bottom", "distance", units="µm")
        self.prof_plot.getAxis("bottom").enableAutoSIPrefix(False)
        h0, h1 = (self.prof_roi.mapToParent(hd.pos()) for hd in self.prof_roi.getHandles())
        x0, y0, x1, y1 = h0.x() / um, h0.y() / um, h1.x() / um, h1.y() / um
        n = int(max(2, math.hypot(x1 - x0, y1 - y0)))
        xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
        ok = (xs >= 0) & (xs < data.shape[1] - 1) & (ys >= 0) & (ys < data.shape[0] - 1)
        prof = np.full(n, np.nan, np.float32)
        if not ok.any():  # outside the image: nothing stale may remain
            self.prof_curve.setData([], [])
            self.prof_label.setText("The profile line is outside the image.")
            return
        prof[ok] = cv2.remap(data.astype(np.float32), xs[ok].astype(np.float32).reshape(1, -1),
                             ys[ok].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).ravel()
        dist = np.hypot(xs - x0, ys - y0) * um
        self.prof_curve.setData(dist, prof, connect="finite")
        v = prof[np.isfinite(prof)]
        if mode == 2:
            self.prof_label.setText(f"Confidence along {dist[-1]:.1f} µm: mean {np.mean(v):.2f}" if v.size
                                    else "No confidence values along the line.")
        elif v.size > 4:
            v = v - v.mean()
            self.prof_label.setText(f"Profile over {dist[-1]:.1f} µm ({v.size} of {n} points valid):  "
                                    f"Ra {np.mean(np.abs(v)):.4g} {zl}  ·  Rq {np.sqrt(np.mean(v * v)):.4g} {zl}  ·  "
                                    f"peak-to-valley {np.ptp(v):.4g} {zl}  (unfiltered profile; for orientation only)")
        else:
            self.prof_label.setText("Too few valid heights along the line for profile values.")

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
        zf = np.where(finite, zs, fill)  # display-only fill: those facets are fully transparent (real holes)
        lo, hi = np.percentile(zf[finite], [1, 99])
        cm = pg.colormap.get(self.cmb_cmap.currentText())
        norm = np.clip((zf - lo) / max(hi - lo, 1e-9), 0, 1)
        exag = self.sl_exag.value() / 10.0
        is_um = units.is_um(r["z_unit"])
        note = (f"z exaggeration ×{exag:g}" + ("" if is_um else " · z in focus frames × pixel size (not µm)")
                + f" · {1 - finite.mean():.0%} without depth shown as holes")
        if self.gl is not None:
            zscale = exag * (1.0 if is_um else um)
            cols = cm.map(norm.ravel(), mode="float").reshape(zs.shape + (4,))
            cols[~finite, 3] = 0.0
            xs = np.arange(zs.shape[1]) * um * step
            ys = np.arange(zs.shape[0]) * um * step
            self.gl_surf.setData(x=xs - xs.mean(), y=ys - ys.mean(), z=(zf.T - fill) * zscale,
                                 colors=np.transpose(cols, (1, 0, 2)))
            self.gl_surf.setVisible(True)
            span = max(xs.max(), ys.max())
            key = (zs.shape, round(span, 3))
            if key != getattr(self, "_gl_key", None):  # keep the user's view when only colours or exaggeration change
                self._gl_key = key
                self.gl_grid.setSize(span * 1.2, span * 1.2)
                self.gl_grid.setSpacing(span / 10, span / 10)
                self.gl.setCameraPosition(distance=span * 1.4)
            self.lbl_3d.setText(note)
        else:
            gy, gx = np.gradient(zf * exag)
            shade = np.clip(0.6 + 0.4 * (-gx - gy) / (np.hypot(gx, gy) + 1e-3 * (hi - lo) + 1e-12), 0, 1)
            rgb = (cm.map(norm.ravel(), mode="float").reshape(zs.shape + (4,))[..., :3] * shade[..., None] * 255)
            rgb[~finite] = hex_rgb(self.t["nan"])
            self.relief.set_image(rgb.astype(np.uint8), um * step, rgb=True)
            self.lbl_3d.setText("Shaded relief (no OpenGL) · " + note)

    def _render_report(self):
        t = self.t
        r = self.result
        css = (f"body{{color:{t['text']};font-size:13px}} h2,h3{{margin:6px 0}} table{{border-collapse:collapse}}"
               f" td,th{{padding:4px 10px;border-bottom:1px solid {t['border']}}} th{{text-align:left;color:{t['muted']}}}"
               f" .w{{color:{t['warning']}}} .m{{color:{t['muted']}}} .ok{{color:{t['success']}}}")
        if r is None:
            self.report.setHtml(f"<style>{css}</style><h2>Report</h2><p class='m'>No measurement yet. Turn the fine "
                                f"focus through the sample to create a height map.</p>{OPTICS_NOTES}")
            return
        zu = units.label(r["z_unit"])
        elig = r.get("eligibility", {})

        def row(name, key, unit, desc, fmt="{:.4g}", group=None):
            v = r.get(key)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                why = elig.get(group, "") if group else ""
                why = "" if why == "ok" else why
                return (f"<tr><td>{name}</td><td class='m'>—</td><td>{unit}</td><td class='m'>{html.escape(desc)}"
                        f"{(' · <i>' + html.escape(why) + '</i>') if why else ''}</td></tr>")
            return (f"<tr><td>{name}</td><td><b>{fmt.format(v)}</b></td><td>{unit}</td>"
                    f"<td class='m'>{html.escape(desc)}</td></tr>")

        def table(title, rows):
            return (f"<h3>{title}</h3><table><tr><th>Parameter</th><th>Value</th><th>Unit</th><th>Meaning</th></tr>"
                    + "".join(row(*x) for x in rows) + "</table>")

        # 1. eligibility first: what these numbers are and are not
        status = []
        zs = r.get("z_status")
        if zs == CalStatus.SIMULATED.value:
            status.append("Simulated specimen: heights are simulation truth, not a measurement.")
        elif units.canon(r["z_unit"]) == units.FRAMES:
            status.append("Z is not calibrated: heights are relative focus progression in camera frames (constant "
                          "knob speed assumed). Use 'Set Z…' to state this sweep's knob travel.")
        elif zs == CalStatus.ESTIMATED.value:
            status.append("Z scale is an ESTIMATE carried over from another sweep or the DOF ruler "
                          f"({html.escape(str(r.get('z_source') or ''))}); it assumes the same knob speed.")
        if r.get("z_orientation") == "unknown":
            status.append("Sweep direction unknown: the relief may be inverted.")
        if r.get("lateral_status") == "NOMINAL":
            status.append("Lateral scale is nominal (uncalibrated): µm lengths, Sdr and grain sizes are indicative.")
        head = "".join(f"<p class='w'>⚠ {html.escape(x)}</p>" for x in status)
        head += "".join(f"<p class='w'>⚠ {html.escape(w)}</p>" for w in r.get("warnings", []))
        head += "".join(f"<p class='m'>{html.escape(n)}</p>" for n in r.get("filter_notes", [])
                        if n not in r.get("warnings", []))
        form = metrology.FORM_NAMES.get(r.get("form_order"), str(r.get("form_order")))
        lam_s = r.get("lambda_s_um") or 0
        operator = (f"Surface operator: form removal ({form}), "
                    + (f"S-filter λs {lam_s:.3g} µm" if lam_s else "no S-filter")
                    + (f", L-filter λc {r['lambda_c_um']:.4g} µm" if r.get("lambda_c_um") else ", no L-filter")
                    + f"; {r['valid_fraction']:.0%} of pixels valid; parameters use valid pixels only.")
        body = table("Height (ISO 25178-2 amplitude)", [
            ("Sa", "Sa", zu, "Arithmetic mean height", "{:.4g}", "height"),
            ("Sq", "Sq", zu, "Root-mean-square height", "{:.4g}", "height"),
            ("Sp", "Sp", zu, "Maximum peak height", "{:.4g}", "height"),
            ("Sv", "Sv", zu, "Maximum pit depth", "{:.4g}", "height"),
            ("Sz", "Sz", zu, "Maximum height (Sp + Sv)", "{:.4g}", "height"),
            ("Ssk", "Ssk", "", "Skewness", "{:.4g}", "height"), ("Sku", "Sku", "", "Kurtosis", "{:.4g}", "height"),
            ("Sdq", "Sdq", "rad", "Mean surface slope (gradient magnitude)", "{:.4g}", "height"),
            ("Sdr", "Sdr_pct", "%", "Developed interfacial area ratio (needs z in µm)", "{:.4g}", "Sdr")])
        body += table("Volume parameters", [
            ("Vmp", "Vmp", "µm³/µm²", "Material volume of the peaks (above the core plane)", "{:.4g}", "volume"),
            ("Vmc", "Vmc", "µm³/µm²", "Material volume of the core", "{:.4g}", "volume"),
            ("Vvc", "Vvc", "µm³/µm²", "Void volume of the core valleys", "{:.4g}", "volume"),
            ("Vvv", "Vvv", "µm³/µm²", "Total void volume (valleys below the core plane)", "{:.4g}", "volume")])
        body += table("Feature / summit parameters", [
            ("Sds", "Sds_per_mm2", "summits/mm²", "Summit density (Isaacson summit definition)", "{:.4g}", "features"),
            ("Ssc", "Ssc_per_um", "1/µm", "Mean summit curvature", "{:.4g}", "features"),
            ("Spd", "Spd_per_mm2", "summits/mm²", "Density of significant (Wolf-pruned) peaks", "{:.4g}", "features"),
            ("Spc", "Spc_per_um", "1/µm", "Mean curvature of the significant peaks", "{:.4g}", "features"),
            ("S5p", "S5p", zu, "Mean height of the five highest peaks", "{:.4g}", "features"),
            ("S5v", "S5v", zu, "Mean depth of the five deepest pits", "{:.4g}", "features"),
            ("S10z", "S10z", zu, "S5p + S5v: peak-to-valley of the ten extremes", "{:.4g}", "features")])
        body += table("Material ratio (functional)", [
            ("Sk", "Sk", zu, "Core height (40% secant of the material ratio curve)", "{:.4g}", "functional"),
            ("Spk", "Spk", zu, "Reduced peak height", "{:.4g}", "functional"),
            ("Svk", "Svk", zu, "Reduced valley depth", "{:.4g}", "functional"),
            ("Smr1", "Smr1_pct", "%", "Material ratio at the top of the core", "{:.4g}", "functional"),
            ("Smr2", "Smr2_pct", "%", "Material ratio at the bottom of the core", "{:.4g}", "functional")])
        body += table("Spatial and spectral", [
            ("Sal", "Sal_um", "µm", "Fastest autocorrelation decay length (to 0.2)", "{:.4g}", "spatial"),
            ("Str", "Str", "", "Texture aspect ratio: ~1 isotropic, ~0 strongly directional", "{:.4g}", "spatial"),
            ("Std", "Std_deg", "°", "Texture direction (lay), from the angular spectrum", "{:.4g}", "spatial"),
            ("PSD slope β", "PSD_beta", "", "Log-log slope of the radial power spectrum", "{:.4g}", "spectral"),
            ("Fractal dimension", "fractal_dimension", "", "(8−β)/2, only when 2 < β < 4", "{:.4g}", "spectral")])
        if r.get("sigma_inst") is not None or any("Repeatability" in w for w in r.get("warnings", [])):
            body += table("Repeatability (two sweeps, same spot)", [
                ("σ_inst", "sigma_inst", zu, "Relative shape repeatability, std(Δz − mean)/√2"),
                ("Mean offset", "repeat_offset", zu, "Datum difference between the sweeps (removed above)"),
                ("Overlap", "repeat_overlap", "", "Fraction of pixels valid in both", "{:.0%}")])
        body += table("Image texture and grains (from the all-in-focus image, not heights)", [
            ("GLCM contrast", "GLCM_contrast", "", "Haralick texture, 64 grey levels"),
            ("GLCM homogeneity", "GLCM_homogeneity", "", ""), ("GLCM correlation", "GLCM_correlation", "", ""),
            ("Regions", "grain_count", "", "Watershed regions not touching the border (not verified grains)", "{:d}"),
            ("Mean region area", "mean_grain_area_um2", "µm²", ""),
            ("Mean max Feret", "mean_max_feret_um", "µm", ""), ("Mean circularity", "mean_circularity", "", "4πA/P²"),
            ("Mean lineal intercept", "mean_intercept_um", "µm", "Heyn intercept on the region boundaries"),
            ("ASTM E112-style G", "ASTM_E112_G", "", ("From the mean intercept; needs calibrated X-Y and a real "
             "grain structure"))])
        if self.chk_topo_grains.isChecked() and r is not None:
            body += table("Topographic grains", [
                ("Grain count", "topo_grain_count", "", "Summits on the surface (Isaacson, watershed basins)", "{:d}"),
                ("Mean grain area", "topo_mean_grain_area_um2", "µm²", "Mean watershed basin area")])
        prov = [("Result", r.get("label", "")), ("Result id", r.get("result_id", "—")), ("Objective", r["objective"]),
                ("Source", f"{r.get('source_id', '—')} (generation {r.get('source_generation', '—')})"),
                ("Frames used", f"{r['frames_used']} of {r['frames_in']}"),
                ("Z", z_status_text(r) + (f" · {r['z_source']}" if r.get("z_source") else "")),
                ("Lateral status", r.get("lateral_status")), ("Pixel size", f"{r['um_per_px']:.4f} µm"),
                ("Calibration revision", r.get("calibration_revision", "—")), ("Focus measure", r["measure"]),
                ("Compute backend", r.get("backend", "cpu")),
                ("Registration", f"ECC mean {r['ecc_mean']:.3f}, fallbacks {r['registration_fallbacks']}, "
                 f"unobservable {r.get('registration_unobservable', 0)}" if r.get("ecc_mean") is not None else "—"),
                ("Observed in every frame", f"{r['support_fraction']:.1%}" if r.get("support_fraction") is not None
                 else "—"),
                ("DOF ruler (experimental)", f"{r['dof_ruler_um_per_frame']:.4f} µm/frame"
                 if r.get("dof_ruler_um_per_frame") else "—"),
                ("Session file", f"schema {r['session_schema']}" if r.get("session_schema") else "—"),
                ("Time", f"{r['timings']['total']:.1f} s" if r.get("timings") else "—")]
        pv = "".join(f"<tr><td>{k}</td><td>{html.escape(str(v))}</td></tr>" for k, v in prov)
        self.report.setHtml(f"<style>{css}</style><h2>Surface report</h2>{head}"
                            f"<p class='m'>{html.escape(operator)} ISO 25178-style parameters (not a certified "
                            f"implementation).</p>{body}<h3>Provenance</h3><table>{pv}</table>{OPTICS_NOTES}")
        if r.get("timings"):
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

    def _confirm_export(self, path, what, unit, invalid) -> bool:
        """Say exactly what will be written before writing it."""
        r = self.result or {}
        txt = (f"<b>{html.escape(what)}</b><br>Result: {html.escape(str(r.get('label', '—')))} · "
               f"{html.escape(r.get('objective', '—'))}<br>Heights in <b>{unit}</b>"
               + (f" · {html.escape(z_status_text(r))}" if r else "")
               + f"<br>Pixels without a reliable depth: {html.escape(invalid)}<br>A validity mask (.mask.png) and "
               f"provenance (.json) are written beside it.<br><br>{html.escape(str(path))}")
        box = QMessageBox(QMessageBox.Question, "Export", txt, QMessageBox.Save | QMessageBox.Cancel, self)
        box.setTextFormat(Qt.RichText)
        return box.exec() == QMessageBox.Save

    def _export(self, fmt, title, name, filt, what, invalid):
        if not (self._need_result() and (p := self._save_path(title, name, filt))):
            return
        r = self.result
        if not self._confirm_export(p, what, units.label(r["z_unit"]), invalid):
            return
        prov = {k: r.get(k) for k in ("label", "result_id", "objective", "z_status", "z_orientation",
                                      "lateral_status", "form_name", "lambda_s_um", "lambda_c_um", "timestamp")}
        prov["software"] = f"MicroTopos {__version__}"
        try:
            paths = export.export_height(p, fmt, r["sl_surface"], r["um_per_px"], r["z_unit"], prov,
                                         f"MicroTopos {r['objective']} {r.get('z_status')}")
            self.notify(f"Saved {paths[0]} (+ mask and provenance)", "success")
        except OSError as e:
            self.notify(f"Export failed, nothing written: {e}", "danger")

    def _export_gsf(self):
        self._export("gsf", "Export Gwyddion field", "height.gsf", "GSF (*.gsf)", "Filtered height map, Gwyddion .gsf",
                     "filled with the median in the .gsf (Gwyddion needs a value); the mask records them")

    def _export_tiff(self):
        self._export("tif", "Export float TIFF", "height.tif", "TIFF (*.tif)", "Filtered height map, 32-bit float TIFF",
                     "NaN")

    def _export_stl(self):
        self._export("stl", "Export STL", "surface.stl", "STL (*.stl)", "Visualisation mesh (STL), not a measurement "
                     "archive", "filled with the median (display surface)")

    def _export_report(self):
        if not self._need_result():
            return
        d = QFileDialog.getExistingDirectory(self, "Folder for the report bundle",
                                             self.settings.value("export_dir", str(Path.home())))
        if d:
            import scipy
            import skimage
            try:
                files = export.write_report_bundle(d, self.result, {
                    "microtopos": __version__, "numpy": np.__version__, "opencv": cv2.__version__,
                    "scipy": scipy.__version__, "scikit-image": skimage.__version__})
                self.notify(f"Saved {len(files)} files to {d}", "success")
            except OSError as e:
                self.notify(f"Report not written (nothing was left half-done): {e}", "danger")

    def _save_session(self):
        (img, origin), parts = self.engine.mosaic_full()
        mosaic_ok = np.isfinite(img).any() and len(self.engine.mosaic.keyframes) > 1
        if self.result is None and not mosaic_ok and self.loaded_mosaic is None:
            self.notify("Nothing to save yet: no height map and no mosaic.", "warning")
            return
        p = self._save_path("Save session", "session.npz", "Session (*.npz)")
        if not p:
            return
        extra = {"um_preview": getattr(self, "um_preview", None)}
        if len(self.engine.mosaic.keyframes) >= 1:
            extra["mosaic"] = self.engine.mosaic
        if mosaic_ok:
            extra.update(mosaic_image=img, mosaic_origin=list(origin))
            if parts:
                extra.update(height_mosaic=parts[0][1], height_mosaic_origin=list(parts[0][2]),
                             height_mosaic_key=parts[0][0])
        elif self.loaded_mosaic is not None:  # re-saving an opened session keeps its mosaic
            (limg, lorig, _d), lparts, lum = self.loaded_mosaic
            if lum is not None:
                extra["um_preview"] = lum
            extra.update(mosaic_image=limg, mosaic_origin=list(lorig))
            if lparts:
                extra.update(height_mosaic=lparts[0][1], height_mosaic_origin=list(lparts[0][2]),
                             height_mosaic_key=lparts[0][0])
        try:
            export.save_session(p, self.result, {"objective": self.obj.name, "objective_spec": self.obj.to_dict(),
                                                 "measure": self.cmb_measure.currentText(),
                                                 "software": f"MicroTopos {__version__}"}, extra)
            self.notify(f"Saved {p}" + ("" if self.result is not None else " (mosaic only)"), "success")
        except OSError as e:
            self.notify(f"Session not saved: {e}", "danger")

    def _load_session(self):
        d = self.settings.value("export_dir", str(Path.home()))
        p, _ = QFileDialog.getOpenFileName(self, "Open session", d, "Session (*.npz)")
        if not p:
            return
        try:
            r, _s = export.load_session(p)  # validated before anything in the window changes
            mosaic = r.pop("mosaic", None)  # restore the full pose graph into the live engine
            if mosaic is not None:
                mmeta, marrays = mosaic.to_state_arrays()
                self.engine.load_mosaic_state(mmeta, marrays)
                self.last_mosaic_version = -1
            has = "height" in r
            if has:  # pin the optics it was made with; an unknown objective fails here with a clear message
                r["objective_spec"] = resolve_result(r).to_dict()
            if has and r["migrated"]:  # schema 1: recompute what it did not store, and say so
                if "edf_proc" not in r:
                    r["edf_proc"] = cv2.resize(r["edf"], (r["height"].shape[1], r["height"].shape[0]),
                                               interpolation=cv2.INTER_AREA)
                m = metrology.analyze(r["height"], r["edf_proc"], r["um_per_px"], r["z_unit"],
                                      resolve_result(r).lateral_cutoff_um, r.get("form_order", 1),
                                      r.get("lambda_s_um"), r.get("lambda_c_um", 0.0))
                kept = [w for w in r.get("warnings", []) if not metrology.is_metrology_warning(w)]
                r["warnings"] = kept + m.pop("warnings") + [
                    ("Opened from an older session file: the processing image and the filtered surface were "
                     "recomputed from the stored heights (the original processing image was not stored).")]
                r.update({k: v for k, v in m.items() if k not in r or k in metrology.PARAMETER_KEYS})
            if isinstance(r.get("mosaic_image"), np.ndarray):
                parts = []
                if isinstance(r.get("height_mosaic"), np.ndarray):
                    parts = [(r.get("height_mosaic_key", "um" if units.is_um(r.get("z_unit")) else "frames#loaded"),
                              r["height_mosaic"], tuple(r["height_mosaic_origin"]), 1)]
                um_loaded = r.get("um_preview")
                if not (isinstance(um_loaded, (int, float)) and np.isfinite(um_loaded) and um_loaded > 0):
                    um_loaded = None
                    if has and isinstance(r.get("preview_shape"), (list, tuple)) and len(r["preview_shape"]) == 2 and r["preview_shape"][1]:
                        um_loaded = float(r["um_per_px"]) * float(r["height"].shape[1]) / float(r["preview_shape"][1])
                self.loaded_mosaic = ((r["mosaic_image"], tuple(r["mosaic_origin"]), 1), parts, um_loaded)
                self.last_mosaic_version = -1
            if has:
                r.setdefault("result_id", uuid.uuid4().hex[:12])
                r["mosaic_generation"] = -1  # never pinned into the live mosaic
                r["label"] = f"{Path(p).name}: {r.get('label', '')}".strip(": ")
                self.result = r
                self._remember(r)
                self._show_result()
                self.tabs.setCurrentIndex(3)
            else:
                self.tabs.setCurrentIndex(1)
            self.notify(f"Opened {p}" + ("" if has else " (mosaic only)"), "success")
        except Exception as e:  # noqa: BLE001 - a damaged file is reported, the window is unchanged
            self.notify(f"Could not open session: {e}", "danger")

    # ================================================================== Slice 5: capability wiring
    def _set_color_mode(self, on: bool):
        self.color_mode = bool(on)

    # ---------------------------------------------------------------- replay transport bar
    def _replay_bar(self):
        h = QHBoxLayout(); h.setContentsMargins(6, 4, 6, 4)
        self.btn_replay_pause = QPushButton("‖ Pause")
        self.btn_replay_pause.setCheckable(True)
        self.btn_replay_pause.clicked.connect(self._replay_toggle)
        self.btn_replay_step_b = QPushButton("◀ Step −1")
        self.btn_replay_step_b.clicked.connect(lambda: self._replay_step(-1))
        self.btn_replay_step_f = QPushButton("Step +1 ▶")
        self.btn_replay_step_f.clicked.connect(lambda: self._replay_step(1))
        self.replay_seek = QSlider(Qt.Horizontal)
        self.replay_seek.setRange(0, 0)
        self.replay_seek.sliderReleased.connect(self._replay_seek_done)
        h.addWidget(self.btn_replay_pause)
        h.addWidget(self.btn_replay_step_b)
        h.addWidget(self.btn_replay_step_f)
        h.addWidget(self.replay_seek, 1)
        h.addWidget(QLabel("Speed"))
        self.cmb_replay_speed = QComboBox()
        self.cmb_replay_speed.addItems(["0.25×", "0.5×", "1×", "2×", "4×"])
        self.cmb_replay_speed.setCurrentText("1×")
        self.cmb_replay_speed.setFixedWidth(56)
        self.cmb_replay_speed.currentTextChanged.connect(self._replay_speed)
        h.addWidget(self.cmb_replay_speed)
        self.chk_replay_loop = QCheckBox("Loop")
        self.chk_replay_loop.toggled.connect(self._replay_loop)
        h.addWidget(self.chk_replay_loop)
        return h

    def _set_replay_transport_enabled(self, on: bool):
        """Pause/step/seek/speed act only on a paced replay; an exhaustive (consumer-paced) one refuses them."""
        for wdg in (self.btn_replay_pause, self.btn_replay_step_b, self.btn_replay_step_f, self.replay_seek,
                    self.cmb_replay_speed):
            wdg.setEnabled(on)
            wdg.setToolTip("" if on else "Not available: this recording is replayed frame by frame as fast as it is processed")

    def _replay_toggle(self, checked: bool):
        s = self.source
        if not isinstance(s, ReplaySource):
            return
        if checked:
            s.pause()
            self.btn_replay_pause.setText("▶ Play")
        else:
            s.resume()
            self.btn_replay_pause.setText("‖ Pause")

    def _replay_step(self, delta: int):
        s = self.source
        if isinstance(s, ReplaySource):
            s.step_frames(delta)
            self.btn_replay_pause.setChecked(True)
            self.btn_replay_pause.setText("▶ Play")

    def _replay_seek_done(self):
        s = self.source
        if isinstance(s, ReplaySource):
            s.seek(self.replay_seek.value())
            self.btn_replay_pause.setChecked(True)
            self.btn_replay_pause.setText("▶ Play")

    def _replay_speed(self, text: str):
        s = self.source
        if isinstance(s, ReplaySource):
            s.set_speed(float(text.replace("×", "")))

    def _replay_loop(self, on: bool):
        s = self.source
        if isinstance(s, ReplaySource):
            s.set_loop(on)

    # ---------------------------------------------------------------- mosaic bar
    def _mosaic_blend(self, text: str):
        self.engine.mosaic.set_blending(text)
        self.last_mosaic_version = -1

    def _mosaic_track_rotation(self, on: bool):
        self.track_rotation = bool(on)
        self.engine.mosaic.track_rotation = bool(on)
        self.last_mosaic_version = -1

    def _mosaic_rgb_mode(self, on: bool):
        self.mosaic_rgb = bool(on)
        self.last_mosaic_version = -1

    def _mosaic_add_inset(self):
        s = self.engine.snapshot()
        if s.preview is None:
            self.notify("No live image to inset yet.", "warning")
            return
        um = s.um_preview if (s.um_preview and s.um_preview > 0) else self.obj.lateral_um_per_px_full
        if not (um and um > 0):
            um = self.obj.nominal_um_per_px(s.preview.shape[1])
        x_um, y_um = s.mosaic_origin
        self.engine.mosaic.add_inset(self.obj.name, um, x_um, y_um, s.preview)
        self.last_mosaic_version = -1
        self.notify(f"Inset added: {s.preview.shape[1]}×{s.preview.shape[0]} px")

    # ---------------------------------------------------------------- height interactive tools
    def _level_mode(self, on: bool):
        self.level_picking = bool(on)
        self.level_pts = []
        self.btn_level3.setText("Finish: click 3 points" if on else "Level: pick 3 points")
        self.notify("Click 3 points on the height map to level it (local plane fit around each)." if on
                    else "3-point levelling cancelled.")

    def _on_height_clicked(self, x_px: float, y_px: float):
        if not self.level_picking:
            return
        r = self.result
        if r is None:
            return
        self.level_pts.append((float(x_px), float(y_px)))
        self.notify(f"Level point {len(self.level_pts)}/3 at ({x_px:.0f}, {y_px:.0f}) px")
        if len(self.level_pts) < 3:
            return
        self.level_picking = False
        self.btn_level3.blockSignals(True)
        self.btn_level3.setChecked(False)
        self.btn_level3.setText("Level: pick 3 points")
        self.btn_level3.blockSignals(False)
        mode = self.cmb_height.currentIndex()
        if mode == 1:
            z = r["height"].copy()
        elif mode == 2:
            z = r["confidence"].copy()
        else:
            z = r["sl_surface"].copy()
        um = r.get("um_per_px", 1.0)
        levelled = metrology.level_three_points(z, self.level_pts, radius_px=max(3, round(10 / um)))
        self.level_pts = []
        if levelled.size == 0:
            self.notify("3-point levelling produced nothing usable.", "warning")
            return
        new = dict(r)
        new["height"] = levelled
        self.result = new
        self._show_result()
        self.notify("Height map levelled through the 3 picked points.", "success")

    def _apply_mask_roi(self):
        r = self.result
        if r is None:
            return
        mode = self.cmb_height.currentIndex()
        if mode == 1:
            z = r["height"]
        elif mode == 2:
            z = r["confidence"]
        else:
            z = r["sl_surface"]
        um = r.get("um_per_px", 1.0)
        rect_px = self._roi_rect_px(self.mask_roi, um)
        try:
            masked = metrology.apply_exclusion_mask(z, rect_px)
        except ValueError as e:
            self.notify(str(e), "warning")
            return
        new = dict(r)
        new["height"] = masked
        self.result = new
        self._show_result()
        self.notify("Mask ROI excluded from the height map.", "success")

    def _measure_step_roi(self):
        r = self.result
        if r is None:
            return
        mode = self.cmb_height.currentIndex()
        if mode == 1:
            z = r["height"]
        elif mode == 2:
            self.notify("Step height needs a height map (not the confidence view).", "warning")
            return
        else:
            z = r["sl_surface"]
        um = r.get("um_per_px", 1.0)
        ra = self._roi_rect_px(self.step_roi_a, um)
        rb = self._roi_rect_px(self.step_roi_b, um)
        try:
            m = metrology.step_height_regions(z, ra, rb)
        except ValueError as e:
            self.notify(str(e), "warning")
            return
        self.lbl_height.setText(f"Step A→B: {m['step_height']:+.4g} µm  (SE ±{m['standard_error']:.4g} µm, "
                                f"{m['pixels_a']}/{m['pixels_b']} px)")
        self.notify(f"Step height {m['step_height']:+.4g} µm.", "success")

    @staticmethod
    def _roi_rect_px(roi, um):
        """ROI item rect (group/µm coords) converted to an integer pixel rect (x0, y0, x1, y1)."""
        p = roi.pos(); s = roi.size()
        return (round(p.x() / um), round(p.y() / um),
                round((p.x() + s.x()) / um), round((p.y() + s.y()) / um))

    # ================================================================== shutdown
    def closeEvent(self, ev):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("state", self.saveState())
        self.timer.stop()
        for j in list(self.jobs):  # ask every worker to stop, then wait until it has (they poll the flag)
            j.cancel = True
        t0 = time.perf_counter()
        for j in list(self.jobs):
            while j.isRunning():
                QApplication.processEvents()
                j.wait(50)
                if time.perf_counter() - t0 > 60:
                    print(f"warning: {j.label} did not stop within 60 s", flush=True)
                    t0 = float("inf")
        if isinstance(self.source, CameraSource) and getattr(self.source, "recording", False):
            self.source.stop_recording()
        self.engine.stop()
        if self.source is not None:
            self.source.stop()
        super().closeEvent(ev)
