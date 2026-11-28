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

from .. import __version__, export, metrology, response, stills, units, zstack
from .. import objectives as objlib
from ..calibration import (
    FLAT_ESTIMATED,
    CalibrationStore,
    CalStatus,
    bgr_to_rgb_u8,
    flat_source,
    radiometric_key,
    today,
    white_balance_gains,
)
from ..camera import (
    CameraSource,
    ReplaySource,
    SimulatorSource,
    camera_modes,
    list_cameras,
    opencv_threads,
    pick_resolution,
)
from ..focus import MEASURES
from ..kinematics import Override, State, guess_objective
from ..live import MIN_COMPLETENESS, LiveEngine, SweepCapture
from ..objectives import resolve, resolve_result
from ..stacking import stack_frames
from ..stage import GrblStageDriver, SimulatedStageDriver, StageDriver, capture_z_sweep, list_serial_ports
from ..superres import dwell_superres, max_useful_scale
from .objectives_dialog import ObjectivesDialog
from .theme import DARK, LIGHT, hex_rgb, stylesheet
from .widgets import (
    DARK_P99_DN,
    Banner,
    Chip,
    FlowBar,
    FlowWidget,
    ImagePanel,
    Toast,
    frame_percentile,
    section,
    stretch_levels,
)

DEFAULT_LEFT_DOCK_WIDTH = 420  # the session panel and the telemetry panel keep these widths (never below their
DEFAULT_RIGHT_DOCK_WIDTH = 340  # minimum); the middle panel, which shows the image, gets all the rest

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
<p><b>Focus breathing</b> (a small magnification change with focus) is not corrected by default. An experimental
correction (3D tab, "Correct focus breathing") measures it from the frame pairs and improves the lateral alignment of
the frames; on simulated sweeps it did not change the height error measurably. A drift larger than 1.5 % over the
sweep is limited to 1.5 % and reported as a warning.</p>
"""

STEPS = ["Camera", "Objective", "Dark && flat", "Lateral scale", "Z scale", "Inspect"]

# The profile-roughness cut-off used when the L-filter is off, in words built from the constants the code divides by
# (the λc tooltip and the profile label both say this; neither may state another rule).
DEFAULT_PROFILE_CUTOFF_TEXT = (f"the profile length / {metrology.PROFILE_TRAVERSE_CUTOFFS:g} "
                               f"({metrology.PROFILE_EVAL_CUTOFFS:g} sections of λc plus "
                               f"{metrology.PROFILE_RUN_CUTOFFS / 2:g} λc run-in and run-out)")


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


BRACKET_TIMEOUT_S = 180.0
BRACKET_SETTLE_FRAMES = 5   # settled: the last 5 frame means lie within 1 % of each other ...
BRACKET_SETTLE_REL = 0.01
BRACKET_SETTLE_MIN = 6      # ... after at least 6 frames (the MikrOkular overshoots for ~10 frames after a change)
BRACKET_SETTLE_MAX = 60     # ... or after 60 frames at most
RESPONSE_EXPOSURES, RESPONSE_FRAMES = 8, 3
HDR_EXPOSURES, HDR_FRAMES = 5, 2


def _response_job(frames, times, progress=None, cancelled=None):
    if progress:
        progress(0.1, "Solving for the camera response...")
    return response.estimate_response(frames, times)


def _hdr_job(frames, times, curve, progress=None, cancelled=None):
    if progress:
        progress(0.1, "Merging the exposure bracket...")
    m = response.merge_hdr(frames, times, curve)
    rad = m["radiance"]
    disp = response.tonemap_for_display(np.nan_to_num(rad, nan=0.0)).astype(np.float32)
    w = np.array([0.0722, 0.7152, 0.2126], np.float32)
    lum = rad @ w if rad.ndim == 3 else rad
    return {"image": disp @ w if disp.ndim == 3 else disp, "colour": disp if disp.ndim == 3 else None,
            "radiance": lum.astype(np.float32), "radiance_colour": rad if rad.ndim == 3 else None,
            "unusable_fraction": float(m["unusable"].mean()), "frames_used": len(frames), "frames_rejected": 0,
            "rl_iterations": 0, "values": "relative radiance (measured camera response; arbitrary unit)",
            "notes": [(f"{len(set(times))} exposures {min(times) * 1e3:.1f}-{max(times) * 1e3:.1f} ms, "
                       f"{float(m['unusable'].mean()):.1%} of pixels unusable in every exposure"),
                      "display: tone-mapped"]}


def _stack_job(still, psf, progress=None, cancelled=None):
    if progress:
        progress(0.1, "Aligning and merging the still frames...")
    prep = stills.prepare(still)
    r = stack_frames(prep["frames"], colour=prep["colour"], deconvolve_psf_sigma=psf)
    return {**r, "notes": prep["notes"]}


SR_CROP = 512  # super-resolution works on the centre crop: its output has scale^2 times the pixels


def _sr_job(still, psf, progress=None, cancelled=None, scale=2, pixfrac=0.7, deconvolve=True, oversampling=None):
    if progress:
        progress(0.1, "Registering dwell frames and drizzling...")
    frames = stills.prepare(still)
    h, w = frames["frames"][0].shape
    c = min(SR_CROP, h, w)
    y0, x0 = (h - c) // 2, (w - c) // 2
    crops = [f[y0:y0 + c, x0:x0 + c] for f in frames["frames"]]
    r = dwell_superres(crops, psf, scale=scale, pixfrac=pixfrac, deconvolve=deconvolve, oversampling=oversampling)
    return {**r, "input": crops[0], "notes": frames["notes"]}


def _metrology_job(r, form_order, lam_s, lam_c, progress=None, cancelled=None):
    if progress:
        progress(0.2, "Recomputing metrology...")
    return metrology.analyze(r["height"], r["edf_proc"], r["um_per_px"], r["z_unit"],
                             resolve_result(r).lateral_cutoff_um, form_order, lam_s, lam_c)


def _feature_job(r, progress=None, cancelled=None):
    """Spd/Spc/S5p/S5v/S10z and the topographic hills of the filtered surface (seconds per megapixel)."""
    if progress:
        progress(0.2, "Finding significant peaks and hills...")
    # the summit/peak noise floor comes from the measured heights before form removal, as in analyze()
    return metrology.feature_set(r["sl_surface"], r["um_per_px"], r["z_unit"],
                                 noise_floor=metrology.height_noise_floor(r["height"]))


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


SETTINGS_DIR_ENV = "MICROTOPOS_SETTINGS_DIR"


def app_settings() -> QSettings:
    """The window's saved settings (theme, objective, layout): the one place that decides where they live.

    By default the platform's native store. On Windows that is the registry, which QSettings.setPath cannot
    redirect, so a test or smoke run that only set a path read and wrote the user's real settings. When
    MICROTOPOS_SETTINGS_DIR names a folder, the settings are an ini file in that folder instead (the test suite,
    the GUI smoke run, a portable copy)."""
    folder = os.environ.get(SETTINGS_DIR_ENV)
    if folder:
        return QSettings(str(Path(folder) / "MicroTopos.ini"), QSettings.Format.IniFormat)
    return QSettings("MicroTopos", "MicroTopos")


class MainWindow(QMainWindow):
    def __init__(self, simulator_only: bool = False, replay: str | None = None, verbose: bool = False):
        super().__init__()
        opencv_threads()
        self.settings = app_settings()
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
        self._load_white_balance()
        self.engine.on_sweep = self.bridge.sweep.emit
        self.engine.on_objective_hint = self.bridge.objective_hint.emit
        self.bridge.sweep.connect(self._on_sweep)
        self.bridge.objective_hint.connect(self._on_objective_hint)
        self.source = None
        self._readback_seq: int | None = None  # CameraSource.applied_seq shown in the "Device reports" label
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
        self._replay_shown_pos: int | None = None  # the replay frame the seek slider was last set to by _tick
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
        # without this Qt sizes the docks by their size hints (the telemetry panel's is 600+ px), which left the live
        # image 480 px on a 1536 px wide screen
        self.resizeDocks([self.dock_left, self.dock_right], [DEFAULT_LEFT_DOCK_WIDTH, DEFAULT_RIGHT_DOCK_WIDTH],
                         Qt.Horizontal)
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
        h = FlowBar()
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
        self.btn_wb = QPushButton("White balance")
        self.btn_wb.setToolTip("Set the colour balance from the current view (grey world: the mean of the pixels that "
                               "are neither dark nor clipped becomes neutral). Applies to colour images only, never to "
                               "the measurement. Right-click to reset.")
        self.btn_wb.clicked.connect(self._set_white_balance)
        self.btn_wb.setContextMenuPolicy(Qt.CustomContextMenu)
        self.btn_wb.customContextMenuRequested.connect(lambda _p: self._set_white_balance(reset=True))
        h.addWidget(self.btn_wb)
        self.cmb_sr_scale = QComboBox()
        self.cmb_sr_scale.addItems(["2×", "3×", "4×"])
        self.cmb_sr_scale.setCurrentText("2×")
        self.cmb_sr_scale.setFixedWidth(52)
        self.cmb_sr_scale.currentTextChanged.connect(
            lambda t: setattr(self, "sr_scale", int(t.replace("×", ""))))
        h.addLabelled("SR scale", self.cmb_sr_scale)
        self.sl_sr_pixfrac = QSlider(Qt.Horizontal)
        self.sl_sr_pixfrac.setRange(3, 10)
        self.sl_sr_pixfrac.setValue(7)
        self.sl_sr_pixfrac.setFixedWidth(70)
        self.sl_sr_pixfrac.valueChanged.connect(lambda v: setattr(self, "sr_pixfrac", v / 10))
        h.addLabelled("pixfrac", self.sl_sr_pixfrac)
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
        return h.layout()

    def _mosaic_bar(self):
        h = FlowBar()
        self.cmb_mview = QComboBox()
        self.cmb_mview.addItems(["Image", "Image + height overlay", "Height only"])
        self.cmb_mview.setCurrentIndex(1)
        self.cmb_mview.setToolTip("Every height map stays pinned to its place in the mosaic; overlapping maps are "
                                  "levelled to each other (least squares) and blended")
        self.cmb_mview.currentIndexChanged.connect(lambda: setattr(self, "last_mosaic_version", -1))
        h.addWidget(self.cmb_mview)
        self.sl_hop = QSlider(Qt.Horizontal)
        self.sl_hop.setRange(10, 100)
        self.sl_hop.setValue(55)
        self.sl_hop.setFixedWidth(110)
        self.sl_hop.valueChanged.connect(lambda: setattr(self, "last_mosaic_version", -1))
        h.addLabelled("Opacity", self.sl_hop)
        self.cmb_mblend = QComboBox()
        self.cmb_mblend.addItems(["feather", "multiband"])
        self.cmb_mblend.setCurrentText("feather")
        self.cmb_mblend.currentTextChanged.connect(self._mosaic_blend)
        h.addLabelled("Blend", self.cmb_mblend)
        self.chk_rot_track = QCheckBox("Track rotation")
        self.chk_rot_track.toggled.connect(self._mosaic_track_rotation)
        h.addWidget(self.chk_rot_track)
        self.chk_mosaic_rgb = QCheckBox("Colour mosaic")
        self.chk_mosaic_rgb.toggled.connect(self._mosaic_rgb_mode)
        h.addWidget(self.chk_mosaic_rgb)
        self.lbl_hlegend = QLabel()
        self.lbl_hlegend.setObjectName("mono")
        h.addWidget(self.lbl_hlegend, 1)
        h2 = FlowBar()  # second bar: actions and status
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
        v.addLayout(h.layout())
        v.addLayout(h2.layout())
        return v

    def _edf_bar(self):
        h = FlowBar()
        self.chk_grains = QCheckBox("Overlay grain boundaries")
        self.chk_grains.toggled.connect(self._show_result)
        h.addWidget(self.chk_grains)
        self.cmb_edf = QComboBox()
        self.cmb_edf.addItems(["All-in-focus", "Still image (stack / super-resolution)"])
        self.cmb_edf.currentIndexChanged.connect(self._show_result)
        h.addWidget(self.cmb_edf)
        self.lbl_edf = QLabel()
        self.lbl_edf.setObjectName("muted")
        h.addWidget(self.lbl_edf, 1)
        return h.layout()

    def _height_bar(self):
        h = FlowBar()
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
        self.btn_level3.setToolTip("Click 3 points on the height map; the measured height is levelled through them "
                                   "and every layer and parameter is recomputed")
        self.btn_level3.toggled.connect(self._level_mode)
        h.addWidget(self.btn_level3)
        self.btn_mask = QPushButton("Exclude ROI")
        self.btn_mask.setToolTip("First press shows the pink rectangle to place; the next press excludes its pixels "
                                 "from the measured height and recomputes every parameter")
        self.btn_mask.clicked.connect(lambda: self._tool_pressed("mask"))
        h.addWidget(self.btn_mask)
        self.btn_step = QPushButton("Step height A→B")
        self.btn_step.setToolTip("First press shows the two orange rectangles to place on the two terraces; the next "
                                 "press measures the step between them (common-tilt removed)")
        self.btn_step.clicked.connect(lambda: self._tool_pressed("step"))
        h.addWidget(self.btn_step)
        self.btn_tool_cancel = QPushButton("Cancel tool")
        self.btn_tool_cancel.clicked.connect(self._disarm_tools)
        self.btn_tool_cancel.hide()
        h.addWidget(self.btn_tool_cancel)
        self.btn_tools_reset = QPushButton("Reset height tools")
        self.btn_tools_reset.setToolTip("Restore the reconstructed height (undo levelling and exclusions)")
        self.btn_tools_reset.setEnabled(False)
        self.btn_tools_reset.clicked.connect(self._reset_height_tools)
        h.addWidget(self.btn_tools_reset)
        self.chk_topo_grains = QCheckBox("Feature parameters")
        self.chk_topo_grains.setToolTip("Compute Spd, Spc, S5p, S5v, S10z and the topographic hills (Wolf-pruned peaks, "
                                        "5 % of Sz) of the filtered surface; takes seconds per megapixel")
        self.chk_topo_grains.toggled.connect(self._show_result)
        h.addWidget(self.chk_topo_grains)
        return h.layout()

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
        bar = FlowBar()
        self.sl_exag = QSlider(Qt.Horizontal)
        self.sl_exag.setRange(1, 100)
        self.sl_exag.setValue(10)
        self.sl_exag.setMinimumWidth(120)
        self.sl_exag.valueChanged.connect(self._update_3d)
        bar.addLabelled("Z exaggeration", self.sl_exag)
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["viridis", "cividis", "inferno", "plasma", "magma", "turbo"])
        self.cmb_cmap.currentIndexChanged.connect(self._update_3d)
        bar.addLabelled("Colour map", self.cmb_cmap)
        self.cmb_3d_backend = QComboBox()
        self.cmb_3d_backend.addItems(["CPU", "OpenCL"])
        self.cmb_3d_backend.currentTextChanged.connect(lambda t: setattr(self.zst, "backend", t.lower()))
        bar.addLabelled("Backend", self.cmb_3d_backend)
        self.chk_3d_breathing = QCheckBox("Correct focus breathing (experimental)")
        self.chk_3d_breathing.setToolTip("Estimates the magnification change with focus from the frame pairs and folds it "
                                         "into the frame alignment. Experimental: it improves the lateral alignment, but "
                                         "did not change the simulated height error measurably.")
        self.chk_3d_breathing.toggled.connect(lambda on: setattr(self.zst, "correct_breathing", on))
        bar.addWidget(self.chk_3d_breathing)
        self.chk_3d_tiling = QCheckBox("Strip-wise reconstruction")
        self.chk_3d_tiling.setToolTip("Reconstruct in horizontal strips when the whole stack would not fit in free "
                                      "memory: bounds peak memory, gives the same result.")
        self.chk_3d_tiling.setChecked(True)
        self.chk_3d_tiling.toggled.connect(lambda on: setattr(self.zst, "allow_tiling", on))
        bar.addWidget(self.chk_3d_tiling)
        bar.addWidget(self.lbl_3d, 1)
        lay.addLayout(bar.layout())


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
        # Filled per camera in _on_source with the sizes that camera offers (camera.camera_modes); a lateral
        # calibration applies only to the camera mode it was made in (LateralCalibration.covers).
        self.cmb_res.setToolTip("Capture size. Full resolution is used for Z-stacks; live analysis uses a 640 px preview")
        self.cmb_res.currentIndexChanged.connect(self._on_resolution)
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
        self.chk_linear.setToolTip("Off: Rec.601 luma of the camera's gamma-encoded output. On: linear luminance, "
                                   "through this camera's measured response when there is one, else assuming "
                                   "gamma 2.2")
        self.btn_measure_resp = self._btn("Measure camera response", self._measure_response,
                                          "Keep the view still: the exposure is stepped through its range and the "
                                          "camera's tone curve is measured from the frames (stored per camera and "
                                          "capture size). Refused when the frames do not agree with the exposures.")
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
        drv_row = QHBoxLayout()
        drv_row.addWidget(QLabel("Stage driver"))
        self.cmb_stg = QComboBox()
        # "None" first: most setups focus and pan by hand, and the stage controls only appear once a driver is chosen
        self.cmb_stg.addItems(["None", "Simulated stage", "GRBL serial port"])
        self.cmb_stg.currentTextChanged.connect(self._stage_driver_changed)
        drv_row.addWidget(self.cmb_stg, 1)
        self.stage_details = QWidget()
        stg = QGridLayout(self.stage_details)
        stg.setContentsMargins(0, 0, 0, 0)
        stg.addWidget(QLabel("Serial port"), 1, 0)
        port_row = QHBoxLayout()
        self.cmb_stg_port = QComboBox()
        self.cmb_stg_port.setEnabled(False)
        port_row.addWidget(self.cmb_stg_port, 1)
        self.btn_stg_rescan = self._btn("Rescan", self._stage_rescan_ports, "List the serial ports again (a "
                                        "controller plugged in after MicroTopos started)")
        self.btn_stg_rescan.setEnabled(False)
        port_row.addWidget(self.btn_stg_rescan)
        stg.addLayout(port_row, 1, 1)
        self.btn_stg_connect = self._btn("Connect", self._stage_connect)
        stg.addWidget(self.btn_stg_connect, 2, 0)
        self.lbl_stg_pose = QLabel("stage: disconnected")
        self.lbl_stg_pose.setObjectName("muted")
        stg.addWidget(self.lbl_stg_pose, 2, 1)
        jog = FlowBar()  # wraps: five buttons side by side set the left panel's minimum width
        for name, dx, dy in (("X −10 µm", -10.0, 0.0), ("X +10 µm", 10.0, 0.0),
                             ("Y −10 µm", 0.0, -10.0), ("Y +10 µm", 0.0, 10.0)):
            # clicked(bool) passes `checked` as the first positional argument: it must not land in dx
            b = self._btn(name, lambda _checked=False, dx=dx, dy=dy: self._stage_jog(dx, dy))
            b.setEnabled(False)
            self.stage_jog_buttons.append(b)
            jog.addWidget(b)
        self.btn_stg_zero = self._btn("Zero X/Y/Z", self._stage_zero)
        self.btn_stg_zero.setEnabled(False)
        jog.addWidget(self.btn_stg_zero)
        stg.addLayout(jog.layout(), 3, 0, 1, 2)
        sweep = FlowBar()
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
        stg.addLayout(sweep.layout(), 4, 0, 1, 2)
        stage_hint = QLabel("Move a real (GRBL) or simulated stage. Jog moves are relative to the last "
                            "reported pose. Zero makes the current position the origin of a GRBL stage; the "
                            "simulated stage keeps the simulator's coordinates, so Zero moves it back to "
                            "their origin. Z sweep: the stage runs through the "
                            "Z range while the camera frames, labelled with the stage's Z, are captured "
                            "and reconstructed (the simulated stage drives the simulator's focus; the GRBL "
                            "path is verified against a simulated controller only).")
        stage_hint.setObjectName("muted")
        stage_hint.setWordWrap(True)
        stg.addWidget(stage_hint, 0, 0, 1, 2)
        self.stage_details.setVisible(False)
        self.stage_box = section("Stage", drv_row, self.stage_details)
        return section("1. Camera", self.cmb_source, self.cmb_res, self.chk_autoexp, exp, rot, self.chk_linear,
                       self.btn_measure_resp, self.lbl_readback, self.demo_box, self.stage_box,
                       hint="Pick the microscope camera. Exposure is locked and in-camera sharpening is switched off, "
                            "because both would corrupt focus measurements.")

    def _stage_driver_changed(self, text: str):
        grbl = text.startswith("GRBL")
        self.cmb_stg_port.setEnabled(grbl)
        self.btn_stg_rescan.setEnabled(grbl)
        if grbl:
            self._stage_rescan_ports()
        if text.startswith("None") and self.stage_driver is not None:
            self._stage_disconnect()
        self.stage_details.setVisible(not text.startswith("None") or self.stage_driver is not None)

    def _stage_rescan_ports(self):
        """Fill the serial-port list now (ports appear and disappear as controllers are plugged in)."""
        keep = self.cmb_stg_port.currentText()
        self.cmb_stg_port.clear()
        self.cmb_stg_port.addItems(list_serial_ports())
        if keep:
            self.cmb_stg_port.setCurrentText(keep)

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
            if not poses:
                self.notify("Z sweep cancelled before the first stage position; nothing was captured.", "warning")
                return
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
        self.btn_flat_est = self._btn("Estimate flat while panning", self._toggle_flat_estimate,
                                      "No blank at hand: start, then pan the specimen slowly in two directions over "
                                      "2-3 field widths. The shading is separated from the moving specimen and stored "
                                      "as this objective's flat, labelled 'estimated'. Needs a locked exposure.")
        self.btn_clear_rad = self._btn("Clear for this objective", self._clear_radiometric)
        self.lbl_rad = QLabel()
        self.lbl_rad.setObjectName("muted")
        self.lbl_rad.setWordWrap(True)
        return section("3. Dark & flat", self.btn_dark, self.btn_flat, self.btn_flat_est, self.btn_clear_rad,
                       self.lbl_rad,
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
        self.btn_stack = self._btn("Stack still frames", self._run_stack,
                                   "Align the last still frames and merge them, rejecting outliers: lower noise on "
                                   "the camera's own pixel grid. With 'Deconvolve' on (SR strip), Richardson-Lucy "
                                   "sharpening up to the optics' resolution follows.")
        self.btn_hdr = self._btn("HDR still (exposure bracket)", self._run_hdr,
                                 "Keep the view still: frames at several exposures are merged into one relative "
                                 "radiance image through the measured camera response (needs one for this camera "
                                 "and capture size), so shiny areas that clip and dark areas both keep detail")
        self.btn_export_still = self._btn("Export still image…", self._export_still,
                                          "Save the shown still image (stack / super-resolution): 32-bit float TIFF "
                                          "of the luma, 16-bit RGB TIFF of the colour when there is one, and "
                                          "provenance JSON")
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
        self.sp_lc.setToolTip("L-filter nesting index λc (removes longer wavelengths, i.e. waviness). 0 = off. "
                              "Also the cut-off of the profile roughness on the Height tab; "
                              f"at 0 that cut-off is {DEFAULT_PROFILE_CUTOFF_TEXT}")
        self.sp_lc.valueChanged.connect(lambda _v: self._update_profile())
        f = QFormLayout()
        f.addRow("Mode", self.cmb_override)
        f.addRow("Focus measure", self.cmb_measure)
        f.addRow("Form removal order", self.sp_form)
        f.addRow("S-filter λs", self.sp_ls)
        f.addRow("L-filter λc", self.sp_lc)
        self.btn_recompute = self._btn("Recompute metrology", self._recompute)
        return section("6. Inspect", f, self.btn_manual, self.btn_repeat, self.btn_recompute, self.btn_stack,
                       self.btn_sr, self.btn_hdr, self.btn_export_still,
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

    def _on_resolution(self, _idx: int):
        """The user picked a capture size: remember it for the next launch and reopen the camera at that size."""
        size = self.cmb_res.currentText()
        if size:
            self.settings.setValue("resolution", size)
            self._on_source(self.cmb_source.currentIndex())

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
            modes = camera_modes(info)
            self.cmb_res.blockSignals(True)
            self.cmb_res.clear()
            self.cmb_res.addItems(modes)
            self.cmb_res.setCurrentText(pick_resolution(modes, self.settings.value("resolution", None)))
            self.cmb_res.blockSignals(False)
            w, h = (int(v) for v in self.cmb_res.currentText().split("x"))
            self.sim_source = None
            self.source = CameraSource(info, w, h, 25)
        self.source.start()
        self.engine.set_source(self.source)
        if isinstance(self.stage_driver, SimulatedStageDriver) and self.stage_driver.sim_source is not self.sim_source:
            self._stage_disconnect()  # the simulated stage moves the simulator it was created for
        if self.stage_driver is None:  # offer the simulated stage with the simulator, no stage with a camera
            cur = self.cmb_stg.currentText()
            if self.sim_source is not None and cur == "None":
                self.cmb_stg.setCurrentText("Simulated stage")
            elif self.sim_source is None and cur == "Simulated stage":
                self.cmb_stg.setCurrentText("None")
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
        self.cmb_res.setVisible(isinstance(self.source, CameraSource))  # a capture size means nothing to a replay
        self.replay_widget.setVisible(isinstance(self.source, ReplaySource))
        if isinstance(self.source, ReplaySource):
            self.replay_seek.setRange(0, max(0, self.source.total_frames - 1))
            self._replay_shown_pos = None  # a new player: the slider follows it from its first frame
            self._set_replay_transport_enabled(self.source.transport_supported)
            # a new player starts with the speed and looping the controls show, and playing
            self.source.set_loop(self.chk_replay_loop.isChecked())
            if self.source.transport_supported:
                self.source.set_speed(float(self.cmb_replay_speed.currentText().replace("×", "")))
            self._replay_sync_button()
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

    # ---------------------------------------------------------------- exposure brackets (response, HDR)
    def _bracket_camera_ok(self, what: str) -> bool:
        if not isinstance(self.source, CameraSource) or self.source.exposure_range() is None:
            self.notify(f"{what} needs a real camera whose exposure time can be set.", "warning")
            return False
        if getattr(self, "_brk", None) is not None:
            self.notify("An exposure bracket is already running.", "warning")
            return False
        return True

    def _bracket(self, exposures: list[int], n_frames: int, on_done, label: str):
        """Step the camera through `exposures` (device units, 100 us), wait at each until the brightness has
        settled (BRACKET_SETTLE_*), keep `n_frames` new frames, then restore the user's exposure and call
        on_done(frames (uint8 BGR), exposure times in s). Runs on GUI timers; never blocks."""
        self._brk = {"exp": list(exposures), "i": 0, "n": n_frames, "frames": [], "times": [], "state": "set",
                     "seq": -1, "means": [], "t0": time.time(), "on_done": on_done, "label": label,
                     "gen": self.engine.source_generation, "override": self.engine.sm.override}
        # the brightness steps are not motion: motion detection is paused, so they start no sweep, cut or mosaic
        # event, and the user's override comes back afterwards
        self.engine.sm.override = Override.PAUSE
        self.btn_measure_resp.setEnabled(False)
        self.btn_hdr.setEnabled(False)
        QTimer.singleShot(0, self._bracket_step)

    def _bracket_end(self, error: str | None = None):
        b, self._brk = self._brk, None
        self.engine.sm.override = b["override"]
        self.btn_measure_resp.setEnabled(True)
        self.btn_hdr.setEnabled(True)
        self._apply_exposure()  # the user's exposure (or automatic exposure) again
        if error:
            self.notify(f"{b['label']} stopped: {error}", "warning")
            return
        b["on_done"](b["frames"], b["times"])

    def _bracket_step(self):
        b = getattr(self, "_brk", None)
        if b is None:
            return
        if self.engine.source_generation != b["gen"] or not isinstance(self.source, CameraSource):
            self._bracket_end("the camera changed")
            return
        if time.time() - b["t0"] > BRACKET_TIMEOUT_S:
            self._bracket_end(f"no settled frames within {BRACKET_TIMEOUT_S:.0f} s")
            return
        item = self.source.latest()
        e = b["exp"][b["i"]]
        if b["state"] == "set":
            self.source.set_manual_exposure(e)
            b.update(state="settle", means=[], seq=item[2] if item else -1)
        elif item is not None and item[2] != b["seq"]:
            b["seq"] = item[2]
            if b["state"] == "settle":
                m = b["means"]
                m.append(float(item[0][::8, ::8].mean()))
                tail = m[-BRACKET_SETTLE_FRAMES:]
                if (len(m) >= BRACKET_SETTLE_MIN and max(tail) - min(tail) <= BRACKET_SETTLE_REL * max(np.mean(tail), 1.0)
                        or len(m) >= BRACKET_SETTLE_MAX):
                    b["state"] = "collect"
            else:
                b["frames"].append(np.ascontiguousarray(item[0]))
                b["times"].append(e * 1e-4)
                if len(b["frames"]) >= b["n"] * (b["i"] + 1):
                    b["i"] += 1
                    if b["i"] == len(b["exp"]):
                        self._bracket_end()
                        return
                    b["state"] = "set"
        self.lbl_job.setText(f"{b['label']}: exposure {b['i'] + 1}/{len(b['exp'])} ({e / 10:.1f} ms, {b['state']})")
        QTimer.singleShot(30, self._bracket_step)

    def _exposure_series(self, n: int) -> list[int]:
        lo, hi = self.source.exposure_range()
        return sorted({round(float(v)) for v in np.geomspace(lo, hi, n)})

    def _measure_response(self):
        if not self._bracket_camera_ok("Measuring the camera response"):
            return
        self.notify("Keep the view still: stepping the exposure through its range…")
        self._bracket(self._exposure_series(RESPONSE_EXPOSURES), RESPONSE_FRAMES, self._on_response_frames,
                      "Camera response")

    def _on_response_frames(self, frames, times):
        shape = frames[0].shape[:2]
        cam = self.source.camera_name
        small = [f[::2, ::2] for f in frames]  # the curve needs levels, not detail: half size is ample
        job = Job(_response_job, small, times, label="Camera response")

        def done(curve):
            self.store.set_response(cam, shape, curve)
            self.engine.forget_response()
            self.notify(f"Camera response measured for {cam} at {shape[1]}x{shape[0]}: gamma-equivalent "
                        f"{curve.gamma_equivalent():.2f}, consistency {curve.inconsistency_dn:.1f} DN.", "success")
        self._start_job(job, done)

    def _run_hdr(self):
        if not self._bracket_camera_ok("HDR"):
            return
        item = self.source.latest()
        shape = item[0].shape[:2] if item else None
        curve = self.store.get_response(self.source.camera_name, shape) if shape else None
        if curve is None:
            self.notify("HDR needs a measured response of this camera at this capture size: use 'Measure camera "
                        "response' on the Camera page first (it is refused for cameras whose exposure setting does "
                        "not give the brightness it states).", "warning")
            return
        self.notify("Keep the view still: taking an exposure bracket…")
        self._bracket(self._exposure_series(HDR_EXPOSURES), HDR_FRAMES,
                      lambda f, t: self._on_hdr_frames(f, t, curve), "HDR")

    def _on_hdr_frames(self, frames, times, curve):
        h, w = frames[0].shape[:2]
        cal = self.store.get(self.obj)
        um = cal.um_per_px(w, self.obj, h)
        job = Job(_hdr_job, frames, times, curve, label="HDR merge")
        self._start_job(job, lambda r: self._on_sr({**r, "kind": "hdr", "um_per_px": um, "scale": 1}))

    def _apply_exposure(self):
        auto = self.chk_autoexp.isChecked()
        self.sl_exp.setEnabled(not auto)
        self.settings.setValue("exposure", self.sl_exp.value())
        self.engine.exposure = None if auto else float(self.sl_exp.value())
        if isinstance(self.source, CameraSource):
            # While streaming, the capture thread applies this with its next frame; _tick refreshes the readback
            # when the device's answer arrives (CameraSource.applied_seq), so the label never shows a stale report.
            self.source.set_manual_exposure(None if auto else self.sl_exp.value())
            self._show_readback()
        self.engine._corrector_key = None

    def _show_readback(self):
        if isinstance(self.source, CameraSource):
            rb = dict(self.source.readback)
            rb.update(self.source.applied)
            self._readback_seq = self.source.applied_seq
            err = f"  Error: {self.source.error}" if self.source.error else ""
            self.lbl_readback.setText("Device reports: " + ", ".join(f"{k}={v}" for k, v in rb.items()) + err)
        else:
            self._readback_seq = None
            self.lbl_readback.setText("Simulator: 640×360, randomised noise, MJPEG, tremor and focus breathing.")

    def _refresh_readback(self):
        """Show the device's answer as soon as the capture thread has applied a control request."""
        if isinstance(self.source, CameraSource) and self.source.applied_seq != self._readback_seq:
            self._show_readback()

    def _on_objective(self, name: str):
        obj = self.objset.get(name)
        if obj is None:  # the combo is being refilled
            return
        self.obj = obj
        self.settings.setValue("objective", name)
        # a mosaic made with another objective restarts (keeping insets as the checkbox says, plus the old mosaic as
        # an overview inset); one made with this objective (an opened session) is kept and tracked into again
        restarted = self.engine.set_objective(self.obj, restart_mosaic=True,
                                              keep_insets=self.chk_keep_insets.isChecked())
        if self.sim_source is not None:
            self.sim_source.set_objective(self.obj)
        self.sp_ls.blockSignals(True)
        self.sp_ls.setValue(self.obj.lateral_cutoff_um)
        self.sp_ls.blockSignals(False)
        self._ls_user_set = False
        self._cancel_rotation_measure()
        self._refresh_checklist()
        self.notify(f"Objective {name}. " + ("Mosaic restarted for the new magnification." if restarted
                                              else "The mosaic was made with this objective: mapping continues."))

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

    def _toggle_flat_estimate(self):
        if self.engine.flat_estimate_progress() is not None or self.engine._flat_req is not None:
            self.engine.cancel_flat_estimate("cancelled")
            self.btn_flat_est.setText("Estimate flat while panning")
            return
        if self.sim_source is None and self.chk_autoexp.isChecked():
            self.notify("Lock the exposure first (untick Automatic exposure): automatic exposure changes the "
                        "brightness between frames, which the estimate would take for shading.", "warning")
            return
        self.engine.start_flat_estimate()
        self.btn_flat_est.setText("Cancel flat estimate")
        self.notify("Pan the specimen slowly in two directions over 2-3 field widths.")

    def _clear_radiometric(self):
        if self._reject_simulator_cal():
            return
        c = self.store.get(self.obj)
        c.darks.clear()
        c.flats.clear()
        c.flat_sources.clear()
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
        prev_h, prev_w = self.p_live.data.shape[:2]
        full = (val / px) * prev_w / 1920.0
        c = self.store.get(self.obj)
        c.lateral_um_per_px_full = full
        c.lateral_frame = (int(prev_w), int(prev_h))  # the camera mode this scale covers (same aspect ratio)
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
        c.lateral_frame = None
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
        if not self._supersede(old, new):
            self.notify(f"Z not applied: {old.get('label') or 'the result'} was closed while it was rescaled.",
                        "warning")
            return
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
        fh, fw = (live.frame_shape[0], live.frame_shape[1]) if live.frame_shape[0] else (None, 1920)
        lat_status = c.lateral_status_for(fw, fh)  # CALIBRATED only where it covers the current camera mode
        lat_ok = lat_status is CalStatus.CALIBRATED
        states = [
            ("Simulator" if sim else ("Live" if cam_ok else "No frames"), "info" if sim else ("success" if cam_ok else "danger")),
            (f"{self.obj.magnification:g}×", "success"),
            ("Done" if rad else ("Partial" if rkey in c.darks or rkey in c.flats else "Optional"),
             "success" if rad else "warning"),
            ("Calibrated" if lat_ok else "Nominal", "success" if lat_ok else "warning"),
            ("Estimate" if c.z_um_per_frame else "Frames", "warning"),
            ("Ready" if cam_ok else "Waiting", "success" if cam_ok else "neutral"),
        ]
        for chip, (txt, kind) in zip(self.step_chips, states):
            chip.set(txt, kind)
        nxt = next((i for i, (_, k) in enumerate(states) if k in ("danger",)), None)
        for i, b in enumerate(self.step_btns):
            b.setToolTip("Next step" if i == nxt else "")
        um_full = c.um_per_px(fw, self.obj, fh) * fw / 1920.0  # per pixel at 1920 px width, for this camera mode
        mode_note = ""
        if c.lateral_um_per_px_full and not lat_ok:
            mode_note = (f"Not calibrated for this camera mode ({fw}x{fh}): the calibration was measured at "
                         f"{c.lateral_frame[0]}x{c.lateral_frame[1]}, a different aspect ratio. The nominal scale "
                         "is used; calibrate again in this mode.")
        self.chip_xy.set(f"XY {'CAL' if lat_ok else 'NOMINAL'} {um_full:.3f} µm/px",
                         "success" if lat_ok else "warning",
                         mode_note or c.lateral_source
                         or "Nameplate optics only; calibrate with a stage micrometer (step 4)")
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
            f"Pixel {um_full:.3f} µm at 1920 px ({lat_status.value.lower()})")
        lines = [f"{k}: dark {'✓' if k in c.darks else '—'}, flat "
                 + (("✓ (estimated from panning)" if flat_source(c, k).startswith(FLAT_ESTIMATED) else "✓")
                    if k in c.flats else "—")
                 for k in sorted(set(c.darks) | set(c.flats))]
        self.lbl_rad.setText("\n".join(lines) or "Nothing recorded for this objective.")
        self.lbl_lat.setText(f"{lat_status.value}: {c.lateral_source or 'nominal'}"
                             + (f" ({c.lateral_date})" if c.lateral_date else "")
                             + (f"\n{mode_note}" if mode_note else ""))
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
        self._refresh_readback()
        s = self.engine.snapshot()
        if s.preview is None:
            return
        cal = self.store.get(self.obj)
        # the preview covers the full field of view; the simulator's truth scale is the nominal one
        pw = s.preview.shape[1]
        um_prev = (self.obj.nominal_um_per_px(pw) if self.sim_source is not None
                   else cal.um_per_px(pw, self.obj, s.preview.shape[0]))
        self.um_preview = um_prev
        if self.chk_stretch.isChecked():
            self._stretch = stretch_levels(s.preview, self._stretch)
            levels = self._stretch
        else:
            self._stretch = None
            levels = (0, 255)
        if self.chk_rgb.isChecked() and s.preview_rgb is not None and s.preview_rgb.shape[:2] == s.preview.shape:
            # colour view of the same preview; the stretch (display only) maps the same DN range onto 0-255
            lo, hi = levels
            rgb = np.clip((s.preview_rgb.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255)
            self.p_live.set_image(rgb.astype(np.uint8), um_prev, unit="DN", rgb=True)
        else:
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
            self._replay_sync_button()
            # The slider follows every frame the player publishes, playing or stepped while paused. It is left alone
            # while the user drags it and while the position has not changed, so a paused seek that has not been
            # published yet does not pull the slider back to the frame before it.
            pos = self.source.position
            if pos != self._replay_shown_pos and not self.replay_seek.isSliderDown():
                self._replay_shown_pos = pos
                self.replay_seek.blockSignals(True)
                self.replay_seek.setValue(min(self.replay_seek.maximum(), pos))
                self.replay_seek.blockSignals(False)
        self.lbl_motion.setText(f"shift {s.shift_px:5.2f} px · corr {s.response:4.2f} · sharpness z {s.z_score:+5.1f}"
                                + ("\nMosaic tracking lost: move back over a mapped area" if s.mosaic_lost else ""))
        if s.brenner_hist:
            self.spark_curve.setData(np.asarray(s.brenner_hist, float))
        fp = self.engine.flat_estimate_progress()
        if fp is not None:
            self.lbl_job.setText(f"Estimating flat: {fp['pairs']} pairs, {fp['travel']:.1f} fields of travel"
                                 + (f" · still needs: {'; '.join(fp['missing'])}" if fp["missing"] else ""))
        elif self.engine.flat_status and self.engine.flat_status != getattr(self, "_flat_status_shown", None):
            self._flat_status_shown = self.engine.flat_status
            self.btn_flat_est.setText("Estimate flat while panning")
            self.notify(self.engine.flat_status, "warning" if "cancelled" in self.engine.flat_status else "success")
            self._refresh_checklist()
        cp = self.engine.calibration_progress()
        if cp:
            self.lbl_job.setText(f"Recording {cp[0]}: {cp[1]}/{cp[2]} frames")
        elif getattr(self, "_cal_was", None):
            if not self.engine.calib_error:
                self.notify(f"{self._cal_was[0].capitalize()} frame recorded and saved.", "success")
            self._refresh_checklist()
        self._cal_was = cp
        if (s.mosaic_version != self.last_mosaic_version or s.mosaic_hold != getattr(self, "_mosaic_hold_shown", None)) \
                and self.tabs.currentIndex() == 1:
            self.last_mosaic_version = s.mosaic_version
            self._mosaic_hold_shown = s.mosaic_hold
            (img, origin, d), parts, nkf, npatch, um_view = self.engine.mosaic_view()
            showing_loaded = nkf <= 1 and self.loaded_mosaic is not None
            if showing_loaded:  # nothing mapped yet: show the opened session's mosaic
                (img, origin, d), parts, um_loaded = self.loaded_mosaic
                m_um = um_loaded or um_prev
            else:  # the mosaic's own scale (an opened one may be another objective's); the live one until it has one
                m_um = um_view or um_prev
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
                if self.mosaic_rgb and not showing_loaded:
                    rgb = self.engine.mosaic_image_rgb()  # same box and origin as the grey view
                    if rgb is not None:
                        rgba = rgb
                self.p_mosaic.set_image(rgba, m_um * d, rgb=True)
                self.mosaic_origin = origin
                self._draw_height_overlay(parts if mode == 1 else [], origin, m_um)
            span = (img.shape[1] * d * m_um, img.shape[0] * d * m_um)
            self.lbl_mosaic.setText(f"{nkf} keyframes · {npatch} height patches · {span[0]:.0f} × {span[1]:.0f} µm"
                                    + (f" · shown at 1/{d}" if d > 1 else "")
                                    + (f" · not tracking: {s.mosaic_hold} (Reset mosaic to map with the current "
                                       "objective)" if s.mosaic_hold and not showing_loaded else ""))
        if self.tabs.currentIndex() == 1:
            showing_loaded = self.loaded_mosaic is not None and self.engine.mosaic_image()[2] <= 1
            if showing_loaded or s.mosaic_hold:  # the camera is not in this mosaic
                self.cam_box.hide()
            else:
                self._draw_camera_box(s, getattr(self, "mosaic_um", um_prev))
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
                   + (" (enough)" if pct >= MIN_COMPLETENESS else f" (needs {MIN_COMPLETENESS:.0%})"))
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
        if not cap.usable and self.engine.sm.override != Override.Z and not force:
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
        um = obj.nominal_um_per_px(w) if cap.simulated else cal.um_per_px(w, obj, cap.shape[0])
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
        meta = {"z_status": zstat, "lateral_status": cal.lateral_status_for(w, cap.shape[0]).value if not cap.simulated else "SIMULATED",
                "pose": cap.pose, "capture_shape": cap.shape, "result_id": uuid.uuid4().hex[:12],
                "source_id": cap.source_id, "source_generation": cap.source_generation,
                "mosaic_generation": cap.mosaic_generation, "calibration_revision": cal.revision,
                "preview_shape": list(cap.preview_shape), "exposure": cap.exposure, "linearize": cap.linearize,
                "white_balance": list(cap.white_balance) if cap.white_balance else None,
                "linearization": cap.linearization,
                "z_source": zsrc or (cal.z_source if zstat == CalStatus.ESTIMATED.value else ""),
                "sweep_seq_range": [cap.seqs[0], cap.seqs[-1]] if cap.seqs else None}
        sample_pos = np.asarray(cap.seqs, float) if cap.seqs and z is None else None
        job = Job(zstack.reconstruct, cap.frames, z, obj, um, unit, zcal, st, label="Reconstruction",
                  sample_pos=sample_pos, z_sign=sign if z is None else None, color_frames=cap.color_frames)
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
        self.prev_result = self.result
        self._set_result(r)
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
            self._set_result(self.history[i])
            self._show_result()

    def _set_result(self, r: dict):
        """Make `r` the displayed result: the one step every path that replaces the result goes through.

        Height tools act on the displayed map: when its height is not the one on screen (a new reconstruction, an
        opened session, another history entry, an edited or rescaled height), armed tools and picked levelling
        points were placed on the previous map and must not carry over. A recompute that keeps the height (a
        filter change) keeps them."""
        cur = self.result
        if cur is None or cur.get("height") is not r.get("height"):
            self._disarm_tools()
            self.level_pts = []
        self.result = r

    def _supersede(self, old: dict, new: dict) -> bool:
        """Put `new` in place of `old` in the history and, when `old` is displayed, on screen. Results are found by
        identity; the jobs that replace a result (metrology, Z rescale) run one at a time, and the feature job merges
        into its result instead of replacing it, so `old` is still there unless it was closed meanwhile. Returns
        False, installing nothing, when it is no longer open."""
        found = False
        for i, h in enumerate(self.history):
            if h is old:
                self.history[i] = new
                found = True
        if self.result is old:
            self._set_result(new)
            found = True
        return found

    def _recompute(self):
        r = self.result
        if r is None:
            return
        job = Job(_metrology_job, r, self.sp_form.value(), self._lam_s(resolve_result(r)),
                  self.sp_lc.value(), label="Metrology")
        self._start_job(job, lambda m, r=r: self._on_recomputed(r, m), main=True)

    def _on_recomputed(self, r, m, base=None, message="Metrology recomputed with the new filter settings."):
        """Install the metrology `m` computed from `base` (a height-edited copy of `r`) or from `r` itself."""
        new = dict(r if base is None else base)
        for k in metrology.PARAMETER_KEYS:
            new.pop(k, None)
        new["warnings"] = [x for x in new.get("warnings", []) if not metrology.is_metrology_warning(x)] + m.pop("warnings")
        new.update(m)
        if not self._supersede(r, new):
            self.lbl_job.setText("Metrology result discarded.")
            self.notify(f"Not applied: {r.get('label') or 'the result'} was closed while its metrology was "
                        "recomputed.", "warning")
            return
        self.prog.setValue(1000)
        self.lbl_job.setText("Metrology recomputed.")
        self._show_result()
        self.notify(message if self.result is new else
                    f"{message} (to {new.get('label') or 'a result'} in the history, not the one displayed)", "success")

    def _ensure_features(self):
        """Start the on-demand feature job for the current result when the box is ticked and they are missing."""
        r = self.result
        if (r is None or not self.chk_topo_grains.isChecked() or r.get("features_done")
                or any(getattr(j, "feature_for", None) is r and j.isRunning() for j in self.jobs)
                or "sl_surface" not in r):
            return
        job = Job(_feature_job, r, label="Feature parameters")
        job.feature_for = r
        self._start_job(job, lambda m, r=r: self._on_features(r, m))

    def _on_features(self, r, m):
        if not any(h is r for h in self.history) and self.result is not r:
            return  # the result was replaced (height tool, recompute) while this ran: its features are stale
        # Merged INTO r, not into a copy: a metrology or Z job started from r finds it by identity when it finishes
        # (_supersede). Replacing r here made such a job, finishing second, drop its height edit or filter change.
        # The features are kept only by results with this height and filter: _on_recomputed drops them
        # (metrology.PARAMETER_KEYS) and _ensure_features computes them again for the new result.
        r.update(m)
        r["features_done"] = True
        r["eligibility"] = {**r.get("eligibility", {}), "feature": "ok"}
        self.lbl_job.setText("Feature parameters computed.")
        self._show_result()

    def _run_stack(self):
        if self.sr_job is not None and self.sr_job.isRunning():
            self.notify("A still-image job is already running.", "warning")
            return
        still = self.engine.dwell_still()
        if len(still.frames) < 8:
            self.notify("Hold still for a moment (needs 8 still frames).", "warning")
            return
        h, w = still.frames[0].shape
        cal = self.store.get(self.obj)
        um = self.obj.nominal_um_per_px(w) if self.sim_source is not None else cal.um_per_px(w, self.obj, h)
        psf = self.obj.psf_sigma_px(um) if self.sr_deconvolve else None
        job = Job(_stack_job, still, psf, label="Stack")
        self.sr_job = job
        self._start_job(job, lambda r: self._on_sr({**r, "kind": "stack", "um_per_px": um, "scale": 1,
                                                     "input": still.frames[0]}))

    def _run_sr(self):
        if self.sr_job is not None and self.sr_job.isRunning():
            self.notify("Super-resolution is already running.", "warning")
            return
        still = self.engine.dwell_still()
        if len(still.frames) < 8:
            self.notify("Hold still for a moment (needs 8 still frames).", "warning")
            return
        h, w = still.frames[0].shape
        cal = self.store.get(self.obj)
        um = (self.obj.nominal_um_per_px(w) if self.sim_source is not None
              else cal.um_per_px(w, self.obj, h))
        osr = self.obj.optical_oversampling(um)
        if self.sr_scale > max_useful_scale(osr):
            scale_kind = ("measured" if self.sim_source is None and cal.lateral_status_for(w, h) == CalStatus.CALIBRATED
                          else "nominal")
            self.notify(f"{self.sr_scale}× super-resolution adds no detail here: at {um:.3g} µm/px ({scale_kind}) the "
                        f"camera samples {self.obj.name} {osr:.1f}× finer than the optics resolve (λ/4NA = "
                        f"{self.obj.nyquist_pixel_um:.3g} µm). Use 'Stack still frames' instead.", "warning")
            return
        job = Job(_sr_job, still, self.obj.psf_sigma_px(um),
                  scale=self.sr_scale, pixfrac=self.sr_pixfrac, deconvolve=self.sr_deconvolve, oversampling=osr,
                  label="Super-resolution")
        self.sr_job = job
        scale = int(self.sr_scale)  # the drizzle output grid is `scale` times finer in each direction
        self._start_job(job, lambda r: self._on_sr({**r, "um_per_px": um / scale, "scale": scale}))

    def _on_sr(self, r):
        self.sr_result = r
        self.sr_count = getattr(self, "sr_count", 0) + 1
        if self.job is None or not self.job.isRunning():
            self.prog.setValue(1000)
            k = r.get("scale", 2)
            if r.get("kind") == "hdr":
                self.lbl_job.setText(f"HDR: {r['frames_used']} frames merged; " + "; ".join(r["notes"][:1]))
            elif r.get("kind") == "stack":
                self.lbl_job.setText(f"Stack: {r['frames_used']} frames used, {r['frames_rejected']} rejected, "
                                     f"noise {r['noise_sigma_single']:.2f} → {r['noise_sigma_stack']:.2f} DN, "
                                     f"{r['pixels_clipped_fraction']:.1%} samples rejected as outliers"
                                     + (f", RL {r['rl_iterations']} iterations" if r["rl_iterations"] else ""))
            else:
                self.lbl_job.setText(f"Super-resolution ({k}× linear, {k * k}× pixels): {r['frames_used']} frames used, "
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
            if sr.get("colour") is not None and self.chk_rgb.isChecked():
                self.p_edf.set_image(bgr_to_rgb_u8(np.clip(sr["colour"], 0, 255)), sr["um_per_px"], unit="DN", rgb=True)
            else:
                self.p_edf.set_image(np.clip(sr["image"], 0, 255), sr["um_per_px"], levels=(0, 255), unit="DN",
                                     autorange=True)
            self.p_edf.set_overlay(None)
            if sr.get("kind") == "hdr":
                self.lbl_edf.setText("HDR still: relative radiance through the measured camera response, shown "
                                     "tone-mapped · " + "; ".join(sr.get("notes", [])[:1]))
            elif sr.get("kind") == "stack":
                self.lbl_edf.setText(f"Stack of {sr['frames_used']} still frames on the camera's pixel grid"
                                     + (" + Richardson-Lucy" if sr["rl_iterations"] else "")
                                     + " · " + "; ".join(sr.get("notes", [])))
            else:
                self.lbl_edf.setText(f"Experimental: {sr.get('scale', 2)}× (linear) drizzle + Richardson-Lucy of the "
                                     "centre crop; a display image, not a measured resolution gain")
        elif r is not None:
            um_cap = r["um_per_px_capture"]
            edf_rgb = r.get("edf_rgb")  # colour all-in-focus (fused with the grey weights) when the sweep was colour
            if edf_rgb is not None:
                self.p_edf.set_image(edf_rgb, um_cap, unit="DN", rgb=True)
            else:
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
                                 "watershed regions (image-based)" + (" · colour" if edf_rgb is not None else ""))
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
        self.btn_tools_reset.setEnabled(bool(r.get("height_tools")))
        self._ensure_features()
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
            self.prof_label.setText(self._roughness_text(r, xs, ys, ok, dist, n))
        else:
            self.prof_label.setText("Too few valid heights along the line for profile values.")

    def _roughness_text(self, r, xs, ys, ok, dist, n) -> str:
        """ISO 21920 roughness along the profile line, computed by metrology.profile_parameters from the MEASURED
        height (not the display layer): line levelling, the applied S-filter, and a Gaussian L-filter at λc."""
        zl = units.label(r["z_unit"])
        prof = np.full(n, np.nan, np.float32)
        prof[ok] = cv2.remap(r["height"].astype(np.float32), xs[ok].astype(np.float32).reshape(1, -1),
                             ys[ok].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).ravel()
        step = float(dist[-1]) / (n - 1)
        length = float(dist[-1])
        lam_c = self.sp_lc.value()
        origin = "L-filter setting"
        if lam_c <= 0:
            lam_c, origin = length / metrology.PROFILE_TRAVERSE_CUTOFFS, f"default: {DEFAULT_PROFILE_CUTOFF_TEXT}"
        lam_s = float(r.get("lambda_s_um") or 0.0)
        why = metrology.profile_cutoff_problem(n, step, lam_c)
        if why is not None:
            return f"Profile too short for λc = {lam_c:.3g} µm ({origin}): {why}. No roughness reported."
        p = metrology.profile_parameters(prof, step, lam_s_um=lam_s, lam_c_um=lam_c)
        if not p:
            return "Too few valid heights along the line for profile values."
        s_txt = f"λs {p['lambda_s_applied_um']:.3g} µm" if p["lambda_s_applied_um"] else "no λs"
        ev = p.get("evaluation_length_um") or length
        return (f"Roughness over {ev:.1f} µm of {length:.1f} µm measured (ISO 21920 Gaussian filter: {s_txt}, "
                f"λc {p['lambda_c_applied_um']:.3g} µm [{origin}], line levelled):  "
                f"Ra {p['Ra']:.4g} {zl}  ·  Rq {p['Rq']:.4g} {zl}  ·  Rz {p['Rz']:.4g} {zl}  ·  "
                f"Rp {p['Rp']:.4g} {zl}  ·  Rv {p['Rv']:.4g} {zl}")

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
            ("Ssk", "Ssk", "", "Skewness", "{:.4g}", "height"), ("Sku", "Sku", "", "Kurtosis", "{:.4g}", "height")])
        body += table("Hybrid (ISO 25178-2)", [
            ("Sdq", "Sdq", "", "Root mean square gradient (dimensionless, 1 = 45° plane; needs z in µm)", "{:.4g}",
             "height"),
            ("Sdr", "Sdr_pct", "%", "Developed interfacial area ratio (needs z in µm)", "{:.4g}", "Sdr")])
        vol_unit = "µm³/µm²" if units.is_um(r["z_unit"]) else f"{zu} (per unit area)"
        body += table("Volume parameters (per unit area, p = 10 %, q = 80 %)", [
            ("Vmp", "Vmp", vol_unit, "Material volume of the peaks: Vm(10 %)", "{:.4g}", "functional"),
            ("Vmc", "Vmc", vol_unit, "Material volume of the core: Vm(80 %) − Vm(10 %)", "{:.4g}", "functional"),
            ("Vvc", "Vvc", vol_unit, "Void volume of the core: Vv(10 %) − Vv(80 %)", "{:.4g}", "functional"),
            ("Vvv", "Vvv", vol_unit, "Void volume of the valleys: Vv(80 %)", "{:.4g}", "functional")])
        body += table("Feature / summit parameters", [
            ("Sds", "Sds_per_mm2", "summits/mm²", "Density of ALL 8-neighbour maxima (unpruned, noise-sensitive)",
             "{:.4g}", "feature"),
            ("Ssc", "Ssc_per_um", "1/µm", "Mean curvature at those summits", "{:.4g}", "feature"),
            ("Spd", "Spd_per_mm2", "peaks/mm²", "Density of significant peaks (Wolf pruning, 5 % of Sz)",
             "{:.4g}", "feature"),
            ("Spc", "Spc_per_um", "1/µm", "Mean curvature of the significant peaks", "{:.4g}", "feature"),
            ("S5p", "S5p", zu, "Mean height of the five highest significant peaks", "{:.4g}", "feature"),
            ("S5v", "S5v", zu, "Mean depth of the five deepest significant pits", "{:.4g}", "feature"),
            ("S10z", "S10z", zu, "S5p + S5v", "{:.4g}", "feature")])
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
            body += table("Topographic hills", [
                ("Hill count", "topo_grain_count", "", ("Significant hills (watershed of the pruned peaks) not "
                 "touching the border"), "{:d}", "feature"),
                ("Mean hill area", "topo_mean_grain_area_um2", "µm²", "Mean catchment area of those hills", "{:.4g}",
                 "feature")])
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

    def _export_still(self):
        sr = self.sr_result
        if sr is None:
            self.notify("No still image yet: use 'Stack still frames' or super-resolution first.", "warning")
            return
        kind = sr.get("kind", "super-resolution")
        p = self._save_path("Export still image", f"{kind}.tif", "TIFF (*.tif)")
        if not p:
            return
        prov = {"kind": kind, "objective": self.obj.name, "software": f"MicroTopos {__version__}",
                "scale_linear": sr.get("scale", 1), "frames_used": sr.get("frames_used"),
                "frames_rejected": sr.get("frames_rejected"), "notes": sr.get("notes", []),
                "noise_sigma_single": sr.get("noise_sigma_single"), "noise_sigma_stack": sr.get("noise_sigma_stack"),
                "rl_iterations": sr.get("rl_iterations"), "values": sr.get("values", "camera DN")}
        try:
            if sr.get("kind") == "hdr":  # the radiance itself, not the tone-mapped display image
                paths = export.export_still(p, sr["radiance"], sr["um_per_px"], prov, sr.get("radiance_colour"),
                                            colour_float=True)
            else:
                paths = export.export_still(p, sr["image"], sr["um_per_px"], prov, sr.get("colour"))
            self.notify(f"Saved {paths[0]} (+ {len(paths) - 1} file(s))", "success")
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
            extra["mosaic"] = self.engine.mosaic_state()  # under the engine lock: the engine thread mutates it
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
            mosaic = r.pop("mosaic", None)  # the saved pose graph; applied to the live engine only at the end
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
                # the scale the mosaic recorded itself, else the session's preview scale (older files)
                um_loaded = mosaic.um_per_px if mosaic is not None and mosaic.um_per_px else r.get("um_preview")
                if not (isinstance(um_loaded, (int, float)) and np.isfinite(um_loaded) and um_loaded > 0):
                    um_loaded = None
                    if has and isinstance(r.get("preview_shape"), (list, tuple)) and len(r["preview_shape"]) == 2 and r["preview_shape"][1]:
                        um_loaded = float(r["um_per_px"]) * float(r["height"].shape[1]) / float(r["preview_shape"][1])
                loaded_mosaic = ((r["mosaic_image"], tuple(r["mosaic_origin"]), 1), parts, um_loaded)
            else:
                loaded_mosaic = None
            if has:
                r.setdefault("result_id", uuid.uuid4().hex[:12])
                r["mosaic_generation"] = -1  # never pinned into the live mosaic
                r["label"] = f"{Path(p).name}: {r.get('label', '')}".strip(": ")
            # everything is validated: apply. The live mosaic is replaced under the engine lock, in one step
            # (load_mosaic_state is all-or-nothing); the saved mosaic is the loader's private copy.
            if mosaic is not None:
                # a mosaic saved before it recorded its objective and scale: the session's (it was made with the
                # objective selected when it was saved, at that preview scale)
                if not mosaic.objective and isinstance(_s.get("objective"), str):
                    mosaic.objective = _s["objective"]
                um_s = r.get("um_preview")
                if mosaic.um_per_px is None and isinstance(um_s, (int, float)) and np.isfinite(um_s) and um_s > 0:
                    mosaic.um_per_px = float(um_s)
                mmeta, marrays = mosaic.to_state_arrays()
                self.engine.load_mosaic_state(mmeta, marrays)
                self._sync_mosaic_controls()
                self.last_mosaic_version = -1
            if loaded_mosaic is not None:
                self.loaded_mosaic = loaded_mosaic
                self.last_mosaic_version = -1
            if has:
                self._set_result(r)
                self._remember(r)
                self._show_result()
                self.tabs.setCurrentIndex(3)
            else:
                self.tabs.setCurrentIndex(1)
            self.notify(f"Opened {p}" + ("" if has else " (mosaic only)"), "success")
        except Exception as e:  # noqa: BLE001 - a damaged file is reported, the window is unchanged
            self.notify(f"Could not open session: {e}", "danger")

    # ================================================================== Slice 5: capability wiring
    def _load_white_balance(self):
        g = self.settings.value("white_balance", None)
        try:
            gains = tuple(float(v) for v in g) if g else None
        except (TypeError, ValueError):
            gains = None
        self.engine.wb_gains = gains if gains is not None and len(gains) == 3 and all(v > 0 for v in gains) else None

    def _set_white_balance(self, reset: bool = False):
        """Grey-world white balance from the newest camera frame (calibration.white_balance_gains), stored in the
        settings; reset=True removes it."""
        if reset:
            self.engine.wb_gains = None
            self.settings.remove("white_balance")
            self.notify("White balance reset: colour images show the camera's own balance.", "success")
            return
        item = self.source.latest() if self.source is not None else None
        if item is None or item[0] is None or item[0].ndim != 3:
            self.notify("White balance needs a colour frame from the camera.", "warning")
            return
        try:
            gains = white_balance_gains(item[0][::2, ::2])
        except ValueError as e:
            self.notify(f"White balance not set: {e}.", "warning")
            return
        self.engine.wb_gains = gains
        self.settings.setValue("white_balance", list(gains))
        self.notify(f"White balance set from this view: gains B {gains[0]:.3f}, G 1, R {gains[2]:.3f} "
                    "(colour images only).", "success")

    def _set_color_mode(self, on: bool):
        self.color_mode = bool(on)
        self.engine.color_mode = bool(on)  # without this the engine never kept colour frames and the colour EDF was unreachable

    # ---------------------------------------------------------------- replay transport bar
    def _replay_bar(self):
        h = FlowBar()  # the seek slider gets its own row above (below); the buttons wrap
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
        self.replay_seek.valueChanged.connect(self._replay_seek_moved)  # programmatic updates block signals
        h.addWidget(self.btn_replay_pause)
        h.addWidget(self.btn_replay_step_b)
        h.addWidget(self.btn_replay_step_f)
        self.replay_seek.setMinimumWidth(120)
        self.cmb_replay_speed = QComboBox()
        self.cmb_replay_speed.addItems(["0.25×", "0.5×", "1×", "2×", "4×"])
        self.cmb_replay_speed.setCurrentText("1×")
        self.cmb_replay_speed.setFixedWidth(56)
        self.cmb_replay_speed.currentTextChanged.connect(self._replay_speed)
        h.addLabelled("Speed", self.cmb_replay_speed)
        self.chk_replay_loop = QCheckBox("Loop")
        self.chk_replay_loop.toggled.connect(self._replay_loop)
        h.addWidget(self.chk_replay_loop)
        v = QVBoxLayout()
        v.setContentsMargins(6, 4, 6, 4)
        v.setSpacing(4)
        v.addWidget(self.replay_seek)
        v.addLayout(h.layout())
        return v

    def _set_replay_transport_enabled(self, on: bool):
        """Pause/step/seek/speed act only on a paced replay; an exhaustive (consumer-paced) one refuses them."""
        for wdg in (self.btn_replay_pause, self.btn_replay_step_b, self.btn_replay_step_f, self.replay_seek,
                    self.cmb_replay_speed):
            wdg.setEnabled(on)
            wdg.setToolTip("" if on else "Not available: this recording is replayed frame by frame as fast as it is processed")

    # The play/pause button always shows the player's own state (ReplaySource.paused / .done), never a state of its
    # own: seek keeps playing or paused as it was, a step pauses, and the end of a recording reads "Play" (which
    # restarts it from the first frame).
    def _replay_stopped(self) -> bool:
        s = self.source
        return isinstance(s, ReplaySource) and (s.paused or (s.done and not s.loop))

    def _replay_sync_button(self):
        stopped = self._replay_stopped()
        self.btn_replay_pause.blockSignals(True)
        self.btn_replay_pause.setChecked(stopped)
        self.btn_replay_pause.blockSignals(False)
        self.btn_replay_pause.setText("▶ Play" if stopped else "‖ Pause")

    def _replay_toggle(self, _checked: bool = False):
        s = self.source
        if not isinstance(s, ReplaySource):
            return
        if self._replay_stopped():
            s.resume()  # at the end of the recording this restarts from frame 0
        else:
            s.pause()
        self._replay_sync_button()

    def _replay_step(self, delta: int):
        s = self.source
        if isinstance(s, ReplaySource):
            s.step_frames(delta)
            self._replay_sync_button()

    def _replay_seek_done(self):
        s = self.source
        if isinstance(s, ReplaySource):
            s.seek(self.replay_seek.value())
            self._replay_sync_button()

    def _replay_seek_moved(self, _value: int):
        """A click on the slider's track or a key press moves it without a drag: seek then too."""
        if not self.replay_seek.isSliderDown():
            self._replay_seek_done()

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
        self.engine.set_mosaic_blending(text)  # under the engine lock: the engine thread splats into the same tiles
        self.last_mosaic_version = -1

    def _mosaic_track_rotation(self, on: bool):
        self.track_rotation = bool(on)
        self.engine.set_mosaic_track_rotation(on)
        self.last_mosaic_version = -1

    def _sync_mosaic_controls(self):
        """Show the mosaic's own blending and rotation tracking (an opened session restores both)."""
        m = self.engine.mosaic
        for wdg, setter, value in ((self.cmb_mblend, self.cmb_mblend.setCurrentText, m.blending),
                                   (self.chk_rot_track, self.chk_rot_track.setChecked, bool(m.track_rotation))):
            wdg.blockSignals(True)
            setter(value)
            wdg.blockSignals(False)
        self.track_rotation = bool(m.track_rotation)

    def _mosaic_rgb_mode(self, on: bool):
        self.mosaic_rgb = bool(on)
        self.last_mosaic_version = -1

    def _mosaic_add_inset(self):
        s = self.engine.snapshot()
        if s.preview is None:
            self.notify("No live image to inset yet.", "warning")
            return
        if s.mosaic_hold:  # the live view's position is in no mosaic coordinates
            self.notify(f"No inset: the live view is not tracked into this mosaic ({s.mosaic_hold}).", "warning")
            return
        # The mosaic is in preview pixels, so the inset's scale is the preview's (the same one the Live view uses,
        # calibrated for this camera mode or nominal) and its position is the live view's mosaic position.
        pw = s.preview.shape[1]
        um = (self.obj.nominal_um_per_px(pw) if self.sim_source is not None
              else self.store.get(self.obj).um_per_px(pw, self.obj, s.preview.shape[0]))
        mx, my = s.mosaic_xy
        self.engine.add_mosaic_inset(self.obj.name, um, mx * um, my * um, s.preview)
        self.last_mosaic_version = -1
        self.notify(f"Inset added: {s.preview.shape[1]}×{s.preview.shape[0]} px")

    # ---------------------------------------------------------------- height interactive tools
    # The tools act on the MEASURED height (result["height"]), whatever layer is displayed, and then the result's
    # metrology is recomputed through the same job as a filter change (_metrology_job -> _on_recomputed), so the
    # filtered surface, every displayed layer, every parameter and the report agree. The reconstructed height is
    # kept in result["height_reconstructed"]; every edit is listed in result["height_tools"] and in the warnings.
    TOOL_WARNING = "Height edited by the user"

    def _tool_warning(self, rec: dict) -> str:
        if rec["tool"] == "level3":
            pts = ", ".join(f"({x:.0f}, {y:.0f})" for x, y in rec["points_px"])
            return f"{self.TOOL_WARNING}: surface levelled through the 3 points {pts} px (plane subtracted)."
        x0, y0, x1, y1 = rec["rect_px"]
        return f"{self.TOOL_WARNING}: pixels x {x0}–{x1}, y {y0}–{y1} excluded (set to no data)."

    def _commit_height_tool(self, new_height: np.ndarray, rec: dict, message: str):
        """Replace the measured height by `new_height` and recompute all metrology from it (existing job)."""
        r = self.result
        base = dict(r)
        if "height_reconstructed" not in base:
            base["height_reconstructed"] = r["height"]
        base["height"] = new_height
        base["height_tools"] = list(r.get("height_tools", [])) + [rec]
        base["warnings"] = [w for w in r.get("warnings", []) if not w.startswith(self.TOOL_WARNING)]
        base["warnings"] += [self._tool_warning(x) for x in base["height_tools"]]
        job = Job(_metrology_job, base, self.sp_form.value(), self._lam_s(resolve_result(r)), self.sp_lc.value(),
                  label="Metrology")
        self._start_job(job, lambda m, r=r, base=base: self._on_recomputed(r, m, base=base, message=message),
                        main=True)

    @staticmethod
    def _tools_reset_blocker(r: dict) -> str | None:
        """Why "Reset height tools" cannot restore the reconstructed height of `r`, or None when it can."""
        if "height_reconstructed" not in r:
            return ("This result does not hold the height from before the edits (a session saved by an older version); "
                    "reconstruct again to get the original.")
        return None  # a Z rescale scales the reconstructed height with the others (zstack.HEIGHT_ARRAYS)

    def _reset_height_tools(self):
        r = self.result
        if r is None or not r.get("height_tools"):
            return
        why = self._tools_reset_blocker(r)
        if why is not None:
            self.notify(why, "warning")
            return
        base = {k: v for k, v in r.items() if k not in ("height_reconstructed", "height_tools")}
        base["height"] = r["height_reconstructed"]
        base["warnings"] = [w for w in r.get("warnings", []) if not w.startswith(self.TOOL_WARNING)]
        job = Job(_metrology_job, base, self.sp_form.value(), self._lam_s(resolve_result(r)), self.sp_lc.value(),
                  label="Metrology")
        self._start_job(job, lambda m, r=r, base=base: self._on_recomputed(
            r, m, base=base, message="Height tools reset: the reconstructed height is restored."), main=True)

    def _disarm_tools(self):
        """Hide every tool ROI and leave levelling mode."""
        self._armed_tool = None
        for roi in (self.mask_roi, self.step_roi_a, self.step_roi_b):
            roi.hide()
        self.btn_mask.setText("Exclude ROI")
        self.btn_step.setText("Step height A→B")
        self.btn_tool_cancel.hide()
        if self.btn_level3.isChecked():
            self.btn_level3.setChecked(False)  # leaves levelling mode through _level_mode

    def _place_roi(self, roi, fx, fy, fw, fh):
        """Position a tool ROI at a fraction of the height map (in µm group coordinates)."""
        r = self.result
        h, w = r["height"].shape
        um = r["um_per_px"]
        roi.setPos((fx * w * um, fy * h * um))
        roi.setSize((fw * w * um, fh * h * um))

    def _tool_pressed(self, which: str):
        """First press arms the tool (its ROIs appear); the second press applies it."""
        r = self.result
        if r is None:
            self.notify("No height map yet.", "warning")
            return
        if getattr(self, "_armed_tool", None) != which:
            self._disarm_tools()
            self._armed_tool = which
            if which == "mask":
                self._place_roi(self.mask_roi, 0.35, 0.35, 0.3, 0.3)
                self.mask_roi.show()
                self.btn_mask.setText("Apply exclusion")
                self.notify("Move and resize the pink rectangle, then press “Apply exclusion”.")
            else:
                self._place_roi(self.step_roi_a, 0.1, 0.4, 0.3, 0.2)
                self._place_roi(self.step_roi_b, 0.6, 0.4, 0.3, 0.2)
                self.step_roi_a.show()
                self.step_roi_b.show()
                self.btn_step.setText("Measure step")
                self.notify("Place the two orange rectangles on the two terraces, then press “Measure step”.")
            self.btn_tool_cancel.show()
            return
        (self._apply_mask_roi if which == "mask" else self._measure_step_roi)()

    def _level_mode(self, on: bool):
        self.level_picking = bool(on)
        self.level_pts = []
        if on:
            for roi in (self.mask_roi, self.step_roi_a, self.step_roi_b):
                roi.hide()
            self._armed_tool = None
            self.btn_mask.setText("Exclude ROI")
            self.btn_step.setText("Step height A→B")
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
        pts, self.level_pts = self.level_pts, []
        self.level_picking = False
        self.btn_level3.blockSignals(True)
        self.btn_level3.setChecked(False)
        self.btn_level3.setText("Level: pick 3 points")
        self.btn_level3.blockSignals(False)
        um = r.get("um_per_px", 1.0)
        radius = max(3, round(10 / um))
        try:
            levelled = metrology.level_three_points(r["height"], pts, radius_px=radius)
        except ValueError as e:
            self.notify(f"3-point levelling failed: {e}", "warning")
            return
        self._commit_height_tool(levelled, {"tool": "level3", "points_px": [list(p) for p in pts],
                                            "radius_px": radius},
                                 "Height map levelled through the 3 picked points; all parameters recomputed.")

    def _apply_mask_roi(self):
        r = self.result
        if r is None:
            return
        rect_px = self._roi_rect_px(self.mask_roi, r.get("um_per_px", 1.0))
        masked = metrology.apply_exclusion_mask(r["height"], rect_px)
        self._disarm_tools()
        self._commit_height_tool(masked, {"tool": "exclude", "rect_px": list(rect_px)},
                                 "Mask ROI excluded from the height map; all parameters recomputed.")

    def _measure_step_roi(self):
        r = self.result
        if r is None:
            return
        um = r.get("um_per_px", 1.0)
        ra = self._roi_rect_px(self.step_roi_a, um)
        rb = self._roi_rect_px(self.step_roi_b, um)
        try:
            m = metrology.step_height_regions(r["height"], ra, rb)
        except ValueError as e:
            self.notify(str(e), "warning")
            return
        zl = units.label(r["z_unit"])
        self.lbl_height.setText(f"Step A→B: {m['step_height']:+.4g} {zl}  (SE ±{m['standard_error']:.4g} {zl}, "
                                f"{m['pixels_a']}/{m['pixels_b']} px)")
        self.notify(f"Step height {m['step_height']:+.4g} {zl}.", "success")
        self._disarm_tools()

    @staticmethod
    def _roi_rect_px(roi, um):
        """ROI item rect (group/µm coords) converted to an integer pixel rect (x0, y0, x1, y1)."""
        p = roi.pos()
        s = roi.size()
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
        if self.stage_driver is not None:  # after the source: its thread reads the stage for every frame
            self.stage_driver.close()
            self.stage_driver = None
        super().closeEvent(ev)
