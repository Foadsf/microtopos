"""Z-stack reconstruction: segment -> register -> focus volume -> depth -> refine -> all-in-focus -> metrology.

Depth and metrology run at a processing resolution (<= proc_width px wide); the all-in-focus image is fused
at the capture resolution. Qt-free, so it runs in a worker thread and in the self-tests.

Z coordinates
  * z_labels given (simulator truth or a measured drive): physical, signed z per frame; orientation "known".
  * hand sweep (z_labels None): the coordinate of each kept frame is the number of SOURCE frames since the
    first kept frame (`sample_pos`, e.g. the camera sequence numbers), not its index in the thinned list, so
    thinning and dropped frames keep their spacing. Frames detected as a pause add no travel. This assumes a
    constant knob speed while moving. Multiplied by `z_um_per_frame` (a scale for THIS sweep) and `z_sign`
    (+1: the focal plane moved up, toward the objective; -1: down). Without a stated sign the orientation is
    "unknown" and the relief may be inverted; the result says so.
"""

from __future__ import annotations

import datetime
import math
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import metrology, sff, units
from .backend import focus_map_backend
from .calibration import bgr_to_rgb_u8
from .focus import focus_map, tile_scores
from .optics import Objective
from .registration import RegistrationCancelled, register_sweep, warp_scale_shift, warp_support, warp_to_reference


class Cancelled(Exception):
    pass


class _OnDemandFrames:
    """Read-only sequence of n frames produced by `make(k)` when asked (no resident stack)."""

    def __init__(self, n: int, make):
        self._n, self._make = n, make

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, k: int) -> np.ndarray:
        return self._make(k)


@dataclass
class ZSettings:
    measure: str = "DoG band-pass"
    proc_width: int = 960
    form_order: int = 1
    lam_s_um: float | None = None  # None = optical cutoff lambda/(2 NA)
    lam_c_um: float = 0.0
    workers: int = 4
    conf_threshold: float = 0.25
    memory_fraction: float = 0.8  # refuse a job whose estimated peak exceeds this share of available memory
    backend: str = "cpu"  # "cpu" (reference) or "opencl" (focus measure + guided filter; CPU fallback; see backend.py)
    correct_breathing: bool = False  # EXPERIMENTAL: per-frame magnification drift from the pair scales, folded into the warps
    allow_tiling: bool = True  # strip-wise (tiled) reconstruction to bound peak memory when RAM is tight


def rescale_warp(W: np.ndarray, f: float) -> np.ndarray:
    """Warp estimated on an image downsampled by f (INTER_AREA) -> warp at full resolution."""
    A = W[:, :2].astype(np.float64)
    c = np.full(2, (f - 1) / 2.0)
    t = f * W[:, 2].astype(np.float64) + (np.eye(2) - A) @ c
    return np.hstack([A, t[:, None]]).astype(np.float32)


def _windows_available_bytes() -> int | None:
    """Available physical memory from GlobalMemoryStatusEx (Windows only)."""
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return None
    return int(stat.ullAvailPhys)


def available_memory_bytes() -> int | None:
    """Memory available to a new allocation: MemAvailable on Linux, GlobalMemoryStatusEx on Windows, sysconf
    where the OS offers it. None when unknown (macOS has no standard-library query: there the memory guard and
    the strip-wise path are not used, the sweep always runs in one piece)."""
    if sys.platform == "win32":
        try:
            return _windows_available_bytes()
        except (OSError, AttributeError):
            return None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return None


def estimate_peak_bytes(n: int, ph: int, pw: int) -> int:
    """Lower-bound peak of the processing stage: processing + aligned + focus volumes and the fit's log volume."""
    return int(4 * n * ph * pw * 4 + 12 * ph * pw * 4)


def strip_halo_rows(measure: str, window: int, psf_px: float) -> int:
    """Rows of context a strip needs on each side so its focus map equals the full-frame one: the focus window's
    half width plus the measure's own filter reach (DoG: Gaussian kernel radius of the wide blur, ~4 sigma; Sobel and
    the SML kernel: 1 px; Vollath shifts columns only). The warp needs no halo: each strip is warped straight
    from the full frame (see registration.warp_to_reference rows=)."""
    reach = {"DoG band-pass": math.ceil(4.0 * 2.5 * max(0.8, 0.8 * psf_px)) + 1, "Tenengrad": 1, "SML": 1}.get(measure, 0)
    return int(reach + window // 2 + 2)


def estimate_tiled_peak_bytes(n: int, ph: int, pw: int, n_strips: int, halo: int) -> int:
    """Lower-bound peak of the strip-wise path: the weights of every frame kept as uint16 (2 bytes) plus, per strip
    with its halo rows, the aligned, focus and fit (log) volumes and one temporary (4 float32 volumes)."""
    rows = min(ph, math.ceil(ph / n_strips) + 2 * halo)
    return int(2 * n * ph * pw + 4 * 4 * n * rows * pw + 12 * ph * pw * 4)


def plan_strips(n: int, ph: int, pw: int, halo: int, avail: int | None, memory_fraction: float,
                allow_tiling: bool) -> int:
    """Number of horizontal strips the reconstruction runs in (1 = in one piece).

    Raises MemoryError when the job does not fit in memory_fraction x avail: untiled if tiling is not allowed,
    otherwise only when even the smallest possible strip (one row plus its halo) does not fit. Unknown memory
    (avail None) means no guard and one piece."""
    if avail is None:
        return 1
    budget = memory_fraction * avail
    need = estimate_peak_bytes(n, ph, pw)
    if need <= budget:
        return 1
    if allow_tiling:
        for s in range(2, ph + 1):
            if estimate_tiled_peak_bytes(n, ph, pw, s, halo) <= budget:
                return s
        floor = estimate_tiled_peak_bytes(n, ph, pw, ph, halo)
        raise MemoryError(f"This sweep needs about {need / 2**30:.1f} GiB, and still {floor / 2**30:.1f} GiB when "
                          f"processed in strips, but only {avail / 2**30:.1f} GiB is available; lower the "
                          "processing width or sweep with fewer frames.")
    raise MemoryError(f"This sweep needs about {need / 2**30:.1f} GiB but only {avail / 2**30:.1f} GiB is "
                      "available; lower the processing width or sweep with fewer frames.")


def hand_sweep_z(positions: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Coordinate of each kept frame: source frames travelled since the first one, pauses excluded.

    positions: sample position (source frame number or time in frame periods) of every candidate frame.
    keep: indices into positions that are not pause frames (keep[0] == 0). The interval that ends at a dropped
    (pause) frame contributes no travel; every other interval contributes its full length.
    """
    p = np.asarray(positions, np.float64)
    step = np.diff(p)
    moving = np.zeros(len(p), bool)
    moving[np.asarray(keep)] = True
    contrib = np.where(moving[1:], step, 0.0)
    cum = np.concatenate([[0.0], np.cumsum(contrib)])
    return cum[np.asarray(keep)]


def reconstruct(frames: list[np.ndarray], z_labels: np.ndarray | None, objective: Objective, um_per_px: float,
                z_unit: str, z_um_per_frame: float | None = None, settings: ZSettings | None = None,
                progress=None, cancelled=None, sample_pos: np.ndarray | None = None,
                z_sign: int | None = None,
                color_frames: list[np.ndarray] | None = None) -> dict:
    """frames: radiometrically corrected luma (uint8 or float32), all the same size; um_per_px is THEIR pixel size.

    z_labels: physical z per frame (simulator ground truth or a motorised drive), or None for a hand sweep.
    z_um_per_frame: µm per source frame for this sweep; used only when z_labels is None.
    sample_pos: source frame number of each frame (default 0, 1, 2, ...); used only when z_labels is None.
    z_sign: +1 / -1 when the sweep direction is known (hand sweep); None = orientation unknown.
    color_frames: full-resolution colour (BGR or grey; uint8, uint16 or float on the 0..255 scale) image per frame,
    aligned with `frames`; when given, an all-in-focus colour image is fused with the same warps and weights and
    returned as `edf_rgb` (uint8 RGB).
    """
    st = settings or ZSettings()
    z_unit = units.canon(z_unit)
    t_all = time.perf_counter()
    timings: dict[str, float] = {}
    warnings: list[str] = []

    def check_cancel():
        if cancelled and cancelled():
            raise Cancelled()

    def step(name, frac, msg):
        check_cancel()
        if progress:
            progress(frac, msg)
        timings[name] = time.perf_counter()

    def done(name):
        timings[name] = round(time.perf_counter() - timings[name], 3)

    if len(frames) == 0:
        raise ValueError("empty sweep")
    H, W = frames[0].shape
    if any(fr.shape != (H, W) for fr in frames):
        raise ValueError("frames of a sweep must all have the same size")
    if z_labels is not None and len(z_labels) != len(frames):
        raise ValueError(f"{len(z_labels)} z labels for {len(frames)} frames")
    if color_frames is not None and len(color_frames) != len(frames):
        raise ValueError(f"{len(color_frames)} colour frames for {len(frames)} frames")
    if sample_pos is not None:
        sample_pos = np.asarray(sample_pos, np.float64)
        if len(sample_pos) != len(frames) or not np.all(np.isfinite(sample_pos)) or np.any(np.diff(sample_pos) <= 0):
            raise ValueError("sample positions must be finite, strictly increasing and one per frame")
    if z_sign not in (None, 1, -1):
        raise ValueError("z_sign must be +1, -1 or None")
    f = max(1.0, W / float(st.proc_width))
    pw, ph = round(W / f), round(H / f)
    f = W / pw
    um_proc = um_per_px * f

    # 1 ---- segmentation on small tiles -------------------------------------------------------------
    step("segment", 0.02, "Analysing sweep direction...")
    small_w = 320
    sw, sh = small_w, round(H * small_w / W)
    tiles = np.array([tile_scores(cv2.resize(fr, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32), grid=6)
                      for fr in frames])
    n_in = len(frames)
    reversal_at = None
    if z_labels is not None:
        z_all = np.asarray(z_labels, np.float64)
        a, b = sff.monotonic_segments_from_z(z_all)[0]
        idx = np.arange(a, b)
        z = z_all[idx]
        if b - a < n_in:
            reversal_at = int(b if a == 0 else a)
        orientation, scale_used = "known", None
    else:
        t = sff.find_reversal(tiles)
        if t is not None:
            reversal_at = t
            idx = np.arange(0, t + 1) if t + 1 >= n_in - t else np.arange(t, n_in)
        else:
            idx = np.arange(n_in)
        keep = sff.drop_pauses(tiles[idx])
        pos = (sample_pos if sample_pos is not None else np.arange(n_in, dtype=np.float64))[idx]
        z = hand_sweep_z(pos, keep)
        idx = idx[keep]
        scale_used = z_um_per_frame or 1.0
        z = z * scale_used * (z_sign or 1)
        orientation = "stated" if z_sign else "unknown"
        if z_sign is None:
            warnings.append("Sweep direction unknown: heights are relative focus progression and the relief may be "
                            "inverted. State the direction (Z scale step) to fix the sign.")
        if z_unit == units.UM:
            warnings.append("Hand-sweep Z scale assumes a constant knob speed during the sweep.")
    if reversal_at is not None:
        part = "first pass" if int(np.min(idx)) < reversal_at else "return pass"
        warnings.append(f"Focus direction reversed at frame {reversal_at}; used the longer monotonic part, the "
                        f"{part} ({len(idx)} of {n_in} frames)"
                        + ("; a stated direction refers to that pass." if z_labels is None else "."))
    if len(idx) < 7:
        raise ValueError(f"Too few usable frames in the sweep ({len(idx)}); sweep more slowly or further.")
    if z[-1] < z[0]:
        idx, z = idx[::-1], z[::-1]
    n = len(idx)
    window = objective.focus_window_px(um_proc)
    psf_px = objective.psf_sigma_px(um_proc)
    halo = strip_halo_rows(st.measure, window, psf_px)
    n_strips = plan_strips(n, ph, pw, halo, available_memory_bytes(), st.memory_fraction, st.allow_tiling)
    done("segment")

    # 2 ---- registration at processing resolution -------------------------------------------------------
    step("register", 0.08, "Registering frames...")

    def proc_frame(k: int) -> np.ndarray:
        r = np.empty((ph, pw), np.float32)
        r[...] = cv2.resize(frames[idx[k]], (pw, ph), interpolation=cv2.INTER_AREA)
        return r

    if n_strips == 1:
        proc = np.empty((n, ph, pw), np.float32)
        for k in range(n):
            proc[k] = proc_frame(k)
        proc_seq = list(proc)
    else:  # strip-wise: no resident n x ph x pw stack; frames are resized when a pair or a strip needs them
        proc_seq = _OnDemandFrames(n, proc_frame)
    reg_factor = 2 if pw >= 480 else 1
    try:
        warps, pairs = register_sweep(proc_seq, factor=reg_factor, workers=st.workers, cancelled=cancelled,
                                      progress=(lambda q: progress(0.08 + 0.3 * q, "Registering frames..."))
                                      if progress else None,
                                      correct_breathing=st.correct_breathing,
                                      z=None if not st.correct_breathing else [float(v) for v in z], notes=warnings)
    except RegistrationCancelled:
        raise Cancelled() from None
    eccs = np.array([p.ecc for p in pairs if p.ok])
    fails = int(sum(not p.ok for p in pairs))
    unobservable = int(sum(not p.observable for p in pairs))
    scales = [warp_scale_shift(Wk, (ph, pw))[0] for Wk in warps]
    if fails:
        warnings.append(f"{fails} of {len(pairs)} frame pairs fell back to translation-only registration"
                        + (f" ({unobservable} without observable motion; assumed still)." if unobservable else "."))
    if unobservable > 0.3 * len(pairs):
        warnings.append("Registration could not observe the motion of most frames (little texture): lateral "
                        "alignment is unsupported, treat the result with caution.")
    done("register")

    # 3 ---- focus volume, depth fit, fusion weights -----------------------------------------------------------
    step("focus", 0.4, "Computing focus volume...")
    support = np.ones((ph, pw), bool)
    for k in range(n):
        support &= warp_support((ph, pw), warps[k])
    # the focus window reaches window//2 px: pixels that close to unobserved border are not supported either
    r = window // 2
    support = cv2.erode(support.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8),
                        borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(bool)
    ran = set()
    edf_proc = np.zeros((ph, pw), np.float32)

    if n_strips == 1:
        aligned = np.empty_like(proc)
        for k in range(n):
            warp_to_reference(proc[k], warps[k], out=aligned[k])
        del proc, proc_seq
        fv = np.empty_like(aligned)
        for k in range(n):
            if k % 16 == 0:
                check_cancel()
            fv[k], used = focus_map_backend(aligned[k], st.measure, window, psf_px, st.backend)
            ran.add(used)
        done("focus")

        # 4 ---- depth -------------------------------------------------------------------------------------
        step("depth", 0.5, "Fitting depth and confidence...")
        fit = sff.fit_depth(fv, z, st.conf_threshold)
        wts = sff.softmax_weights_inplace(fv)  # fv is not needed any more: becomes the fusion weights
        del fv
        for k in range(n):
            edf_proc += wts[k] * aligned[k]
        del aligned

        def wts_of(j: int) -> np.ndarray:
            return wts[j]
    else:
        # Strip-wise (tiled) reconstruction to bound peak memory. Each strip of rows [y0, y1) is computed with
        # `halo` extra rows on each side (taken from the real frame, clamped at the image edge), so everything a
        # pixel's value depends on (focus window, filter reach) is present and the core rows equal the
        # full-frame computation. Per-pixel steps (fit, softmax weights, weighted sum) run per strip; spatially
        # global steps (support erosion above, refine_depth and everything after it) run once on the stitched
        # 2-D maps. The weights are kept for all frames as uint16 for the capture-resolution fusion.
        z_fit = np.empty((ph, pw), np.float32)
        z_raw = np.empty((ph, pw), np.float32)
        conf = np.empty((ph, pw), np.float32)
        peak_idx = np.empty((ph, pw), np.int32)
        wts16 = np.empty((n, ph, pw), np.uint16)
        fwhm_samples = []
        for rows in np.array_split(np.arange(ph), n_strips):
            y0, y1 = int(rows[0]), int(rows[-1]) + 1
            r0, r1 = max(0, y0 - halo), min(ph, y1 + halo)
            core = slice(y0 - r0, y1 - r0)
            aligned = np.empty((n, r1 - r0, pw), np.float32)
            fv = np.empty_like(aligned)
            full = np.empty((ph, pw), np.float32)
            for k in range(n):
                if k % 16 == 0:
                    check_cancel()
                # the whole frame is warped (bit-identical to the one-piece path: a sub-frame warp differs in the
                # last bits of OpenCV's fixed-point coordinates) and only the strip's rows are kept
                warp_to_reference(proc_frame(k), warps[k], out=full)
                aligned[k] = full[r0:r1]
                fv[k], used = focus_map_backend(aligned[k], st.measure, window, psf_px, st.backend)
                ran.add(used)
            sfit = sff.fit_depth(fv[:, core], z, st.conf_threshold)
            w = sff.softmax_weights_inplace(fv[:, core])
            for k in range(n):
                edf_proc[y0:y1] += w[k] * aligned[k, core]
            wts16[:, y0:y1] = np.clip(np.rint(w * 65535.0), 0, 65535).astype(np.uint16)
            z_fit[y0:y1], z_raw[y0:y1], conf[y0:y1], peak_idx[y0:y1] = (sfit.z, sfit.z_raw, sfit.confidence,
                                                                          sfit.peak_index)
            fwhm_samples.append(sfit.fwhm_z_samples)
            del aligned, fv, w, sfit, full
        samples = np.concatenate(fwhm_samples)
        fwhm = float(np.nanmedian(samples)) if samples.size else float("nan")
        fit = sff.DepthFit(z_fit, z_raw, conf, peak_idx, fwhm, samples)
        done("focus")
        step("depth", 0.5, "Fitting depth and confidence...")

        def wts_of(j: int) -> np.ndarray:
            return wts16[j].astype(np.float32) * np.float32(1.0 / 65535.0)

    fit.z[~support] = np.nan
    fit.confidence[~support] = 0.0
    check_cancel()
    height = sff.refine_depth(fit.z, fit.confidence, edf_proc, radius=max(2, window // 2), backend=st.backend)
    height[~support] = np.nan
    done("depth")

    # 5 ---- all-in-focus at capture resolution ---------------------------------------------------------
    step("edf", 0.62, "Fusing all-in-focus image...")
    acc = sff.EDFAccumulator((H, W))
    for j, i in enumerate(idx):
        check_cancel()
        Wf = rescale_warp(warps[j], f)
        fr = warp_to_reference(frames[i].astype(np.float32), Wf)
        acc.add(fr, cv2.resize(wts_of(j), (W, H), interpolation=cv2.INTER_LINEAR))
        if progress and j % 8 == 0:
            progress(0.62 + 0.23 * j / n, "Fusing all-in-focus image...")
    edf = np.clip(acc.result(), 0, 255).astype(np.float32)

    # colour all-in-focus fusion with the same warps and weights
    edf_rgb = None
    if color_frames is not None:
        acc_rgb = sff.EDFAccumulator((H, W, 3))
        for j, i in enumerate(idx):
            check_cancel()
            # one conversion to 0..255 RGB (uint16 / 257, grey -> 3 channels): calibration.bgr_to_rgb_u8
            c_fr = bgr_to_rgb_u8(color_frames[i]).astype(np.float32)
            Wf = rescale_warp(warps[j], f)
            acc_rgb.add(warp_to_reference(c_fr, Wf),
                        cv2.resize(wts_of(j), (W, H), interpolation=cv2.INTER_LINEAR))
            if progress and j % 8 == 0:
                progress(0.84 + 0.04 * j / n, "Fusing the colour all-in-focus image...")
        edf_rgb = np.clip(np.rint(acc_rgb.result()), 0, 255).astype(np.uint8)
        del acc_rgb
    del acc
    done("edf")

    # 6 ---- metrology ---------------------------------------------------------------------------------
    step("metrology", 0.88, "Surface metrology...")
    m = metrology.analyze(height, edf_proc, um_proc, z_unit, objective.lateral_cutoff_um, st.form_order,
                          st.lam_s_um, st.lam_c_um)
    warnings += m.pop("warnings")
    done("metrology")

    # DOF ruler (only meaningful for an uncalibrated hand sweep): theoretical FWHM / measured FWHM in frames. The
    # numerator is a model estimate (sff.theoretical_fwhm_um), so the result is an ESTIMATE of um/frame, never a calibration.
    dof_ruler = None
    if z_labels is None and np.isfinite(fit.fwhm_z_median) and fit.fwhm_z_median > 0:
        fw_um = sff.theoretical_fwhm_um(objective, um_proc, lambda im: focus_map(im, st.measure, window, psf_px),
                                       edf_proc)
        if fw_um is not None:
            dof_ruler = fw_um / (fit.fwhm_z_median / scale_used)

    timings["total"] = round(time.perf_counter() - t_all, 3)
    if progress:
        progress(1.0, f"Done in {timings['total']:.1f} s")
    return {
        **m,
        "edf": edf, "edf_proc": edf_proc.astype(np.float32), "edf_rgb": edf_rgb,
        "height": height, "height_raw": fit.z_raw,
        "confidence": fit.confidence, "support": support, "um_per_px_capture": um_per_px, "objective": objective.name,
        "objective_spec": objective.to_dict(),
        "frames_in": n_in, "frames_used": int(n), "reversal_at": reversal_at,
        "z_first": float(z[0]), "z_last": float(z[-1]), "fwhm_z": fit.fwhm_z_median,
        "z_orientation": orientation, "z_sign": z_sign if z_labels is None else None,
        "z_um_per_frame_used": scale_used if z_labels is None and z_unit == units.UM else None,
        "dof_ruler_um_per_frame": dof_ruler, "ecc_mean": float(eccs.mean()) if eccs.size else None,
        "ecc_min": float(eccs.min()) if eccs.size else None, "registration_fallbacks": fails,
        "registration_unobservable": unobservable, "support_fraction": float(support.mean()), "n_strips": n_strips,
        "breathing_scale_range": float(max(scales) - min(scales)), "focus_window_px": window,
        "measure": st.measure, "backend": "+".join(sorted(ran)), "timings": timings, "warnings": warnings,
        "frame_indices": idx.tolist(), "z_values": z.tolist(),
        "sample_positions": (None if z_labels is not None else
                             [float(v) for v in (sample_pos if sample_pos is not None else np.arange(n_in))[idx]]),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }


HEIGHT_ARRAYS = ("height", "height_raw")


def rescale_heights(result: dict, factor: float, new_unit: str, settings: ZSettings | None = None) -> dict:
    """A new result with every height scaled by `factor` (sign included) and metrology recomputed.

    Used when the user states the travel or direction of a hand sweep after it was reconstructed. All height
    operators (form removal, Gaussian filters) are linear, so scaling the fitted depth is exact.
    """
    if not np.isfinite(factor) or factor == 0:
        raise ValueError("scale factor must be finite and non-zero")
    out = dict(result)
    for k in HEIGHT_ARRAYS:
        if isinstance(result.get(k), np.ndarray):
            out[k] = (result[k] * factor).astype(np.float32)
    z_vals = [v * factor for v in result.get("z_values", [])]
    if factor < 0:
        z_vals.reverse()
        if isinstance(result.get("frame_indices"), list):
            out["frame_indices"] = list(reversed(result["frame_indices"]))
        if isinstance(result.get("sample_positions"), list):
            out["sample_positions"] = list(reversed(result["sample_positions"]))
    out["z_values"] = z_vals
    if "z_first" in result:  # keep z ascending, as reconstruct returns it
        a, b = result["z_first"] * factor, result["z_last"] * factor
        out["z_first"], out["z_last"] = min(a, b), max(a, b)
    if isinstance(result.get("fwhm_z"), (int, float)) and np.isfinite(result["fwhm_z"]):
        out["fwhm_z"] = float(result["fwhm_z"]) * abs(factor)
    rep_cleared = any(k in out for k in ("sigma_inst", "repeat_offset", "repeat_overlap", "sigma_inst_lt_Sq_over_10"))
    for k in ("sigma_inst", "repeat_offset", "repeat_overlap", "sigma_inst_lt_Sq_over_10"):
        out.pop(k, None)
    out["z_unit"] = units.canon(new_unit)
    st = settings or ZSettings()
    from .objectives import resolve_result
    lam_s = result.get("lambda_s_requested_um")
    if lam_s is None:
        if any("No S-filter: the default" in n for n in result.get("filter_notes", [])):
            lam_s = None
        else:
            lam_s = result.get("lambda_s_um")
    lam_c = result.get("lambda_c_requested_um", result.get("lambda_c_um", 0.0))
    m = metrology.analyze(out["height"], out["edf_proc"], out["um_per_px"], out["z_unit"],
                          resolve_result(out).lateral_cutoff_um, out.get("form_order", st.form_order),
                          lam_s, lam_c)
    for k in metrology.PARAMETER_KEYS:
        out.pop(k, None)
    metro_warn = m.pop("warnings")
    out.update(m)
    extra_warn = (["Repeatability cleared: Z scale of this sweep was changed after comparison."]
                  if rep_cleared else [])
    out["warnings"] = [w for w in result.get("warnings", []) if not metrology.is_metrology_warning(w)
                       and "Sweep direction unknown" not in w] + extra_warn + metro_warn
    return out
