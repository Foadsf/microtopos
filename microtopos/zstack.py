"""Z-stack reconstruction: segment -> register -> focus volume -> depth -> refine -> all-in-focus -> metrology.

Depth and metrology run at a processing resolution (<= proc_width px wide); the all-in-focus image is fused
at the capture resolution. Qt-free, so it runs in a worker thread and in the self-tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import metrology, sff
from .focus import focus_map, tile_scores
from .optics import Objective
from .registration import register_sweep, warp_scale_shift, warp_to_reference


class Cancelled(Exception):
    pass


@dataclass
class ZSettings:
    measure: str = "DoG band-pass"
    proc_width: int = 960
    form_order: int = 1
    lam_s_um: float | None = None  # None = optical cutoff lambda/(2 NA)
    lam_c_um: float = 0.0
    workers: int = 4
    conf_threshold: float = 0.25


def rescale_warp(W: np.ndarray, f: float) -> np.ndarray:
    """Warp estimated on an image downsampled by f (INTER_AREA) -> warp at full resolution."""
    A = W[:, :2].astype(np.float64)
    c = np.full(2, (f - 1) / 2.0)
    t = f * W[:, 2].astype(np.float64) + (np.eye(2) - A) @ c
    return np.hstack([A, t[:, None]]).astype(np.float32)


def reconstruct(frames: list[np.ndarray], z_labels: np.ndarray | None, objective: Objective, um_per_px: float,
                z_unit: str, z_um_per_frame: float | None = None, settings: ZSettings | None = None,
                progress=None, cancelled=None) -> dict:
    """frames: radiometrically corrected luma (uint8 or float32), all the same size; um_per_px is THEIR pixel size.

    z_labels: physical z per frame (simulator ground truth or a motorised drive), or None for a hand sweep.
    z_um_per_frame: user's knob calibration; used only when z_labels is None.
    """
    st = settings or ZSettings()
    t_all = time.perf_counter()
    timings: dict[str, float] = {}
    warnings: list[str] = []

    def step(name, frac, msg):
        if cancelled and cancelled():
            raise Cancelled()
        if progress:
            progress(frac, msg)
        timings[name] = time.perf_counter()

    def done(name):
        timings[name] = round(time.perf_counter() - timings[name], 3)

    H, W = frames[0].shape
    f = max(1.0, W / float(st.proc_width))
    pw, ph = int(round(W / f)), int(round(H / f))
    f = W / pw
    um_proc = um_per_px * f

    # 1 ---- segmentation on small tiles -------------------------------------------------------------
    step("segment", 0.02, "Analysing sweep direction...")
    small_w = 320
    sw, sh = small_w, int(round(H * small_w / W))
    tiles = np.array([tile_scores(cv2.resize(fr.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA), grid=6)
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
    else:
        t = sff.find_reversal(tiles)
        if t is not None:
            reversal_at = t
            idx = np.arange(0, t + 1) if t + 1 >= n_in - t else np.arange(t, n_in)
        else:
            idx = np.arange(n_in)
        keep = sff.drop_pauses(tiles[idx])
        idx = idx[keep]
        z = np.arange(len(idx), dtype=np.float64) * (z_um_per_frame or 1.0)
    if reversal_at is not None:
        warnings.append(f"Focus direction reversed at frame {reversal_at}; used the longer monotonic part "
                        f"({len(idx)} of {n_in} frames).")
    if len(idx) < 7:
        raise ValueError(f"Too few usable frames in the sweep ({len(idx)}); sweep more slowly or further.")
    if z[-1] < z[0]:
        idx, z = idx[::-1], z[::-1]
    done("segment")

    # 2 ---- registration at processing resolution -------------------------------------------------------
    step("register", 0.08, "Registering frames...")
    proc = [cv2.resize(frames[i].astype(np.float32), (pw, ph), interpolation=cv2.INTER_AREA) for i in idx]
    reg_factor = 2 if pw >= 480 else 1
    warps, pairs = register_sweep(proc, factor=reg_factor, workers=st.workers,
                                  progress=(lambda q: progress(0.08 + 0.3 * q, "Registering frames...")) if progress else None)
    eccs = np.array([p.ecc for p in pairs])
    fails = int(sum(not p.ok for p in pairs))
    scales = [warp_scale_shift(Wk, (ph, pw))[0] for Wk in warps]
    if fails:
        warnings.append(f"{fails} of {len(pairs)} frame pairs fell back to translation-only registration.")
    done("register")

    # 3 ---- focus volume ------------------------------------------------------------------------------
    step("focus", 0.4, "Computing focus volume...")
    aligned = np.stack([warp_to_reference(p, Wk) for p, Wk in zip(proc, warps)])
    window = objective.focus_window_px(um_proc)
    psf_px = objective.psf_sigma_px(um_proc)
    fv = np.stack([focus_map(a, st.measure, window, psf_px) for a in aligned])
    done("focus")

    # 4 ---- depth -------------------------------------------------------------------------------------
    step("depth", 0.5, "Fitting depth and confidence...")
    fit = sff.fit_depth(fv, z, st.conf_threshold)
    wts = sff.softmax_weights(fv)
    edf_proc = np.sum(wts * aligned, axis=0)
    height = sff.refine_depth(fit.z, fit.confidence, edf_proc, radius=max(2, window // 2))
    done("depth")

    # 5 ---- all-in-focus at capture resolution ---------------------------------------------------------
    step("edf", 0.62, "Fusing all-in-focus image...")
    acc = sff.EDFAccumulator((H, W))
    for j, i in enumerate(idx):
        if cancelled and cancelled():
            raise Cancelled()
        Wf = rescale_warp(warps[j], f)
        fr = warp_to_reference(frames[i].astype(np.float32), Wf)
        acc.add(fr, cv2.resize(wts[j], (W, H), interpolation=cv2.INTER_LINEAR))
        if progress and j % 8 == 0:
            progress(0.62 + 0.23 * j / len(idx), "Fusing all-in-focus image...")
    edf = np.clip(acc.result(), 0, 255).astype(np.float32)
    done("edf")

    # 6 ---- metrology ---------------------------------------------------------------------------------
    step("metrology", 0.88, "Surface metrology...")
    m = metrology.analyze(height, edf_proc, um_proc, z_unit, objective.lateral_cutoff_um, st.form_order,
                          st.lam_s_um, st.lam_c_um)
    warnings += m.pop("warnings")
    done("metrology")

    # DOF ruler (only meaningful for an uncalibrated hand sweep): theoretical FWHM / measured FWHM in frames
    dof_ruler = None
    if z_labels is None and np.isfinite(fit.fwhm_z_median) and fit.fwhm_z_median > 0:
        fw_um = sff.theoretical_fwhm_um(objective, um_proc, lambda im: focus_map(im, st.measure, window, psf_px),
                                       edf_proc)
        dof_ruler = fw_um / (fit.fwhm_z_median / (z_um_per_frame or 1.0))

    timings["total"] = round(time.perf_counter() - t_all, 3)
    if progress:
        progress(1.0, f"Done in {timings['total']:.1f} s")
    return {
        **m,
        "edf": edf, "edf_proc": edf_proc.astype(np.float32), "height": height, "height_raw": fit.z_raw,
        "confidence": fit.confidence, "um_per_px_capture": um_per_px, "objective": objective.name,
        "frames_in": n_in, "frames_used": int(len(idx)), "reversal_at": reversal_at,
        "z_first": float(z[0]), "z_last": float(z[-1]), "fwhm_z": fit.fwhm_z_median,
        "dof_ruler_um_per_frame": dof_ruler, "ecc_mean": float(eccs.mean()) if eccs.size else None,
        "ecc_min": float(eccs.min()) if eccs.size else None, "registration_fallbacks": fails,
        "breathing_scale_range": float(max(scales) - min(scales)), "focus_window_px": window,
        "measure": st.measure, "timings": timings, "warnings": warnings,
        "frame_indices": idx.tolist(), "z_values": z.tolist(),
    }
