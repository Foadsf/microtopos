"""Ground-truth self-tests. Thresholds are fixed here, before any run; a failing check is reported, not tuned.

    uv run microtopos-selftest            # algorithm checks (headless)
    uv run microtopos-selftest --quick    # one seed
    uv run microtopos-selftest --gui      # also drive the GUI offscreen for ~20 s
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from . import export, metrology, sff, zstack
from .calibration import bgr_to_luma
from .focus import MEASURES, focus_map, tile_scores
from .kinematics import State, StateMachine
from .mosaic import Mosaic
from .optics import OBJECTIVES
from .registration import phase_shift, register_pair, register_sweep, warp_scale_shift
from .simulator import SpecimenSimulator
from .superres import dwell_superres

THRESH = {
    "geom_offset_px": 0.05,
    "geom_r": 0.99,
    "reg_shift_rms_px": 0.25,
    "reg_scale_err": 2e-3,
    "sff_r": 0.90,
    "sff_slope_tol": 0.05,
    "sff_valid": 0.70,
    "rank_r": 0.90,
    "reversal_tol_frames": 3,
    "textureless_nan": 0.85,
    "false_sweeps": 0,
    "mosaic_rms_px": 1.5,
    "sfilter_tx": (0.48, 0.52),
    "sr_shift_rms_px": 0.1,
    "timing_1080p_s": 15.0,
}

Row = tuple  # (check, seed, measured, threshold, passed | None)


def _frames(sim, zs, **kw):
    fr, tr = [], []
    for z in zs:
        sim.focus_z_um = float(z)
        bgr, t = sim.render(**kw)
        fr.append(bgr_to_luma(bgr))
        tr.append(t)
    return fr, tr


def check_conventions(seed):
    sim = SpecimenSimulator(seed, OBJECTIVES["40x / 0.85"])
    alb, _ = sim.truth_patch()

    def flat(bgr):  # undo the simulated dark level, gain and vignette (radiometry, not geometry)
        return ((bgr_to_luma(bgr) - sim.dark_level) / (sim.vignette * sim.gain)).astype(np.float32)

    a = flat(sim.render(tremor=False, noise=False, breathing=False, blur=False)[0])
    dx, dy = (float(v) for v in register_pair(alb, a, 1).warp[:, 2])
    off = math.hypot(dx, dy)
    r = float(np.corrcoef(alb.ravel(), a.ravel())[0, 1])
    rows = [("simulator geometry: render vs truth offset (px)", seed, f"{off:.3f}", f"< {THRESH['geom_offset_px']}",
             off < THRESH["geom_offset_px"]),
            ("simulator geometry: correlation (blur/noise off)", seed, f"{r:.4f}", f">= {THRESH['geom_r']}",
             r >= THRESH["geom_r"])]
    win = cv2.createHanningWindow(alb.shape[::-1], cv2.CV_32F)
    pdx, _, _ = phase_shift(alb, np.roll(alb, 5, axis=1), win)
    rows.append(("phaseCorrelate sign: content +5 px -> dx", seed, f"{pdx:+.2f}", "+5 +/- 0.1", abs(pdx - 5) < 0.1))
    a0, b0 = alb.copy(), np.roll(alb, 3, axis=0)
    a1, b1 = a0.copy(), b0.copy()
    phase_shift(a1, b1, win)
    intact = np.array_equal(a0, a1) and np.array_equal(b0, b1)
    rows.append(("phase_shift leaves inputs unmodified (OpenCV 5 bug)", seed, str(intact), "True", intact))
    sim.tremor_px = 0.8
    errs = []
    for _ in range(3):
        bgr, t = sim.render(noise=False, breathing=False, blur=False)
        ex, ey = (float(v) for v in register_pair(alb, flat(bgr), 1).warp[:, 2])
        errs.append(math.hypot(ex - t.content_shift_px[0], ey - t.content_shift_px[1]))
    rows.append(("simulator content-shift sign (ECC, 3 frames)", seed, f"max err {max(errs):.3f} px", "< 0.05",
                 max(errs) < 0.05))
    return rows


def check_focus_measures(seed):
    sim = SpecimenSimulator(seed, OBJECTIVES["40x / 0.85"])
    alb, _ = sim.truth_patch()
    ok, worst = True, ""
    for m in MEASURES:
        s = [float(focus_map(cv2.GaussianBlur(alb, (0, 0), sg) if sg else alb, m, 9).mean()) for sg in (0, 1.5, 3.5)]
        if not (s[0] > s[1] > s[2] > 0):
            ok, worst = False, m
    return [("focus measures sharp > blurred (5 measures)", seed, "all monotonic" if ok else f"{worst} fails",
             "all monotonic", ok)]


def check_registration(seed):
    sim = SpecimenSimulator(seed, OBJECTIVES["40x / 0.85"])
    zs = sim.sweep_trajectory()
    fr, tr = _frames(sim, zs)
    warps, pairs = register_sweep(fr, factor=2)
    ref = len(fr) // 2
    errs, serr = [], []
    for W, t in zip(warps, tr):
        s, dx, dy = warp_scale_shift(W, fr[0].shape)
        tdx = t.content_shift_px[0] - tr[ref].content_shift_px[0]
        tdy = t.content_shift_px[1] - tr[ref].content_shift_px[1]
        errs.append((dx - tdx) ** 2 + (dy - tdy) ** 2)
        serr.append(abs(s - t.breathing_scale / tr[ref].breathing_scale))
    sharp = np.array([float(np.mean(np.diff(f, axis=1) ** 2)) for f in fr])
    inf = sharp >= 0.3 * sharp.max()  # frames that carry focus information
    rms_inf = math.sqrt(float(np.mean(np.array(errs)[inf])))
    rms_all = math.sqrt(float(np.mean(errs)))
    return [("registration: shift RMS, informative frames (px)", seed, f"{rms_inf:.3f} ({inf.sum()} frames)",
             f"< {THRESH['reg_shift_rms_px']}", rms_inf < THRESH["reg_shift_rms_px"]),
            ("registration: shift RMS, all frames incl. blurred (px)", seed, f"{rms_all:.3f}", "info", None),
            ("registration: uncorrected breathing, max scale error", seed, f"{max(serr):.2e}", "info (not modelled)",
             None)]


def check_sff(seed, objective="40x / 0.85"):
    obj = OBJECTIVES[objective]
    sim = SpecimenSimulator(seed, obj)
    zs = sim.sweep_trajectory()
    fr, _ = _frames(sim, zs)
    res = zstack.reconstruct(fr, zs, obj, obj.nominal_um_per_px(fr[0].shape[1]), "um")
    _, gt = sim.truth_patch()
    h = res["height"]
    g = cv2.resize(gt, (h.shape[1], h.shape[0]), interpolation=cv2.INTER_AREA)
    c = (slice(16, -16), slice(16, -16))
    v = np.isfinite(h[c])
    r = float(np.corrcoef(g[c][v], h[c][v])[0, 1])
    slope = float(np.polyfit(g[c][v], h[c][v], 1)[0])
    rmse = float(np.sqrt(np.mean(((h[c][v] - g[c][v]) - np.mean(h[c][v] - g[c][v])) ** 2)))
    tag = f"SFF {objective} known z"
    return [(f"{tag}: Pearson r", seed, f"{r:.4f}", f">= {THRESH['sff_r']}", r >= THRESH["sff_r"]),
            (f"{tag}: height slope", seed, f"{slope:.3f}", f"1 +/- {THRESH['sff_slope_tol']}",
             abs(slope - 1) <= THRESH["sff_slope_tol"]),
            (f"{tag}: valid pixels", seed, f"{v.mean():.2f}", f">= {THRESH['sff_valid']}", v.mean() >= THRESH["sff_valid"]),
            (f"{tag}: RMSE (um) / DOF", seed, f"{rmse:.3f} / {obj.dof_um:.2f}", "info", None),
            (f"{tag}: runtime (s, 640x360)", seed, f"{res['timings']['total']:.2f}", "info", None)]


def check_hand_sweep(seed):
    """Unknown z (hand-turned knob, variable speed, pause): depth in frame units must rank-correlate with truth."""
    obj = OBJECTIVES["40x / 0.85"]
    sim = SpecimenSimulator(seed, obj)
    zs = sim.sweep_trajectory()
    fr, _ = _frames(sim, zs)
    res = zstack.reconstruct(fr, None, obj, obj.nominal_um_per_px(fr[0].shape[1]), "frames")
    _, gt = sim.truth_patch()
    h = res["height"]
    g = cv2.resize(gt, (h.shape[1], h.shape[0]), interpolation=cv2.INTER_AREA)
    v = np.isfinite(h)
    from scipy.stats import spearmanr
    rho = abs(float(spearmanr(g[v], h[v]).statistic))
    used = np.array(res["frame_indices"])
    true_step = abs(zs[used[-1]] - zs[used[0]]) / max(1, len(used) - 1)
    est = res["dof_ruler_um_per_frame"]
    err = (est / true_step - 1) * 100 if est else float("nan")
    return [("hand sweep (z unknown): rank correlation", seed, f"{rho:.3f}", f">= {THRESH['rank_r']}",
             rho >= THRESH["rank_r"]),
            ("DOF ruler (experimental): um/frame error", seed, f"{err:+.0f}% (est {est or float('nan'):.3f}, "
             f"true mean {true_step:.3f})", "info", None)]


def check_reversal(seed):
    obj = OBJECTIVES["40x / 0.85"]
    sim = SpecimenSimulator(seed, obj)
    zs = sim.sweep_trajectory(reversal=True, back_fraction=(0.6, 0.8))
    up = zs[len(zs) // 3] > zs[0]
    turn = int(np.argmax(zs) if up else np.argmin(zs))
    fr, _ = _frames(sim, zs, tremor=True)
    segs = sff.monotonic_segments_from_z(zs)
    tiles = np.array([tile_scores(cv2.resize(f, (320, 180), interpolation=cv2.INTER_AREA), grid=6) for f in fr])
    t_opt = sff.find_reversal(tiles)
    ok_lbl = len(segs) == 2
    ok_opt = t_opt is not None and abs(t_opt - turn) <= THRESH["reversal_tol_frames"]
    # a reversal inside the blank (defocused) margin cannot be seen; check it does no harm
    sim2 = SpecimenSimulator(seed + 1000, obj)
    zs2 = sim2.sweep_trajectory(reversal=True, back_fraction=(0.1, 0.2))
    fr2, _ = _frames(sim2, zs2)
    res2 = zstack.reconstruct(fr2, None, obj, obj.nominal_um_per_px(fr[0].shape[1]), "frames")
    _, gt2 = sim2.truth_patch()
    h2 = res2["height"]
    g2 = cv2.resize(gt2, (h2.shape[1], h2.shape[0]), interpolation=cv2.INTER_AREA)
    v2 = np.isfinite(h2)
    from scipy.stats import spearmanr
    rho2 = abs(float(spearmanr(g2[v2], h2[v2]).statistic))
    return [("reversal split from z labels", seed, f"{len(segs)} segments", "2", ok_lbl),
            ("blank-margin reversal (z unknown): rank correlation", seed, f"{rho2:.3f}", f">= {THRESH['rank_r']}",
             rho2 >= THRESH["rank_r"]),
            ("reversal found optically (z unknown)", seed, f"at {t_opt} (true {turn})",
             f"+/- {THRESH['reversal_tol_frames']}", ok_opt)]


def check_textureless(seed):
    obj = OBJECTIVES["40x / 0.85"]
    sim = SpecimenSimulator(seed, obj, textureless_patch=True)
    zs = sim.sweep_trajectory()
    fr, _ = _frames(sim, zs)
    res = zstack.reconstruct(fr, zs, obj, obj.nominal_um_per_px(fr[0].shape[1]), "um")
    h = res["height"]
    H, W = h.shape
    patch = h[H // 3 + 8: 2 * H // 3 - 8, W // 3 + 8: 2 * W // 3 - 8]
    nanf = float(np.mean(~np.isfinite(patch)))
    return [("confidence mask on textureless patch (NaN fraction)", seed, f"{nanf:.2f}",
             f">= {THRESH['textureless_nan']}", nanf >= THRESH["textureless_nan"])]


def check_state_machine(seed):
    obj = OBJECTIVES["10x / 0.25"]
    sim = SpecimenSimulator(seed, obj)
    fov = sim.out_w * sim.um_per_px

    def run(path):
        sm, sweeps, states = StateMachine(), 0, []
        for x, y, z in path:
            sim.stage_x_um, sim.stage_y_um, sim.focus_z_um = x, y, z
            st, _, ev = sm.update(bgr_to_luma(sim.render()[0]))
            sweeps += sum(e.kind == "sweep_started" for e in ev)
            states.append(st)
        return sweeps, states

    static = [(0, 0, 0)] * 300
    t = np.linspace(0, 2 * np.pi, 200)
    pan = [(0.4 * fov * np.sin(a), 0.25 * fov * (1 - np.cos(a)), 0) for a in t]
    zs = sim.sweep_trajectory()
    sweep = [(0, 0, zs[0])] * 30 + [(0, 0, z) for z in zs] + [(0, 0, zs[-1])] * 30
    s1, _ = run(static)
    s2, st2 = run(pan)
    s3, _ = run(sweep)
    pan_frac = np.mean([s == State.PAN for s in st2[20:-20]])
    return [("state machine: false Z sweeps, 300 static frames", seed, str(s1), "0", s1 == 0),
            ("state machine: false Z sweeps, 200 panning frames", seed, f"{s2} (PAN {pan_frac:.0%} of frames)", "0",
             s2 == 0),
            ("state machine: real sweep detected", seed, f"{s3} sweep(s)", "1", s3 == 1)]


def check_objective_change(seed):
    """Switch from a 10x to a 40x view mid-stream: expect one objective-change event and a 40x guess."""
    from .kinematics import guess_objective
    s10 = SpecimenSimulator(seed, OBJECTIVES["10x / 0.25"])
    s40 = SpecimenSimulator(seed + 1, OBJECTIVES["40x / 0.85"])
    _, h40 = s40.truth_patch()
    s40.focus_z_um = float(np.median(h40))
    _, h10 = s10.truth_patch()
    s10.focus_z_um = float(np.median(h10))
    rows = []
    for variant in ("instant switch", "turret: 6 dark frames"):
        sm, events, sweeps = StateMachine(), [], 0
        for i in range(120):
            sim = s10 if i < 60 else s40
            img = bgr_to_luma(sim.render()[0])
            if variant.startswith("turret") and 60 <= i < 66:
                img = img * 0.08 + 3.0
            _, _, ev = sm.update(img)
            events += [e for e in ev if e.kind == "objective_change"]
            sweeps += sum(e.kind == "sweep_started" for e in ev)
        rows.append((f"objective change detected ({variant})", seed, f"{len(events)} event(s), {sweeps} false sweeps",
                     "1, 0", len(events) == 1 and sweeps == 0))
    mags = [o.magnification for o in OBJECTIVES.values()]
    g = guess_objective(10.0, events[0].detail.get("magnification_ratio"), mags) if events else None
    guess = list(OBJECTIVES)[g] if g is not None else None
    return rows + [("objective guess from texture scale (heuristic)", seed, str(guess), "info", None)]


def check_mosaic(seed):
    obj = OBJECTIVES["10x / 0.25"]
    sim = SpecimenSimulator(seed, obj)
    fov = sim.out_w * sim.um_per_px
    m = Mosaic()
    errs = []
    for a in np.linspace(0, 2 * np.pi, 160):
        sim.stage_x_um, sim.stage_y_um = 0.4 * fov * np.sin(a), 0.25 * fov * (1 - np.cos(a))
        m.track(bgr_to_luma(sim.render()[0]))
        errs.append((m.x - sim.stage_x_um / sim.um_per_px) ** 2 + (m.y - sim.stage_y_um / sim.um_per_px) ** 2)
    rms = math.sqrt(float(np.mean(errs)))
    return [("mosaic: camera position RMS over a closed loop (px)", seed, f"{rms:.2f} ({len(m.keyframes)} keyframes)",
             f"< {THRESH['mosaic_rms_px']}", rms < THRESH["mosaic_rms_px"])]


def check_sfilter(_seed):
    um, lam = 0.5, 20.0
    x = np.arange(512) * um
    z = np.tile(np.sin(2 * np.pi * x / lam), (64, 1)).astype(np.float32)
    f = metrology.gaussian_filter_nan(z, lam, um)
    tx = float(np.std(f[:, 100:-100]) / np.std(z[:, 100:-100]))
    lo, hi = THRESH["sfilter_tx"]
    return [("ISO 16610-61 Gaussian: transmission at cutoff", 0, f"{tx:.3f}", f"{lo}..{hi}", lo <= tx <= hi)]


def check_superres(seed):
    obj = OBJECTIVES["4x / 0.10"]
    sim = SpecimenSimulator(seed, obj, 320, 180)
    sim.tremor_px = 0.6
    _, hz = sim.truth_patch()
    sim.focus_z_um = float(np.median(hz))
    frames, tr = [], []
    for _ in range(16):
        bgr, t = sim.render(breathing=False)
        frames.append(bgr_to_luma(bgr))
        tr.append(t)
    res = dwell_superres(frames, obj.psf_sigma_px(sim.um_per_px), scale=2, deconvolve=True)
    rel = [(t.content_shift_px[0] - tr[0].content_shift_px[0], t.content_shift_px[1] - tr[0].content_shift_px[1]) for t in tr]
    sh_rms = math.sqrt(np.mean([(a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 for a, b in zip(res["shifts"], rel)]))
    truth = sim.truth_albedo_upsampled(2)
    c = (slice(24, -24), slice(24, -24))

    def nrmse(img):
        """Normalised RMSE after aligning img to the truth grid (each candidate gets the same treatment)."""
        img = img.astype(np.float32)
        W = register_pair(truth, img, 1).warp
        img = cv2.warpAffine(img, W, img.shape[::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
        a = (img - img[c].mean()) / img[c].std()
        t = (truth - truth[c].mean()) / truth[c].std()
        return float(np.sqrt(np.mean((a[c] - t[c]) ** 2)))

    up = truth.shape[::-1]
    e_sr = nrmse(res["image"])
    e_bic = nrmse(cv2.resize(frames[0], up, interpolation=cv2.INTER_CUBIC))
    e_mean = nrmse(cv2.resize(np.mean(frames, axis=0), up, interpolation=cv2.INTER_CUBIC))
    return [("drizzle: recovered sub-pixel shifts RMS (px)", seed, f"{sh_rms:.3f}", f"< {THRESH['sr_shift_rms_px']}",
             sh_rms < THRESH["sr_shift_rms_px"]),
            ("super-resolution beats bicubic single frame (4x, nRMSE)", seed, f"SR {e_sr:.3f} vs bicubic {e_bic:.3f}",
             "SR < bicubic", e_sr < e_bic),
            ("super-resolution vs bicubic of 16-frame mean (nRMSE)", seed, f"SR {e_sr:.3f} vs {e_mean:.3f}", "info",
             None)]


def check_exports(seed):
    rng = np.random.default_rng(seed)
    h = rng.normal(0, 1, (90, 160)).astype(np.float32)
    h[10:20, 10:20] = np.nan
    rows = []
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        export.write_gwyddion_gsf(d / "a.gsf", h, 0.25, True)
        meta, arr = export.read_gwyddion_gsf(d / "a.gsf")
        ok = arr.shape == h.shape and meta["XYUnits"] == "m" and abs(float(meta["XReal"]) - 160 * 0.25e-6) < 1e-12
        rows.append(("export: Gwyddion .gsf round trip", seed, "ok" if ok else "mismatch", "ok", ok))
        n, (sx, sy) = export.write_binary_stl(d / "a.stl", h, 0.25, max_grid_dim=64)
        n2, (rx, ry), normals = export.read_binary_stl_bounds(d / "a.stl")
        ok = n == n2 and abs(rx - sx) < 1e-3 and abs(ry - sy) < 1e-3 and np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-3)
        rows.append(("export: binary STL round trip (spans, unit normals)", seed, f"{n} triangles", "ok", ok))
        res = {"edf": rng.uniform(0, 255, (90, 160)).astype(np.float32), "height": h, "confidence": np.ones_like(h),
               "Sq": 1.5, "z_unit": "um"}
        export.save_session(d / "s.npz", res, {"objective": "40x / 0.85"})
        r2, s2 = export.load_session(d / "s.npz")
        ok = np.array_equal(r2["height"], h, equal_nan=True) and r2["Sq"] == 1.5 and s2["objective"] == "40x / 0.85"
        rows.append(("export: session save/load round trip", seed, "ok" if ok else "mismatch", "ok", ok))
    return rows


def check_timing(seed):
    obj = OBJECTIVES["40x / 0.85"]
    sim = SpecimenSimulator(seed, obj, 960, 540, world_margin=1.4)
    zs = sim.sweep_trajectory(n_frames=48)
    fr, _ = _frames(sim, zs)
    fr = [np.clip(cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8) for f in fr]
    t = time.perf_counter()
    res = zstack.reconstruct(fr, zs, obj, obj.nominal_um_per_px(fr[0].shape[1]), "um")
    dt = time.perf_counter() - t
    stages = ", ".join(f"{k} {v:.1f}" for k, v in res["timings"].items() if k != "total")
    return [("timing: 48 frames 1920x1080 end to end (s)", seed, f"{dt:.1f} [{stages}]",
             f"<= {THRESH['timing_1080p_s']}", dt <= THRESH["timing_1080p_s"])]


CHECKS = [check_conventions, check_focus_measures, check_registration, check_sff,
          lambda s: check_sff(s, "10x / 0.25"), check_hand_sweep, check_reversal, check_textureless,
          check_state_machine, check_objective_change, check_mosaic, check_superres]
ONCE = [check_sfilter, check_exports, check_timing]


def run(seeds=(11, 22, 33), out=print) -> bool:
    rows = []
    t0 = time.perf_counter()
    for seed in seeds:
        for chk in CHECKS:
            rows += chk(seed)
    for chk in ONCE:
        rows += chk(seeds[0])
    w = max(len(r[0]) for r in rows)
    out(f"{'check':<{w}}  seed  {'measured':<44} threshold      result")
    out("-" * (w + 80))
    for name, seed, meas, thr, ok in rows:
        res = "info" if ok is None else ("PASS" if ok else "FAIL")
        out(f"{name:<{w}}  {seed:<4}  {meas:<44} {thr:<14} {res}")
    n_fail = sum(1 for r in rows if r[4] is False)
    n_pass = sum(1 for r in rows if r[4] is True)
    out(f"\n{n_pass} passed, {n_fail} failed, {sum(1 for r in rows if r[4] is None)} info  "
        f"({time.perf_counter() - t0:.0f} s)")
    return n_fail == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="one seed only")
    ap.add_argument("--gui", action="store_true", help="also drive the GUI offscreen")
    a = ap.parse_args()
    ok = run((11,) if a.quick else (11, 22, 33))
    if a.gui:
        from .gui.smoke import run_gui_smoke
        ok = run_gui_smoke() and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
