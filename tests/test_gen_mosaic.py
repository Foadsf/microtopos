"""Generalisation of live mosaicking (Mosaic.track): closed-loop tracking at several speeds and objectives, loop
closure under a systematic registration bias, and revisit sharpness under defocus. Seeds and conditions vary;
nothing is initialised at the truth (positions come only from tracking). Bounds are set from the distribution
measured on main (5242ef1) with about 2x margin on the worst case plus a median bound; measured values are next
to each bound. `pytest -m slow` runs the full matrix.
"""

import math

import numpy as np
import pytest
from _gen import closed_loop, out_and_back, revisit

# ------------------------------------------------------------------------------------------ closed-loop tracking
# Elliptical lap (0.8 x 0.5 FOV), tremor, breathing, sensor noise, motion blur 0.3 x the per-frame travel.
# Measured on main, 10 seeds per group (position RMS / max error per lap, px):
#   10x: 160 frames (~5 px/frame) RMS med 0.54 max 1.18, max err 2.16; 80 (~10) 0.55 / 0.88, 1.55;
#        40 (~20) 0.65 / 1.06, 1.75
#   40x: 160 frames RMS med 0.65 max 1.15, max err 2.31; 80: 0.58 / 1.59, 2.54; 40: 0.51 / 0.88, 1.53
LOOP_RMS_MAX = 3.2  # 2 x 1.59
LOOP_ERR_MAX = 5.0  # 2 x 2.54
LOOP_RMS_MEDIAN = 1.2  # full-group medians 0.51-0.65; the 2-seed fast groups 0.58-0.85


def _closed_loop(oname, n_frames, seeds):
    rms = []
    for seed in seeds:
        e, m = closed_loop(seed, oname, n_frames)
        r = math.sqrt(float(np.mean(e ** 2)))
        rms.append(r)
        tag = f"{oname} {n_frames} frames seed {seed}: RMS {r:.3f} max {e.max():.3f} px"
        assert m.areas == 1 and not m.lost, tag + f" - lost track / phantom new area ({m.areas} areas)"
        assert r <= LOOP_RMS_MAX and e.max() <= LOOP_ERR_MAX, tag
    assert np.median(rms) <= LOOP_RMS_MEDIAN, f"{oname} {n_frames}: median RMS {np.median(rms):.3f} ({rms})"


@pytest.mark.parametrize("oname,n_frames", [("10x / 0.25", 40), ("40x / 0.85", 80)])
def test_closed_loop_tracking(oname, n_frames):
    _closed_loop(oname, n_frames, seeds=(1, 2))


@pytest.mark.slow
@pytest.mark.parametrize("n_frames", [160, 80, 40])
@pytest.mark.parametrize("oname", ["10x / 0.25", "40x / 0.85"])
def test_closed_loop_tracking_full(oname, n_frames):
    _closed_loop(oname, n_frames, seeds=range(1, 11))


# ------------------------------------------------------------------------------------------ loop closure + bias
# Pan 360 px out along x and back (10x, noise and motion blur on), every Mosaic._match x estimate biased by `bias`
# px. A constant bias cannot be fully removed by the pose graph, but loop closure must bound it: without loop
# closure the keyframe RMS at bias 0.5 is 1.8 px (step 10) and 3.6 px (step 4) on main.
# (step, bias) -> (bound on per-run keyframe RMS, bound on the median over seeds); measured on main, 8 seeds:
OAB = {
    (4.0, 0.0): (0.32, 0.15),  # RMS med 0.075 max 0.158; loop edges 34-42
    (4.0, 0.5): (1.38, 1.0),  # med 0.661 max 0.692; 37
    (4.0, 1.0): (2.55, 1.9),  # med 1.247 max 1.275; 37
    (10.0, 0.0): (0.18, 0.09),  # med 0.044 max 0.086; 7
    (10.0, 0.5): (1.24, 0.9),  # med 0.583 max 0.618; 7
    (10.0, 1.0): (2.3, 1.65),  # med 1.098 max 1.149; 7
}


def _out_and_back(step, bias, seeds):
    rms_bound, med_bound = OAB[(step, bias)]
    rms = []
    for seed in seeds:
        err, loops, final, m = out_and_back(seed, step, bias)
        r = math.sqrt(float(np.mean(err ** 2)))
        rms.append(r)
        tag = f"step {step} bias {bias} seed {seed}: keyframe RMS {r:.3f} px, {loops} loop edges, final {final:.2f}"
        assert loops >= 1, tag + " - the path returned over mapped ground but no loop edge was made"
        assert m.areas == 1, tag
        assert r <= rms_bound, tag
    assert np.median(rms) <= med_bound, f"step {step} bias {bias}: median RMS {np.median(rms):.3f} ({rms})"


@pytest.mark.parametrize("bias", [0.0, 0.5])
def test_loop_closure_under_bias(bias):
    _out_and_back(10.0, bias, seeds=(1, 2))


@pytest.mark.slow
@pytest.mark.parametrize("bias", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("step", [4.0, 10.0])
def test_loop_closure_under_bias_full(step, bias):
    _out_and_back(step, bias, seeds=range(1, 7))


# ------------------------------------------------------------------------------------------ revisit sharpness
# 40x: map the origin sharply, pan 64 px away, return to `offset` px and dwell defocused by dz DOF. Ratio =
# Laplacian variance of the mosaic over the original view / that of the original sharp frame (noise-free frames,
# see _gen.revisit). Measured on main, 8 seeds, offsets 1/3/6/15 px:
#   0 DOF:   min 0.949, medians 0.965-0.979 (the revisit keyframe is placed: 2 keyframes near the origin)
#   0.5 DOF: min 0.592, medians 0.828-0.844 (the defocused revisit passes the sharpness gate and is blended)
#   1.5 DOF: min 0.957, medians 0.969 (the defocused frame is refused: 1 keyframe near the origin)
REVISIT = {0.0: (0.90, 0.93, 2), 0.5: (0.50, 0.65, 2), 1.5: (0.90, 0.93, 1)}  # (min, median bound, keyframes)
# 0.5 DOF median bound: 0.83 over 8 seeds, 0.73 over the fast seeds (2, 7), which include the worst seed 7
FAST_SEEDS = (2, 7)


def _revisit(offset, dz, seeds):
    lo, med, near = REVISIT[dz]
    out = [revisit(seed, offset, dz) for seed in seeds]
    ratios = [r for r, _ in out]
    tag = f"offset {offset} px, defocus {dz} DOF: ratios {np.round(ratios, 3).tolist()}"
    assert all(n == near for _, n in out), tag + f", keyframes near origin {[n for _, n in out]} (expected {near})"
    assert min(ratios) >= lo, tag
    assert np.median(ratios) >= med, tag


@pytest.mark.parametrize("dz", [0.0, 0.5, 1.5])
@pytest.mark.parametrize("offset", [1.0, 15.0])
def test_revisit_sharpness(offset, dz):
    _revisit(offset, dz, FAST_SEEDS)


@pytest.mark.slow
@pytest.mark.parametrize("dz", [0.0, 0.5, 1.5])
@pytest.mark.parametrize("offset", [1.0, 3.0, 6.0, 15.0])
def test_revisit_sharpness_full(offset, dz):
    _revisit(offset, dz, range(1, 9))


_BLUR = ("known weakness on main: a revisit frame defocused by 0.5 DOF passes the keyframe sharpness gate "
         "(brenner >= 0.8 x recent median) and is feather-blended over the sharp original, so the mosaic over a "
         "revisited spot loses up to 41% of its Laplacian variance (min ratio 0.59, seed 7; median 0.83)")


@pytest.mark.xfail(strict=True, reason=_BLUR)
@pytest.mark.parametrize("offset", [1.0, 15.0])
def test_revisit_keeps_the_sharp_view(offset):
    ratios = [revisit(seed, offset, 0.5)[0] for seed in FAST_SEEDS]
    assert min(ratios) >= 0.75, f"offset {offset}: ratios {np.round(ratios, 3).tolist()}"


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=_BLUR)
@pytest.mark.parametrize("offset", [1.0, 3.0, 6.0, 15.0])
def test_revisit_keeps_the_sharp_view_full(offset):
    ratios = [revisit(seed, offset, dz)[0] for seed in range(1, 9) for dz in (0.0, 0.5, 1.5)]
    assert min(ratios) >= 0.75, f"offset {offset}: min ratio {min(ratios):.3f}"
