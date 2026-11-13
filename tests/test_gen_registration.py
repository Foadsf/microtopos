"""Generalisation of frame registration: register_pair (sweep / super-resolution path, factor 1, translation) and
Mosaic._match (live tracking, revisit and loop-closure path) over seeds x objectives x conditions, plus negative
controls that must be rejected.

Pairs come from the package simulator (tests/_gen.py): 320 x 240 frames, simulated sensor noise and JPEG, a random
sub-pixel content shift per pair, every instrument unknown drawn from the seed. The default run uses a
representative subset (3 seeds x 2 objectives x 2 draws = 12 pairs per condition); `pytest -m slow` runs the full
matrix (12 seeds x 2 objectives x 4 draws = 96 pairs per condition) against the same bounds.

Every bound below was set from the full 96-pair distribution measured on main (5242ef1) with about 2x margin on
the maximum and on the median; the measured values are next to each bound (px, Euclidean error vs truth).
"""

import numpy as np
import pytest
from _gen import Cond, run_matrix

from microtopos.mosaic import Mosaic

FAST = {"seeds": (1, 5, 9), "draws": 2}
FULL = {"seeds": tuple(range(1, 13)), "draws": 4}

# cond -> (register_pair median, max, Mosaic._match median, max) bounds in px. Comments: measured on main, full
# matrix (n=96) "med / max", and the fast subset's median where the bound is close to it.
ACCURACY = {
    # rp 0.0112 / 0.0222 | match 0.0199 / 0.0489
    Cond("shift +-3 px"): (0.025, 0.045, 0.04, 0.10),
    # rp 0.0157 / 0.0620 | match 0.0369 / 0.1277
    Cond("shift +-20 px", shift=20): (0.035, 0.125, 0.075, 0.26),
    # rp 0.0125 / 0.0409 | match 0.0284 / 0.1163
    Cond("defocus +1 DOF", dz_dof=1.0): (0.025, 0.085, 0.06, 0.23),
    # rp 0.0335 / 0.1708 | match 0.0568 / 0.2649
    Cond("defocus +3 DOF", dz_dof=3.0): (0.07, 0.34, 0.12, 0.53),
    # rp 0.0103 / 0.0260 | match 0.0233 / 0.0643
    Cond("noise sigma 6", noise=6.0): (0.021, 0.052, 0.047, 0.13),
    # rp 0.0127 / 0.0433 | match 0.0219 / 0.0573
    Cond("motion blur 8 px", motion=8.0): (0.026, 0.087, 0.044, 0.115),
    # translation model on a rotated pair: error measured at the image centre
    # rp 0.1489 / 0.4317 (fast median 0.245) | match 0.1475 / 0.3595 (fast median 0.211)
    Cond("rotation 1 deg", deg=1.0): (0.31, 0.87, 0.30, 0.72),
    # rp 0.1842 / 0.4987 (fast median 0.251) | match 0.1560 / 0.4819
    Cond("scale 1.02", scale=1.02): (0.37, 1.0, 0.31, 0.96),
    # rp 0.0421 / 1.656 (max not bounded here: 3/96 accepted wrong, see test_accepted_results_are_correct)
    # | match 0.0633 / 0.2319
    Cond("contrast 5%", contrast=0.05): (0.085, None, 0.13, 0.46),
    # rp 0.0107 / 0.0232 (fast median 0.0122) | match 0.0194 / 0.0562
    Cond("textureless patch", textureless=True): (0.022, 0.047, 0.04, 0.11),
    # 75 % overlap (|shift| 80, 60 px): register_pair refuses by design (shift > 0.25 x frame; ok 0/96), so only
    # "refused or correct" applies to it | match 0.0486 / 0.1425, all 96 accepted (NCC >= 0.92)
    Cond("overlap 75%", fixed=(80.0, 60.0)): (None, None, 0.10, 0.29),
}


def _check_accuracy(cond, run):
    rp_med, rp_max, m_med, m_max = ACCURACY[cond]
    res = run_matrix(cond, **run)
    rp = np.array([r.rp_err for r in res])
    m = np.array([r.m_err for r in res])
    msg = (f"{cond.name}: register_pair med {np.median(rp):.4f} max {rp.max():.4f} "
           f"({sum(r.rp_ok for r in res)} ok); _match med {np.median(m):.4f} max {m.max():.4f} (n={len(res)})")
    # a correctly matchable pair must be accepted by _match (and by register_pair within its design range)
    assert all(r.m_q >= Mosaic.TRACK_NCC for r in res), msg + f"; min NCC {min(r.m_q for r in res):.3f}"
    if rp_med is None:  # outside register_pair's design range: refusing is right, a confident wrong answer is not
        assert not any(r.rp_ok and r.rp_err > 0.5 for r in res), msg
    else:
        assert all(r.rp_ok for r in res), msg + f"; register_pair rejected {sum(not r.rp_ok for r in res)}"
        assert np.median(rp) <= rp_med, msg
        assert rp_max is None or rp.max() <= rp_max, msg
    assert np.median(m) <= m_med, msg
    assert m.max() <= m_max, msg


@pytest.mark.parametrize("cond", list(ACCURACY), ids=lambda c: c.name)
def test_accuracy(cond):
    _check_accuracy(cond, FAST)


@pytest.mark.slow
@pytest.mark.parametrize("cond", list(ACCURACY), ids=lambda c: c.name)
def test_accuracy_full(cond):
    _check_accuracy(cond, FULL)


# ------------------------------------------------------------------------------------- accepted means correct
# Conditions where an estimate can be badly wrong. Whatever a pipeline ACCEPTS must still be within 0.5 px:
# register_pair accepts when ok is True; _match is accepted for tracking at NCC >= TRACK_NCC and for loop closure
# at NCC >= LOOP_NCC. Measured on main, full matrix: accepted with error > 0.5 px out of 96 (largest such error).
HARD = {
    "grating": Cond("periodic grating 9 x 13 px", grating=True),  # rp 18 (45 px), track 17 (38 px), loop 17
    # 50 % overlap (|dx| 160 px): rp ok 0/96 (refuses); _match accepts about half, all within 0.06 px, and rejects
    # the rest (the phase peak wraps to the other half-frame)
    "overlap50": Cond("overlap 50%", fixed=(160.0, 0.0)),
    "rot3": Cond("rotation 3 deg", deg=3.0),  # rp 85 (3.6 px), track 73 (2.2 px), loop 26 (1.3 px)
    "contrast5": Cond("contrast 5%", contrast=0.05),  # rp 3 (1.7 px), track 0, loop 0
    "contrast2": Cond("contrast 2%", contrast=0.02),  # rp 18 (17 px), track 0 (NCC 0.56-0.80), loop 0
    "contrast1": Cond("contrast 1%", contrast=0.01),  # rp 34 (53 px), track 0 (NCC max 0.53), loop 0
}
GATES = {
    "register_pair": lambda r: r.rp_ok,
    "match_track": lambda r: r.m_q >= Mosaic.TRACK_NCC,
    "match_loop": lambda r: r.m_q >= Mosaic.LOOP_NCC,
}
_RP_BUG = ("register_pair accepts (ok=True) a wrong translation: with a qualified phase peak it keeps any ECC "
           "result near it, with no minimum correlation (registration.py:94)")
# (gate, cond) -> reason, for combinations that fail on main (strict: a fix turns them red, so the mark is removed
# with the fix). The fast subset fails on the same combinations (fast counts out of 12 in brackets).
KNOWN_BAD = {
    ("register_pair", "grating"): _RP_BUG + "; periodic texture, off by whole periods (18/96 pairs, up to 45 px) [3]",
    ("match_track", "grating"): "phase correlation locks onto a wrong period of a grating (17/96, up to 38 px) [3]",
    ("match_loop", "grating"): "phase correlation locks onto a wrong period of a grating (17/96, up to 38 px) [3]",
    ("register_pair", "rot3"): _RP_BUG + "; translation model under 3 deg rotation (85/96, up to 3.6 px) [11]",
    ("match_track", "rot3"): "NCC 0.59-0.69 under 3 deg rotation passes TRACK_NCC 0.5 (73/96, up to 2.2 px) [12]",
    ("match_loop", "rot3"): "NCC up to 0.69 under 3 deg rotation passes LOOP_NCC 0.65 (26/96, up to 1.3 px) [1]",
    ("register_pair", "contrast5"): _RP_BUG + "; 5% contrast (3/96, up to 1.7 px) [2]",
    ("register_pair", "contrast2"): _RP_BUG + "; 2% contrast (18/96, up to 17 px) [4]",
    ("register_pair", "contrast1"): _RP_BUG + "; 1% contrast (34/96, up to 53 px) [3]",
}


def _accepted_correct(gate, key, run):
    res = run_matrix(HARD[key], **run)
    acc = [r for r in res if GATES[gate](r)]
    bad = [r for r in acc if r.m_err > 0.5] if gate != "register_pair" else [r for r in acc if r.rp_err > 0.5]
    errs = [r.rp_err if gate == "register_pair" else r.m_err for r in bad]
    assert not bad, (f"{gate} on {HARD[key].name}: {len(bad)}/{len(res)} accepted with error > 0.5 px "
                     f"(max {max(errs):.2f} px); {len(acc)} accepted")


def _params(known):
    out = []
    for gate in GATES:
        for key in HARD:
            reason = known.get((gate, key))
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            out.append(pytest.param(gate, key, marks=marks, id=f"{gate}-{key}"))
    return out


@pytest.mark.parametrize("gate,key", _params(KNOWN_BAD))
def test_accepted_results_are_correct(gate, key):
    _accepted_correct(gate, key, FAST)


@pytest.mark.slow
@pytest.mark.parametrize("gate,key", _params(KNOWN_BAD))
def test_accepted_results_are_correct_full(gate, key):
    _accepted_correct(gate, key, FULL)


# ------------------------------------------------------------------------------------- unrelated frames: reject
# No true shift exists, so nothing may be accepted. The far frames come from a world 3.5 fields wide, so they are
# real specimen 1.2 (or 1.1 x 1.1) fields away, not a reflection of the border.
UNRELATED = {
    "no-overlap": Cond("no overlap, 1.2 FOV in x", far=(1.2 * 320, 0.0)),
    "no-overlap-diag": Cond("no overlap, 1.1 FOV diagonal", far=(1.1 * 320, 1.1 * 240)),
    "other-specimen": Cond("different specimen", other_specimen=True),
}
# seeds 7 and 10 with all 4 draws contain reproducers of the register_pair false accepts on main (no overlap:
# 40x seed 7 draw 3, 40x seed 10 draw 2; different specimen: 40x seed 10 draw 3)
NEG_FAST = {"seeds": (7, 10), "draws": 4}
_RP_NEG = ("register_pair reports ok=True (method 'ecc') for frames that share no content: ECC correlation as low "
           "as 0.04-0.39 is accepted because the random phase peak qualified (registration.py:94); measured on main ")
RP_NEG_BUGS = {"no-overlap": _RP_NEG + "6/96 pairs (ECC 0.11-0.39) [fast 2/16]",
               "no-overlap-diag": _RP_NEG + "7/96 pairs (ECC 0.04-0.32)",
               "other-specimen": _RP_NEG + "5/96 pairs (ECC 0.19-0.26) [fast 1/16]"}


def _unrelated(gate, key, run):
    res = run_matrix(UNRELATED[key], **run)
    if gate == "register_pair":
        acc = [r for r in res if r.rp_ok]
    else:
        # not even the weak, prediction-gated path may take it: NCC must stay below WEAK_NCC (0.25)
        # measured on main, full matrix: max NCC 0.128 (no overlap), 0.114 (diagonal), 0.108 (different specimen)
        acc = [r for r in res if r.m_q >= Mosaic.WEAK_NCC]
    assert not acc, f"{gate} accepted {len(acc)}/{len(res)} unrelated pairs ({UNRELATED[key].name})"


def _neg_params(fast: bool):
    out = []
    for gate in ("register_pair", "match"):
        for key in UNRELATED:
            if fast and key == "no-overlap-diag":
                continue
            marks = [pytest.mark.xfail(strict=True, reason=RP_NEG_BUGS[key])] if gate == "register_pair" else []
            out.append(pytest.param(gate, key, marks=marks, id=f"{gate}-{key}"))
    return out


@pytest.mark.parametrize("gate,key", _neg_params(fast=True))
def test_unrelated_frames_rejected(gate, key):
    _unrelated(gate, key, NEG_FAST)


@pytest.mark.slow
@pytest.mark.parametrize("gate,key", _neg_params(fast=False))
def test_unrelated_frames_rejected_full(gate, key):
    _unrelated(gate, key, FULL)
