"""Property-based testing for MicroTopos using Hypothesis."""

from __future__ import annotations

import math

import cv2
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from microtopos.mosaic import BacklashModel, PoseGraph, sharpness_score
from microtopos.validation import parse_crop


@given(
    x=st.integers(min_value=0, max_value=4000),
    y=st.integers(min_value=0, max_value=4000),
    w=st.integers(min_value=1, max_value=4000),
    h=st.integers(min_value=1, max_value=4000),
)
@settings(max_examples=50)
def test_crop_parsing_roundtrip(x: int, y: int, w: int, h: int):
    """Property: formatted 'x,y,w,h' strings parse deterministically back to their integer components."""
    spec = f"{x},{y},{w},{h}"
    px, py, pw, ph = parse_crop(spec)
    assert (px, py, pw, ph) == (x, y, w, h)


@given(
    offset=st.floats(min_value=-20.0, max_value=20.0),
)
@settings(max_examples=30)
def test_sharpness_dc_invariance(offset: float):
    """Property: adding a constant uniform illumination offset (DC shift) within linear sensor range does not alter Laplacian sharpness."""
    rng = np.random.default_rng(42)
    base = rng.uniform(80, 160, (64, 64)).astype(np.float32)
    cv2.circle(base, (32, 32), 15, 200, 2)
    s0 = sharpness_score(base)
    shifted = base + offset
    s_shifted = sharpness_score(shifted)
    # Inside the unclipped linear sensor range, spatial derivatives are invariant to DC offset
    assert math.isclose(s0, s_shifted, rel_tol=1e-4, abs_tol=1e-4)


@given(
    bx=st.floats(min_value=0.1, max_value=20.0),
    by=st.floats(min_value=0.1, max_value=20.0),
    step=st.floats(min_value=1.0, max_value=50.0),
)
@settings(max_examples=30)
def test_backlash_reversal_symmetry(bx: float, by: float, step: float):
    """Property: BacklashModel strictly applies backlash correction on direction reversal."""
    bm = BacklashModel(backlash_x=bx, backlash_y=by)
    # Forward motion: no correction
    cx0, cy0 = bm.apply(step, step)
    assert cx0 == 0.0 and cy0 == 0.0

    # Reverse motion: applies exactly negative backlash
    cx_rev, cy_rev = bm.apply(-step, -step)
    assert math.isclose(cx_rev, -bx, rel_tol=1e-5)
    assert math.isclose(cy_rev, -by, rel_tol=1e-5)


@given(
    val_um=st.floats(min_value=0.001, max_value=1e5),
)
@settings(max_examples=50)
def test_unit_conversion_roundtrip(val_um: float):
    """Property: dimensional unit roundtrips (um <-> mm <-> m) are strictly invertible."""
    val_mm = val_um / 1000.0
    val_m = val_mm / 1000.0
    recovered_um = val_m * 1e6
    assert math.isclose(val_um, recovered_um, rel_tol=1e-9)


@given(
    dx=st.floats(min_value=-200.0, max_value=200.0),
    dy=st.floats(min_value=-200.0, max_value=200.0),
)
@settings(max_examples=30)
def test_pose_graph_single_edge_recovery(dx: float, dy: float):
    """Property: PoseGraph relaxation on a single relative constraint recovers the exact target displacement."""
    pg = PoseGraph()
    pg.add_node(0, 0.0, 0.0)
    pg.add_node(1, 0.0, 0.0)
    pg.add_edge(0, 1, dx, dy, weight=10.0)
    opt = pg.optimize(anchor_id=0, anchor_weight=100.0, default_prior=0.001)

    rec_dx = opt[1][0] - opt[0][0]
    rec_dy = opt[1][1] - opt[0][1]
    assert math.isclose(rec_dx, dx, abs_tol=1e-2)
    assert math.isclose(rec_dy, dy, abs_tol=1e-2)
