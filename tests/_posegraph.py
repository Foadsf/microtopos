"""Synthetic pose graphs for Mosaic._relax: true keyframe positions on a lawnmower path, odometry and loop edges
measured with noise, bias and optional outliers, and keyframes initialised at the DEAD-RECKONED (drifted)
positions, never at the truth."""

from __future__ import annotations

import math
import zlib

import numpy as np

from microtopos.mosaic import Mosaic

SPACING = 60.0  # px between neighbouring keyframes (a quarter of a 240 px field)
LOOP_RADIUS = 0.6 * 240  # Mosaic._loop_closure links keyframes closer than 0.6 x the frame's short side


def lawnmower(rows: int = 3, cols: int = 6) -> np.ndarray:
    """Serpentine path: row 0 left->right, row 1 right->left, ... (true camera positions, px)."""
    pts = []
    for r in range(rows):
        cs = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        pts += [(c * SPACING, r * SPACING) for c in cs]
    return np.array(pts, float)


def build(seed: int, noise: float = 0.2, bias: tuple = (0.0, 0.0), outlier: float = 0.0, rows: int = 3,
          cols: int = 6):
    """Mosaic whose keyframes sit at dead-reckoned positions (odometry integrated from keyframe 0), with odometry
    edges (i, i+1) and loop edges between non-consecutive keyframes within LOOP_RADIUS, all measured as
    truth + bias + N(0, noise). `outlier` px is added (perpendicular to the path) to one loop edge.
    Returns (mosaic, true positions)."""
    rng = np.random.default_rng(zlib.crc32(repr((seed, noise, bias, outlier, rows, cols)).encode()))
    P = lawnmower(rows, cols)
    n = len(P)
    meas = lambda i, j: P[j] - P[i] + np.asarray(bias) + rng.normal(0, noise, 2)
    odo = [meas(i, i + 1) for i in range(n - 1)]
    init = np.vstack([P[0], P[0] + np.cumsum(odo, axis=0)])
    m = Mosaic()
    img = np.full((8, 8), 128, np.uint8).astype(np.float32)
    fb = m._bp(np.full((16, 16), 128, np.float32))
    for i in range(n):
        m._add_keyframe(img, fb, 1.0, float(init[i, 0]), float(init[i, 1]), None, (0.0, 0.0))
    for i in range(n - 1):
        m.edges.append((i, i + 1, float(odo[i][0]), float(odo[i][1]), 1.0))
    loops = [(i, j) for i in range(n) for j in range(i + 2, n) if np.hypot(*(P[j] - P[i])) <= LOOP_RADIUS]
    for k, (i, j) in enumerate(loops):
        e = meas(i, j)
        if outlier and k == len(loops) // 2:
            e = e + np.array([0.0, outlier])
        m.edges.append((i, j, float(e[0]), float(e[1]), 1.0))
    m.x, m.y = float(init[-1, 0]), float(init[-1, 1])
    m.prev_xy = (m.x, m.y)
    m.ref = n - 1
    return m, P


def errors(m: Mosaic, P: np.ndarray) -> np.ndarray:
    return np.array([math.hypot(k.x - p[0], k.y - p[1]) for k, p in zip(m.keyframes, P)])
