"""Generalisation of the pose-graph relaxation (Mosaic._relax). Keyframes start at DEAD-RECKONED positions
(odometry integrated with noise and bias, never the truth); odometry and loop edges carry noise and a bias.
Two graph shapes (3 x 6 and 2 x 8 lawnmower paths, 60 px spacing, loop edges between keyframes within 0.6 of a
240 px field as in Mosaic._loop_closure), 20 seeds in the slow run, 4 in the default run.

Measured on main (5242ef1), 20 seeds per row, px (RMS / max error over keyframes after relaxation):
    noise  bias          3x6: RMS max  err max   2x8: RMS max  err max   dead-reckoned RMS (min..max)
    0.2    (0, 0)        0.327         0.740     0.452         0.666     0.33 .. 1.90
    0.2    (0.3, 0)      0.581         0.981     0.607         0.847     1.64 .. 4.21
    0.5    (0.3, -0.2)   1.143         1.807     0.973         1.661     1.43 .. 5.86
    0.2    (1.0, 0)      1.422         2.520     1.132         1.910     6.22 .. 10.87
"""

import numpy as np
import pytest
from _posegraph import build, errors

SHAPES = [(3, 6), (2, 8)]
# (noise, bias) -> (bound on per-graph RMS error, bound on per-graph max error, bound on the median RMS)
CASES = {
    (0.2, (0.0, 0.0)): (0.9, 1.5, 0.5),  # RMS max 0.452, err max 0.740, RMS median 0.24-0.27
    (0.2, (0.3, 0.0)): (1.2, 2.0, 0.8),  # 0.607, 0.981, medians 0.32-0.41
    (0.5, (0.3, -0.2)): (2.3, 3.6, 1.0),  # 1.143, 1.807, medians 0.52
    (0.2, (1.0, 0.0)): (2.9, 5.0, 2.0),  # 1.422, 2.520, medians 1.00-1.31
}


def _relax(case, shape, seeds):
    rms_b, max_b, med_b = CASES[case]
    noise, bias = case
    rms = []
    for seed in seeds:
        m, P = build(seed, noise, bias, rows=shape[0], cols=shape[1])
        before = errors(m, P)
        k0 = (m.keyframes[0].x, m.keyframes[0].y)
        m._relax()
        e = errors(m, P)
        r = float(np.sqrt(np.mean(e ** 2)))
        r0 = float(np.sqrt(np.mean(before ** 2)))
        rms.append(r)
        tag = f"{shape} noise {noise} bias {bias} seed {seed}: RMS {r0:.3f} -> {r:.3f}, max {e.max():.3f} px"
        assert r <= rms_b and e.max() <= max_b, tag
        # the gauge: keyframe 0 carries the strong prior and must not move
        assert abs(m.keyframes[0].x - k0[0]) < 0.05 and abs(m.keyframes[0].y - k0[1]) < 0.05, tag
        if r0 > 1.0:  # a clearly drifted start must be pulled toward the truth, not left where it was
            assert r < 0.6 * r0, tag
    assert np.median(rms) <= med_b, f"{shape} noise {noise} bias {bias}: median RMS {np.median(rms):.3f}"


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("case", list(CASES), ids=str)
def test_relax_from_drifted_start(case, shape):
    _relax(case, shape, seeds=range(1, 5))


@pytest.mark.slow
@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("case", list(CASES), ids=str)
def test_relax_from_drifted_start_full(case, shape):
    _relax(case, shape, seeds=range(1, 21))


_OUTLIER = ("Mosaic._relax is plain least squares: one false loop edge 20 px off (it passes _loop_closure's gate, "
            "which admits up to 0.1 x 240 = 24 px of disagreement) spreads over the whole graph. Measured on main, "
            "20 seeds, noise 0.2: max error 0.67-0.74 px clean -> 1.79-1.98 (3x6) and 3.38-3.99 px (2x8) with the "
            "outlier (2.4-6x the clean worst case; the unrelaxed dead-reckoned start has RMS median 0.55-0.71 px)")


def _outlier(shape, seeds):
    worst = []
    for seed in seeds:
        m, P = build(seed, 0.2, (0.0, 0.0), outlier=20.0, rows=shape[0], cols=shape[1])
        m._relax()
        worst.append(float(errors(m, P).max()))
    # 2x the clean max error (0.74 px): a robust relaxation would leave the outlier's neighbours near it
    assert max(worst) <= 1.5, f"{shape}: max error per seed {np.round(worst, 2).tolist()}"


@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_relax_survives_one_false_loop_edge(shape):
    _outlier(shape, seeds=range(1, 5))


@pytest.mark.slow
@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_relax_survives_one_false_loop_edge_full(shape):
    _outlier(shape, seeds=range(1, 21))
