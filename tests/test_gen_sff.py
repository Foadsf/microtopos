"""Generalisation of shape-from-focus (zstack.reconstruct, known z) and its height metrology over analytic surfaces
(tilted plane, step, spherical cap, sinusoid), seeds (random geometry and texture), sensor noise and sweep
spacing. The sweep is rendered with a blur law that is not the reconstruction's optical model (tests/_surfaces.py),
and the truth Sa/Sq are computed with plain NumPy, not microtopos.metrology.

Measured on main (5242ef1), 6 seeds x noise {1, 4} grey levels x spacing {0.1, 0.2} um, 40x / 0.85, 0.5 um/px,
320 x 180 px; height RMS after removing the mean offset, 16 px border excluded; Sa/Sq errors in um:
    surface   RMS max   largest condition median   |Sq err| max   |Sa err| max   valid (min)
    slope     0.0225    0.0214 (noise 4)          0.0226 (Sq 0)  0.0176         0.98
    step      0.0246    0.0231                    0.0035         0.0026         0.94
    sphere    0.0287    0.0263                    0.0053         0.0058         0.80
    sinusoid  0.0470    0.0349                    0.0270 (-4%)   0.0249         0.98
Bounds: 2x the RMS and Sa/Sq maxima, 1.5x the largest condition median; valid >= 0.70 (as the self-test).
"""

import numpy as np
import pytest
from _surfaces import UM_PER_PX, sa_sq, sweep

from microtopos import zstack
from microtopos.objectives import default_set, resolve

OBJ = resolve("40x / 0.85", objset=default_set())
B = 16
# surface -> (RMS bound per sweep, bound on the median RMS of a run, |Sq err| bound, |Sa err| bound)
BOUNDS = {
    "slope": (0.045, 0.032, 0.045, 0.035),
    "step": (0.05, 0.035, 0.007, 0.0055),
    "sphere": (0.058, 0.04, 0.011, 0.012),
    "sinusoid": (0.095, 0.053, 0.055, 0.05),
}


def _sff(kind, runs):
    rms_b, med_b, sq_b, sa_b = BOUNDS[kind]
    rms = []
    for seed, noise, dz in runs:
        frames, zs, truth = sweep(kind, seed, noise, dz)
        res = zstack.reconstruct(frames, zs, OBJ, UM_PER_PX, "um")
        h = res["height"]
        c = (slice(B, -B), slice(B, -B))
        v = np.isfinite(h[c])
        d = h[c][v] - truth[c][v]
        r = float(np.sqrt(np.mean((d - d.mean()) ** 2)))
        rms.append(r)
        tsa, tsq = sa_sq(truth, np.isfinite(h))
        tag = (f"{kind} seed {seed} noise {noise} dz {dz}: RMS {r:.4f} um, valid {v.mean():.2f}, "
               f"Sq {res.get('Sq', float('nan')):.4f} vs {tsq:.4f}, Sa {res.get('Sa', float('nan')):.4f} vs {tsa:.4f}")
        assert v.mean() >= 0.70, tag
        assert r <= rms_b, tag
        assert abs(res["Sq"] - tsq) <= sq_b and abs(res["Sa"] - tsa) <= sa_b, tag
    assert np.median(rms) <= med_b, f"{kind}: median RMS {np.median(rms):.4f} ({np.round(rms, 4).tolist()})"


@pytest.mark.parametrize("kind", list(BOUNDS))
def test_sff_surface(kind):
    _sff(kind, [(1, 4.0, 0.2), (2, 1.0, 0.2)])


@pytest.mark.slow
@pytest.mark.parametrize("kind", list(BOUNDS))
def test_sff_surface_full(kind):
    _sff(kind, [(seed, noise, dz) for seed in range(1, 7) for noise in (1.0, 4.0) for dz in (0.1, 0.2)])
