"""N5: empty / mismatched sweeps are already rejected with a clear ValueError on this stack; these tests pin it.
(N6 and N7 are real defects; their failing tests are in tests/pending_neon_patches.py.)"""

from __future__ import annotations

import numpy as np
import pytest

from microtopos import objectives, zstack

OBJ = objectives.resolve("10x / 0.25", objset=objectives.default_set())


# ----------------------------------------------------------------------------------------------- N5
@pytest.mark.parametrize(
    ("frames", "labels", "match"),
    [
        ([], None, "empty sweep"),
        ([np.zeros((48, 64), np.float32)], np.array([0.0]), "Too few usable frames"),
        ([np.zeros((48, 64), np.float32), np.zeros((40, 64), np.float32)], np.array([0.0, 1.0]), "same size"),
        ([np.zeros((48, 64), np.float32)] * 2, np.array([0.0]), "z labels"),
    ],
)
def test_reconstruct_rejects_degenerate_sweeps_with_a_clear_error(frames, labels, match):
    with pytest.raises(ValueError, match=match):
        zstack.reconstruct(frames, labels, OBJ, 1.0, "um")
