from __future__ import annotations

import numpy as np

from ezquant.qc.checks.drift_checks import DriftCheck


def test_drift_check_detects_shift():
    base = np.zeros((64, 64), dtype=np.float32)
    base[20:30, 20:30] = 1.0
    shifted = np.roll(base, shift=5, axis=1)
    stack = np.stack([base, shifted], axis=0)
    policy = {"qc_thresholds": {"drift": {"max_pixels": 2.0}}}
    item = DriftCheck().run(stack, policy)
    assert item.status.value == "FAIL"
