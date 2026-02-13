from __future__ import annotations

import numpy as np

from ezquant.qc.checks.saturation_checks import SaturationCheck


def test_saturation_check_fails_when_clipped():
    image = np.full((32, 32), 65535, dtype=np.uint16)
    policy = {"qc_thresholds": {"saturation": {"max_fraction_global": 0.005}}}
    item = SaturationCheck().run(image, policy)
    assert item.status.value == "FAIL"
