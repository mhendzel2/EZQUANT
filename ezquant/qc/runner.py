from __future__ import annotations

import numpy as np

from ezquant.qc.checks.drift_checks import DriftCheck
from ezquant.qc.checks.focus_checks import FocusStabilityCheck
from ezquant.qc.checks.metadata_checks import MetadataCompletenessCheck
from ezquant.qc.checks.saturation_checks import SaturationCheck
from ezquant.qc.checks.snr_checks import SNRCheck
from ezquant.qc.qc_types import QCReport


def run_global_qc(image: np.ndarray, metadata: dict, policy: dict, recipe_name: str) -> QCReport:
    checks = [
        MetadataCompletenessCheck().run(metadata, policy, recipe_name),
        SaturationCheck().run(image, policy),
        DriftCheck().run(image, policy),
        FocusStabilityCheck().run(image, policy),
        SNRCheck().run(image, policy),
    ]
    return QCReport.from_items(checks)
