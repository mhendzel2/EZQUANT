from __future__ import annotations

import numpy as np
from scipy.ndimage import laplace

from ezquant.qc.qc_types import QCItem, QCStatus


class FocusStabilityCheck:
    id = "qc.focus.laplacian_variance"

    def run(self, image: np.ndarray, policy: dict) -> QCItem:
        thresholds = policy.get("qc_thresholds", {}).get("focus", {})
        max_drop_fraction = float(thresholds.get("max_drop_fraction", 0.4))

        if image.ndim < 3:
            vars_ = [float(np.var(laplace(image.astype(float))))]
        else:
            vars_ = [float(np.var(laplace(frame.astype(float)))) for frame in image]

        first = max(vars_[0], 1e-12)
        minv = min(vars_)
        drop = 1.0 - (minv / first)
        if drop > max_drop_fraction:
            status = QCStatus.FAIL if drop > (max_drop_fraction + 0.2) else QCStatus.WARN
        else:
            status = QCStatus.PASS

        return QCItem(
            id=self.id,
            status=status,
            message="Focus stability acceptable" if status == QCStatus.PASS else "Focus appears unstable",
            metric={"drop_fraction": float(drop)},
            threshold={"max_drop_fraction": max_drop_fraction},
            suggested_fix="Refocus acquisition or trim out-of-focus frames",
        )
