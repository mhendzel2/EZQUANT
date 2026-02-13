from __future__ import annotations

import numpy as np

from ezquant.qc.qc_types import QCItem, QCStatus


class SaturationCheck:
    id = "qc.saturation.max_fraction"

    def run(self, image: np.ndarray, policy: dict) -> QCItem:
        thresholds = policy.get("qc_thresholds", {}).get("saturation", {})
        max_fraction_global = float(thresholds.get("max_fraction_global", 0.005))
        maxv = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else float(image.max())
        frac = float(np.mean(image >= maxv))
        status = QCStatus.PASS if frac <= max_fraction_global else QCStatus.FAIL
        msg = "Saturation within threshold" if status == QCStatus.PASS else "Image appears saturated/clipped"
        return QCItem(
            id=self.id,
            status=status,
            message=msg,
            metric={"fraction_at_max": frac},
            threshold={"max_fraction_global": max_fraction_global},
            suggested_fix="Lower exposure or gain during acquisition",
        )
