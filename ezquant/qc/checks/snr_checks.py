from __future__ import annotations

import numpy as np

from ezquant.qc.qc_types import QCItem, QCStatus


class SNRCheck:
    id = "qc.snr.robust_proxy"

    def run(self, image: np.ndarray, policy: dict) -> QCItem:
        thresholds = policy.get("qc_thresholds", {}).get("snr", {})
        warn_min = float(thresholds.get("warn_min", 2.0))
        fail_min = float(thresholds.get("fail_min", 1.2))

        sig = float(np.percentile(image, 95))
        bg = float(np.percentile(image, 10))
        noise = float(np.percentile(image, 60) - np.percentile(image, 40))
        noise = max(noise, 1e-6)
        snr = (sig - bg) / noise

        if snr < fail_min:
            status = QCStatus.FAIL
        elif snr < warn_min:
            status = QCStatus.WARN
        else:
            status = QCStatus.PASS

        return QCItem(
            id=self.id,
            status=status,
            message="SNR acceptable" if status == QCStatus.PASS else "SNR is low",
            metric={"snr_proxy": snr},
            threshold={"warn_min": warn_min, "fail_min": fail_min},
            suggested_fix="Increase signal, reduce background, or denoise carefully",
        )
