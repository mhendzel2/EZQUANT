from __future__ import annotations

import numpy as np

from ezquant.qc.qc_types import QCItem, QCStatus


def _estimate_shift_2d(ref: np.ndarray, mov: np.ndarray) -> tuple[float, float]:
    f_ref = np.fft.fft2(ref)
    f_mov = np.fft.fft2(mov)
    cps = f_ref * np.conj(f_mov)
    cps /= np.maximum(np.abs(cps), 1e-12)
    corr = np.fft.ifft2(cps)
    y, x = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    h, w = ref.shape
    if y > h // 2:
        y -= h
    if x > w // 2:
        x -= w
    return float(y), float(x)


class DriftCheck:
    id = "qc.drift.translation"

    def run(self, image: np.ndarray, policy: dict) -> QCItem:
        thresholds = policy.get("qc_thresholds", {}).get("drift", {})
        max_pixels = float(thresholds.get("max_pixels", 2.0))

        if image.ndim < 3:
            drift_max = 0.0
            drift_rms = 0.0
        else:
            frames = image.shape[0]
            shifts = []
            ref = image[0].astype(float)
            for i in range(1, frames):
                sh = _estimate_shift_2d(ref, image[i].astype(float))
                shifts.append((sh[0] ** 2 + sh[1] ** 2) ** 0.5)
            drift_max = float(np.max(shifts)) if shifts else 0.0
            drift_rms = float(np.sqrt(np.mean(np.square(shifts)))) if shifts else 0.0

        status = QCStatus.PASS if drift_max <= max_pixels else QCStatus.FAIL
        return QCItem(
            id=self.id,
            status=status,
            message="Drift within threshold" if status == QCStatus.PASS else "Drift exceeds threshold",
            metric={"max_pixels": drift_max, "rms_pixels": drift_rms},
            threshold={"max_pixels": max_pixels},
            suggested_fix="Enable stage drift correction or exclude unstable frames",
        )
