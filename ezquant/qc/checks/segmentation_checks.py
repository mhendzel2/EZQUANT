from __future__ import annotations

import numpy as np

from ezquant.qc.qc_types import QCItem, QCStatus


class SegmentationAreaCheck:
    id = "qc.segmentation.area_median"

    def run(self, masks: np.ndarray, policy: dict) -> QCItem:
        cfg = policy.get("qc_thresholds", {}).get("segmentation", {})
        min_area = float(cfg.get("min_median_area", 20.0))
        max_area = float(cfg.get("max_median_area", 5000.0))

        labels = np.unique(masks)
        labels = labels[labels > 0]
        if len(labels) == 0:
            median_area = 0.0
        else:
            areas = [float(np.sum(masks == lb)) for lb in labels]
            median_area = float(np.median(areas))

        status = QCStatus.PASS if min_area <= median_area <= max_area else QCStatus.FAIL
        return QCItem(
            id=self.id,
            status=status,
            message="Segmentation area in expected range" if status == QCStatus.PASS else "Median object area implausible",
            metric={"median_area_px": median_area},
            threshold={"min": min_area, "max": max_area},
            suggested_fix="Tune segmentation diameter and threshold",
        )
