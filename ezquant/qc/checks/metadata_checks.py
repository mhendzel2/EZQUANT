from __future__ import annotations

from ezquant.qc.qc_types import QCItem, QCStatus


class MetadataCompletenessCheck:
    id = "qc.metadata.completeness"

    def run(self, metadata: dict, policy: dict, recipe_name: str) -> QCItem:
        thresholds = policy.get("qc_thresholds", {}).get("metadata", {})
        px_range = thresholds.get("pixel_size_um_range", [0.01, 5.0])

        pixel_size = metadata.get("pixel_size_um")
        if pixel_size is None:
            return QCItem(
                id=self.id,
                status=QCStatus.FAIL,
                message="Pixel size is missing",
                metric=None,
                threshold={"required": "pixel_size_um"},
                suggested_fix="Provide calibrated pixel size metadata",
            )

        if not (px_range[0] <= float(pixel_size) <= px_range[1]):
            return QCItem(
                id=self.id,
                status=QCStatus.FAIL,
                message="Pixel size is outside plausible range",
                metric=float(pixel_size),
                threshold={"min": px_range[0], "max": px_range[1]},
                suggested_fix="Verify microscope calibration",
            )

        needs_time = recipe_name in {"frap_basic"}
        if needs_time and metadata.get("time_interval_s") is None:
            return QCItem(
                id=self.id,
                status=QCStatus.FAIL,
                message="Time interval metadata required for selected recipe",
                metric=None,
                threshold={"required": "time_interval_s"},
                suggested_fix="Provide frame interval metadata",
            )

        return QCItem(
            id=self.id,
            status=QCStatus.PASS,
            message="Required metadata present",
            metric={"pixel_size_um": pixel_size, "time_interval_s": metadata.get("time_interval_s")},
            threshold={"pixel_size_um_range": px_range},
            suggested_fix="",
        )
