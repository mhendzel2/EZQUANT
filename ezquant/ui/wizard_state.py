from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WizardStage(str, Enum):
    SELECT_DATA = "Select data"
    CHOOSE_RECIPE = "Choose recipe"
    CONFIRM_METADATA = "Confirm metadata"
    PREFLIGHT_QC = "Preflight QC"
    SEGMENTATION_PREVIEW = "Segmentation preview"
    RUN_ANALYSIS = "Run analysis"
    REVIEW_RESULTS = "Review results"
    EXPORT = "Export"
    SAVE_BUNDLE = "Save report bundle"


@dataclass
class WizardState:
    role: str
    qc_status: str | None = None
    override_reason: str | None = None
    stage: WizardStage = WizardStage.SELECT_DATA

    def can_run(self) -> bool:
        if self.qc_status in {"PASS", "WARN"}:
            return True
        if self.qc_status == "FAIL" and self.role != "student" and bool(self.override_reason):
            return True
        return False

    def next_stage(self) -> None:
        order = list(WizardStage)
        idx = order.index(self.stage)
        if idx < len(order) - 1:
            if order[idx + 1] == WizardStage.RUN_ANALYSIS and not self.can_run():
                return
            self.stage = order[idx + 1]
