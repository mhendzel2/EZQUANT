from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QCStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class QCItem:
    id: str
    status: QCStatus
    message: str
    metric: Any
    threshold: Any
    evidence_artifacts: list[str] = field(default_factory=list)
    suggested_fix: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "message": self.message,
            "metric": self.metric,
            "threshold": self.threshold,
            "evidence_artifacts": self.evidence_artifacts,
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class QCReport:
    overall_status: QCStatus
    checks: list[QCItem]

    @staticmethod
    def from_items(items: list[QCItem]) -> "QCReport":
        statuses = [item.status for item in items]
        if QCStatus.FAIL in statuses:
            overall = QCStatus.FAIL
        elif QCStatus.WARN in statuses:
            overall = QCStatus.WARN
        else:
            overall = QCStatus.PASS
        return QCReport(overall_status=overall, checks=items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "checks": [c.to_dict() for c in self.checks],
        }
