from __future__ import annotations

from pathlib import Path

import pandas as pd

from ezquant.core.exceptions import ExportBlockedError
from ezquant.io.readers import sha256_file
from ezquant.qc.qc_types import QCStatus


class Exporter:
    def __init__(self, policy: dict):
        self.policy = policy

    def export(self, outputs, qc_report, role: str, output_dir: str, override_reason: str | None = None) -> dict:
        role_cfg = self.policy.get("roles", {}).get(role, {})
        export_mode = role_cfg.get("export", "PASS_ONLY")
        allow_overrides = bool(role_cfg.get("allow_overrides", False))
        require_reason = bool(self.policy.get("require_override_reason", True))

        if qc_report.overall_status == QCStatus.FAIL:
            if not allow_overrides:
                raise ExportBlockedError("Export blocked: QC FAIL")
            if require_reason and not override_reason:
                raise ExportBlockedError("Export blocked: override_reason required")
            export_status = "INCLUDED_FAIL_WITH_OVERRIDE"
        elif qc_report.overall_status == QCStatus.WARN and export_mode not in {"PASS_WARN", "PASS_ONLY"}:
            raise ExportBlockedError("Export mode not recognized")
        elif qc_report.overall_status == QCStatus.WARN and export_mode == "PASS_ONLY":
            raise ExportBlockedError("Export blocked: WARN not allowed for this role")
        else:
            export_status = export_mode

        artifacts = []
        qc_reasons = "; ".join([f"{c.id}:{c.status.value}" for c in qc_report.checks if c.status != QCStatus.PASS])

        for table in outputs.tables:
            df = pd.DataFrame(table.rows)
            df["qc_status"] = qc_report.overall_status.value
            df["qc_reasons"] = qc_reasons
            out_path = Path(output_dir) / f"{table.name}.csv"
            df.to_csv(out_path, index=False)
            artifacts.append({"path": str(out_path), "sha256": sha256_file(out_path), "kind": "csv"})

        return {"export_status": export_status, "artifacts": artifacts}
