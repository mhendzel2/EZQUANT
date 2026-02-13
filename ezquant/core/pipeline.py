from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

from ezquant.core.exceptions import ExportBlockedError, PolicyViolationError
from ezquant.core.manifest import InputManifest, PolicyManifest, RecipeManifest, RunManifest
from ezquant.core.policy import load_policy, resolve_recipe_params
from ezquant.core.registry import get_recipe_registry
from ezquant.export.exporter import Exporter
from ezquant.io.readers import detect_reader, read_image_with_metadata, sha256_file
from ezquant.qc.runner import run_global_qc
from ezquant.qc.qc_types import QCItem, QCReport, QCStatus
from ezquant.reporting.report_builder import build_report_html
from ezquant.segmentation.classical_nuclei import segment_nuclei


class Pipeline:
    def __init__(self, policy_path: str):
        self.policy = load_policy(policy_path)

    def run(
        self,
        input_path: str,
        recipe_name: str,
        role: str,
        output_dir: str,
        requested_params: dict[str, Any] | None = None,
        override_reason: str | None = None,
    ) -> dict[str, Any]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        reader = detect_reader(input_path)
        image, metadata = read_image_with_metadata(input_path)
        file_hash = sha256_file(input_path)

        manifest = RunManifest(user_role=role, platform=platform.platform())
        manifest.inputs.append(
            InputManifest(
                path=input_path,
                sha256=file_hash,
                reader_used=reader,
                parsed_metadata=metadata,
            )
        )
        manifest.lab_policy = PolicyManifest(
            policy_name=self.policy.name,
            policy_version=self.policy.version,
            policy_hash=self.policy.policy_hash,
        )

        registry = get_recipe_registry()
        if recipe_name not in registry:
            raise PolicyViolationError(f"Unknown recipe: {recipe_name}")
        recipe = registry[recipe_name]()
        defaults = recipe.get_default_params(self.policy.raw)
        params = resolve_recipe_params(recipe_name, defaults, requested_params, self.policy, role)
        manifest.recipe = RecipeManifest(name=recipe_name, version=recipe.describe().version, parameters=params)

        qc_pre = run_global_qc(image=image, metadata=metadata, policy=self.policy.raw, recipe_name=recipe_name)

        context: dict[str, Any] = {"image": image, "metadata": metadata, "output_dir": str(out_dir)}

        if recipe_name == "nuclear_intensity":
            masks = segment_nuclei(image, expected_diameter_px=int(params.get("expected_nucleus_diameter_px", 20)))
            context["masks"] = masks
            seg_items = self._segmentation_qc(masks, self.policy.raw)
            qc_pre.checks.extend(seg_items)
            qc_pre = QCReport.from_items(qc_pre.checks)

        manifest.qc_report = qc_pre.to_dict()

        if qc_pre.overall_status == QCStatus.FAIL and role == "student":
            report_path = out_dir / "report.html"
            report_path.write_text(build_report_html(manifest, qc_pre, None, watermark_failed=True), encoding="utf-8")
            manifest.artifacts.append({"path": str(report_path), "sha256": sha256_file(report_path), "kind": "report"})
            return {"manifest": manifest, "qc_report": qc_pre, "result": None, "report_path": str(report_path)}

        if qc_pre.overall_status == QCStatus.FAIL:
            role_cfg = self.policy.role(role)
            require_reason = self.policy.raw.get("require_override_reason", True)
            if not role_cfg.allow_overrides:
                raise ExportBlockedError("QC failed and overrides are disabled for role")
            if require_reason and not override_reason:
                raise ExportBlockedError("Override reason is required for failed QC export")
            manifest.override_reason = override_reason

        raw = recipe.run(inputs={"image": image}, params=params, context=context)
        post_items = recipe.postflight_checks(raw, context)
        all_items = qc_pre.checks + post_items
        qc_final = QCReport.from_items(all_items)
        manifest.qc_report = qc_final.to_dict()

        outputs = recipe.build_outputs(raw, context)
        report_html = build_report_html(
            manifest,
            qc_final,
            outputs,
            watermark_failed=(qc_final.overall_status == QCStatus.FAIL),
        )
        report_path = out_dir / "report.html"
        report_path.write_text(report_html, encoding="utf-8")

        exporter = Exporter(self.policy.raw)
        export_info = exporter.export(
            outputs=outputs,
            qc_report=qc_final,
            role=role,
            output_dir=str(out_dir),
            override_reason=override_reason,
        )

        manifest.export_status = export_info["export_status"]
        manifest.results_summary = outputs.summary
        manifest.artifacts.extend(export_info["artifacts"])
        manifest.artifacts.append({"path": str(report_path), "sha256": sha256_file(report_path), "kind": "report"})

        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

        return {
            "manifest": manifest,
            "qc_report": qc_final,
            "result": outputs,
            "report_path": str(report_path),
            "manifest_path": str(manifest_path),
        }

    def _segmentation_qc(self, masks, policy: dict[str, Any]) -> list[QCItem]:
        import numpy as np

        obj_count = int(masks.max())
        min_count = policy.get("qc_thresholds", {}).get("segmentation", {}).get("min_objects", 1)
        max_count = policy.get("qc_thresholds", {}).get("segmentation", {}).get("max_objects", 10000)
        status = QCStatus.PASS if min_count <= obj_count <= max_count else QCStatus.FAIL
        return [
            QCItem(
                id="qc.segmentation.object_count",
                status=status,
                message=f"Detected {obj_count} objects",
                metric=obj_count,
                threshold={"min": min_count, "max": max_count},
                suggested_fix="Adjust threshold/diameter and verify image contrast",
            )
        ]
