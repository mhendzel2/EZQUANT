from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from ezquant.core.exceptions import ExportBlockedError
from ezquant.core.pipeline import Pipeline


def _write_saturated_tiff(path: Path) -> None:
    arr = np.full((64, 64), 65535, dtype=np.uint16)
    tifffile.imwrite(path, arr)


def test_student_blocked_on_fail_qc(tmp_path: Path):
    inp = tmp_path / "bad_saturated.ome.tif"
    _write_saturated_tiff(inp)

    pipe = Pipeline("configs/lab_policy.default.yaml")
    result = pipe.run(
        input_path=str(inp),
        recipe_name="nuclear_intensity",
        role="student",
        output_dir=str(tmp_path / "out_student"),
    )

    assert result["qc_report"].overall_status.value == "FAIL"
    csvs = list((tmp_path / "out_student").glob("*.csv"))
    assert not csvs


def test_instructor_override_required_on_fail_qc(tmp_path: Path):
    inp = tmp_path / "bad_saturated.ome.tif"
    _write_saturated_tiff(inp)

    pipe = Pipeline("configs/lab_policy.default.yaml")

    with pytest.raises(ExportBlockedError):
        pipe.run(
            input_path=str(inp),
            recipe_name="nuclear_intensity",
            role="instructor",
            output_dir=str(tmp_path / "out_instr_fail"),
        )

    ok = pipe.run(
        input_path=str(inp),
        recipe_name="nuclear_intensity",
        role="instructor",
        output_dir=str(tmp_path / "out_instr_ok"),
        override_reason="Training demo override",
    )

    manifest = ok["manifest"]
    assert manifest.override_reason == "Training demo override"
    assert any(a["kind"] == "report" for a in manifest.artifacts)
