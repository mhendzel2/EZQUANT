from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from ezquant.core.manifest import stable_manifest_hash
from ezquant.core.pipeline import Pipeline


def _write_good(path: Path) -> None:
    rng = np.random.default_rng(0)
    arr = (rng.normal(500, 40, size=(64, 64)).clip(0, 1000)).astype(np.uint16)
    tifffile.imwrite(path, arr)


def test_manifest_stable_for_same_input(tmp_path: Path):
    inp = tmp_path / "good_dataset.ome.tif"
    _write_good(inp)
    pipe = Pipeline("configs/lab_policy.default.yaml")

    r1 = pipe.run(
        input_path=str(inp),
        recipe_name="nuclear_intensity",
        role="instructor",
        output_dir=str(tmp_path / "run1"),
        override_reason="allowed",
    )
    r2 = pipe.run(
        input_path=str(inp),
        recipe_name="nuclear_intensity",
        role="instructor",
        output_dir=str(tmp_path / "run2"),
        override_reason="allowed",
    )

    h1 = stable_manifest_hash(r1["manifest"])
    h2 = stable_manifest_hash(r2["manifest"])
    assert h1 == h2
    assert r1["manifest"].inputs[0].sha256 == r2["manifest"].inputs[0].sha256
