from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ezquant.core.recipe_base import RecipeBase, RecipeDescription
from ezquant.core.results import AnalysisResult, AnalysisTable
from ezquant.qc.qc_types import QCItem, QCStatus


def _annulus_background(image: np.ndarray, mask: np.ndarray, inner: int = 2, outer: int = 6) -> float:
    dil_outer = ndi.binary_dilation(mask, iterations=outer)
    dil_inner = ndi.binary_dilation(mask, iterations=inner)
    ring = dil_outer & (~dil_inner)
    if np.sum(ring) == 0:
        return float(np.median(image))
    return float(np.median(image[ring]))


class NuclearIntensityRecipe(RecipeBase):
    def describe(self) -> RecipeDescription:
        return RecipeDescription(
            name="nuclear_intensity",
            version="1.0.0",
            summary="Per-nucleus mean intensity with annulus background correction",
            required_inputs=["pixel_size_um", "channel_names"],
        )

    def get_default_params(self, policy=None) -> dict:
        return {
            "expected_nucleus_diameter_px": 20,
            "background_method": "annulus",
            "flatfield": "off",
        }

    def preflight_requirements(self) -> dict:
        return {"required_metadata": ["pixel_size_um", "channel_names"], "channels": "single_or_first"}

    def run(self, inputs: dict, params: dict, context: dict) -> dict:
        image = inputs["image"]
        if image.ndim > 2:
            image2d = image[0].astype(float)
        else:
            image2d = image.astype(float)
        masks = context["masks"]

        labels = np.unique(masks)
        labels = labels[labels > 0]
        rows = []
        neg_count = 0
        for lb in labels:
            obj = masks == lb
            raw_mean = float(np.mean(image2d[obj]))
            bg = _annulus_background(image2d, obj)
            corr = raw_mean - bg
            if corr < 0:
                neg_count += 1
            rows.append(
                {
                    "cell_id": int(lb),
                    "mean_intensity_raw": raw_mean,
                    "background": bg,
                    "mean_intensity_bg_corrected": corr,
                    "area_px": int(np.sum(obj)),
                }
            )

        return {"rows": rows, "negative_fraction": (neg_count / max(len(rows), 1))}

    def postflight_checks(self, raw_results: dict, context: dict) -> list[QCItem]:
        frac = float(raw_results.get("negative_fraction", 0.0))
        status = QCStatus.PASS if frac <= 0.05 else QCStatus.WARN if frac <= 0.2 else QCStatus.FAIL
        return [
            QCItem(
                id="qc.recipe.nuclear_intensity.negative_bg_corrected_fraction",
                status=status,
                message="Background-corrected intensity plausibility",
                metric={"negative_fraction": frac},
                threshold={"warn": 0.05, "fail": 0.2},
                suggested_fix="Verify background annulus and acquisition dynamic range",
            )
        ]

    def build_outputs(self, raw_results: dict, context: dict) -> AnalysisResult:
        rows = raw_results["rows"]
        table = AnalysisTable(
            name="nuclear_intensity_per_cell",
            rows=rows,
            units={
                "mean_intensity_raw": "a.u.",
                "background": "a.u.",
                "mean_intensity_bg_corrected": "a.u.",
                "area_px": "px^2",
            },
        )
        vals = [r["mean_intensity_bg_corrected"] for r in rows] if rows else [0.0]
        summary = {
            "n_cells": len(rows),
            "mean_intensity_bg_corrected": float(np.mean(vals)),
            "normalization": "background-subtracted annulus",
        }
        return AnalysisResult(tables=[table], overlays=[], plots=[], summary=summary)
