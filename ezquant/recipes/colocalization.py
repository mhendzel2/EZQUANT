from __future__ import annotations

from ezquant.core.recipe_base import RecipeBase, RecipeDescription
from ezquant.core.results import AnalysisResult, AnalysisTable
from ezquant.qc.qc_types import QCItem


class ColocalizationRecipe(RecipeBase):
    def describe(self) -> RecipeDescription:
        return RecipeDescription("colocalization", "0.1.0", "Placeholder recipe", ["pixel_size_um", "channel_names"])

    def get_default_params(self, policy=None) -> dict:
        return {}

    def preflight_requirements(self) -> dict:
        return {"required_metadata": ["pixel_size_um", "channel_names"]}

    def run(self, inputs, params, context) -> dict:
        return {"rows": []}

    def postflight_checks(self, raw_results, context) -> list[QCItem]:
        return []

    def build_outputs(self, raw_results, context) -> AnalysisResult:
        return AnalysisResult(tables=[AnalysisTable(name="colocalization", rows=[])])
