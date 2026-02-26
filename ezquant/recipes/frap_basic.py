from __future__ import annotations

from ezquant.core.recipe_base import RecipeBase, RecipeDescription
from ezquant.core.results import AnalysisResult, AnalysisTable
from ezquant.qc.qc_types import QCItem


class FrapBasicRecipe(RecipeBase):
    def describe(self) -> RecipeDescription:
        return RecipeDescription("frap_basic", "0.1.0", "Optional FRAP recipe placeholder", ["time_interval_s"])

    def get_default_params(self, policy=None) -> dict:
        return {}

    def preflight_requirements(self) -> dict:
        return {"required_metadata": ["time_interval_s"]}

    def run(self, inputs, params, context) -> dict:
        return {"rows": []}

    def postflight_checks(self, raw_results, context) -> list[QCItem]:
        return []

    def build_outputs(self, raw_results, context) -> AnalysisResult:
        return AnalysisResult(tables=[AnalysisTable(name="frap", rows=[])])
