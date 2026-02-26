from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ezquant.core.results import AnalysisResult
from ezquant.qc.qc_types import QCItem


@dataclass
class RecipeDescription:
    name: str
    version: str
    summary: str
    required_inputs: list[str]


class RecipeBase(ABC):
    @abstractmethod
    def describe(self) -> RecipeDescription:
        raise NotImplementedError

    @abstractmethod
    def get_default_params(self, policy: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def preflight_requirements(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def run(self, inputs: dict[str, Any], params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def postflight_checks(self, raw_results: dict[str, Any], context: dict[str, Any]) -> list[QCItem]:
        raise NotImplementedError

    @abstractmethod
    def build_outputs(self, raw_results: dict[str, Any], context: dict[str, Any]) -> AnalysisResult:
        raise NotImplementedError
