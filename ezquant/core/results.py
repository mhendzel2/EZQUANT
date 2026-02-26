from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalysisTable:
    name: str
    rows: list[dict[str, Any]]
    units: dict[str, str] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    tables: list[AnalysisTable]
    plots: list[str] = field(default_factory=list)
    overlays: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportArtifact:
    path: str
    sha256: str
    kind: str
