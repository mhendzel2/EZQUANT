from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class InputManifest:
    path: str
    sha256: str
    reader_used: str
    parsed_metadata: dict[str, Any]


@dataclass
class RecipeManifest:
    name: str
    version: str
    parameters: dict[str, Any]


@dataclass
class PolicyManifest:
    policy_name: str
    policy_version: str
    policy_hash: str


@dataclass
class RunManifest:
    run_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ezquant_version: str = "0.1.0"
    git_commit: str | None = None
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    dependencies: list[str] = field(default_factory=list)
    user_role: str = "student"
    lab_policy: PolicyManifest | None = None
    inputs: list[InputManifest] = field(default_factory=list)
    recipe: RecipeManifest | None = None
    preprocessing_steps: list[dict[str, Any]] = field(default_factory=list)
    qc_report: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    results_summary: dict[str, Any] = field(default_factory=dict)
    export_status: str = "PASS_ONLY"
    override_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def stable_manifest_hash(manifest: RunManifest, ignored_fields: set[str] | None = None) -> str:
    ignored_fields = ignored_fields or {"run_id", "timestamp_utc"}
    data = manifest.to_dict()
    for key in ignored_fields:
        data.pop(key, None)
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
