from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ezquant.core.exceptions import PolicyViolationError


@dataclass
class RolePolicy:
    allowed_recipes: list[str]
    export: str
    allow_overrides: bool


@dataclass
class ResolvedPolicy:
    raw: dict[str, Any]
    path: str
    policy_hash: str

    @property
    def name(self) -> str:
        return self.raw.get("policy_name", "unknown")

    @property
    def version(self) -> str:
        return self.raw.get("policy_version", "unknown")

    def role(self, role_name: str) -> RolePolicy:
        role = self.raw.get("roles", {}).get(role_name)
        if not role:
            raise PolicyViolationError(f"Role not configured: {role_name}")
        return RolePolicy(
            allowed_recipes=role.get("allowed_recipes", []),
            export=role.get("export", "PASS_ONLY"),
            allow_overrides=bool(role.get("allow_overrides", False)),
        )


def load_policy(policy_path: str | Path) -> ResolvedPolicy:
    path = Path(policy_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    normalized = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    policy_hash = hashlib.sha256(normalized).hexdigest()
    return ResolvedPolicy(raw=raw, path=str(path), policy_hash=policy_hash)


def resolve_recipe_params(
    recipe_name: str,
    default_params: dict[str, Any],
    requested_params: dict[str, Any] | None,
    policy: ResolvedPolicy,
    role: str,
) -> dict[str, Any]:
    requested_params = requested_params or {}
    role_cfg = policy.role(role)
    if recipe_name not in role_cfg.allowed_recipes:
        raise PolicyViolationError(f"Recipe '{recipe_name}' is not allowed for role '{role}'")

    recipe_cfg = policy.raw.get("recipes", {}).get(recipe_name, {})
    locked = recipe_cfg.get("locked_params", {})
    editable = set(recipe_cfg.get("student_editable_params", []))

    resolved = dict(default_params)
    for key, value in requested_params.items():
        if role == "student" and key not in editable:
            continue
        resolved[key] = value

    resolved.update(locked)

    bounds = recipe_cfg.get("bounds", {})
    for key, spec in bounds.items():
        if key not in resolved:
            continue
        val = resolved[key]
        if "min" in spec and val < spec["min"]:
            raise PolicyViolationError(f"Parameter '{key}' below minimum")
        if "max" in spec and val > spec["max"]:
            raise PolicyViolationError(f"Parameter '{key}' above maximum")

    return resolved
