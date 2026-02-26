"""
Role-based policy enforcement for EZQUANT.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import copy

import yaml

try:
    import bcrypt
except ImportError:  # pragma: no cover - runtime fallback if bcrypt missing
    bcrypt = None


@dataclass
class GateResult:
    """Outcome of a policy gate check."""

    allowed: bool
    message: str
    require_reason: bool
    log_entry: Dict[str, Any]


class PolicyEngine:
    """Loads lab policy YAML and enforces role-based restrictions."""

    def __init__(self, policy_path: Path):
        self.policy_path = Path(policy_path)
        self.policy = self._load_policy(self.policy_path)
        self.current_role: str = "student"

        if self.current_role not in self.policy.get("roles", {}):
            available = sorted(self.policy.get("roles", {}).keys())
            if not available:
                raise ValueError("Policy file does not define any roles")
            self.current_role = available[0]

    @staticmethod
    def _load_policy(policy_path: Path) -> Dict[str, Any]:
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

        data = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
        if "roles" not in data or not isinstance(data["roles"], dict):
            raise ValueError("Invalid policy file: missing 'roles' mapping")
        return data

    @property
    def role_config(self) -> Dict[str, Any]:
        """Return configuration for the active role."""
        roles = self.policy.get("roles", {})
        return copy.deepcopy(roles.get(self.current_role, {}))

    def set_role(self, role: str, auth_token: Optional[str] = None):
        """Switch active role. Elevated roles require authentication token."""
        roles = self.policy.get("roles", {})
        if role not in roles:
            raise ValueError(f"Unknown role: {role}")

        role_auth = self.policy.get("role_auth", {})
        expected_hash = role_auth.get(role)
        if expected_hash:
            if not auth_token:
                raise PermissionError(f"Authentication token required for role '{role}'")
            if not self._verify_token(auth_token, expected_hash):
                raise PermissionError("Invalid authentication token")

        self.current_role = role

    def can(self, action: str) -> bool:
        """Check whether the current role generally allows an action."""
        return self.gate(action).allowed

    def gate(self, action: str, context: Optional[Dict[str, Any]] = None) -> GateResult:
        """Evaluate action with optional context and return GateResult."""
        ctx = dict(context or {})
        require_reason = False
        allowed = True
        message = "Allowed"

        cfg = self.role_config
        needs_reason_policy = bool(self.policy.get("require_override_reason", False))

        if action == "export":
            export_mode = str(cfg.get("export", "ALL")).upper()
            qc_status = str(ctx.get("qc_status", "PASS")).upper()

            if export_mode == "PASS_ONLY":
                if qc_status != "PASS":
                    allowed = False
                    message = "Export blocked: this role can only export PASS QC data."
                else:
                    allowed = True
                    message = "Export allowed: QC status is PASS."
            elif export_mode == "PASS_WARN":
                if qc_status == "FAIL":
                    allowed = False
                    message = "Export blocked: QC FAIL must be resolved before export."
                elif qc_status == "WARN":
                    require_reason = needs_reason_policy
                    message = "Export allowed with warnings."
            elif export_mode == "ALL":
                message = "Export allowed by role policy."
            else:
                allowed = False
                message = f"Export blocked: unsupported export policy '{export_mode}'."

        elif action == "override_qc":
            allowed = bool(cfg.get("allow_overrides", False))
            require_reason = allowed and needs_reason_policy
            message = (
                "QC override allowed."
                if allowed
                else "QC override blocked for the current role."
            )

        elif action == "set_segmentation_params":
            allowed = bool(cfg.get("allow_manual_segmentation_params", True))
            message = (
                "Manual segmentation parameter edits are allowed."
                if allowed
                else "Locked: Student role requires A/B parameter optimization."
            )

        elif action == "batch_without_preview":
            if bool(cfg.get("allow_batch_without_preview", True)):
                allowed = True
                message = "Batch without preview is allowed."
            else:
                preview_completed = bool(ctx.get("preview_completed", False))
                allowed = preview_completed
                if allowed:
                    message = "Batch allowed after preview segmentation."
                else:
                    message = "Batch blocked: preview segmentation is required for this role."

        elif action == "export_raw_masks":
            allowed = bool(cfg.get("allow_raw_mask_export", False))
            message = (
                "Raw mask export is allowed."
                if allowed
                else "Raw mask export is disabled for this role."
            )

        elif action == "skip_tournament":
            if bool(cfg.get("require_tournament", False)):
                completed = bool(ctx.get("tournament_completed", False))
                allowed = completed
                message = (
                    "Tournament requirement satisfied."
                    if allowed
                    else "A/B tournament must be completed before segmentation."
                )
            else:
                allowed = True
                message = "Tournament is optional for this role."

        elif action == "edit_winning_params":
            max_dev = cfg.get("max_parameter_deviation", None)
            deviation = float(ctx.get("deviation", 0.0) or 0.0)

            if max_dev is None:
                allowed = True
                message = "Winning parameter edits are allowed."
            else:
                max_dev = float(max_dev)
                allowed = deviation <= max_dev
                if allowed:
                    message = "Winning parameter edit is within allowed bounds."
                else:
                    message = (
                        "Winning parameter edits exceed role limits and are blocked."
                    )

        elif action.startswith("use_recipe:"):
            recipe_name = action.split(":", 1)[1]
            allowed_recipes = cfg.get("allowed_recipes", [])

            if allowed_recipes == "ALL":
                allowed = True
            elif isinstance(allowed_recipes, list):
                allowed = "ALL" in allowed_recipes or recipe_name in allowed_recipes
            else:
                allowed = False

            message = (
                f"Recipe '{recipe_name}' is allowed for role '{self.current_role}'."
                if allowed
                else f"Recipe '{recipe_name}' is not allowed for role '{self.current_role}'."
            )

        else:
            allowed = False
            message = f"Unknown policy action: {action}"

        log_entry = {
            "timestamp": self._utc_now_iso(),
            "role": self.current_role,
            "action": action,
            "context": copy.deepcopy(ctx),
            "outcome": "allowed" if allowed else "denied",
            "message": message,
        }

        return GateResult(
            allowed=allowed,
            message=message,
            require_reason=require_reason,
            log_entry=log_entry,
        )

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _verify_token(token: str, expected_hash: str) -> bool:
        if not token or not expected_hash:
            return False

        if bcrypt is None:
            # Fallback for environments without bcrypt; compare directly.
            return token == expected_hash

        try:
            return bcrypt.checkpw(token.encode("utf-8"), expected_hash.encode("utf-8"))
        except ValueError:
            # Invalid hash format
            return False
