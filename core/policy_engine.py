"""
PolicyEngine: Role-based access control for EZQUANT.
Loads lab_policy.yaml and enforces role-based restrictions at decision points.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

VALID_ROLES = ("student", "instructor", "research")

# Sentinel value used in policy YAML to indicate no passphrase is configured
_PLACEHOLDER_HASH_PREFIX = "$2b$12$placeholder"

ACTIONS = {
    "export",
    "override_qc",
    "set_segmentation_params",
    "batch_without_preview",
    "export_raw_masks",
    "skip_tournament",
    "edit_winning_params",
}


@dataclass
class GateResult:
    allowed: bool
    message: str
    require_reason: bool = False
    log_entry: dict = field(default_factory=dict)


class PolicyEngine:
    """Loads lab_policy.yaml and enforces role-based restrictions."""

    def __init__(self, policy_path: Path | str):
        self._path = Path(policy_path)
        self.policy: dict = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        self.current_role: str = "student"

    def set_role(self, role: str, auth_token: str | None = None) -> bool:
        """Set role. instructor/research roles require a passphrase."""
        if role not in VALID_ROLES:
            raise ValueError(f"Unknown role: {role!r}. Valid roles: {VALID_ROLES}")
        if role == "student":
            self.current_role = "student"
            return True

        role_auth = self.policy.get("role_auth", {}).get(role, {})
        stored_hash = role_auth.get("passphrase_hash", "")

        # If no hash configured or it's a placeholder, allow switching freely
        if not stored_hash or stored_hash.startswith(_PLACEHOLDER_HASH_PREFIX):
            self.current_role = role
            return True

        if auth_token is None:
            return False

        # Use bcrypt if available, otherwise fallback to sha256
        try:
            import bcrypt  # type: ignore
            token_bytes = auth_token.encode("utf-8")
            hash_bytes = stored_hash.encode("utf-8")
            if bcrypt.checkpw(token_bytes, hash_bytes):
                self.current_role = role
                return True
            return False
        except ImportError:
            # Fallback: compare sha256 of token against stored hash
            token_hash = hashlib.sha256(auth_token.encode()).hexdigest()
            if token_hash == stored_hash:
                self.current_role = role
                return True
            return False

    @property
    def role_config(self) -> dict:
        """Return the full config dict for the current role."""
        return self.policy.get("roles", {}).get(self.current_role, {})

    def can(self, action: str) -> bool:
        """Check if current role permits a given action."""
        return self.gate(action).allowed

    def gate(self, action: str, context: dict | None = None) -> GateResult:
        """Return a GateResult with allow/deny plus a user-facing message."""
        context = context or {}
        role_cfg = self.role_config
        require_reason = bool(self.policy.get("require_override_reason", True))

        log_entry = {
            "role": self.current_role,
            "action": action,
            "context": context,
        }

        # recipe-specific action
        if action.startswith("use_recipe:"):
            recipe_name = action.split(":", 1)[1]
            allowed_recipes = role_cfg.get("allowed_recipes", [])
            if allowed_recipes == "ALL" or recipe_name in allowed_recipes:
                return GateResult(True, f"Recipe '{recipe_name}' allowed.", log_entry=log_entry)
            return GateResult(
                False,
                f"Locked: '{self.current_role}' role cannot use recipe '{recipe_name}'.",
                log_entry=log_entry,
            )

        if action == "export":
            qc_status = context.get("qc_status", "PASS")
            export_mode = role_cfg.get("export", "PASS_ONLY")
            if export_mode == "ALL":
                return GateResult(True, "Export allowed.", log_entry=log_entry)
            if qc_status == "FAIL":
                if not role_cfg.get("allow_overrides", False):
                    return GateResult(
                        False,
                        "Locked: Student role cannot export data that failed QC. "
                        "All QC checks must pass before export.",
                        require_reason=False,
                        log_entry=log_entry,
                    )
                return GateResult(
                    True,
                    "Export allowed with QC override.",
                    require_reason=require_reason,
                    log_entry=log_entry,
                )
            if qc_status == "WARN" and export_mode == "PASS_ONLY":
                return GateResult(
                    False,
                    "Locked: Student role cannot export data with QC warnings.",
                    log_entry=log_entry,
                )
            return GateResult(True, "Export allowed.", log_entry=log_entry)

        if action == "override_qc":
            if not role_cfg.get("allow_overrides", False):
                return GateResult(
                    False,
                    "Locked: Student role cannot override QC failures. "
                    "Contact your instructor.",
                    log_entry=log_entry,
                )
            return GateResult(
                True,
                "QC override allowed.",
                require_reason=require_reason,
                log_entry=log_entry,
            )

        if action == "set_segmentation_params":
            if not role_cfg.get("allow_manual_segmentation_params", False):
                return GateResult(
                    False,
                    "Locked: Student role requires A/B parameter optimization. "
                    "Use the 'Optimize Parameters (A/B)' button.",
                    log_entry=log_entry,
                )
            return GateResult(True, "Manual segmentation parameters allowed.", log_entry=log_entry)

        if action == "skip_tournament":
            if role_cfg.get("require_tournament", False):
                return GateResult(
                    False,
                    "Locked: Student role must complete the A/B parameter tournament "
                    "before running segmentation.",
                    log_entry=log_entry,
                )
            return GateResult(True, "Tournament not required for this role.", log_entry=log_entry)

        if action == "edit_winning_params":
            max_dev = role_cfg.get("max_parameter_deviation")
            if max_dev == 0.0:
                return GateResult(
                    False,
                    "Locked: Student role cannot edit winning parameters after the tournament.",
                    log_entry=log_entry,
                )
            return GateResult(True, "Parameter editing allowed.", log_entry=log_entry)

        if action == "batch_without_preview":
            if not role_cfg.get("allow_batch_without_preview", False):
                return GateResult(
                    False,
                    "Locked: Student role must preview segmentation before running batch.",
                    log_entry=log_entry,
                )
            return GateResult(True, "Batch without preview allowed.", log_entry=log_entry)

        if action == "export_raw_masks":
            if not role_cfg.get("allow_raw_mask_export", False):
                return GateResult(
                    False,
                    "Locked: Student role cannot export raw segmentation masks.",
                    log_entry=log_entry,
                )
            return GateResult(True, "Raw mask export allowed.", log_entry=log_entry)

        logger.warning("PolicyEngine.gate: unknown action %r", action)
        return GateResult(True, f"Action '{action}' not gated.", log_entry=log_entry)
