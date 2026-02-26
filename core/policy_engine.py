"""
Policy engine for role-based access control in EZQUANT.

Loads ``lab_policy.yaml`` (or the default) and enforces role-based
restrictions at every gated action point in the GUI.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_POLICY_PATH = (
    Path(__file__).parent.parent / "configs" / "lab_policy.default.yaml"
)

# Mapping from action string → policy field (bool flags)
_ACTION_TO_FIELD: Dict[str, str] = {
    "set_segmentation_params": "allow_manual_segmentation_params",
    "batch_without_preview": "allow_batch_without_preview",
    "export_raw_masks": "allow_raw_mask_export",
    "skip_tournament": "allow_manual_segmentation_params",  # require_tournament inverse
    "edit_winning_params": "allow_manual_segmentation_params",
    "override_qc": "allow_overrides",
}


@dataclass
class GateResult:
    """Result of a policy gate check."""

    allowed: bool
    message: str
    require_reason: bool = False
    log_entry: Dict[str, Any] = field(default_factory=dict)


class PolicyEngine:
    """
    Loads ``lab_policy.yaml`` and enforces role-based restrictions.

    Parameters
    ----------
    policy_path:
        Path to the YAML policy file.  Defaults to
        ``configs/lab_policy.default.yaml``.
    """

    def __init__(self, policy_path: Optional[Path] = None) -> None:
        self._path = Path(policy_path) if policy_path else _DEFAULT_POLICY_PATH
        self.policy: Dict = {}
        self.current_role: str = "student"
        self._load_policy()

    # ------------------------------------------------------------------
    # Policy loading
    # ------------------------------------------------------------------

    def _load_policy(self) -> None:
        """Load and parse the YAML policy file."""
        try:
            self.policy = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError:
            logger.warning(
                "Policy file not found at %s; using empty policy.", self._path
            )
            self.policy = {}
        except yaml.YAMLError as exc:
            logger.error("Failed to parse policy file: %s", exc)
            self.policy = {}

    def reload(self) -> None:
        """Reload the policy file from disk."""
        self._load_policy()

    # ------------------------------------------------------------------
    # Role management
    # ------------------------------------------------------------------

    def set_role(self, role: str, auth_token: Optional[str] = None) -> bool:
        """
        Set the active role.

        ``instructor`` and ``research`` roles require an ``auth_token``
        matching the hashed passphrase stored in the policy under
        ``role_auth.<role>``.  ``student`` can always be set without
        authentication.

        Returns ``True`` on success, ``False`` if authentication fails.
        """
        if role not in self._known_roles():
            logger.warning("Unknown role: %s", role)
            return False

        if role == "student":
            self.current_role = role
            return True

        # Elevated roles require authentication
        expected_hash = self._role_hash(role)
        if expected_hash is None:
            # No hash configured — allow freely (open lab policy)
            self.current_role = role
            return True

        if auth_token is None:
            return False

        token_hash = hashlib.sha256(auth_token.encode()).hexdigest()
        if hmac.compare_digest(token_hash, expected_hash):
            self.current_role = role
            return True

        logger.warning("Authentication failed for role '%s'.", role)
        return False

    def _known_roles(self) -> list:
        return list(self.policy.get("roles", {}).keys()) or ["student", "instructor", "research"]

    def _role_hash(self, role: str) -> Optional[str]:
        """Return the expected hex-digest for *role*, or None if not set."""
        return (
            self.policy.get("role_auth", {}).get(role) or
            self.policy.get("roles", {}).get(role, {}).get("auth_hash")
        )

    # ------------------------------------------------------------------
    # Permission checks
    # ------------------------------------------------------------------

    @property
    def role_config(self) -> Dict:
        """Return the full config dict for the current role."""
        return self.policy.get("roles", {}).get(self.current_role, {})

    def can(self, action: str) -> bool:
        """
        Return ``True`` if the current role permits *action*.

        Supported actions:

        * ``export``
        * ``override_qc``
        * ``set_segmentation_params``
        * ``batch_without_preview``
        * ``export_raw_masks``
        * ``skip_tournament``
        * ``edit_winning_params``
        * ``use_recipe:<name>``
        """
        return self.gate(action).allowed

    def gate(
        self, action: str, context: Optional[Dict] = None
    ) -> GateResult:
        """
        Like :meth:`can`, but returns a :class:`GateResult` with a
        human-readable message and audit entry.

        Parameters
        ----------
        action:
            One of the known action strings (see :meth:`can`).
        context:
            Optional dict with extra information (e.g. ``{"qc_status": "FAIL"}``).
        """
        context = context or {}
        rc = self.role_config

        log_entry: Dict[str, Any] = {
            "role": self.current_role,
            "action": action,
            "context": context,
        }

        # --- Recipe actions ---
        if action.startswith("use_recipe:"):
            recipe_name = action[len("use_recipe:"):]
            allowed_recipes = rc.get("allowed_recipes", [])
            if allowed_recipes == "ALL" or recipe_name in allowed_recipes:
                log_entry["outcome"] = "allowed"
                return GateResult(
                    allowed=True,
                    message=f"Recipe '{recipe_name}' is available for role '{self.current_role}'.",
                    log_entry=log_entry,
                )
            log_entry["outcome"] = "denied"
            return GateResult(
                allowed=False,
                message=(
                    f"Recipe '{recipe_name}' is not available for role '{self.current_role}'. "
                    f"Allowed: {allowed_recipes}."
                ),
                log_entry=log_entry,
            )

        # --- Export action ---
        if action == "export":
            export_policy = rc.get("export", "PASS_ONLY")
            if export_policy == "ALL":
                log_entry["outcome"] = "allowed"
                return GateResult(allowed=True, message="Export allowed.", log_entry=log_entry)

            qc_status = context.get("qc_status", "PASS")
            if export_policy == "PASS_ONLY" and qc_status != "PASS":
                log_entry["outcome"] = "denied"
                return GateResult(
                    allowed=False,
                    message=(
                        "Export blocked: student role requires all QC checks to PASS. "
                        f"Current QC status: {qc_status}."
                    ),
                    log_entry=log_entry,
                )
            if export_policy == "PASS_WARN" and qc_status == "FAIL":
                require_reason = self.policy.get("require_override_reason", False)
                log_entry["outcome"] = "requires_reason" if require_reason else "allowed_with_warning"
                return GateResult(
                    allowed=True,
                    message=(
                        "Export allowed with QC warnings. "
                        "Results may be unreliable — please document your reason."
                    ),
                    require_reason=require_reason,
                    log_entry=log_entry,
                )
            log_entry["outcome"] = "allowed"
            return GateResult(allowed=True, message="Export allowed.", log_entry=log_entry)

        # --- Boolean flag actions ---
        if action in _ACTION_TO_FIELD:
            field_name = _ACTION_TO_FIELD[action]
            allowed = bool(rc.get(field_name, True))
            if allowed:
                log_entry["outcome"] = "allowed"
                return GateResult(allowed=True, message=f"Action '{action}' permitted.", log_entry=log_entry)
            else:
                require_reason = self.policy.get("require_override_reason", False)

                # Special message for tournament requirement
                if action in ("skip_tournament", "edit_winning_params") and rc.get("require_tournament", False):
                    msg = (
                        "Locked: Student role requires A/B parameter optimization. "
                        "Please complete the tournament to set segmentation parameters."
                    )
                else:
                    msg = (
                        f"Action '{action}' is not permitted for role '{self.current_role}'. "
                        "Contact your instructor to perform this action."
                    )
                log_entry["outcome"] = "denied"
                return GateResult(
                    allowed=False,
                    message=msg,
                    require_reason=require_reason,
                    log_entry=log_entry,
                )

        # Unknown action — allow by default but log
        logger.debug("Unknown policy action '%s'; allowing by default.", action)
        log_entry["outcome"] = "allowed_unknown"
        return GateResult(
            allowed=True,
            message=f"Action '{action}' not gated by policy (allowed by default).",
            log_entry=log_entry,
        )

    # ------------------------------------------------------------------
    # Guardrail checks
    # ------------------------------------------------------------------

    def check_guardrails(
        self, recipe_name: str, context: Dict
    ) -> Optional[GateResult]:
        """
        Check recipe-level guardrails.

        Returns a :class:`GateResult` with ``allowed=False`` and a
        user-friendly message if any guardrail fires, otherwise ``None``.

        Parameters
        ----------
        recipe_name:
            The name of the active recipe (e.g. ``nuclear_intensity``).
        context:
            Dict that may contain keys like ``nucleus_count``, ``area_cv``,
            ``saturated_fraction``, ``dna_channel_set``.
        """
        recipes = self.policy.get("recipes", {})
        recipe = recipes.get(recipe_name, {})
        guardrails = recipe.get("guardrails", {})

        checks = [
            (
                "min_nuclei_for_export",
                lambda v, c: c.get("nucleus_count", v + 1) < v,
                lambda v, c: (
                    f"Only {c.get('nucleus_count', 0)} nuclei passed quality control. "
                    f"At least {v} are needed for reliable measurements. "
                    "Try adjusting the segmentation or check image quality."
                ),
            ),
            (
                "max_cv_area_percent",
                lambda v, c: c.get("area_cv_percent", 0.0) > v,
                lambda v, c: (
                    f"Area coefficient of variation ({c.get('area_cv_percent', 0):.0f}%) "
                    f"exceeds the maximum allowed ({v}%). "
                    "The segmentation may be unreliable — check nucleus outlines."
                ),
            ),
            (
                "require_dna_channel_set",
                lambda v, c: v and not c.get("dna_channel_set", False),
                lambda v, c: (
                    "Cannot proceed: no DNA (DAPI/Hoechst) channel has been selected. "
                    "Please assign the DNA channel before continuing."
                ),
            ),
            (
                "max_saturated_fraction",
                lambda v, c: c.get("saturated_fraction", 0.0) > v,
                lambda v, c: (
                    f"Image saturation ({c.get('saturated_fraction', 0):.3%}) "
                    f"exceeds the allowed maximum ({v:.3%}). "
                    "Saturated pixels will corrupt intensity measurements."
                ),
            ),
        ]

        for key, predicate, message_fn in checks:
            value = guardrails.get(key)
            if value is not None and predicate(value, context):
                return GateResult(
                    allowed=False,
                    message=message_fn(value, context),
                    log_entry={
                        "role": self.current_role,
                        "action": f"guardrail:{key}",
                        "recipe": recipe_name,
                        "context": context,
                        "outcome": "blocked",
                    },
                )
        return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def available_roles(self) -> list:
        """Return the list of roles defined in the policy."""
        return self._known_roles()

    def to_json(self) -> str:
        """Serialise current policy state to JSON (for audit logs)."""
        return json.dumps(
            {
                "policy_name": self.policy.get("policy_name", ""),
                "policy_version": self.policy.get("policy_version", ""),
                "current_role": self.current_role,
            }
        )
