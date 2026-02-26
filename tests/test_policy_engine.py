"""
Tests for the policy engine (role-based access control).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml


# We import PolicyEngine after writing a temp policy file so we can
# test with controlled data without touching the repo's default policy.
from core.policy_engine import PolicyEngine, GateResult


# ---------------------------------------------------------------------------
# Helper to write a minimal policy file
# ---------------------------------------------------------------------------

_MINIMAL_POLICY = {
    "policy_name": "TestLab",
    "policy_version": "2026-01-01",
    "require_override_reason": True,
    "role_auth": {
        "instructor": "",  # empty → no auth required in tests
        "research": "",
    },
    "roles": {
        "student": {
            "allowed_recipes": ["nuclear_intensity"],
            "export": "PASS_ONLY",
            "allow_overrides": False,
            "allow_manual_segmentation_params": False,
            "allow_batch_without_preview": False,
            "allow_raw_mask_export": False,
            "require_tournament": True,
            "max_parameter_deviation": 0.0,
        },
        "instructor": {
            "allowed_recipes": ["nuclear_intensity", "colocalization"],
            "export": "PASS_WARN",
            "allow_overrides": True,
            "allow_manual_segmentation_params": True,
            "allow_batch_without_preview": True,
            "allow_raw_mask_export": True,
            "require_tournament": False,
        },
        "research": {
            "allowed_recipes": "ALL",
            "export": "ALL",
            "allow_overrides": True,
            "allow_manual_segmentation_params": True,
            "allow_batch_without_preview": True,
            "allow_raw_mask_export": True,
            "require_tournament": False,
        },
    },
    "recipes": {
        "nuclear_intensity": {
            "guardrails": {
                "min_nuclei_for_export": 30,
                "max_cv_area_percent": 80,
                "require_dna_channel_set": True,
                "max_saturated_fraction": 0.005,
            }
        }
    },
}


def _write_policy(d: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(d, tmp)
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPolicyEngineLoading(unittest.TestCase):

    def setUp(self):
        self.policy_path = _write_policy(_MINIMAL_POLICY)
        self.engine = PolicyEngine(policy_path=self.policy_path)

    def tearDown(self):
        self.policy_path.unlink(missing_ok=True)

    def test_default_role_is_student(self):
        self.assertEqual(self.engine.current_role, "student")

    def test_policy_name_loaded(self):
        self.assertEqual(self.engine.policy.get("policy_name"), "TestLab")

    def test_reload(self):
        """Reload should succeed without error."""
        self.engine.reload()
        self.assertEqual(self.engine.current_role, "student")


class TestRoleSwitching(unittest.TestCase):

    def setUp(self):
        self.policy_path = _write_policy(_MINIMAL_POLICY)
        self.engine = PolicyEngine(policy_path=self.policy_path)

    def tearDown(self):
        self.policy_path.unlink(missing_ok=True)

    def test_switch_to_student_always_allowed(self):
        self.assertTrue(self.engine.set_role("student"))
        self.assertEqual(self.engine.current_role, "student")

    def test_switch_to_instructor_no_auth_required_when_hash_empty(self):
        """When role_auth hash is empty, no token is needed."""
        result = self.engine.set_role("instructor")
        self.assertTrue(result)
        self.assertEqual(self.engine.current_role, "instructor")

    def test_switch_to_unknown_role_fails(self):
        result = self.engine.set_role("super_admin")
        self.assertFalse(result)

    def test_role_config_reflects_current_role(self):
        self.engine.set_role("instructor")
        config = self.engine.role_config
        self.assertTrue(config.get("allow_overrides"))

    def test_available_roles(self):
        roles = self.engine.available_roles()
        self.assertIn("student", roles)
        self.assertIn("instructor", roles)
        self.assertIn("research", roles)


class TestStudentPermissions(unittest.TestCase):

    def setUp(self):
        self.policy_path = _write_policy(_MINIMAL_POLICY)
        self.engine = PolicyEngine(policy_path=self.policy_path)
        # Stays as student

    def tearDown(self):
        self.policy_path.unlink(missing_ok=True)

    def test_student_cannot_set_segmentation_params(self):
        result = self.engine.gate("set_segmentation_params")
        self.assertFalse(result.allowed)
        self.assertIsInstance(result.message, str)
        self.assertTrue(len(result.message) > 0)

    def test_student_cannot_override_qc(self):
        self.assertFalse(self.engine.can("override_qc"))

    def test_student_cannot_batch_without_preview(self):
        self.assertFalse(self.engine.can("batch_without_preview"))

    def test_student_cannot_export_raw_masks(self):
        self.assertFalse(self.engine.can("export_raw_masks"))

    def test_student_export_blocked_on_qc_fail(self):
        result = self.engine.gate("export", context={"qc_status": "FAIL"})
        self.assertFalse(result.allowed)

    def test_student_export_blocked_on_qc_warn(self):
        result = self.engine.gate("export", context={"qc_status": "WARN"})
        self.assertFalse(result.allowed)

    def test_student_export_allowed_on_qc_pass(self):
        result = self.engine.gate("export", context={"qc_status": "PASS"})
        self.assertTrue(result.allowed)

    def test_student_recipe_allowed(self):
        result = self.engine.gate("use_recipe:nuclear_intensity")
        self.assertTrue(result.allowed)

    def test_student_recipe_denied(self):
        result = self.engine.gate("use_recipe:colocalization")
        self.assertFalse(result.allowed)

    def test_skip_tournament_denied_for_student(self):
        result = self.engine.gate("skip_tournament")
        self.assertFalse(result.allowed)
        # Message should explain tournament requirement
        self.assertIn("tournament", result.message.lower())


class TestInstructorPermissions(unittest.TestCase):

    def setUp(self):
        self.policy_path = _write_policy(_MINIMAL_POLICY)
        self.engine = PolicyEngine(policy_path=self.policy_path)
        self.engine.set_role("instructor")

    def tearDown(self):
        self.policy_path.unlink(missing_ok=True)

    def test_instructor_can_set_segmentation_params(self):
        self.assertTrue(self.engine.can("set_segmentation_params"))

    def test_instructor_can_override_qc(self):
        self.assertTrue(self.engine.can("override_qc"))

    def test_instructor_can_batch_without_preview(self):
        self.assertTrue(self.engine.can("batch_without_preview"))

    def test_instructor_export_allowed_with_warn(self):
        result = self.engine.gate("export", context={"qc_status": "WARN"})
        self.assertTrue(result.allowed)

    def test_instructor_export_blocked_on_fail_requires_reason(self):
        result = self.engine.gate("export", context={"qc_status": "FAIL"})
        # PASS_WARN policy: FAIL is allowed but requires a reason
        self.assertTrue(result.allowed)
        self.assertTrue(result.require_reason)

    def test_instructor_recipe_colocalization_allowed(self):
        result = self.engine.gate("use_recipe:colocalization")
        self.assertTrue(result.allowed)


class TestResearchPermissions(unittest.TestCase):

    def setUp(self):
        self.policy_path = _write_policy(_MINIMAL_POLICY)
        self.engine = PolicyEngine(policy_path=self.policy_path)
        self.engine.set_role("research")

    def tearDown(self):
        self.policy_path.unlink(missing_ok=True)

    def test_research_export_always_allowed(self):
        for status in ("PASS", "WARN", "FAIL"):
            result = self.engine.gate("export", context={"qc_status": status})
            self.assertTrue(result.allowed, f"Expected export allowed for status={status}")
            self.assertFalse(result.require_reason, f"Expected require_reason=False for status={status}")

    def test_research_all_recipes_allowed(self):
        for recipe in ("nuclear_intensity", "colocalization", "frap_basic", "anything"):
            result = self.engine.gate(f"use_recipe:{recipe}")
            self.assertTrue(result.allowed)


class TestGuardrails(unittest.TestCase):

    def setUp(self):
        self.policy_path = _write_policy(_MINIMAL_POLICY)
        self.engine = PolicyEngine(policy_path=self.policy_path)

    def tearDown(self):
        self.policy_path.unlink(missing_ok=True)

    def test_min_nuclei_guardrail_fires(self):
        result = self.engine.check_guardrails(
            "nuclear_intensity", {"nucleus_count": 10}
        )
        self.assertIsNotNone(result)
        self.assertFalse(result.allowed)
        self.assertIn("30", result.message)

    def test_min_nuclei_guardrail_passes(self):
        result = self.engine.check_guardrails(
            "nuclear_intensity", {"nucleus_count": 50, "dna_channel_set": True}
        )
        self.assertIsNone(result)

    def test_saturation_guardrail_fires(self):
        result = self.engine.check_guardrails(
            "nuclear_intensity",
            {"nucleus_count": 50, "saturated_fraction": 0.01, "dna_channel_set": True},
        )
        self.assertIsNotNone(result)
        self.assertFalse(result.allowed)

    def test_dna_channel_not_set_blocks(self):
        result = self.engine.check_guardrails(
            "nuclear_intensity",
            {"nucleus_count": 50, "dna_channel_set": False},
        )
        self.assertIsNotNone(result)
        self.assertFalse(result.allowed)

    def test_dna_channel_set_passes(self):
        result = self.engine.check_guardrails(
            "nuclear_intensity",
            {"nucleus_count": 50, "dna_channel_set": True},
        )
        self.assertIsNone(result)

    def test_unknown_recipe_no_guardrails(self):
        """Unknown recipe should not raise — no guardrails apply."""
        result = self.engine.check_guardrails("unknown_recipe", {})
        self.assertIsNone(result)


class TestGateResultType(unittest.TestCase):

    def setUp(self):
        self.policy_path = _write_policy(_MINIMAL_POLICY)
        self.engine = PolicyEngine(policy_path=self.policy_path)

    def tearDown(self):
        self.policy_path.unlink(missing_ok=True)

    def test_gate_returns_gate_result(self):
        result = self.engine.gate("export")
        self.assertIsInstance(result, GateResult)
        self.assertIsInstance(result.allowed, bool)
        self.assertIsInstance(result.message, str)
        self.assertIsInstance(result.require_reason, bool)
        self.assertIsInstance(result.log_entry, dict)

    def test_log_entry_contains_role(self):
        result = self.engine.gate("export")
        self.assertIn("role", result.log_entry)
        self.assertEqual(result.log_entry["role"], "student")

    def test_unknown_action_allowed_by_default(self):
        result = self.engine.gate("completely_unknown_action_xyz")
        self.assertTrue(result.allowed)


if __name__ == "__main__":
    unittest.main()
