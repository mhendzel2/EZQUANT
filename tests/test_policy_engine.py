"""Tests for core/policy_engine.py"""
from __future__ import annotations

import pytest

from core.policy_engine import PolicyEngine, GateResult


POLICY_PATH = "configs/lab_policy.default.yaml"


class TestRoleManagement:
    def test_default_role_is_student(self):
        engine = PolicyEngine(POLICY_PATH)
        assert engine.current_role == "student"

    def test_switch_to_instructor_no_hash(self):
        engine = PolicyEngine(POLICY_PATH)
        result = engine.set_role("instructor")
        assert result is True
        assert engine.current_role == "instructor"

    def test_switch_to_research_no_hash(self):
        engine = PolicyEngine(POLICY_PATH)
        result = engine.set_role("research")
        assert result is True
        assert engine.current_role == "research"

    def test_switch_to_student_always_allowed(self):
        engine = PolicyEngine(POLICY_PATH)
        engine.set_role("instructor")
        result = engine.set_role("student")
        assert result is True
        assert engine.current_role == "student"

    def test_invalid_role_raises_error(self):
        engine = PolicyEngine(POLICY_PATH)
        with pytest.raises(ValueError):
            engine.set_role("superadmin")

    def test_role_config_returns_dict(self):
        engine = PolicyEngine(POLICY_PATH)
        cfg = engine.role_config
        assert isinstance(cfg, dict)
        assert "export" in cfg


class TestStudentGating:
    def setup_method(self):
        self.engine = PolicyEngine(POLICY_PATH)
        # Already student by default

    def test_student_cannot_export_fail_qc(self):
        result = self.engine.gate("export", {"qc_status": "FAIL"})
        assert result.allowed is False
        assert "Student" in result.message or "student" in result.message.lower()

    def test_student_cannot_set_segmentation_params(self):
        result = self.engine.gate("set_segmentation_params")
        assert result.allowed is False

    def test_student_cannot_skip_tournament(self):
        result = self.engine.gate("skip_tournament")
        assert result.allowed is False

    def test_student_cannot_override_qc(self):
        result = self.engine.gate("override_qc")
        assert result.allowed is False

    def test_student_cannot_export_raw_masks(self):
        result = self.engine.gate("export_raw_masks")
        assert result.allowed is False

    def test_student_cannot_batch_without_preview(self):
        result = self.engine.gate("batch_without_preview")
        assert result.allowed is False

    def test_student_cannot_edit_winning_params(self):
        result = self.engine.gate("edit_winning_params")
        assert result.allowed is False

    def test_student_can_export_pass_qc(self):
        result = self.engine.gate("export", {"qc_status": "PASS"})
        assert result.allowed is True

    def test_student_can_use_allowed_recipe(self):
        result = self.engine.gate("use_recipe:nuclear_intensity")
        assert result.allowed is True

    def test_student_cannot_use_disallowed_recipe(self):
        result = self.engine.gate("use_recipe:colocalization")
        assert result.allowed is False


class TestInstructorGating:
    def setup_method(self):
        self.engine = PolicyEngine(POLICY_PATH)
        self.engine.set_role("instructor")

    def test_instructor_can_export_pass_qc(self):
        result = self.engine.gate("export", {"qc_status": "PASS"})
        assert result.allowed is True

    def test_instructor_can_export_fail_qc_with_reason(self):
        result = self.engine.gate("export", {"qc_status": "FAIL"})
        assert result.allowed is True
        assert result.require_reason is True

    def test_instructor_can_override_qc(self):
        result = self.engine.gate("override_qc")
        assert result.allowed is True

    def test_instructor_can_set_segmentation_params(self):
        result = self.engine.gate("set_segmentation_params")
        assert result.allowed is True

    def test_instructor_can_skip_tournament(self):
        result = self.engine.gate("skip_tournament")
        assert result.allowed is True

    def test_instructor_can_export_raw_masks(self):
        result = self.engine.gate("export_raw_masks")
        assert result.allowed is True

    def test_instructor_can_batch_without_preview(self):
        result = self.engine.gate("batch_without_preview")
        assert result.allowed is True

    def test_instructor_can_edit_winning_params(self):
        result = self.engine.gate("edit_winning_params")
        assert result.allowed is True

    def test_instructor_can_use_colocalization(self):
        result = self.engine.gate("use_recipe:colocalization")
        assert result.allowed is True


class TestResearchGating:
    def setup_method(self):
        self.engine = PolicyEngine(POLICY_PATH)
        self.engine.set_role("research")

    def test_research_can_export_anything(self):
        result = self.engine.gate("export", {"qc_status": "FAIL"})
        assert result.allowed is True

    def test_research_can_override_qc(self):
        result = self.engine.gate("override_qc")
        assert result.allowed is True

    def test_research_can_use_any_recipe(self):
        for recipe in ["nuclear_intensity", "puncta_counting", "colocalization",
                       "frap_basic", "some_custom_recipe"]:
            # research has allowed_recipes: ALL
            result = self.engine.gate(f"use_recipe:{recipe}")
            assert result.allowed is True, f"Research should allow recipe {recipe}"

    def test_research_can_export_raw_masks(self):
        result = self.engine.gate("export_raw_masks")
        assert result.allowed is True


class TestGateResultStructure:
    def test_gate_result_has_required_fields(self):
        engine = PolicyEngine(POLICY_PATH)
        result = engine.gate("export", {"qc_status": "PASS"})
        assert isinstance(result, GateResult)
        assert isinstance(result.allowed, bool)
        assert isinstance(result.message, str)
        assert isinstance(result.require_reason, bool)
        assert isinstance(result.log_entry, dict)

    def test_can_method_matches_gate(self):
        engine = PolicyEngine(POLICY_PATH)
        # Student: cannot skip tournament
        assert engine.can("skip_tournament") == engine.gate("skip_tournament").allowed
