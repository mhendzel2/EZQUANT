import unittest
from pathlib import Path

from core.policy_engine import PolicyEngine


class TestPolicyEngine(unittest.TestCase):
    def setUp(self):
        policy_path = Path(__file__).resolve().parent.parent / "configs" / "lab_policy.default.yaml"
        self.engine = PolicyEngine(policy_path)

    def test_role_authentication_required(self):
        with self.assertRaises(PermissionError):
            self.engine.set_role("instructor")

        with self.assertRaises(PermissionError):
            self.engine.set_role("instructor", auth_token="wrong-passphrase")

        self.engine.set_role("instructor", auth_token="instructor-passphrase")
        self.assertEqual(self.engine.current_role, "instructor")

    def test_student_tournament_gate(self):
        self.engine.set_role("student")

        denied = self.engine.gate("skip_tournament", context={"tournament_completed": False})
        allowed = self.engine.gate("skip_tournament", context={"tournament_completed": True})

        self.assertFalse(denied.allowed)
        self.assertTrue(allowed.allowed)

    def test_export_gate_by_qc_status(self):
        self.engine.set_role("student")
        self.assertTrue(self.engine.gate("export", {"qc_status": "PASS"}).allowed)
        self.assertFalse(self.engine.gate("export", {"qc_status": "WARN"}).allowed)

        self.engine.set_role("instructor", auth_token="instructor-passphrase")
        warn_gate = self.engine.gate("export", {"qc_status": "WARN"})
        fail_gate = self.engine.gate("export", {"qc_status": "FAIL"})

        self.assertTrue(warn_gate.allowed)
        self.assertTrue(warn_gate.require_reason)
        self.assertFalse(fail_gate.allowed)

    def test_recipe_gate(self):
        self.engine.set_role("student")
        self.assertTrue(self.engine.gate("use_recipe:nuclear_intensity").allowed)
        self.assertFalse(self.engine.gate("use_recipe:colocalization").allowed)

        self.engine.set_role("research", auth_token="research-passphrase")
        self.assertTrue(self.engine.gate("use_recipe:colocalization").allowed)

    def test_edit_winning_params_gate(self):
        self.engine.set_role("student")
        self.assertTrue(self.engine.gate("edit_winning_params", {"deviation": 0.0}).allowed)
        self.assertFalse(self.engine.gate("edit_winning_params", {"deviation": 0.1}).allowed)

    def test_gate_log_entry_shape(self):
        gate = self.engine.gate("set_segmentation_params")
        self.assertIn("timestamp", gate.log_entry)
        self.assertIn("role", gate.log_entry)
        self.assertIn("action", gate.log_entry)
        self.assertIn("outcome", gate.log_entry)


if __name__ == "__main__":
    unittest.main()
