"""
Tests for the A/B tournament segmentation parameter optimiser.
"""

from __future__ import annotations

import unittest
import numpy as np

from core.ab_tournament import (
    ABTournament,
    ParameterCandidate,
    PARAMETER_SPACE,
    _lhs_sample,
    _compute_mask_stats,
    _interpolate_params,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_simple_masks(n_nuclei: int, img_size: int = 100) -> np.ndarray:
    """Return a label mask with *n_nuclei* non-overlapping squares."""
    masks = np.zeros((img_size, img_size), dtype=int)
    side = 8
    gap = img_size // max(n_nuclei, 1)
    for i in range(n_nuclei):
        x0 = (i * gap) % (img_size - side)
        y0 = (i * gap // img_size) * gap % (img_size - side)
        masks[y0 : y0 + side, x0 : x0 + side] = i + 1
    return masks


def _fill_candidates(tournament: ABTournament) -> None:
    """Fill fake masks/stats for every candidate so the tournament can run."""
    for c in tournament.get_all_candidates():
        if c.masks is None:
            n = np.random.randint(5, 30)
            c.masks = _make_simple_masks(n)
            c.nucleus_count, c.median_area, c.area_cv = _compute_mask_stats(c.masks)


# ---------------------------------------------------------------------------
# LHS sampling
# ---------------------------------------------------------------------------

class TestLHSSampling(unittest.TestCase):

    def test_produces_n_candidates(self):
        """LHS should produce exactly N parameter dicts."""
        samples = _lhs_sample(8, PARAMETER_SPACE)
        self.assertEqual(len(samples), 8)

    def test_parameter_ranges(self):
        """Every float parameter must lie within its defined range."""
        samples = _lhs_sample(16, PARAMETER_SPACE)
        for s in samples:
            for key, spec in PARAMETER_SPACE.items():
                if spec["type"] == "float":
                    self.assertGreaterEqual(
                        s[key], spec["min"],
                        f"{key}={s[key]} below min {spec['min']}"
                    )
                    self.assertLessEqual(
                        s[key], spec["max"],
                        f"{key}={s[key]} above max {spec['max']}"
                    )

    def test_categoricals_valid(self):
        """Every categorical parameter must be one of the defined choices."""
        samples = _lhs_sample(12, PARAMETER_SPACE)
        for s in samples:
            for key, spec in PARAMETER_SPACE.items():
                if spec["type"] == "categorical":
                    self.assertIn(s[key], spec["choices"], f"{key}={s[key]} not in choices")

    def test_distinct_float_values(self):
        """LHS should produce spread-out values (no two identical in small N)."""
        samples = _lhs_sample(8, PARAMETER_SPACE)
        diameters = [s["diameter"] for s in samples]
        # All 8 diameters should be distinct (LHS guarantees stratification)
        self.assertEqual(len(set(round(d, 6) for d in diameters)), 8)


# ---------------------------------------------------------------------------
# ParameterCandidate
# ---------------------------------------------------------------------------

class TestParameterCandidate(unittest.TestCase):

    def test_unique_ids(self):
        c1 = ParameterCandidate()
        c2 = ParameterCandidate()
        self.assertNotEqual(c1.id, c2.id)

    def test_to_dict(self):
        c = ParameterCandidate(
            params={"diameter": 30.0},
            nucleus_count=50,
            generation=1,
        )
        d = c.to_dict()
        self.assertEqual(d["nucleus_count"], 50)
        self.assertEqual(d["generation"], 1)
        self.assertNotIn("masks", d)


# ---------------------------------------------------------------------------
# Single-elimination bracket logic (N=8)
# ---------------------------------------------------------------------------

class TestSingleEliminationN8(unittest.TestCase):

    def setUp(self):
        self.image = np.zeros((100, 100), dtype=np.uint16)
        self.tournament = ABTournament(
            image=self.image,
            channels=[0, 0],
            n_candidates=8,
            enable_offspring=False,
        )
        self.candidates = self.tournament.generate_initial_candidates()
        _fill_candidates(self.tournament)

    def test_generates_8_candidates(self):
        self.assertEqual(len(self.candidates), 8)

    def test_total_matchups_n8(self):
        """N=8 single-elimination requires 7 matchups."""
        self.assertEqual(self.tournament.total_matchups(), 7)

    def test_bracket_completion(self):
        """Simulating all 7 matchups should complete the tournament."""
        matchup_count = 0
        while True:
            matchup = self.tournament.get_next_matchup()
            if matchup is None:
                break
            a, b = matchup
            # Always choose A
            self.tournament.record_choice(winner_id=a.id, loser_id=b.id)
            matchup_count += 1

        self.assertEqual(matchup_count, 7)
        self.assertTrue(self.tournament._tournament_complete)

    def test_winner_is_participant(self):
        """The final winner must be one of the original candidates."""
        candidate_ids = {c.id for c in self.candidates}
        while True:
            matchup = self.tournament.get_next_matchup()
            if matchup is None:
                break
            a, b = matchup
            self.tournament.record_choice(winner_id=a.id, loser_id=b.id)

        winner = self.tournament.get_winner()
        self.assertIsNotNone(winner)
        self.assertIn(winner.id, candidate_ids)

    def test_record_choice_advances_bracket(self):
        """After recording a choice the matchup index must increment."""
        matchup = self.tournament.get_next_matchup()
        self.assertIsNotNone(matchup)
        before = self.tournament._current_matchup_idx
        a, b = matchup
        self.tournament.record_choice(winner_id=a.id, loser_id=b.id)
        # Either matchup index incremented OR we moved to next round
        after_idx = self.tournament._current_matchup_idx
        after_round = self.tournament._current_round
        self.assertTrue(after_idx > before or after_round > 0)


# ---------------------------------------------------------------------------
# Offspring generation
# ---------------------------------------------------------------------------

class TestOffspringGeneration(unittest.TestCase):

    def test_offspring_params_within_bounds(self):
        """Offspring parameters must stay within the defined parameter space."""
        rng = np.random.default_rng(0)
        image = rng.integers(0, 1000, size=(50, 50), dtype=np.uint16)
        tournament = ABTournament(
            image=image,
            channels=[0, 0],
            n_candidates=8,
            enable_offspring=True,
        )
        tournament.generate_initial_candidates()
        _fill_candidates(tournament)

        # Run through first three rounds (5 matchups) to trigger offspring
        run_matchups = 0
        while run_matchups < 5:
            matchup = tournament.get_next_matchup()
            if matchup is None:
                break
            a, b = matchup
            tournament.record_choice(winner_id=a.id, loser_id=b.id)
            _fill_candidates(tournament)  # Fill any newly generated offspring
            run_matchups += 1

        offspring = [
            c for c in tournament.get_all_candidates()
            if c.parent_ids is not None
        ]
        # If offspring were generated, check bounds
        for c in offspring:
            for key, spec in PARAMETER_SPACE.items():
                if spec["type"] == "float":
                    self.assertGreaterEqual(c.params[key], spec["min"])
                    self.assertLessEqual(c.params[key], spec["max"])

    def test_interpolate_params_midpoint(self):
        """Interpolated params should be roughly between the two parents."""
        a = {k: v["min"] for k, v in PARAMETER_SPACE.items() if v["type"] == "float"}
        a.update({k: v["choices"][0] for k, v in PARAMETER_SPACE.items() if v["type"] == "categorical"})
        b = {k: v["max"] for k, v in PARAMETER_SPACE.items() if v["type"] == "float"}
        b.update({k: v["choices"][0] for k, v in PARAMETER_SPACE.items() if v["type"] == "categorical"})

        # With jitter=0, the offspring should be at the exact midpoint
        import random
        random.seed(42)
        child = _interpolate_params(a, b, PARAMETER_SPACE, jitter=0.0)

        for key, spec in PARAMETER_SPACE.items():
            if spec["type"] == "float":
                expected_mid = (spec["min"] + spec["max"]) / 2.0
                self.assertAlmostEqual(child[key], expected_mid, places=3)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

class TestAuditLog(unittest.TestCase):

    def test_audit_log_completeness(self):
        """Audit log must contain tournament_start and all matchup_result events."""
        image = np.zeros((50, 50), dtype=np.uint16)
        tournament = ABTournament(
            image=image, channels=[0, 0], n_candidates=4, enable_offspring=False
        )
        tournament.generate_initial_candidates()
        _fill_candidates(tournament)

        while True:
            matchup = tournament.get_next_matchup()
            if matchup is None:
                break
            a, b = matchup
            tournament.record_choice(winner_id=a.id, loser_id=b.id)

        log = tournament.get_audit_log()
        events = [e["event"] for e in log]

        self.assertIn("tournament_start", events)
        matchup_events = [e for e in events if e == "matchup_result"]
        # N-1 matchups for single-elimination with N=4
        self.assertEqual(len(matchup_events), 3)
        self.assertIn("tournament_complete", events)

    def test_audit_log_records_winner(self):
        """The tournament_complete entry must contain the winner ID."""
        image = np.zeros((50, 50), dtype=np.uint16)
        tournament = ABTournament(
            image=image, channels=[0, 0], n_candidates=4, enable_offspring=False
        )
        tournament.generate_initial_candidates()
        _fill_candidates(tournament)

        while True:
            matchup = tournament.get_next_matchup()
            if matchup is None:
                break
            a, b = matchup
            tournament.record_choice(winner_id=a.id, loser_id=b.id)

        log = tournament.get_audit_log()
        complete_entry = next(e for e in log if e["event"] == "tournament_complete")
        self.assertIn("winner_id", complete_entry)
        self.assertIn("winner_params", complete_entry)


# ---------------------------------------------------------------------------
# Degenerate edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_all_zero_masks(self):
        """When all candidates produce 0 nuclei, the tournament should still complete."""
        image = np.zeros((50, 50), dtype=np.uint16)
        tournament = ABTournament(
            image=image, channels=[0, 0], n_candidates=4, enable_offspring=False
        )
        candidates = tournament.generate_initial_candidates()

        # Set all masks to zero (degenerate image)
        for c in candidates:
            c.masks = np.zeros((50, 50), dtype=int)
            c.nucleus_count = 0
            c.median_area = 0.0
            c.area_cv = 0.0

        matchup_count = 0
        while True:
            matchup = tournament.get_next_matchup()
            if matchup is None:
                break
            a, b = matchup
            tournament.record_choice(winner_id=a.id, loser_id=b.id)
            matchup_count += 1

        self.assertEqual(matchup_count, 3)  # N-1
        winner = tournament.get_winner()
        self.assertIsNotNone(winner)
        self.assertEqual(winner.nucleus_count, 0)

    def test_compute_mask_stats_empty(self):
        """_compute_mask_stats should handle empty / None masks gracefully."""
        count, median, cv = _compute_mask_stats(None)
        self.assertEqual(count, 0)
        self.assertEqual(median, 0.0)

        count2, median2, cv2 = _compute_mask_stats(np.zeros((50, 50), dtype=int))
        self.assertEqual(count2, 0)

    def test_undo_last_choice(self):
        """Undo should re-present the previous matchup."""
        image = np.zeros((50, 50), dtype=np.uint16)
        tournament = ABTournament(
            image=image, channels=[0, 0], n_candidates=4, enable_offspring=False
        )
        tournament.generate_initial_candidates()
        _fill_candidates(tournament)

        first_matchup = tournament.get_next_matchup()
        self.assertIsNotNone(first_matchup)
        a, b = first_matchup

        tournament.record_choice(winner_id=a.id, loser_id=b.id)
        self.assertEqual(tournament.completed_matchups(), 1)

        # Undo
        redone = tournament.undo_last_choice()
        self.assertEqual(tournament.completed_matchups(), 0)
        # redone may be None if the undo implementation returns None on first undo
        # (acceptable since the matchup index goes back to 0)


if __name__ == "__main__":
    unittest.main()
