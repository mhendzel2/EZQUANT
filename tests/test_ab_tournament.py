"""Tests for core/ab_tournament.py"""
from __future__ import annotations

import numpy as np
import pytest

from core.ab_tournament import (
    ABTournament,
    ParameterCandidate,
    PARAMETER_SPACE,
    CONTINUOUS_PARAMS,
    CATEGORICAL_PARAMS,
)


class TestParameterSampling:
    def test_lhs_produces_n_candidates(self):
        image = np.zeros((64, 64), dtype=np.uint16)
        t = ABTournament(image, channels=[0], n_candidates=8, seed=42)
        candidates = t.generate_initial_candidates()
        assert len(candidates) == 8

    def test_candidates_have_correct_param_keys(self):
        image = np.zeros((64, 64), dtype=np.uint16)
        t = ABTournament(image, channels=[0], n_candidates=4, seed=0)
        candidates = t.generate_initial_candidates()
        for c in candidates:
            for key in PARAMETER_SPACE:
                assert key in c.params, f"Missing param: {key}"

    def test_continuous_params_in_range(self):
        image = np.zeros((64, 64), dtype=np.uint16)
        t = ABTournament(image, channels=[0], n_candidates=16, seed=7)
        candidates = t.generate_initial_candidates()
        for c in candidates:
            for param in CONTINUOUS_PARAMS:
                spec = PARAMETER_SPACE[param]
                val = c.params[param]
                assert spec["min"] <= val <= spec["max"], (
                    f"{param}={val} out of range [{spec['min']}, {spec['max']}]"
                )

    def test_categorical_params_valid(self):
        image = np.zeros((64, 64), dtype=np.uint16)
        t = ABTournament(image, channels=[0], n_candidates=8, seed=3)
        candidates = t.generate_initial_candidates()
        for c in candidates:
            for param in CATEGORICAL_PARAMS:
                spec = PARAMETER_SPACE[param]
                assert c.params[param] in spec["choices"]

    def test_custom_parameter_ranges(self):
        image = np.zeros((64, 64), dtype=np.uint16)
        custom_ranges = {"diameter": {"min": 20.0, "max": 40.0}}
        t = ABTournament(image, channels=[0], n_candidates=8, seed=1,
                         parameter_ranges=custom_ranges)
        candidates = t.generate_initial_candidates()
        for c in candidates:
            assert 20.0 <= c.params["diameter"] <= 40.0


class TestBracketLogic:
    def _make_tournament(self, n=8, seed=0):
        image = np.zeros((64, 64), dtype=np.uint16)
        t = ABTournament(image, channels=[0], n_candidates=n, seed=seed)
        t.generate_initial_candidates()
        # Give each candidate some dummy stats
        for c in t.candidates:
            c.nucleus_count = 50
            c.median_area = 200.0
        return t

    def test_n8_produces_7_matchups(self):
        t = self._make_tournament(n=8)
        matchup_count = 0
        while True:
            pair = t.get_next_matchup()
            if pair is None:
                break
            t.record_choice(pair[0].id, pair[1].id)
            matchup_count += 1
            if matchup_count > 20:
                break
        assert matchup_count == 7

    def test_single_winner_after_tournament(self):
        t = self._make_tournament(n=8)
        while True:
            pair = t.get_next_matchup()
            if pair is None:
                break
            t.record_choice(pair[0].id, pair[1].id)
        winner = t.get_winner()
        assert winner is not None
        assert isinstance(winner, ParameterCandidate)

    def test_record_choice_advances_bracket(self):
        t = self._make_tournament(n=4)
        pair = t.get_next_matchup()
        assert pair is not None
        t.record_choice(pair[0].id, pair[1].id)
        assert t.completed_matchups() == 1

    def test_get_next_matchup_returns_none_when_done(self):
        t = self._make_tournament(n=2)
        pair = t.get_next_matchup()
        assert pair is not None
        t.record_choice(pair[0].id, pair[1].id)
        assert t.get_next_matchup() is None

    def test_winner_id_is_chosen_candidate(self):
        image = np.zeros((64, 64), dtype=np.uint16)
        t = ABTournament(image, channels=[0], n_candidates=2, seed=99)
        t.generate_initial_candidates()
        pair = t.get_next_matchup()
        expected_winner = pair[0]
        t.record_choice(pair[0].id, pair[1].id)
        winner = t.get_winner()
        assert winner is not None
        assert winner.id == expected_winner.id


class TestUndoFunctionality:
    def test_undo_last_choice_reverts_matchup(self):
        image = np.zeros((64, 64), dtype=np.uint16)
        t = ABTournament(image, channels=[0], n_candidates=4, seed=5)
        t.generate_initial_candidates()
        pair1 = t.get_next_matchup()
        t.record_choice(pair1[0].id, pair1[1].id)
        assert t.completed_matchups() == 1
        result = t.undo_last_choice()
        assert result is True
        assert t.completed_matchups() == 0

    def test_undo_when_empty_returns_false(self):
        image = np.zeros((64, 64), dtype=np.uint16)
        t = ABTournament(image, channels=[0], n_candidates=4, seed=5)
        t.generate_initial_candidates()
        assert t.undo_last_choice() is False


class TestOffspringGeneration:
    def test_offspring_params_bounded(self):
        image = np.zeros((64, 64), dtype=np.uint16)
        t = ABTournament(image, channels=[0], n_candidates=4, seed=2)
        t.generate_initial_candidates()
        parent_a, parent_b = t.candidates[0], t.candidates[1]
        offspring = t.generate_offspring(parent_a, parent_b, n=4)
        assert len(offspring) == 4
        for child in offspring:
            assert child.parent_ids == (parent_a.id, parent_b.id)
            for param in CONTINUOUS_PARAMS:
                spec = PARAMETER_SPACE[param]
                val = child.params[param]
                assert spec["min"] <= val <= spec["max"]

    def test_offspring_generation_field(self):
        image = np.zeros((64, 64), dtype=np.uint16)
        t = ABTournament(image, channels=[0], n_candidates=4, seed=2)
        t.generate_initial_candidates()
        parent_a, parent_b = t.candidates[0], t.candidates[1]
        offspring = t.generate_offspring(parent_a, parent_b, n=2)
        for child in offspring:
            assert child.generation == 1  # generation = round + 1 = 0 + 1


class TestAuditLog:
    def test_audit_log_completeness(self):
        image = np.zeros((64, 64), dtype=np.uint16)
        t = ABTournament(image, channels=[0], n_candidates=4, seed=8)
        t.generate_initial_candidates()
        while True:
            pair = t.get_next_matchup()
            if pair is None:
                break
            t.record_choice(pair[0].id, pair[1].id)
        log = t.get_audit_log()
        assert len(log) == 1
        entry = log[0]
        assert "candidates" in entry
        assert "matchups" in entry
        assert "winner" in entry
        assert entry["winner"] is not None
        assert len(entry["matchups"]) == 3  # n=4 → 3 matchups

    def test_audit_log_has_timestamps(self):
        image = np.zeros((64, 64), dtype=np.uint16)
        t = ABTournament(image, channels=[0], n_candidates=2, seed=0)
        t.generate_initial_candidates()
        pair = t.get_next_matchup()
        t.record_choice(pair[0].id, pair[1].id)
        log = t.get_audit_log()
        matchup = log[0]["matchups"][0]
        assert "timestamp" in matchup
        assert "winner_id" in matchup
        assert "loser_id" in matchup


class TestEdgeCases:
    def test_degenerate_image_zero_nuclei(self):
        """All candidates produce 0 nuclei (degenerate image)."""
        image = np.zeros((64, 64), dtype=np.uint16)
        t = ABTournament(image, channels=[0], n_candidates=4, seed=0)
        t.generate_initial_candidates()
        for c in t.candidates:
            c.nucleus_count = 0
            c.median_area = 0.0
        # Tournament should still complete
        while True:
            pair = t.get_next_matchup()
            if pair is None:
                break
            t.record_choice(pair[0].id, pair[1].id)
        winner = t.get_winner()
        assert winner is not None
        assert winner.nucleus_count == 0

    def test_progress_tracking(self):
        image = np.zeros((64, 64), dtype=np.uint16)
        t = ABTournament(image, channels=[0], n_candidates=8, seed=0)
        t.generate_initial_candidates()
        progress = t.get_progress()
        assert progress["total_completed"] == 0
        assert progress["total_expected"] == 7  # n-1 for single elimination

        pair = t.get_next_matchup()
        t.record_choice(pair[0].id, pair[1].id)
        progress = t.get_progress()
        assert progress["total_completed"] == 1
