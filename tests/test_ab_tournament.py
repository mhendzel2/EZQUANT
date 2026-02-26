import unittest

import numpy as np

from core.ab_tournament import ABTournament


class TestABTournament(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((1, 64, 64), dtype=np.uint16)
        self.parameter_ranges = {
            "diameter": {"type": "float", "min": 10.0, "max": 120.0, "allow_auto": False},
            "flow_threshold": {"type": "float", "min": 0.1, "max": 1.5},
            "cellprob_threshold": {"type": "float", "min": -3.0, "max": 3.0},
            "model_name": {"type": "categorical", "choices": ["nuclei", "cyto", "cyto2", "cyto3"]},
            "restoration_mode": {"type": "categorical", "choices": ["none", "denoise", "deblur"]},
        }

    def test_lhs_sampling_ranges(self):
        tournament = ABTournament(
            image=self.image,
            channels=[0, 0],
            parameter_ranges=self.parameter_ranges,
            n_candidates=8,
            strategy="latin_hypercube",
            seed=7,
        )
        candidates = tournament.generate_initial_candidates()

        self.assertEqual(len(candidates), 8)
        for candidate in candidates:
            p = candidate.params
            self.assertGreaterEqual(p["diameter"], 10.0)
            self.assertLessEqual(p["diameter"], 120.0)
            self.assertGreaterEqual(p["flow_threshold"], 0.1)
            self.assertLessEqual(p["flow_threshold"], 1.5)
            self.assertGreaterEqual(p["cellprob_threshold"], -3.0)
            self.assertLessEqual(p["cellprob_threshold"], 3.0)
            self.assertIn(p["model_name"], self.parameter_ranges["model_name"]["choices"])
            self.assertIn(
                p["restoration_mode"],
                self.parameter_ranges["restoration_mode"]["choices"],
            )

    def test_single_elimination_bracket_n8(self):
        tournament = ABTournament(
            image=self.image,
            channels=[0, 0],
            parameter_ranges=self.parameter_ranges,
            n_candidates=8,
            strategy="latin_hypercube",
            seed=3,
        )
        tournament.generate_initial_candidates()

        comparisons = 0
        while True:
            matchup = tournament.get_next_matchup()
            if matchup is None:
                break
            tournament.record_choice(winner_id=matchup[0].id, loser_id=matchup[1].id)
            comparisons += 1

        self.assertEqual(comparisons, 7)
        winner = tournament.get_winner()
        self.assertIsNotNone(winner)

    def test_record_choice_advances(self):
        tournament = ABTournament(
            image=self.image,
            channels=[0, 0],
            parameter_ranges=self.parameter_ranges,
            n_candidates=4,
            strategy="latin_hypercube",
            seed=2,
        )
        tournament.generate_initial_candidates()

        matchup_1 = tournament.get_next_matchup()
        self.assertIsNotNone(matchup_1)
        tournament.record_choice(winner_id=matchup_1[0].id, loser_id=matchup_1[1].id)

        matchup_2 = tournament.get_next_matchup()
        self.assertIsNotNone(matchup_2)
        self.assertNotEqual({matchup_1[0].id, matchup_1[1].id}, {matchup_2[0].id, matchup_2[1].id})

    def test_offspring_generation_bounds(self):
        tournament = ABTournament(
            image=self.image,
            channels=[0, 0],
            parameter_ranges=self.parameter_ranges,
            n_candidates=4,
            strategy="latin_hypercube",
            seed=13,
        )
        initial = tournament.generate_initial_candidates()
        parent_ids = (initial[0].id, initial[1].id)

        offspring = tournament.generate_offspring_candidates(
            parent_ids=parent_ids,
            n_offspring=5,
            jitter=0.2,
        )

        self.assertEqual(len(offspring), 5)
        for child in offspring:
            p = child.params
            self.assertGreaterEqual(p["diameter"], 10.0)
            self.assertLessEqual(p["diameter"], 120.0)
            self.assertGreaterEqual(p["flow_threshold"], 0.1)
            self.assertLessEqual(p["flow_threshold"], 1.5)
            self.assertGreaterEqual(p["cellprob_threshold"], -3.0)
            self.assertLessEqual(p["cellprob_threshold"], 3.0)
            self.assertIn(p["model_name"], self.parameter_ranges["model_name"]["choices"])
            self.assertIn(
                p["restoration_mode"],
                self.parameter_ranges["restoration_mode"]["choices"],
            )

    def test_audit_log_completeness(self):
        tournament = ABTournament(
            image=self.image,
            channels=[0, 0],
            parameter_ranges=self.parameter_ranges,
            n_candidates=4,
            strategy="latin_hypercube",
            seed=11,
        )
        tournament.generate_initial_candidates()

        while True:
            matchup = tournament.get_next_matchup()
            if matchup is None:
                break
            tournament.record_choice(winner_id=matchup[0].id, loser_id=matchup[1].id)

        audit = tournament.get_audit_log()
        events = [entry.get("event") for entry in audit]

        self.assertIn("initial_candidates_generated", events)
        self.assertIn("tournament_complete", events)
        self.assertEqual(events.count("choice_recorded"), 3)

    def test_degenerate_zero_nuclei_candidates(self):
        tournament = ABTournament(
            image=self.image,
            channels=[0, 0],
            parameter_ranges=self.parameter_ranges,
            n_candidates=4,
            strategy="latin_hypercube",
            seed=17,
        )
        candidates = tournament.generate_initial_candidates()

        for candidate in candidates:
            tournament.attach_segmentation_result(candidate.id, np.zeros((64, 64), dtype=np.uint16))
            self.assertEqual(tournament.candidates[candidate.id].nucleus_count, 0)

        while True:
            matchup = tournament.get_next_matchup()
            if matchup is None:
                break
            tournament.record_choice(winner_id=matchup[0].id, loser_id=matchup[1].id)

        winner = tournament.get_winner()
        self.assertIsNotNone(winner)
        self.assertEqual(winner.nucleus_count, 0)


if __name__ == "__main__":
    unittest.main()
