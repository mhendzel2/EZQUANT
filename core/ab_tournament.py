"""
A/B Tournament Segmentation Optimizer.

Presents two candidate segmentations side by side and the user picks the better one.
A bracket-style tournament converges on optimal parameters in log₂(N) rounds.
"""
from __future__ import annotations

import copy
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

try:
    from scipy.stats.qmc import LatinHypercube
    _HAS_SCIPY_QMC = True
except ImportError:
    _HAS_SCIPY_QMC = False


# ---------------------------------------------------------------------------
# Parameter space definition
# ---------------------------------------------------------------------------

PARAMETER_SPACE: dict[str, dict] = {
    "diameter": {"type": "float", "min": 10.0, "max": 120.0, "default": 30.0},
    "flow_threshold": {"type": "float", "min": 0.1, "max": 1.5, "default": 0.4},
    "cellprob_threshold": {"type": "float", "min": -3.0, "max": 3.0, "default": 0.0},
    "model_name": {
        "type": "categorical",
        "choices": ["nuclei", "cyto", "cyto2", "cyto3"],
        "default": "nuclei",
    },
    "restoration_mode": {
        "type": "categorical",
        "choices": ["none", "denoise", "deblur"],
        "default": "none",
    },
}

CONTINUOUS_PARAMS = [k for k, v in PARAMETER_SPACE.items() if v["type"] == "float"]
CATEGORICAL_PARAMS = [k for k, v in PARAMETER_SPACE.items() if v["type"] == "categorical"]


# ---------------------------------------------------------------------------
# ParameterCandidate
# ---------------------------------------------------------------------------

@dataclass
class ParameterCandidate:
    """A single parameter set with its segmentation result."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    params: dict = field(default_factory=dict)
    masks: Any = None  # np.ndarray | None
    nucleus_count: int = 0
    median_area: float = 0.0
    area_cv: float = 0.0
    generation: int = 0
    parent_ids: tuple | None = None

    def summary(self) -> dict:
        return {
            "id": self.id,
            "params": self.params,
            "nucleus_count": self.nucleus_count,
            "median_area": self.median_area,
            "area_cv": self.area_cv,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids) if self.parent_ids else None,
        }


# ---------------------------------------------------------------------------
# ABTournament
# ---------------------------------------------------------------------------

class ABTournament:
    """
    Manages a bracket-style A/B tournament for segmentation parameter optimization.

    Strategy:
    - 'single_elimination': N candidates → log₂(N) rounds, N-1 matchups total.
    - 'swiss': All candidates play each round; top by win-count advances.
    """

    def __init__(
        self,
        image: np.ndarray,
        channels: list[int],
        parameter_ranges: dict | None = None,
        n_candidates: int = 8,
        strategy: str = "single_elimination",
        seed: int | None = None,
    ):
        self.image = image
        self.channels = channels
        self.parameter_ranges = parameter_ranges or {}
        self.n_candidates = n_candidates
        self.strategy = strategy
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Merge custom ranges into parameter space copy
        self._param_space = copy.deepcopy(PARAMETER_SPACE)
        for param, bounds in self.parameter_ranges.items():
            if param in self._param_space:
                if "min" in bounds:
                    self._param_space[param]["min"] = bounds["min"]
                if "max" in bounds:
                    self._param_space[param]["max"] = bounds["max"]

        self.candidates: list[ParameterCandidate] = []
        self._bracket: list[list[ParameterCandidate]] = []  # rounds
        self._current_round: int = 0
        self._current_matchup_idx: int = 0
        self._matchup_log: list[dict] = []
        self._winner: ParameterCandidate | None = None
        self._undo_stack: list[dict] = []  # for undo

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def generate_initial_candidates(self) -> list[ParameterCandidate]:
        """Generate N initial candidates via Latin Hypercube Sampling."""
        params_list = self._lhs_sample(self.n_candidates)
        candidates = []
        for i, params in enumerate(params_list):
            c = ParameterCandidate(params=params, generation=0)
            candidates.append(c)
        self.candidates = candidates
        self._init_bracket()
        return candidates

    def _lhs_sample(self, n: int) -> list[dict]:
        """Sample n parameter sets using Latin Hypercube Sampling."""
        cont_params = CONTINUOUS_PARAMS
        n_cont = len(cont_params)

        if _HAS_SCIPY_QMC and n_cont > 0:
            sampler = LatinHypercube(d=n_cont, seed=self._np_rng.integers(0, 2**31))
            lhs_samples = sampler.random(n=n)  # shape (n, n_cont) in [0, 1]
        else:
            # Fallback: stratified random sampling
            lhs_samples = self._stratified_sample(n, n_cont)

        results = []
        for i in range(n):
            params: dict[str, Any] = {}
            for j, param_name in enumerate(cont_params):
                spec = self._param_space[param_name]
                lo, hi = spec["min"], spec["max"]
                params[param_name] = lo + lhs_samples[i, j] * (hi - lo)

            for param_name in CATEGORICAL_PARAMS:
                spec = self._param_space[param_name]
                params[param_name] = self._rng.choice(spec["choices"])

            results.append(params)
        return results

    def _stratified_sample(self, n: int, d: int) -> np.ndarray:
        """Fallback stratified sampling when scipy.qmc is unavailable."""
        result = np.zeros((n, d))
        for j in range(d):
            perm = self._np_rng.permutation(n)
            result[:, j] = (perm + self._np_rng.random(n)) / n
        return result

    # ------------------------------------------------------------------
    # Bracket management
    # ------------------------------------------------------------------

    def _init_bracket(self) -> None:
        """Initialize the first round with all candidates."""
        shuffled = list(self.candidates)
        self._rng.shuffle(shuffled)
        # Pad to even number
        if len(shuffled) % 2 != 0:
            shuffled.append(shuffled[0])  # bye — repeat first candidate
        self._bracket = [shuffled]
        self._current_round = 0
        self._current_matchup_idx = 0

    def _current_round_pairs(self) -> list[tuple[ParameterCandidate, ParameterCandidate]]:
        """Return (A, B) pairs for the current round."""
        round_candidates = self._bracket[self._current_round]
        pairs = []
        for i in range(0, len(round_candidates) - 1, 2):
            pairs.append((round_candidates[i], round_candidates[i + 1]))
        return pairs

    def get_next_matchup(self) -> tuple[ParameterCandidate, ParameterCandidate] | None:
        """Return the next A/B pair for the user to judge. None when complete."""
        if self._winner is not None:
            return None

        pairs = self._current_round_pairs()

        while self._current_matchup_idx >= len(pairs):
            # Advance to next round
            winners = [
                self._matchup_log[i]["winner_id"]
                for i in range(len(self._matchup_log))
                if self._matchup_log[i]["round"] == self._current_round
            ]
            winner_candidates = [c for c in self.candidates if c.id in winners]

            if len(winner_candidates) <= 1:
                if winner_candidates:
                    self._winner = winner_candidates[0]
                elif self.candidates:
                    self._winner = self.candidates[0]
                return None

            # Next round
            if len(winner_candidates) % 2 != 0:
                winner_candidates.append(winner_candidates[0])
            self._bracket.append(winner_candidates)
            self._current_round += 1
            self._current_matchup_idx = 0
            pairs = self._current_round_pairs()

        return pairs[self._current_matchup_idx]

    def record_choice(self, winner_id: str, loser_id: str) -> None:
        """Record the user's selection and advance the bracket."""
        # Save state for undo
        undo_state = {
            "round": self._current_round,
            "matchup_idx": self._current_matchup_idx,
            "winner": None,
        }
        self._undo_stack.append(undo_state)

        entry = {
            "round": self._current_round,
            "matchup_index": self._current_matchup_idx,
            "winner_id": winner_id,
            "loser_id": loser_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._matchup_log.append(entry)
        self._current_matchup_idx += 1

        # Check if round is complete
        pairs = self._current_round_pairs()
        if self._current_matchup_idx >= len(pairs):
            # Collect winners of this round
            round_winners = [
                log["winner_id"]
                for log in self._matchup_log
                if log["round"] == self._current_round
            ]
            winner_candidates = [c for c in self.candidates if c.id in round_winners]

            if len(winner_candidates) == 1:
                self._winner = winner_candidates[0]

    def undo_last_choice(self) -> bool:
        """Undo the last choice and re-present the previous matchup. Returns True on success."""
        if not self._undo_stack or not self._matchup_log:
            return False
        state = self._undo_stack.pop()
        self._matchup_log.pop()
        self._current_round = state["round"]
        self._current_matchup_idx = state["matchup_idx"]
        self._winner = None
        return True

    def regenerate_pair(self, better_id: str | None = None) -> tuple[ParameterCandidate, ParameterCandidate]:
        """
        Replace current pair with two fresh candidates.
        If better_id provided, one new candidate is near that candidate's params.
        """
        if better_id:
            better = next((c for c in self.candidates if c.id == better_id), None)
        else:
            better = None

        new_candidates = []
        for _ in range(2):
            if better:
                params = self._jitter_params(better.params, jitter_fraction=0.15)
                c = ParameterCandidate(
                    params=params,
                    generation=self._current_round + 1,
                    parent_ids=(better.id,),
                )
            else:
                params = self._lhs_sample(1)[0]
                c = ParameterCandidate(params=params, generation=self._current_round + 1)
            new_candidates.append(c)
            self.candidates.append(c)

        # Replace current pair in bracket
        round_candidates = self._bracket[self._current_round]
        pair_start = self._current_matchup_idx * 2
        if pair_start + 1 < len(round_candidates):
            round_candidates[pair_start] = new_candidates[0]
            round_candidates[pair_start + 1] = new_candidates[1]

        return (new_candidates[0], new_candidates[1])

    def _jitter_params(self, params: dict, jitter_fraction: float = 0.15) -> dict:
        """Create a new params dict by jittering continuous params ±jitter_fraction."""
        new_params = dict(params)
        for param_name in CONTINUOUS_PARAMS:
            if param_name not in new_params:
                continue
            spec = self._param_space[param_name]
            lo, hi = spec["min"], spec["max"]
            val = new_params[param_name]
            delta = (hi - lo) * jitter_fraction
            jittered = val + self._rng.uniform(-delta, delta)
            new_params[param_name] = max(lo, min(hi, jittered))
        for param_name in CATEGORICAL_PARAMS:
            # Small chance of changing categorical
            if self._rng.random() < 0.2:
                spec = self._param_space[param_name]
                new_params[param_name] = self._rng.choice(spec["choices"])
        return new_params

    def generate_offspring(
        self, parent_a: ParameterCandidate, parent_b: ParameterCandidate, n: int = 2
    ) -> list[ParameterCandidate]:
        """
        Generate n offspring candidates by interpolating between two parents' params.
        Used for optional refinement rounds after semifinals.
        """
        offspring = []
        for _ in range(n):
            new_params: dict[str, Any] = {}
            for param_name in CONTINUOUS_PARAMS:
                spec = self._param_space[param_name]
                lo, hi = spec["min"], spec["max"]
                a_val = parent_a.params.get(param_name, spec["default"])
                b_val = parent_b.params.get(param_name, spec["default"])
                alpha = self._rng.uniform(0.3, 0.7)
                mid = alpha * a_val + (1 - alpha) * b_val
                jitter = (hi - lo) * self._rng.uniform(-0.1, 0.1)
                new_params[param_name] = max(lo, min(hi, mid + jitter))

            for param_name in CATEGORICAL_PARAMS:
                # Inherit from one of the parents
                new_params[param_name] = (
                    parent_a.params.get(param_name) if self._rng.random() < 0.5
                    else parent_b.params.get(param_name)
                )

            c = ParameterCandidate(
                params=new_params,
                generation=self._current_round + 1,
                parent_ids=(parent_a.id, parent_b.id),
            )
            offspring.append(c)
            self.candidates.append(c)
        return offspring

    # ------------------------------------------------------------------
    # Winner and audit
    # ------------------------------------------------------------------

    def get_winner(self) -> ParameterCandidate | None:
        """Return the final tournament winner."""
        return self._winner

    def total_matchups(self) -> int:
        """Return the total expected number of matchups (N-1 for single elimination)."""
        return self.n_candidates - 1

    def completed_matchups(self) -> int:
        """Return how many matchups have been judged."""
        return len(self._matchup_log)

    def get_audit_log(self) -> list[dict]:
        """Return full history of matchups and choices."""
        return [
            {
                "candidates": [c.summary() for c in self.candidates],
                "matchups": list(self._matchup_log),
                "winner": self._winner.summary() if self._winner else None,
            }
        ]

    def get_progress(self) -> dict:
        """Return current progress information."""
        total = max(self.total_matchups(), 1)
        completed = self.completed_matchups()
        rounds_total = max(1, (self.n_candidates - 1).bit_length())
        return {
            "round_current": self._current_round + 1,
            "rounds_total": rounds_total,
            "matchup_current": self._current_matchup_idx + 1,
            "matchups_in_round": len(self._current_round_pairs()),
            "total_completed": completed,
            "total_expected": total,
        }
