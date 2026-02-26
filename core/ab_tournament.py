"""
A/B Tournament system for segmentation parameter optimization.

Uses Latin Hypercube Sampling to generate parameter candidates, then runs
single-elimination or Swiss-system tournament brackets where the user
picks the better segmentation result. Converges on optimal parameters
in log₂(N) rounds.
"""

from __future__ import annotations

import uuid
import copy
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Parameter space definition
PARAMETER_SPACE: Dict[str, dict] = {
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


@dataclass
class ParameterCandidate:
    """A single parameter set with its segmentation result."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    params: Dict = field(default_factory=dict)
    masks: Optional[np.ndarray] = field(default=None, repr=False)
    nucleus_count: int = 0
    median_area: float = 0.0
    area_cv: float = 0.0
    generation: int = 0  # Tournament round this candidate was created in
    parent_ids: Optional[Tuple[str, str]] = None  # For offspring candidates

    def to_dict(self) -> dict:
        """Serialise to a dict (excluding large mask array)."""
        return {
            "id": self.id,
            "params": self.params,
            "nucleus_count": self.nucleus_count,
            "median_area": self.median_area,
            "area_cv": self.area_cv,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids) if self.parent_ids else None,
        }


def _lhs_sample(n: int, param_space: Dict[str, dict]) -> List[Dict]:
    """
    Generate *n* parameter dicts via Latin Hypercube Sampling.

    Continuous parameters are sampled with LHS; categoricals are randomly
    assigned.  Falls back to pure random sampling if scipy is unavailable.
    """
    float_keys = [k for k, v in param_space.items() if v["type"] == "float"]
    cat_keys = [k for k, v in param_space.items() if v["type"] == "categorical"]

    try:
        from scipy.stats.qmc import LatinHypercube

        sampler = LatinHypercube(d=len(float_keys), seed=42)
        unit_samples = sampler.random(n=n)  # shape (n, d), values in [0, 1)
    except ImportError:
        rng = np.random.default_rng(42)
        unit_samples = rng.random((n, len(float_keys)))

    candidates: List[Dict] = []
    for i in range(n):
        p: Dict = {}
        for j, key in enumerate(float_keys):
            spec = param_space[key]
            p[key] = spec["min"] + unit_samples[i, j] * (spec["max"] - spec["min"])
        for key in cat_keys:
            spec = param_space[key]
            p[key] = random.choice(spec["choices"])
        candidates.append(p)
    return candidates


def _compute_mask_stats(masks: Optional[np.ndarray]) -> Tuple[int, float, float]:
    """Return (nucleus_count, median_area, area_cv) from a label mask."""
    if masks is None or masks.max() == 0:
        return 0, 0.0, 0.0
    labels = np.unique(masks)
    labels = labels[labels > 0]
    areas = np.array([np.sum(masks == lbl) for lbl in labels], dtype=float)
    nucleus_count = int(len(areas))
    median_area = float(np.median(areas)) if nucleus_count > 0 else 0.0
    area_cv = float(np.std(areas) / np.mean(areas)) if nucleus_count > 0 else 0.0
    return nucleus_count, median_area, area_cv


def _interpolate_params(
    a: Dict, b: Dict, param_space: Dict[str, dict], jitter: float = 0.15
) -> Dict:
    """
    Create an offspring parameter dict by interpolating between *a* and *b*
    with a random jitter of ±*jitter* relative to the continuous range.
    """
    rng = random.Random()
    p: Dict = {}
    for key, spec in param_space.items():
        if spec["type"] == "float":
            mid = (a[key] + b[key]) / 2.0
            rng_range = (spec["max"] - spec["min"]) * jitter
            p[key] = mid + rng.uniform(-rng_range, rng_range)
            p[key] = max(spec["min"], min(spec["max"], p[key]))
        else:
            # Pick categorical from one of the two parents randomly
            p[key] = rng.choice([a[key], b[key]])
    return p


class ABTournament:
    """
    Manages the bracket and parameter evolution for A/B segmentation
    parameter optimisation.

    Workflow
    --------
    1. Instantiate with an image array and parameter ranges.
    2. Call :meth:`generate_initial_candidates` to get the first generation.
    3. Run segmentation for each candidate externally and set
       ``candidate.masks``, ``candidate.nucleus_count``, etc.
    4. Call :meth:`get_next_matchup` → ``(A, B) | None``.
    5. Show A and B to the user; call :meth:`record_choice`.
    6. Repeat until :meth:`get_next_matchup` returns ``None``.
    7. Retrieve the winner with :meth:`get_winner`.
    """

    def __init__(
        self,
        image: np.ndarray,
        channels: List[int],
        parameter_ranges: Optional[Dict] = None,
        n_candidates: int = 8,
        strategy: str = "latin_hypercube",
        enable_offspring: bool = True,
    ) -> None:
        self.image = image
        self.channels = channels
        self.parameter_ranges = parameter_ranges or PARAMETER_SPACE
        self.n_candidates = n_candidates
        self.strategy = strategy
        self.enable_offspring = enable_offspring

        # State
        self._candidates: Dict[str, ParameterCandidate] = {}
        self._bracket: List[List[str]] = []  # List of rounds; each round is a list of IDs
        self._current_round: int = 0
        self._current_matchup_idx: int = 0
        self._audit_log: List[dict] = []
        self._matchup_history: List[Tuple[str, str, str]] = []  # (A_id, B_id, winner_id)
        self._winner_id: Optional[str] = None
        self._tournament_complete: bool = False
        self._offspring_generated: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_initial_candidates(self) -> List[ParameterCandidate]:
        """
        Generate :attr:`n_candidates` initial candidates via LHS.

        Returns the list so callers can immediately dispatch segmentation
        workers for each candidate.
        """
        if self.strategy == "latin_hypercube":
            param_list = _lhs_sample(self.n_candidates, self.parameter_ranges)
        else:
            # Fallback: pure random
            param_list = [
                {k: random.uniform(v["min"], v["max"])
                 if v["type"] == "float"
                 else random.choice(v["choices"])
                 for k, v in self.parameter_ranges.items()}
                for _ in range(self.n_candidates)
            ]

        candidates = []
        for params in param_list:
            c = ParameterCandidate(params=params, generation=0)
            self._candidates[c.id] = c
            candidates.append(c)

        # Build initial bracket round
        ids = [c.id for c in candidates]
        self._bracket = [ids]
        self._current_round = 0
        self._current_matchup_idx = 0

        self._audit_log.append(
            {
                "event": "tournament_start",
                "n_candidates": self.n_candidates,
                "strategy": self.strategy,
                "candidate_ids": ids,
            }
        )
        return candidates

    def get_next_matchup(
        self,
    ) -> Optional[Tuple[ParameterCandidate, ParameterCandidate]]:
        """
        Return the next (A, B) pair for the user to judge, or ``None`` if
        the tournament is complete.
        """
        if self._tournament_complete:
            return None

        # Ensure we have a bracket
        if not self._bracket:
            return None

        current_round_ids = self._bracket[self._current_round]

        # How many matchups in this round?
        pairs_in_round = len(current_round_ids) // 2
        if self._current_matchup_idx >= pairs_in_round:
            # Advance to the next round
            if not self._advance_round():
                return None
            current_round_ids = self._bracket[self._current_round]
            pairs_in_round = len(current_round_ids) // 2

        idx = self._current_matchup_idx * 2
        id_a = current_round_ids[idx]
        id_b = current_round_ids[idx + 1]
        return self._candidates[id_a], self._candidates[id_b]

    def record_choice(self, winner_id: str, loser_id: str) -> None:
        """
        Record the user's selection.  Advances the internal bracket state.

        Parameters
        ----------
        winner_id:
            The `ParameterCandidate.id` that the user preferred.
        loser_id:
            The `ParameterCandidate.id` that was not chosen.
        """
        self._matchup_history.append((loser_id, winner_id, winner_id))

        log_entry = {
            "event": "matchup_result",
            "round": self._current_round,
            "matchup_index": self._current_matchup_idx,
            "candidate_a": loser_id,
            "candidate_b": winner_id,
            "winner_id": winner_id,
            "loser_id": loser_id,
        }
        self._audit_log.append(log_entry)

        self._current_matchup_idx += 1

        # Check if this round is done
        current_round_ids = self._bracket[self._current_round]
        pairs_in_round = len(current_round_ids) // 2
        if self._current_matchup_idx >= pairs_in_round:
            self._advance_round()

    def undo_last_choice(self) -> Optional[Tuple[ParameterCandidate, ParameterCandidate]]:
        """
        Undo the last recorded choice and return the re-presented matchup.
        Returns ``None`` if there is no history to undo.
        """
        if not self._matchup_history:
            return None

        self._matchup_history.pop()
        self._audit_log.append({"event": "undo_last_choice"})

        # Step back the matchup index
        if self._current_matchup_idx > 0:
            self._current_matchup_idx -= 1
        else:
            # We need to go back a round — rebuild from history
            if self._current_round > 0:
                self._current_round -= 1
                current_round_ids = self._bracket[self._current_round]
                self._current_matchup_idx = len(current_round_ids) // 2 - 1

        self._tournament_complete = False
        self._winner_id = None
        return self.get_next_matchup()

    def get_winner(self) -> Optional[ParameterCandidate]:
        """Return the final tournament winner, or ``None`` if not yet complete."""
        if self._winner_id:
            return self._candidates[self._winner_id]
        return None

    def get_audit_log(self) -> List[dict]:
        """Return the full history of matchups and choices."""
        return list(self._audit_log)

    def get_all_candidates(self) -> List[ParameterCandidate]:
        """Return all candidates (initial + offspring)."""
        return list(self._candidates.values())

    def total_matchups(self) -> int:
        """Total number of matchups needed (N-1 for single-elimination)."""
        return max(0, self.n_candidates - 1)

    def completed_matchups(self) -> int:
        """Number of matchups that have been resolved so far."""
        return len(self._matchup_history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_round(self) -> bool:
        """
        Collect winners from the current round, optionally inject offspring,
        and set up the next round.

        Returns ``True`` if there is a next round; ``False`` if the tournament
        is now complete.
        """
        current_round_ids = self._bracket[self._current_round]
        pairs_in_round = len(current_round_ids) // 2

        # Collect winners from matchup history (last *pairs_in_round* entries)
        recent_history = self._matchup_history[-pairs_in_round:]
        winners = [entry[2] for entry in recent_history]  # winner_id is 3rd element

        # Handle odd IDs (bye rounds)
        odd_id = current_round_ids[-1] if len(current_round_ids) % 2 == 1 else None
        if odd_id:
            winners.append(odd_id)

        if len(winners) <= 1:
            # Final: we have a single winner
            self._winner_id = winners[0] if winners else None
            self._tournament_complete = True
            if self._winner_id:
                self._audit_log.append(
                    {
                        "event": "tournament_complete",
                        "winner_id": self._winner_id,
                        "winner_params": self._candidates[self._winner_id].params,
                    }
                )
            return False

        # Optionally spawn offspring after semifinal (when ≥4 → ≤4 remain)
        if (
            self.enable_offspring
            and not self._offspring_generated
            and len(winners) == 2
            and len(current_round_ids) >= 4
        ):
            offspring = self._generate_offspring(winners[0], winners[1])
            for o in offspring:
                self._candidates[o.id] = o
            winners.extend([o.id for o in offspring])
            self._offspring_generated = True
            self._audit_log.append(
                {
                    "event": "offspring_generated",
                    "parent_ids": winners[:2],
                    "offspring_ids": [o.id for o in offspring],
                }
            )

        self._bracket.append(winners)
        self._current_round += 1
        self._current_matchup_idx = 0
        return True

    def _generate_offspring(
        self, winner_id_a: str, winner_id_b: str, n: int = 2
    ) -> List[ParameterCandidate]:
        """
        Generate *n* offspring candidates by interpolating between the two
        finalist candidates with ±15% jitter.
        """
        a_params = self._candidates[winner_id_a].params
        b_params = self._candidates[winner_id_b].params
        offspring = []
        for _ in range(n):
            params = _interpolate_params(
                a_params, b_params, self.parameter_ranges, jitter=0.15
            )
            c = ParameterCandidate(
                params=params,
                generation=self._current_round + 1,
                parent_ids=(winner_id_a, winner_id_b),
            )
            offspring.append(c)
        return offspring
