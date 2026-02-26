"""
A/B tournament engine for segmentation parameter optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import copy

import numpy as np


DEFAULT_PARAMETER_RANGES: Dict[str, Any] = {
    "diameter": {"type": "float", "min": 10.0, "max": 120.0, "allow_auto": True, "auto_probability": 0.1},
    "flow_threshold": {"type": "float", "min": 0.1, "max": 1.5},
    "cellprob_threshold": {"type": "float", "min": -3.0, "max": 3.0},
    "model_name": {"type": "categorical", "choices": ["nuclei", "cyto", "cyto2", "cyto3"]},
    "restoration_mode": {"type": "categorical", "choices": ["none", "denoise", "deblur"]},
}


@dataclass
class ParameterCandidate:
    """A single parameter set with optional segmentation outputs."""

    id: str
    params: Dict[str, Any]
    masks: Optional[np.ndarray] = None
    nucleus_count: int = 0
    median_area: float = 0.0
    generation: int = 1
    parent_ids: Optional[Tuple[str, str]] = None


class ABTournament:
    """Manages A/B tournament bracket and parameter evolution."""

    def __init__(
        self,
        image: np.ndarray,
        channels: List[int],
        parameter_ranges: Optional[Dict[str, Any]],
        n_candidates: int = 8,
        strategy: str = "latin_hypercube",
        seed: Optional[int] = None,
        enable_offspring: bool = False,
        offspring_jitter: float = 0.15,
    ):
        self.image = image
        self.channels = channels
        self.parameter_ranges = parameter_ranges or copy.deepcopy(DEFAULT_PARAMETER_RANGES)
        self.n_candidates = int(n_candidates)
        self.strategy = strategy
        self.seed = seed
        self.enable_offspring = bool(enable_offspring)
        self.offspring_jitter = float(offspring_jitter)

        self._rng = np.random.default_rng(seed)

        self.candidates: Dict[str, ParameterCandidate] = {}
        self.initial_candidate_ids: List[str] = []

        self._round_number = 1
        self._current_round_candidate_ids: List[str] = []
        self._current_matchups: List[Tuple[str, str]] = []
        self._current_matchup_index = 0
        self._next_round_winners: List[str] = []
        self._winner_id: Optional[str] = None
        self._offspring_added = False

        self._history_snapshots: List[Dict[str, Any]] = []
        self._audit_log: List[Dict[str, Any]] = []

        self._continuous_specs, self._categorical_specs = self._normalize_parameter_ranges(
            self.parameter_ranges
        )

    @property
    def expected_comparisons(self) -> int:
        """Expected number of pairwise comparisons without regenerations."""
        return max(0, self.n_candidates - 1)

    def generate_initial_candidates(self) -> List[ParameterCandidate]:
        """Generate initial tournament candidates."""
        if self.n_candidates < 2:
            raise ValueError("n_candidates must be >= 2")

        self.candidates.clear()
        self.initial_candidate_ids.clear()
        self._history_snapshots.clear()
        self._audit_log.clear()
        self._winner_id = None
        self._round_number = 1
        self._offspring_added = False

        for params in self._sample_parameter_sets(self.n_candidates):
            candidate = ParameterCandidate(
                id=str(uuid4()),
                params=params,
                generation=1,
            )
            self.candidates[candidate.id] = candidate
            self.initial_candidate_ids.append(candidate.id)

        self._current_round_candidate_ids = list(self.initial_candidate_ids)
        self._next_round_winners = []
        self._prepare_round_matchups()

        self._append_audit(
            {
                "event": "initial_candidates_generated",
                "n_candidates": len(self.initial_candidate_ids),
                "candidate_ids": list(self.initial_candidate_ids),
                "strategy": self.strategy,
            }
        )

        return [self.candidates[cid] for cid in self.initial_candidate_ids]

    def get_next_matchup(self) -> Optional[Tuple[ParameterCandidate, ParameterCandidate]]:
        """Return the next A/B matchup, or None when complete."""
        if self._winner_id is not None:
            return None

        if self._current_matchup_index >= len(self._current_matchups):
            # Can happen if the previous action ended a round.
            self._advance_if_round_complete()
            if self._winner_id is not None:
                return None

        if self._current_matchup_index >= len(self._current_matchups):
            return None

        cid_a, cid_b = self._current_matchups[self._current_matchup_index]
        return self.candidates[cid_a], self.candidates[cid_b]

    def record_choice(self, winner_id: str, loser_id: str):
        """Record user choice and advance tournament state."""
        matchup = self.get_next_matchup()
        if matchup is None:
            raise RuntimeError("Tournament is already complete")

        expected_ids = {matchup[0].id, matchup[1].id}
        provided_ids = {winner_id, loser_id}
        if expected_ids != provided_ids:
            raise ValueError("winner_id/loser_id must match the current matchup")

        if winner_id == loser_id:
            raise ValueError("winner_id and loser_id must be different")

        self._history_snapshots.append(self._snapshot_state())

        self._next_round_winners.append(winner_id)

        self._append_audit(
            {
                "event": "choice_recorded",
                "round": self._round_number,
                "matchup_index": self._current_matchup_index,
                "winner_id": winner_id,
                "loser_id": loser_id,
                "winner_params": copy.deepcopy(self.candidates[winner_id].params),
                "loser_params": copy.deepcopy(self.candidates[loser_id].params),
            }
        )

        self._current_matchup_index += 1
        self._advance_if_round_complete()

    def undo_last_choice(self) -> bool:
        """Undo the last recorded choice. Returns True if successful."""
        if not self._history_snapshots:
            return False

        snapshot = self._history_snapshots.pop()
        self._restore_snapshot(snapshot)

        self._append_audit(
            {
                "event": "undo_last_choice",
                "round": self._round_number,
                "matchup_index": self._current_matchup_index,
            }
        )
        return True

    def regenerate_current_matchup(self) -> Optional[Tuple[ParameterCandidate, ParameterCandidate]]:
        """Replace the current matchup with two freshly sampled candidates."""
        matchup = self.get_next_matchup()
        if matchup is None:
            return None

        old_a, old_b = matchup
        new_params = self._sample_parameter_sets(2)

        new_candidates: List[ParameterCandidate] = []
        for params in new_params:
            candidate = ParameterCandidate(
                id=str(uuid4()),
                params=params,
                generation=self._round_number,
                parent_ids=None,
            )
            self.candidates[candidate.id] = candidate
            new_candidates.append(candidate)

        self._current_matchups[self._current_matchup_index] = (
            new_candidates[0].id,
            new_candidates[1].id,
        )

        self._append_audit(
            {
                "event": "matchup_regenerated",
                "round": self._round_number,
                "matchup_index": self._current_matchup_index,
                "replaced_candidate_ids": [old_a.id, old_b.id],
                "new_candidate_ids": [new_candidates[0].id, new_candidates[1].id],
            }
        )

        return new_candidates[0], new_candidates[1]

    def attach_segmentation_result(self, candidate_id: str, masks: np.ndarray):
        """Attach segmentation output and computed summary stats to candidate."""
        if candidate_id not in self.candidates:
            raise KeyError(f"Unknown candidate id: {candidate_id}")

        candidate = self.candidates[candidate_id]
        candidate.masks = masks
        candidate.nucleus_count, candidate.median_area = self._compute_mask_stats(masks)

    def get_winner(self) -> ParameterCandidate:
        """Return the winning candidate when tournament is complete."""
        if self._winner_id is None:
            raise RuntimeError("Tournament is not complete")
        return self.candidates[self._winner_id]

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return the full tournament audit log."""
        return copy.deepcopy(self._audit_log)

    def generate_offspring_candidates(
        self,
        parent_ids: Optional[Tuple[str, str]] = None,
        n_offspring: int = 2,
        jitter: float = 0.15,
        generation: Optional[int] = None,
    ) -> List[ParameterCandidate]:
        """Create offspring candidates interpolating two parents."""
        if n_offspring <= 0:
            return []

        if parent_ids is None:
            parents = self._select_top_two_by_wins()
            if len(parents) < 2:
                raise ValueError("Need at least two parent candidates")
            parent_a, parent_b = parents[0], parents[1]
        else:
            if len(parent_ids) != 2:
                raise ValueError("parent_ids must contain exactly two candidate IDs")
            if parent_ids[0] not in self.candidates or parent_ids[1] not in self.candidates:
                raise KeyError("parent_ids must refer to existing candidates")
            parent_a = self.candidates[parent_ids[0]]
            parent_b = self.candidates[parent_ids[1]]

        jitter = float(jitter)
        generation = int(generation if generation is not None else (self._round_number + 1))

        offspring: List[ParameterCandidate] = []
        for _ in range(n_offspring):
            child_params = self._interpolate_params(parent_a.params, parent_b.params, jitter=jitter)
            child = ParameterCandidate(
                id=str(uuid4()),
                params=child_params,
                generation=generation,
                parent_ids=(parent_a.id, parent_b.id),
            )
            self.candidates[child.id] = child
            offspring.append(child)

        self._append_audit(
            {
                "event": "offspring_generated",
                "generation": generation,
                "parent_ids": [parent_a.id, parent_b.id],
                "offspring_ids": [c.id for c in offspring],
                "jitter": jitter,
            }
        )

        return offspring

    def _prepare_round_matchups(self):
        """Prepare pairings for the current round."""
        ids = list(self._current_round_candidate_ids)
        self._current_matchups = []
        self._current_matchup_index = 0

        if not ids:
            self._winner_id = None
            return

        # Pair sequentially; odd candidate gets a bye.
        if len(ids) % 2 == 1:
            bye_id = ids.pop()
            self._next_round_winners.append(bye_id)
            self._append_audit(
                {
                    "event": "bye_advance",
                    "round": self._round_number,
                    "candidate_id": bye_id,
                }
            )

        for idx in range(0, len(ids), 2):
            self._current_matchups.append((ids[idx], ids[idx + 1]))

        if not self._current_matchups and len(self._next_round_winners) == 1:
            self._winner_id = self._next_round_winners[0]
            self._append_audit(
                {
                    "event": "tournament_complete",
                    "winner_id": self._winner_id,
                    "round": self._round_number,
                }
            )

    def _advance_if_round_complete(self):
        """Advance round state if all matchups in current round are resolved."""
        if self._winner_id is not None:
            return

        if self._current_matchup_index < len(self._current_matchups):
            return

        if self.enable_offspring and not self._offspring_added and len(self._current_round_candidate_ids) == 4:
            if len(self._next_round_winners) >= 2:
                parent_pair = (self._next_round_winners[0], self._next_round_winners[1])
                offspring = self.generate_offspring_candidates(
                    parent_ids=parent_pair,
                    n_offspring=2,
                    jitter=self.offspring_jitter,
                    generation=self._round_number + 1,
                )
                self._offspring_added = True
                self._current_round_candidate_ids = list(parent_pair) + [c.id for c in offspring]
                self._next_round_winners = []
                self._round_number += 1
                self._prepare_round_matchups()
                return

        if len(self._next_round_winners) == 1:
            self._winner_id = self._next_round_winners[0]
            self._append_audit(
                {
                    "event": "tournament_complete",
                    "winner_id": self._winner_id,
                    "round": self._round_number,
                }
            )
            return

        if len(self._next_round_winners) >= 2:
            self._current_round_candidate_ids = list(self._next_round_winners)
            self._next_round_winners = []
            self._round_number += 1
            self._prepare_round_matchups()

    def _snapshot_state(self) -> Dict[str, Any]:
        return {
            "round_number": self._round_number,
            "current_round_candidate_ids": list(self._current_round_candidate_ids),
            "current_matchups": list(self._current_matchups),
            "current_matchup_index": self._current_matchup_index,
            "next_round_winners": list(self._next_round_winners),
            "winner_id": self._winner_id,
            "offspring_added": self._offspring_added,
            "audit_len": len(self._audit_log),
        }

    def _restore_snapshot(self, snapshot: Dict[str, Any]):
        self._round_number = int(snapshot["round_number"])
        self._current_round_candidate_ids = list(snapshot["current_round_candidate_ids"])
        self._current_matchups = list(snapshot["current_matchups"])
        self._current_matchup_index = int(snapshot["current_matchup_index"])
        self._next_round_winners = list(snapshot["next_round_winners"])
        self._winner_id = snapshot["winner_id"]
        self._offspring_added = bool(snapshot.get("offspring_added", False))

        audit_len = int(snapshot.get("audit_len", len(self._audit_log)))
        if audit_len < len(self._audit_log):
            self._audit_log = self._audit_log[:audit_len]

    def _sample_parameter_sets(self, n: int) -> List[Dict[str, Any]]:
        """Sample parameter dictionaries based on configured strategy."""
        if n <= 0:
            return []

        continuous_keys = list(self._continuous_specs.keys())
        categorical_keys = list(self._categorical_specs.keys())
        samples: List[Dict[str, Any]] = []

        if continuous_keys:
            continuous_matrix = self._sample_continuous_matrix(len(continuous_keys), n)
        else:
            continuous_matrix = np.zeros((n, 0), dtype=float)

        for row_idx in range(n):
            params: Dict[str, Any] = {}

            for col_idx, key in enumerate(continuous_keys):
                spec = self._continuous_specs[key]
                min_val = float(spec["min"])
                max_val = float(spec["max"])

                value = min_val + float(continuous_matrix[row_idx, col_idx]) * (max_val - min_val)
                value = float(np.clip(value, min_val, max_val))

                if spec.get("allow_auto"):
                    auto_probability = float(spec.get("auto_probability", 0.0))
                    if self._rng.random() < auto_probability:
                        params[key] = None
                        continue

                params[key] = value

            for key in categorical_keys:
                choices = list(self._categorical_specs[key]["choices"])
                params[key] = choices[int(self._rng.integers(0, len(choices)))]

            samples.append(params)

        return samples

    def _sample_continuous_matrix(self, d: int, n: int) -> np.ndarray:
        if d <= 0:
            return np.zeros((n, 0), dtype=float)

        if self.strategy == "latin_hypercube":
            try:
                from scipy.stats import qmc

                lhs = qmc.LatinHypercube(d=d, seed=self._rng)
                return lhs.random(n=n)
            except Exception:
                # Fall back to random sampling if scipy is unavailable.
                return self._rng.random((n, d))

        return self._rng.random((n, d))

    @staticmethod
    def _normalize_parameter_ranges(parameter_ranges: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        continuous: Dict[str, Dict[str, Any]] = {}
        categorical: Dict[str, Dict[str, Any]] = {}

        for key, spec in parameter_ranges.items():
            if isinstance(spec, dict):
                p_type = spec.get("type")
                if p_type in {"float", "int", "continuous"}:
                    continuous[key] = {
                        "min": float(spec["min"]),
                        "max": float(spec["max"]),
                        "allow_auto": bool(spec.get("allow_auto", False)),
                        "auto_probability": float(spec.get("auto_probability", 0.0)),
                    }
                elif p_type == "categorical":
                    choices = list(spec.get("choices", []))
                    if not choices:
                        raise ValueError(f"Categorical parameter '{key}' has no choices")
                    categorical[key] = {"choices": choices}
                else:
                    # Heuristic fallback for dict-shaped specs.
                    if "choices" in spec:
                        choices = list(spec.get("choices", []))
                        if not choices:
                            raise ValueError(f"Categorical parameter '{key}' has no choices")
                        categorical[key] = {"choices": choices}
                    elif "min" in spec and "max" in spec:
                        continuous[key] = {
                            "min": float(spec["min"]),
                            "max": float(spec["max"]),
                            "allow_auto": bool(spec.get("allow_auto", False)),
                            "auto_probability": float(spec.get("auto_probability", 0.0)),
                        }
                    else:
                        raise ValueError(f"Could not parse parameter spec for '{key}'")

            elif isinstance(spec, (tuple, list)) and len(spec) == 2 and all(
                isinstance(v, (int, float)) for v in spec
            ):
                continuous[key] = {
                    "min": float(spec[0]),
                    "max": float(spec[1]),
                    "allow_auto": False,
                    "auto_probability": 0.0,
                }

            elif isinstance(spec, (tuple, list)) and len(spec) >= 1:
                categorical[key] = {"choices": list(spec)}

            else:
                raise ValueError(f"Unsupported parameter spec for '{key}': {spec}")

        return continuous, categorical

    def _interpolate_params(self, params_a: Dict[str, Any], params_b: Dict[str, Any], jitter: float) -> Dict[str, Any]:
        """Interpolate between two parameter sets with bounded jitter."""
        child: Dict[str, Any] = {}

        for key, spec in self._continuous_specs.items():
            min_val = float(spec["min"])
            max_val = float(spec["max"])
            span = max_val - min_val

            val_a = params_a.get(key)
            val_b = params_b.get(key)

            if val_a is None:
                val_a = (min_val + max_val) / 2.0
            if val_b is None:
                val_b = (min_val + max_val) / 2.0

            val_a = float(val_a)
            val_b = float(val_b)

            midpoint = (val_a + val_b) / 2.0
            local_span = abs(val_a - val_b)
            noise_scale = local_span if local_span > 0 else span * 0.2
            noise = self._rng.uniform(-jitter, jitter) * noise_scale
            candidate_value = float(np.clip(midpoint + noise, min_val, max_val))

            if spec.get("allow_auto"):
                auto_probability = min(0.2, float(spec.get("auto_probability", 0.0)) + 0.05)
                if self._rng.random() < auto_probability:
                    child[key] = None
                    continue

            child[key] = candidate_value

        for key, spec in self._categorical_specs.items():
            choices = list(spec["choices"])
            a = params_a.get(key)
            b = params_b.get(key)

            if a in choices and b in choices:
                if self._rng.random() < 0.45:
                    child[key] = a
                elif self._rng.random() < 0.9:
                    child[key] = b
                else:
                    child[key] = choices[int(self._rng.integers(0, len(choices)))]
            elif a in choices:
                child[key] = a
            elif b in choices:
                child[key] = b
            else:
                child[key] = choices[int(self._rng.integers(0, len(choices)))]

        return child

    def _select_top_two_by_wins(self) -> List[ParameterCandidate]:
        """Select top two candidates by recorded wins."""
        win_counts: Dict[str, int] = {cid: 0 for cid in self.candidates.keys()}
        for entry in self._audit_log:
            if entry.get("event") == "choice_recorded":
                win_id = entry.get("winner_id")
                if win_id in win_counts:
                    win_counts[win_id] += 1

        ranked = sorted(
            self.candidates.values(),
            key=lambda c: (win_counts.get(c.id, 0), c.nucleus_count, -c.generation),
            reverse=True,
        )
        return ranked[:2]

    @staticmethod
    def _compute_mask_stats(masks: Optional[np.ndarray]) -> Tuple[int, float]:
        if masks is None:
            return 0, 0.0

        arr = np.asarray(masks)
        if arr.size == 0:
            return 0, 0.0

        labels, counts = np.unique(arr, return_counts=True)
        valid = labels > 0
        if not np.any(valid):
            return 0, 0.0

        areas = counts[valid]
        return int(areas.size), float(np.median(areas))

    def _append_audit(self, entry: Dict[str, Any]):
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "round": self._round_number,
        }
        payload.update(copy.deepcopy(entry))
        self._audit_log.append(payload)
