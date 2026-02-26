"""
A/B tournament panel for segmentation parameter optimization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.ab_tournament import ABTournament, ParameterCandidate
from core.segmentation import SegmentationEngine
from gui.accessibility import AccessibilityManager
from gui.image_viewer import ImageViewer


class TournamentPrecomputeWorker(QThread):
    """Background worker that segments all candidates for tournament speed."""

    progress = Signal(int, int)
    finished = Signal(object)  # list[ParameterCandidate]
    error = Signal(str)

    def __init__(
        self,
        image: np.ndarray,
        channels: List[int],
        candidates: List[ParameterCandidate],
        gpu_available: bool = False,
    ):
        super().__init__()
        self.image = image
        self.channels = channels
        self.candidates = candidates
        self.gpu_available = gpu_available
        self._cancelled = False

    def run(self):
        try:
            engine = SegmentationEngine(gpu_available=self.gpu_available)
            total = len(self.candidates)

            for idx, candidate in enumerate(self.candidates):
                if self._cancelled:
                    return

                params = candidate.params
                masks, info = engine.segment_cellpose(
                    image=self.image,
                    model_name=params.get("model_name", "nuclei"),
                    diameter=params.get("diameter"),
                    flow_threshold=float(params.get("flow_threshold", 0.4)),
                    cellprob_threshold=float(params.get("cellprob_threshold", 0.0)),
                    do_3d=False,
                    channels=self.channels,
                )

                candidate.masks = masks
                candidate.nucleus_count = int(info.get("nucleus_count", 0))
                candidate.median_area = float(info.get("median_area", 0.0))
                self.progress.emit(idx + 1, total)

            self.finished.emit(self.candidates)

        except Exception as exc:
            self.error.emit(str(exc))

    def cancel(self):
        self._cancelled = True


class ABTournamentPanel(QDialog):
    """Dialog presenting side-by-side A/B segmentation matchups."""

    winner_selected = Signal(dict, object)  # params, audit_log

    def __init__(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        channels: Optional[List[int]] = None,
        parameter_ranges: Optional[Dict[str, Any]] = None,
        n_candidates: int = 8,
        gpu_available: bool = False,
        parent=None,
    ):
        super().__init__(parent)

        self.image = image
        self.metadata = metadata or {}
        self.channels = channels or [0, 0]
        self.gpu_available = gpu_available

        self.tournament = ABTournament(
            image=image,
            channels=self.channels,
            parameter_ranges=parameter_ranges,
            n_candidates=n_candidates,
            strategy="latin_hypercube",
            enable_offspring=False,
        )

        self.current_matchup: Optional[Tuple[ParameterCandidate, ParameterCandidate]] = None
        self.winner_candidate: Optional[ParameterCandidate] = None
        self._sync_guard = False
        self.worker: Optional[TournamentPrecomputeWorker] = None

        self.setWindowTitle("Optimize Parameters (A/B Tournament)")
        self.resize(1600, 900)

        self._setup_ui()
        self._setup_shortcuts()
        self._start_tournament()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self.round_label = QLabel("Preparing tournament...")
        self.round_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.round_label)

        viewers_row = QHBoxLayout()

        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        self.candidate_a_label = QLabel("Candidate A")
        self.viewer_a = ImageViewer()
        self.viewer_a.set_image(self.image, self.metadata)
        self.viewer_a.set_mask_style("solid", "#0072B2")
        self.stats_a = QLabel("-")
        left_layout.addWidget(self.candidate_a_label)
        left_layout.addWidget(self.viewer_a, stretch=1)
        left_layout.addWidget(self.stats_a)

        right_col = QWidget()
        right_layout = QVBoxLayout(right_col)
        self.candidate_b_label = QLabel("Candidate B")
        self.viewer_b = ImageViewer()
        self.viewer_b.set_image(self.image, self.metadata)
        self.viewer_b.set_mask_style("solid", "#E69F00")
        self.stats_b = QLabel("-")
        right_layout.addWidget(self.candidate_b_label)
        right_layout.addWidget(self.viewer_b, stretch=1)
        right_layout.addWidget(self.stats_b)

        viewers_row.addWidget(left_col, stretch=1)
        viewers_row.addWidget(right_col, stretch=1)

        layout.addLayout(viewers_row, stretch=1)

        controls_row = QHBoxLayout()
        self.choose_a_btn = QPushButton("◀ Choose A")
        self.choose_b_btn = QPushButton("Choose B ▶")
        self.zoom_sync_btn = QPushButton("Zoom Sync: ON")
        self.zoom_sync_btn.setCheckable(True)
        self.zoom_sync_btn.setChecked(True)
        self.regenerate_btn = QPushButton("Too Close - Regenerate Both")
        self.undo_btn = QPushButton("Undo Last Choice")

        self.choose_a_btn.clicked.connect(self._on_choose_a)
        self.choose_b_btn.clicked.connect(self._on_choose_b)
        self.zoom_sync_btn.clicked.connect(self._on_zoom_sync_toggled)
        self.regenerate_btn.clicked.connect(self._on_regenerate)
        self.undo_btn.clicked.connect(self._on_undo)

        controls_row.addWidget(self.choose_a_btn)
        controls_row.addWidget(self.zoom_sync_btn)
        controls_row.addWidget(self.choose_b_btn)
        controls_row.addWidget(self.regenerate_btn)
        controls_row.addWidget(self.undo_btn)

        layout.addLayout(controls_row)

        self.progress_label = QLabel("Progress: 0/0 comparisons")
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(max(1, self.tournament.expected_comparisons))
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        footer = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel Tournament")
        self.use_winner_btn = QPushButton("Use Winner ▶")
        self.use_winner_btn.setEnabled(False)

        self.cancel_btn.clicked.connect(self.reject)
        self.use_winner_btn.clicked.connect(self._on_use_winner)

        footer.addWidget(self.cancel_btn)
        footer.addStretch()
        footer.addWidget(self.use_winner_btn)
        layout.addLayout(footer)

        self.viewer_a.view_box.sigRangeChanged.connect(self._sync_view_from_a)
        self.viewer_b.view_box.sigRangeChanged.connect(self._sync_view_from_b)
        self.viewer_a.slice_changed.connect(self._sync_slice_from_a)
        self.viewer_b.slice_changed.connect(self._sync_slice_from_b)

        AccessibilityManager.apply_accessible_names(self)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("A"), self, self._on_choose_a)
        QShortcut(QKeySequence(Qt.Key_Left), self, self._on_choose_a)
        QShortcut(QKeySequence("B"), self, self._on_choose_b)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._on_choose_b)
        QShortcut(QKeySequence("R"), self, self._on_regenerate)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.reject)

    def _set_choice_buttons_enabled(self, enabled: bool):
        self.choose_a_btn.setEnabled(enabled)
        self.choose_b_btn.setEnabled(enabled)
        self.regenerate_btn.setEnabled(enabled)

    def _start_tournament(self):
        candidates = self.tournament.generate_initial_candidates()
        self._segment_candidates_async(candidates)

    def _segment_candidates_async(self, candidates: List[ParameterCandidate]):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(100)

        self._set_choice_buttons_enabled(False)
        self.round_label.setText("Precomputing candidate segmentations...")

        self.worker = TournamentPrecomputeWorker(
            image=self.image,
            channels=self.channels,
            candidates=candidates,
            gpu_available=self.gpu_available,
        )
        self.worker.progress.connect(self._on_precompute_progress)
        self.worker.finished.connect(self._on_precompute_finished)
        self.worker.error.connect(self._on_precompute_error)
        self.worker.start()

    def _on_precompute_progress(self, current: int, total: int):
        self.round_label.setText(f"Precomputing candidate segmentations... {current}/{total}")

    def _on_precompute_finished(self, _candidates: object):
        self._set_choice_buttons_enabled(True)
        self._show_current_matchup()

    def _on_precompute_error(self, error: str):
        self._set_choice_buttons_enabled(False)
        QMessageBox.critical(self, "Tournament Error", f"Failed to precompute candidates:\n{error}")

    def _show_current_matchup(self):
        matchup = self.tournament.get_next_matchup()
        self.current_matchup = matchup

        choice_count = self._choice_count()
        total = max(1, self.tournament.expected_comparisons)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(min(choice_count, total))
        self.progress_label.setText(f"Progress: {choice_count}/{total} comparisons")

        if matchup is None:
            self.winner_candidate = self.tournament.get_winner()
            self.round_label.setText("Tournament complete. Review the winner and apply parameters.")
            self.use_winner_btn.setEnabled(True)
            self._set_choice_buttons_enabled(False)
            return

        cand_a, cand_b = matchup
        self.round_label.setText(
            f"Round {self._current_round_number()} - comparison {choice_count + 1}"
        )

        self.viewer_a.set_mask(cand_a.masks if cand_a.masks is not None else np.zeros(self.image.shape[-2:], dtype=np.uint16))
        self.viewer_b.set_mask(cand_b.masks if cand_b.masks is not None else np.zeros(self.image.shape[-2:], dtype=np.uint16))

        self.stats_a.setText(self._format_candidate_stats(cand_a))
        self.stats_b.setText(self._format_candidate_stats(cand_b))

        description = (
            f"Round {self._current_round_number()}, Candidate A: {cand_a.nucleus_count} nuclei, "
            f"Candidate B: {cand_b.nucleus_count} nuclei. Press A or B to choose."
        )
        self.setAccessibleDescription(description)

    def _format_candidate_stats(self, candidate: ParameterCandidate) -> str:
        params = candidate.params
        return (
            f"{candidate.nucleus_count} nuclei | median {candidate.median_area:.1f} px²\n"
            f"diameter={params.get('diameter')} flow={params.get('flow_threshold', 0):.2f} "
            f"cellprob={params.get('cellprob_threshold', 0):.2f} model={params.get('model_name', 'nuclei')}"
        )

    def _on_choose_a(self):
        if not self.current_matchup:
            return
        cand_a, cand_b = self.current_matchup
        self.tournament.record_choice(winner_id=cand_a.id, loser_id=cand_b.id)
        self._show_current_matchup()

    def _on_choose_b(self):
        if not self.current_matchup:
            return
        cand_a, cand_b = self.current_matchup
        self.tournament.record_choice(winner_id=cand_b.id, loser_id=cand_a.id)
        self._show_current_matchup()

    def _on_regenerate(self):
        regenerated = self.tournament.regenerate_current_matchup()
        if regenerated is None:
            return
        self._segment_candidates_async(list(regenerated))

    def _on_undo(self):
        if self.tournament.undo_last_choice():
            self._show_current_matchup()

    def _on_use_winner(self):
        if self.winner_candidate is None:
            return

        self.winner_selected.emit(
            dict(self.winner_candidate.params),
            self.tournament.get_audit_log(),
        )
        self.accept()

    def _choice_count(self) -> int:
        return sum(1 for entry in self.tournament.get_audit_log() if entry.get("event") == "choice_recorded")

    def _current_round_number(self) -> int:
        audit = self.tournament.get_audit_log()
        if not audit:
            return 1
        return int(audit[-1].get("round", 1))

    def _sync_view_from_a(self, *_args):
        self._sync_views(self.viewer_a, self.viewer_b)

    def _sync_view_from_b(self, *_args):
        self._sync_views(self.viewer_b, self.viewer_a)

    def _sync_views(self, source: ImageViewer, target: ImageViewer):
        if self._sync_guard or not self.zoom_sync_btn.isChecked():
            return

        self._sync_guard = True
        try:
            x_range, y_range = source.view_box.viewRange()
            target.view_box.setRange(xRange=x_range, yRange=y_range, padding=0)
        finally:
            self._sync_guard = False

    def _sync_slice_from_a(self, value: int):
        self._sync_slice(self.viewer_b, value)

    def _sync_slice_from_b(self, value: int):
        self._sync_slice(self.viewer_a, value)

    def _sync_slice(self, target: ImageViewer, value: int):
        if self._sync_guard or not self.zoom_sync_btn.isChecked():
            return

        self._sync_guard = True
        try:
            if target.slice_slider.value() != value:
                target.slice_slider.setValue(value)
        finally:
            self._sync_guard = False

    def _on_zoom_sync_toggled(self, checked: bool):
        self.zoom_sync_btn.setText("Zoom Sync: ON" if checked else "Zoom Sync: OFF")

    def get_winner_params(self) -> Optional[Dict[str, Any]]:
        if self.winner_candidate is None:
            return None
        return dict(self.winner_candidate.params)

    def get_audit_log(self) -> List[Dict[str, Any]]:
        return self.tournament.get_audit_log()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1000)
        super().closeEvent(event)
