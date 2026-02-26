"""
A/B Tournament Panel for segmentation parameter optimization.

Layout:
  Round N of M — Matchup X of Y
  [Candidate A viewer]  [Candidate B viewer]
  [Choose A]  [Zoom Sync]  [Choose B]
  [Too Close — Regenerate Both]
  Progress bar
  [Cancel]  [Use Winner]
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

try:
    from PySide6.QtCore import Qt, Signal, QTimer
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QProgressBar, QGroupBox, QSizePolicy, QMessageBox, QApplication,
    )
    from PySide6.QtGui import QKeySequence, QShortcut
    _PYSIDE6_AVAILABLE = True
except ImportError:
    _PYSIDE6_AVAILABLE = False

from core.ab_tournament import ABTournament, ParameterCandidate


if _PYSIDE6_AVAILABLE:
    class CandidateView(QGroupBox):
        """Displays one candidate's image and statistics."""

        def __init__(self, label: str, parent: QWidget | None = None):
            super().__init__(label, parent)
            self.setMinimumSize(300, 400)
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)

            self.image_label = QLabel("No image")
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label.setMinimumHeight(300)
            self.image_label.setStyleSheet("background-color: #1a1a1a; color: #fff;")
            layout.addWidget(self.image_label)

            self.stats_label = QLabel("—")
            self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.stats_label)

        def update_candidate(self, candidate: ParameterCandidate) -> None:
            count = candidate.nucleus_count
            area = candidate.median_area
            cv = candidate.area_cv
            self.stats_label.setText(
                f"Nuclei: {count}  |  Median area: {area:.0f} px²  |  Area CV: {cv:.1%}"
            )
            # In a full implementation, render mask outlines onto image here
            params = candidate.params
            p_text = (
                f"diameter={params.get('diameter', '?'):.1f}  "
                f"flow={params.get('flow_threshold', '?'):.2f}  "
                f"prob={params.get('cellprob_threshold', '?'):.1f}  "
                f"model={params.get('model_name', '?')}"
            )
            self.image_label.setText(f"[Segmentation preview]\n{p_text}")
            # Update accessible description
            self.setAccessibleDescription(
                f"{self.title()}: {count} nuclei detected, "
                f"median area {area:.0f} px², area CV {cv:.1%}"
            )

        def set_loading(self, loading: bool) -> None:
            self.image_label.setText("Computing segmentation…" if loading else "Ready")

    class ABTournamentPanel(QWidget):
        """
        Side-by-side A/B tournament panel for parameter optimization.

        Signals:
            tournament_complete(dict): Emitted with winning params when done.
            tournament_cancelled(): Emitted when user cancels.
        """

        tournament_complete = Signal(dict)
        tournament_cancelled = Signal()

        def __init__(
            self,
            tournament: ABTournament,
            parent: QWidget | None = None,
        ):
            super().__init__(parent)
            self.tournament = tournament
            self._candidate_a: ParameterCandidate | None = None
            self._candidate_b: ParameterCandidate | None = None
            self._setup_ui()
            self._setup_shortcuts()
            self._load_next_matchup()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)

            # Header
            self.header_label = QLabel("Starting tournament…")
            self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            font = self.header_label.font()
            font.setPointSize(12)
            self.header_label.setFont(font)
            layout.addWidget(self.header_label)

            # Side-by-side views
            views_layout = QHBoxLayout()
            self.view_a = CandidateView("Candidate A")
            self.view_b = CandidateView("Candidate B")
            views_layout.addWidget(self.view_a)
            views_layout.addWidget(self.view_b)
            layout.addLayout(views_layout)

            # Choice buttons
            btn_layout = QHBoxLayout()
            self.btn_choose_a = QPushButton("◀ Choose A  [A / ←]")
            self.btn_choose_a.setMinimumHeight(44)
            self.btn_choose_a.setMinimumWidth(160)
            self.btn_choose_a.setAccessibleName("Choose Candidate A")
            self.btn_choose_a.setAccessibleDescription(
                "Select Candidate A as the better segmentation. Keyboard shortcut: A or Left Arrow."
            )
            self.btn_choose_a.clicked.connect(self._choose_a)

            self.btn_zoom_sync = QPushButton("Zoom Sync")
            self.btn_zoom_sync.setMinimumHeight(44)
            self.btn_zoom_sync.setCheckable(True)
            self.btn_zoom_sync.setChecked(True)
            self.btn_zoom_sync.setAccessibleName("Toggle zoom synchronization")

            self.btn_choose_b = QPushButton("Choose B  [B / →] ▶")
            self.btn_choose_b.setMinimumHeight(44)
            self.btn_choose_b.setMinimumWidth(160)
            self.btn_choose_b.setAccessibleName("Choose Candidate B")
            self.btn_choose_b.setAccessibleDescription(
                "Select Candidate B as the better segmentation. Keyboard shortcut: B or Right Arrow."
            )
            self.btn_choose_b.clicked.connect(self._choose_b)

            btn_layout.addWidget(self.btn_choose_a)
            btn_layout.addStretch()
            btn_layout.addWidget(self.btn_zoom_sync)
            btn_layout.addStretch()
            btn_layout.addWidget(self.btn_choose_b)
            layout.addLayout(btn_layout)

            # Too close / regenerate
            regen_layout = QHBoxLayout()
            self.btn_regenerate = QPushButton("Too Close — Regenerate Both  [R]")
            self.btn_regenerate.setMinimumHeight(44)
            self.btn_regenerate.setAccessibleName("Regenerate both candidates")
            self.btn_regenerate.setAccessibleDescription(
                "Both candidates look equally good or bad. Generate two new candidates. Keyboard: R."
            )
            self.btn_regenerate.clicked.connect(self._regenerate)
            regen_layout.addStretch()
            regen_layout.addWidget(self.btn_regenerate)
            regen_layout.addStretch()
            layout.addLayout(regen_layout)

            # Undo button
            undo_layout = QHBoxLayout()
            self.btn_undo = QPushButton("↩ Undo Last Choice")
            self.btn_undo.setMinimumHeight(44)
            self.btn_undo.setAccessibleName("Undo last choice")
            self.btn_undo.setEnabled(False)
            self.btn_undo.clicked.connect(self._undo)
            undo_layout.addWidget(self.btn_undo)
            undo_layout.addStretch()
            layout.addLayout(undo_layout)

            # Progress bar
            self.progress_bar = QProgressBar()
            self.progress_bar.setMinimum(0)
            self.progress_bar.setAccessibleName("Tournament progress")
            layout.addWidget(self.progress_bar)

            self.progress_label = QLabel("0 / 0 comparisons")
            self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.progress_label)

            # Bottom buttons
            bottom_layout = QHBoxLayout()
            self.btn_cancel = QPushButton("Cancel Tournament")
            self.btn_cancel.setMinimumHeight(44)
            self.btn_cancel.setMinimumWidth(150)
            self.btn_cancel.setAccessibleName("Cancel tournament")
            self.btn_cancel.clicked.connect(self._cancel)

            self.btn_use_winner = QPushButton("Use Winner ▶")
            self.btn_use_winner.setMinimumHeight(44)
            self.btn_use_winner.setMinimumWidth(150)
            self.btn_use_winner.setEnabled(False)
            self.btn_use_winner.setAccessibleName("Apply winning parameters")
            self.btn_use_winner.setAccessibleDescription(
                "Apply the tournament winner's parameters to the segmentation panel."
            )
            self.btn_use_winner.clicked.connect(self._use_winner)

            bottom_layout.addWidget(self.btn_cancel)
            bottom_layout.addStretch()
            bottom_layout.addWidget(self.btn_use_winner)
            layout.addLayout(bottom_layout)

            self.setWindowTitle("A/B Parameter Tournament")
            self.setMinimumSize(700, 600)

        def _setup_shortcuts(self) -> None:
            QShortcut(QKeySequence("A"), self, self._choose_a)
            QShortcut(QKeySequence(Qt.Key.Key_Left), self, self._choose_a)
            QShortcut(QKeySequence("B"), self, self._choose_b)
            QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._choose_b)
            QShortcut(QKeySequence("R"), self, self._regenerate)
            QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self._cancel)
            QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)

        def _load_next_matchup(self) -> None:
            matchup = self.tournament.get_next_matchup()
            if matchup is None:
                self._on_tournament_complete()
                return

            self._candidate_a, self._candidate_b = matchup
            self.view_a.update_candidate(self._candidate_a)
            self.view_b.update_candidate(self._candidate_b)

            progress = self.tournament.get_progress()
            round_cur = progress["round_current"]
            rounds_total = progress["rounds_total"]
            matchup_cur = progress["matchup_current"]
            matchups_in_round = progress["matchups_in_round"]
            completed = progress["total_completed"]
            expected = progress["total_expected"]

            self.header_label.setText(
                f"Round {round_cur} of {rounds_total} — Matchup {matchup_cur} of {matchups_in_round}"
            )
            self.setAccessibleDescription(
                f"Round {round_cur} of {rounds_total}, "
                f"Candidate A: {self._candidate_a.nucleus_count} nuclei, "
                f"Candidate B: {self._candidate_b.nucleus_count} nuclei. "
                f"Press A to choose A, B to choose B."
            )

            self.progress_bar.setMaximum(max(expected, 1))
            self.progress_bar.setValue(completed)
            self.progress_label.setText(f"{completed} / {expected} comparisons")
            self.btn_undo.setEnabled(completed > 0)

        def _choose_a(self) -> None:
            if self._candidate_a and self._candidate_b:
                self.tournament.record_choice(self._candidate_a.id, self._candidate_b.id)
                self._load_next_matchup()

        def _choose_b(self) -> None:
            if self._candidate_a and self._candidate_b:
                self.tournament.record_choice(self._candidate_b.id, self._candidate_a.id)
                self._load_next_matchup()

        def _regenerate(self) -> None:
            self.tournament.regenerate_pair()
            self._load_next_matchup()

        def _undo(self) -> None:
            if self.tournament.undo_last_choice():
                self._load_next_matchup()

        def _cancel(self) -> None:
            self.tournament_cancelled.emit()

        def _on_tournament_complete(self) -> None:
            winner = self.tournament.get_winner()
            self.btn_use_winner.setEnabled(winner is not None)
            if winner:
                self.header_label.setText(
                    f"Tournament complete! Winner: {winner.nucleus_count} nuclei "
                    f"(diameter={winner.params.get('diameter', '?'):.1f})"
                )
            else:
                self.header_label.setText("Tournament complete!")

        def _use_winner(self) -> None:
            winner = self.tournament.get_winner()
            if winner:
                self.tournament_complete.emit(winner.params)

        else:
    # Stub for environments without PySide6
    class ABTournamentPanel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PySide6 is not available. Cannot create ABTournamentPanel. "
                "Install it with: pip install PySide6"
            )
