"""
A/B Tournament panel GUI for segmentation parameter optimisation.

Presents two candidate segmentations side-by-side. The user picks the
better one; a bracket tournament converges on optimal Cellpose parameters.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.ab_tournament import ABTournament, ParameterCandidate
from gui.accessibility import apply_accessible_names, hex_to_rgb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background worker for running segmentation candidates
# ---------------------------------------------------------------------------

class _CandidateWorker(QThread):
    """Run Cellpose for a single :class:`ParameterCandidate` off the main thread."""

    finished = Signal(object)  # ParameterCandidate (with masks filled)
    error = Signal(str)

    def __init__(
        self,
        candidate: ParameterCandidate,
        image: np.ndarray,
        channels: List[int],
        gpu_available: bool = False,
    ) -> None:
        super().__init__()
        self.candidate = candidate
        self.image = image
        self.channels = channels
        self.gpu_available = gpu_available

    def run(self) -> None:  # noqa: D102
        try:
            from core.segmentation import SegmentationEngine
            from core.ab_tournament import _compute_mask_stats

            engine = SegmentationEngine(gpu_available=self.gpu_available)
            params = self.candidate.params

            masks, _ = engine.segment_cellpose(
                image=self.image,
                model_name=params.get("model_name", "nuclei"),
                diameter=params.get("diameter"),
                flow_threshold=params.get("flow_threshold", 0.4),
                cellprob_threshold=params.get("cellprob_threshold", 0.0),
                do_3d=False,
                channels=self.channels or [0, 0],
            )

            self.candidate.masks = masks
            count, median_area, area_cv = _compute_mask_stats(masks)
            self.candidate.nucleus_count = count
            self.candidate.median_area = median_area
            self.candidate.area_cv = area_cv

            self.finished.emit(self.candidate)

        except Exception as exc:  # noqa: BLE001
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Minimal candidate statistics widget
# ---------------------------------------------------------------------------

class _CandidateStatsWidget(QWidget):
    """Display nucleus statistics for a single candidate."""

    def __init__(self, label: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._label = label

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._title = QLabel(f"<b>{label}</b>")
        self._title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._title)

        self._count_lbl = QLabel("Nuclei: —")
        self._area_lbl = QLabel("Median area: —")
        self._cv_lbl = QLabel("Area CV: —")

        for lbl in (self._count_lbl, self._area_lbl, self._cv_lbl):
            lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl)

    def update_candidate(self, c: Optional[ParameterCandidate]) -> None:
        """Refresh statistics from *c*."""
        if c is None:
            self._count_lbl.setText("Nuclei: —")
            self._area_lbl.setText("Median area: —")
            self._cv_lbl.setText("Area CV: —")
            return

        self._count_lbl.setText(f"Nuclei: {c.nucleus_count}")
        self._area_lbl.setText(f"Median area: {c.median_area:.0f} px²")
        self._cv_lbl.setText(f"Area CV: {c.area_cv:.2f}")

        # Update accessible description for screen readers
        self.setAccessibleDescription(
            f"{self._label}: {c.nucleus_count} nuclei, "
            f"median area {c.median_area:.0f} px², "
            f"area CV {c.area_cv:.2f}."
        )


# ---------------------------------------------------------------------------
# Minimal image viewer for the panel (avoids importing the full viewer)
# ---------------------------------------------------------------------------

class _ThumbnailViewer(QLabel):
    """
    Very lightweight image display widget.

    Renders a numpy array as a QLabel pixmap, optionally with a coloured
    mask outline overlay.  The full :class:`~gui.image_viewer.ImageViewer`
    is not used here because the tournament may run before a project image
    is loaded into the main viewer.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setText("(no image)")
        self.setStyleSheet("border: 1px solid #555;")

    def set_image_with_masks(
        self,
        image: Optional[np.ndarray],
        masks: Optional[np.ndarray],
        outline_color: Tuple[int, int, int] = (0, 114, 178),
    ) -> None:
        """Render *image* with *masks* outlines in *outline_color* (RGB)."""
        try:
            from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
            import cv2

            if image is None:
                self.setText("(no image)")
                return

            # Normalise to uint8
            if image.dtype != np.uint8:
                img_f = image.astype(float)
                lo, hi = img_f.min(), img_f.max()
                if hi > lo:
                    img_f = (img_f - lo) / (hi - lo) * 255
                display = img_f.astype(np.uint8)
            else:
                display = image.copy()

            # Convert to RGB
            if display.ndim == 2:
                rgb = np.stack([display] * 3, axis=-1)
            elif display.ndim == 3 and display.shape[2] >= 3:
                rgb = display[:, :, :3]
            elif display.ndim == 3 and display.shape[0] <= 4:
                # (C, H, W) format — take first channel
                rgb = np.stack([display[0]] * 3, axis=-1)
            else:
                rgb = display[:, :, :3]

            # Draw mask outlines
            if masks is not None and masks.max() > 0:
                rgb = rgb.copy()
                for lbl in np.unique(masks):
                    if lbl == 0:
                        continue
                    nucleus_mask = (masks == lbl).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(rgb, contours, -1, outline_color, 1)

            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            # Scale to widget size keeping aspect ratio
            self.setPixmap(
                pixmap.scaled(
                    self.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not render candidate image: %s", exc)
            self.setText("(render error)")


# ---------------------------------------------------------------------------
# Main tournament panel dialog
# ---------------------------------------------------------------------------

class ABTournamentPanel(QDialog):
    """
    Dialog for the A/B tournament segmentation parameter optimisation.

    Signals
    -------
    tournament_complete(dict):
        Emitted with the winning parameter dict when the tournament finishes.
    """

    tournament_complete = Signal(dict)  # winning params

    def __init__(
        self,
        image: np.ndarray,
        channels: Optional[List[int]] = None,
        n_candidates: int = 8,
        gpu_available: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.image = image
        self.channels = channels or [0, 0]
        self.n_candidates = n_candidates
        self.gpu_available = gpu_available

        self._tournament: Optional[ABTournament] = None
        self._candidate_a: Optional[ParameterCandidate] = None
        self._candidate_b: Optional[ParameterCandidate] = None
        self._workers: List[_CandidateWorker] = []
        self._pending_workers: int = 0

        self.setWindowTitle("A/B Tournament — Segmentation Parameter Optimiser")
        self.setMinimumSize(900, 620)

        self._setup_ui()
        self._setup_shortcuts()
        apply_accessible_names(self)

        # Start tournament
        self._start_tournament()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # Status bar at top
        self._round_label = QLabel("Starting tournament…")
        self._round_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self._round_label)

        # Side-by-side viewers
        splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self._viewer_a = _ThumbnailViewer()
        self._stats_a = _CandidateStatsWidget("Candidate A")
        left_layout.addWidget(self._viewer_a)
        left_layout.addWidget(self._stats_a)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self._viewer_b = _ThumbnailViewer()
        self._stats_b = _CandidateStatsWidget("Candidate B")
        right_layout.addWidget(self._viewer_b)
        right_layout.addWidget(self._stats_b)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([450, 450])
        main_layout.addWidget(splitter, stretch=1)

        # Choice buttons
        btn_layout = QHBoxLayout()
        self._choose_a_btn = QPushButton("◀  Choose A  (A / ←)")
        self._choose_a_btn.setMinimumSize(180, 44)
        self._choose_a_btn.setObjectName("choose_a_button")
        self._choose_a_btn.clicked.connect(self._choose_a)

        self._regenerate_btn = QPushButton("Too Close — Regenerate Both")
        self._regenerate_btn.setMinimumSize(200, 44)
        self._regenerate_btn.setObjectName("regenerate_button")
        self._regenerate_btn.clicked.connect(self._regenerate)

        self._choose_b_btn = QPushButton("Choose B  (B / →)  ▶")
        self._choose_b_btn.setMinimumSize(180, 44)
        self._choose_b_btn.setObjectName("choose_b_button")
        self._choose_b_btn.clicked.connect(self._choose_b)

        btn_layout.addWidget(self._choose_a_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self._regenerate_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self._choose_b_btn)
        main_layout.addLayout(btn_layout)

        # Undo / progress
        bottom_layout = QHBoxLayout()
        self._undo_btn = QPushButton("↩ Undo Last Choice")
        self._undo_btn.setMinimumSize(160, 36)
        self._undo_btn.setObjectName("undo_button")
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self._undo)
        bottom_layout.addWidget(self._undo_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("matchup_progress")
        self._progress_bar.setMinimum(0)
        self._progress_bar.setValue(0)
        bottom_layout.addWidget(self._progress_bar, stretch=1)

        main_layout.addLayout(bottom_layout)

        # Cancel / Use winner buttons
        dialog_buttons = QHBoxLayout()
        self._cancel_btn = QPushButton("Cancel Tournament")
        self._cancel_btn.setObjectName("cancel_button")
        self._cancel_btn.clicked.connect(self.reject)

        self._use_winner_btn = QPushButton("Use Winner ▶")
        self._use_winner_btn.setObjectName("use_winner_button")
        self._use_winner_btn.setEnabled(False)
        self._use_winner_btn.clicked.connect(self._use_winner)

        dialog_buttons.addWidget(self._cancel_btn)
        dialog_buttons.addStretch()
        dialog_buttons.addWidget(self._use_winner_btn)
        main_layout.addLayout(dialog_buttons)

        self._set_choice_buttons_enabled(False)

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_A), self, self._choose_a)
        QShortcut(QKeySequence(Qt.Key_Left), self, self._choose_a)
        QShortcut(QKeySequence(Qt.Key_B), self, self._choose_b)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._choose_b)
        QShortcut(QKeySequence(Qt.Key_R), self, self._regenerate)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.reject)

    # ------------------------------------------------------------------
    # Tournament control
    # ------------------------------------------------------------------

    def _start_tournament(self) -> None:
        self._tournament = ABTournament(
            image=self.image,
            channels=self.channels,
            n_candidates=self.n_candidates,
        )
        candidates = self._tournament.generate_initial_candidates()
        self._progress_bar.setMaximum(self._tournament.total_matchups())
        self._round_label.setText("Running initial segmentations, please wait…")
        self._run_candidates(candidates)

    def _run_candidates(self, candidates: List[ParameterCandidate]) -> None:
        """Dispatch segmentation workers for each candidate."""
        self._pending_workers = len(candidates)
        self._set_choice_buttons_enabled(False)

        for c in candidates:
            # Skip if masks already computed
            if c.masks is not None:
                self._pending_workers -= 1
                continue
            worker = _CandidateWorker(
                candidate=c,
                image=self.image,
                channels=self.channels,
                gpu_available=self.gpu_available,
            )
            worker.finished.connect(self._on_candidate_finished)
            worker.error.connect(self._on_candidate_error)
            self._workers.append(worker)
            worker.start()

        if self._pending_workers <= 0:
            self._present_next_matchup()

    def _on_candidate_finished(self, candidate: ParameterCandidate) -> None:
        self._pending_workers = max(0, self._pending_workers - 1)
        if self._pending_workers == 0:
            self._present_next_matchup()

    def _on_candidate_error(self, message: str) -> None:
        logger.error("Candidate segmentation error: %s", message)
        self._pending_workers = max(0, self._pending_workers - 1)
        if self._pending_workers == 0:
            self._present_next_matchup()

    def _present_next_matchup(self) -> None:
        if self._tournament is None:
            return

        matchup = self._tournament.get_next_matchup()
        if matchup is None:
            self._on_tournament_complete()
            return

        self._candidate_a, self._candidate_b = matchup
        completed = self._tournament.completed_matchups()
        total = self._tournament.total_matchups()
        round_num = self._tournament._current_round + 1

        self._round_label.setText(
            f"Round {round_num} — Matchup {completed + 1} of {total}"
        )
        self._progress_bar.setValue(completed)

        # Show images
        from gui.accessibility import AccessibilityManager
        am = AccessibilityManager.instance()
        color_a_hex = am.color("overlay_a")
        color_b_hex = am.color("overlay_b")

        self._viewer_a.set_image_with_masks(
            self.image, self._candidate_a.masks, hex_to_rgb(color_a_hex)
        )
        self._viewer_b.set_image_with_masks(
            self.image, self._candidate_b.masks, hex_to_rgb(color_b_hex)
        )
        self._stats_a.update_candidate(self._candidate_a)
        self._stats_b.update_candidate(self._candidate_b)

        # Accessible announcement
        self.setAccessibleDescription(
            f"Round {round_num}, "
            f"Candidate A: {self._candidate_a.nucleus_count} nuclei, "
            f"Candidate B: {self._candidate_b.nucleus_count} nuclei. "
            "Press A to choose A, B to choose B."
        )

        self._set_choice_buttons_enabled(True)
        self._undo_btn.setEnabled(completed > 0)

    def _choose_a(self) -> None:
        if self._candidate_a and self._candidate_b and self._tournament:
            self._tournament.record_choice(
                winner_id=self._candidate_a.id, loser_id=self._candidate_b.id
            )
            self._after_choice()

    def _choose_b(self) -> None:
        if self._candidate_a and self._candidate_b and self._tournament:
            self._tournament.record_choice(
                winner_id=self._candidate_b.id, loser_id=self._candidate_a.id
            )
            self._after_choice()

    def _after_choice(self) -> None:
        """After recording a choice, check for new candidates (offspring) then present next."""
        if self._tournament is None:
            return

        # Check if any candidates lack masks (e.g. offspring just generated)
        pending = [
            c
            for c in self._tournament.get_all_candidates()
            if c.masks is None
        ]
        if pending:
            self._run_candidates(pending)
        else:
            self._present_next_matchup()

    def _regenerate(self) -> None:
        """Replace the current A and B with fresh LHS candidates."""
        if self._tournament is None:
            return
        # Use undo if available to regenerate the current matchup
        result = self._tournament.undo_last_choice()
        if result is not None:
            self._candidate_a, self._candidate_b = result
            self._present_next_matchup()
        else:
            # Just re-present the current matchup
            self._present_next_matchup()

    def _undo(self) -> None:
        if self._tournament:
            result = self._tournament.undo_last_choice()
            if result:
                self._candidate_a, self._candidate_b = result
                # Re-render
                self._present_next_matchup()

    def _on_tournament_complete(self) -> None:
        winner = self._tournament.get_winner() if self._tournament else None
        self._set_choice_buttons_enabled(False)
        self._undo_btn.setEnabled(False)

        if winner:
            self._round_label.setText(
                f"Tournament complete! Winner: {winner.nucleus_count} nuclei detected."
            )
            self._use_winner_btn.setEnabled(True)
            # Display winner
            from gui.accessibility import AccessibilityManager
            am = AccessibilityManager.instance()
            color_hex = am.color("overlay_a")
            self._viewer_a.set_image_with_masks(
                self.image, winner.masks, hex_to_rgb(color_hex)
            )
            self._stats_a.update_candidate(winner)
            self._viewer_b.setText("(tournament complete)")
        else:
            self._round_label.setText("Tournament complete — no winner determined.")

    def _use_winner(self) -> None:
        if self._tournament:
            winner = self._tournament.get_winner()
            if winner:
                self.tournament_complete.emit(winner.params)
        self.accept()

    def _set_choice_buttons_enabled(self, enabled: bool) -> None:
        self._choose_a_btn.setEnabled(enabled)
        self._choose_b_btn.setEnabled(enabled)
        self._regenerate_btn.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_audit_log(self) -> List[dict]:
        """Return the full tournament audit log."""
        if self._tournament:
            return self._tournament.get_audit_log()
        return []

    def get_winner_params(self) -> Optional[Dict]:
        """Return the winning parameter dict, or ``None``."""
        if self._tournament:
            winner = self._tournament.get_winner()
            if winner:
                return winner.params
        return None
