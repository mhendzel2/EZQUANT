"""
Accessibility support for EZQUANT.

Provides:
- AccessibilityManager: Global accessibility state management
- CVDPalette: Colorblind-safe color palettes
- apply_accessible_names(): Recursive accessible name setter
"""
from __future__ import annotations

from enum import Enum
from typing import Any


class CVDPalette(Enum):
    """Colorblind-safe color palettes."""

    # Okabe-Ito palette (safe for all common CVD types)
    OKABE_ITO = {
        "black": "#000000",
        "orange": "#E69F00",
        "sky_blue": "#56B4E9",
        "bluish_green": "#009E73",
        "yellow": "#F0E442",
        "blue": "#0072B2",
        "vermillion": "#D55E00",
        "reddish_purple": "#CC79A7",
    }

    # IBM colorblind-safe palette
    IBM = {
        "blue": "#648FFF",
        "purple": "#785EF0",
        "pink": "#DC267F",
        "orange": "#FE6100",
        "yellow": "#FFB000",
    }


# Default colors for A/B tournament (distinguishable under all CVD types)
AB_TOURNAMENT_COLORS = {
    "candidate_a": "#0072B2",  # Blue (Okabe-Ito)
    "candidate_b": "#E69F00",  # Orange (Okabe-Ito)
}

# QC status colors with redundant encoding (not red/green only)
QC_STATUS_COLORS = {
    "PASS": "#009E73",    # Bluish green (Okabe-Ito)
    "WARN": "#E69F00",    # Orange (Okabe-Ito)
    "FAIL": "#D55E00",    # Vermillion (Okabe-Ito)
}

QC_STATUS_ICONS = {
    "PASS": "✓",
    "WARN": "⚠",
    "FAIL": "✗",
}


class AccessibilityManager:
    """Global accessibility state for EZQUANT."""

    _instance: AccessibilityManager | None = None

    def __init__(self):
        self.cvd_mode: bool = False
        self.scale_factor: float = 1.0
        self.screen_reader_hints: bool = True
        self._palette = CVDPalette.OKABE_ITO

    @classmethod
    def instance(cls) -> AccessibilityManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_cvd_mode(self, enabled: bool) -> None:
        self.cvd_mode = enabled

    def set_scale_factor(self, factor: float) -> None:
        self.scale_factor = max(1.0, min(2.0, factor))

    def get_overlay_colors(self) -> dict[str, str]:
        """Return overlay colors, CVD-safe if cvd_mode is on."""
        return AB_TOURNAMENT_COLORS

    def get_qc_color(self, status: str) -> str:
        return QC_STATUS_COLORS.get(status, "#000000")

    def get_qc_icon(self, status: str) -> str:
        return QC_STATUS_ICONS.get(status, "?")

    def apply_scale(self, app: Any) -> None:
        """Apply font scaling to a QApplication."""
        try:
            from PySide6.QtGui import QFont
            font = app.font()
            base_size = 10
            font.setPointSizeF(base_size * self.scale_factor)
            app.setFont(font)
        except ImportError:
            pass


def apply_accessible_names(widget: Any, prefix: str = "") -> None:
    """
    Recursively walk a widget tree and set accessible names
    from objectName, toolTip, or text properties as fallback.
    """
    try:
        from PySide6.QtWidgets import QWidget
    except ImportError:
        return

    if not isinstance(widget, QWidget):
        return

    # Set accessible name if not already set
    if not widget.accessibleName():
        name = (
            widget.objectName()
            or getattr(widget, "toolTip", lambda: "")()
            or getattr(widget, "text", lambda: "")()
            or type(widget).__name__
        )
        if name:
            widget.setAccessibleName(name)

    # Recurse into children
    for child in widget.children():
        if isinstance(child, QWidget):
            apply_accessible_names(child, prefix)
