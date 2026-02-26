"""
Accessibility utilities for EZQUANT.

Provides:
- ``AccessibilityManager`` — global state for CVD mode, scale, screen reader hints.
- ``apply_accessible_names()`` — walk a widget tree and set accessible names.
- ``CVDPalette`` — colorblind-safe colour sets (Okabe-Ito palette).
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple

try:
    from PySide6.QtWidgets import QApplication, QWidget, QAbstractButton, QLabel
    from PySide6.QtGui import QFont
    _QT_AVAILABLE = True
except ImportError:  # pragma: no cover - headless/test environments
    _QT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

class CVDPalette(str, Enum):
    """
    Colorblind-safe colour palette identifiers.

    Values are human-readable labels; use :data:`PALETTE_COLORS` to get
    the actual RGB hex strings.
    """

    DEFAULT = "default"          # Standard colours (may be inaccessible)
    OKABE_ITO = "okabe_ito"      # Okabe-Ito (CVD-safe, 8 colours)
    IBM = "ibm"                  # IBM Design Language CVD-safe palette


# Okabe-Ito palette (universally CVD-safe; Okabe & Ito 2008)
# https://jfly.uni-koeln.de/color/
_OKABE_ITO = {
    "black":        "#000000",
    "orange":       "#E69F00",
    "sky_blue":     "#56B4E9",
    "green":        "#009E73",
    "yellow":       "#F0E442",
    "blue":         "#0072B2",
    "vermillion":   "#D55E00",
    "pink":         "#CC79A7",
}

# IBM Design Language CVD-safe palette
_IBM = {
    "blue":         "#648FFF",
    "violet":       "#785EF0",
    "magenta":      "#DC267F",
    "orange":       "#FE6100",
    "yellow":       "#FFB000",
}

# Default (non-CVD) palette used when accessibility mode is off
_DEFAULT = {
    "green":  "#00FF00",
    "red":    "#FF0000",
    "blue":   "#0000FF",
    "yellow": "#FFFF00",
    "white":  "#FFFFFF",
}

PALETTE_COLORS: Dict[CVDPalette, Dict[str, str]] = {
    CVDPalette.DEFAULT:   _DEFAULT,
    CVDPalette.OKABE_ITO: _OKABE_ITO,
    CVDPalette.IBM:       _IBM,
}

# Semantic colour roles for the application
# Each key maps to a (default_hex, okabe_ito_hex) tuple
SEMANTIC_COLORS: Dict[str, Tuple[str, str]] = {
    # A/B tournament overlays
    "overlay_a":      ("#00FF00", "#0072B2"),   # green → blue
    "overlay_b":      ("#FF00FF", "#E69F00"),   # magenta → orange
    # QC indicators
    "qc_pass":        ("#00CC00", "#009E73"),   # green
    "qc_warn":        ("#FFAA00", "#E69F00"),   # orange
    "qc_fail":        ("#CC0000", "#D55E00"),   # red → vermillion
    # Segmentation nucleus outline
    "nucleus_outline": ("#00FF00", "#56B4E9"),  # green → sky blue
}


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert a hex colour string (``#RRGGBB``) to an ``(R, G, B)`` int tuple."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def get_semantic_color(role: str, cvd_mode: bool = False) -> str:
    """
    Return the hex colour for a semantic *role* name.

    Parameters
    ----------
    role:
        One of the keys in :data:`SEMANTIC_COLORS` (e.g. ``"qc_pass"``).
    cvd_mode:
        If ``True``, return the CVD-safe colour.
    """
    colors = SEMANTIC_COLORS.get(role, ("#FFFFFF", "#FFFFFF"))
    return colors[1] if cvd_mode else colors[0]


# ---------------------------------------------------------------------------
# AccessibilityManager
# ---------------------------------------------------------------------------

class AccessibilityManager:
    """
    Singleton-style holder for global accessibility state.

    Attributes
    ----------
    cvd_mode:
        If ``True``, all overlay colours use the CVD-safe palette.
    scale_factor:
        UI scale multiplier (1.0 = default).
    screen_reader_hints:
        If ``True``, accessible descriptions are set on all widgets.
    palette:
        The active :class:`CVDPalette`.
    """

    _instance: Optional["AccessibilityManager"] = None

    def __init__(self) -> None:
        self.cvd_mode: bool = False
        self.scale_factor: float = 1.0
        self.screen_reader_hints: bool = True
        self.palette: CVDPalette = CVDPalette.DEFAULT

    @classmethod
    def instance(cls) -> "AccessibilityManager":
        """Return the global singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_cvd_mode(self, enabled: bool, palette: CVDPalette = CVDPalette.OKABE_ITO) -> None:
        """Enable or disable CVD-safe colour mode."""
        self.cvd_mode = enabled
        self.palette = palette if enabled else CVDPalette.DEFAULT

    def set_scale_factor(self, factor: float) -> None:
        """
        Set the UI scale factor (1.0–2.0) and apply it to the QApplication
        font and stylesheet.
        """
        self.scale_factor = max(1.0, min(2.0, factor))
        if not _QT_AVAILABLE:
            return
        app = QApplication.instance()
        if app is None:
            return
        base_pt = 9
        new_pt = int(base_pt * self.scale_factor)
        font = QFont()
        font.setPointSize(new_pt)
        app.setFont(font)

    def color(self, role: str) -> str:
        """Return the current colour (hex string) for semantic *role*."""
        return get_semantic_color(role, cvd_mode=self.cvd_mode)

    def apply_to_application(self) -> None:
        """Apply current accessibility settings to the running QApplication."""
        if not _QT_AVAILABLE:
            return
        self.set_scale_factor(self.scale_factor)


# ---------------------------------------------------------------------------
# Widget tree walker
# ---------------------------------------------------------------------------

def apply_accessible_names(root: "QWidget") -> None:  # type: ignore[name-defined]
    """
    Recursively walk *root* and its children, setting a reasonable
    ``accessibleName`` on each interactive widget that does not already
    have one.

    The fallback priority is:
    1. Existing ``accessibleName`` (kept unchanged)
    2. Widget ``objectName``
    3. ``text()`` for buttons / labels
    4. ``toolTip()``
    5. Widget class name
    """
    if not _QT_AVAILABLE:
        return

    def _set(widget: "QWidget") -> None:  # type: ignore[name-defined]
        if widget.accessibleName():
            return  # Already set — do not overwrite

        name = ""

        # Try objectName
        obj_name = widget.objectName()
        if obj_name and not obj_name.startswith("qt_"):
            name = obj_name.replace("_", " ").strip()

        # Try .text() for button-like widgets
        if not name and isinstance(widget, (QAbstractButton, QLabel)):
            try:
                name = widget.text().strip()
            except AttributeError:
                pass

        # Try toolTip
        if not name:
            tip = widget.toolTip()
            if tip:
                # Strip rich-text tags
                import re
                name = re.sub(r"<[^>]+>", "", tip).strip()

        # Fallback to class name
        if not name:
            name = type(widget).__name__

        if name:
            widget.setAccessibleName(name)

    def _walk(widget: "QWidget") -> None:  # type: ignore[name-defined]
        _set(widget)
        for child in widget.children():
            if isinstance(child, QWidget):
                _walk(child)

    _walk(root)
