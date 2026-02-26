"""
Accessibility helpers and global accessibility state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QSettings
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QWidget


class CVDPalette(str, Enum):
    """Supported color palettes for color-vision-deficiency-safe rendering."""

    DEFAULT = "default"
    OKABE_ITO = "okabe_ito"
    IBM = "ibm"


PALETTES: Dict[str, Dict[str, object]] = {
    CVDPalette.DEFAULT.value: {
        "candidate_a": "#00AA00",
        "candidate_b": "#CC00CC",
        "qc_pass": "#2E7D32",
        "qc_warn": "#E69F00",
        "qc_fail": "#C62828",
        "channels": [
            "#FF0000",
            "#00FF00",
            "#0000FF",
            "#FFFF00",
            "#FF00FF",
            "#00FFFF",
            "#FFFFFF",
        ],
    },
    CVDPalette.OKABE_ITO.value: {
        "candidate_a": "#0072B2",
        "candidate_b": "#E69F00",
        "qc_pass": "#009E73",
        "qc_warn": "#E69F00",
        "qc_fail": "#D55E00",
        "channels": [
            "#0072B2",
            "#E69F00",
            "#009E73",
            "#56B4E9",
            "#D55E00",
            "#CC79A7",
            "#F0E442",
        ],
    },
    CVDPalette.IBM.value: {
        "candidate_a": "#648FFF",
        "candidate_b": "#FFB000",
        "qc_pass": "#009E73",
        "qc_warn": "#FFB000",
        "qc_fail": "#DC267F",
        "channels": [
            "#648FFF",
            "#785EF0",
            "#DC267F",
            "#FE6100",
            "#FFB000",
            "#009E73",
            "#56B4E9",
        ],
    },
}


@dataclass
class AccessibilityState:
    """Mutable global accessibility preferences."""

    colorblind_safe: bool = False
    palette: str = CVDPalette.OKABE_ITO.value
    scale_factor: float = 1.0
    screen_reader_hints: bool = True
    sticky_keys: bool = False
    double_click_ms: int = 400
    drag_sensitivity: int = 10


class AccessibilityManager:
    """Central accessibility manager for GUI-wide behavior."""

    _state: AccessibilityState = AccessibilityState()
    _base_point_size: Optional[float] = None

    @classmethod
    def load_from_settings(cls):
        settings = QSettings("NucleiSegApp", "EZQUANT")
        cls._state = AccessibilityState(
            colorblind_safe=settings.value("accessibility/colorblind_safe", False, type=bool),
            palette=settings.value("accessibility/palette", CVDPalette.OKABE_ITO.value),
            scale_factor=float(settings.value("accessibility/scale_factor", 1.0, type=float)),
            screen_reader_hints=settings.value("accessibility/screen_reader_hints", True, type=bool),
            sticky_keys=settings.value("accessibility/sticky_keys", False, type=bool),
            double_click_ms=settings.value("accessibility/double_click_ms", 400, type=int),
            drag_sensitivity=settings.value("accessibility/drag_sensitivity", 10, type=int),
        )

        if cls._state.palette not in PALETTES:
            cls._state.palette = CVDPalette.OKABE_ITO.value

    @classmethod
    def save_to_settings(cls):
        settings = QSettings("NucleiSegApp", "EZQUANT")
        settings.setValue("accessibility/colorblind_safe", cls._state.colorblind_safe)
        settings.setValue("accessibility/palette", cls._state.palette)
        settings.setValue("accessibility/scale_factor", cls._state.scale_factor)
        settings.setValue("accessibility/screen_reader_hints", cls._state.screen_reader_hints)
        settings.setValue("accessibility/sticky_keys", cls._state.sticky_keys)
        settings.setValue("accessibility/double_click_ms", cls._state.double_click_ms)
        settings.setValue("accessibility/drag_sensitivity", cls._state.drag_sensitivity)
        settings.sync()

    @classmethod
    def state(cls) -> AccessibilityState:
        return AccessibilityState(**cls._state.__dict__)

    @classmethod
    def update_state(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls._state, key):
                setattr(cls._state, key, value)

    @classmethod
    def palette(cls) -> Dict[str, object]:
        if cls._state.colorblind_safe:
            return PALETTES.get(cls._state.palette, PALETTES[CVDPalette.OKABE_ITO.value])
        return PALETTES[CVDPalette.DEFAULT.value]

    @classmethod
    def get_channel_colors(cls, n_channels: int) -> List[str]:
        palette = cls.palette()
        colors = list(palette.get("channels", []))
        if not colors:
            colors = list(PALETTES[CVDPalette.DEFAULT.value]["channels"])

        result = []
        for i in range(n_channels):
            result.append(colors[i % len(colors)])
        return result

    @classmethod
    def apply_ui_scale(cls, app: Optional[QApplication] = None):
        app = app or QApplication.instance()
        if app is None:
            return

        if cls._base_point_size is None:
            font = app.font()
            cls._base_point_size = font.pointSizeF() if font.pointSizeF() > 0 else 10.0

        scaled_size = max(8.0, cls._base_point_size * max(1.0, cls._state.scale_factor))
        font: QFont = app.font()
        font.setPointSizeF(scaled_size)
        app.setFont(font)

        control_size = max(32, int(round(44 * cls._state.scale_factor)))
        spacing = max(4, int(round(6 * cls._state.scale_factor)))

        app.setStyleSheet(
            "\n".join(
                [
                    f"QPushButton, QToolButton, QCheckBox, QComboBox {{ min-height: {control_size}px; }}",
                    f"QSpinBox, QDoubleSpinBox, QLineEdit {{ min-height: {control_size}px; }}",
                    f"QWidget {{ spacing: {spacing}px; }}",
                ]
            )
        )

        app.setDoubleClickInterval(max(200, int(cls._state.double_click_ms)))

    @classmethod
    def apply_accessible_names(cls, widget_tree: QObject):
        """Recursively apply fallback accessible names/descriptions."""
        if widget_tree is None:
            return

        for obj in widget_tree.findChildren(QObject):
            if not isinstance(obj, QWidget):
                continue

            if not obj.accessibleName():
                fallback_name = cls._guess_accessible_name(obj)
                if fallback_name:
                    obj.setAccessibleName(fallback_name)

            if not obj.accessibleDescription():
                tooltip = obj.toolTip().strip() if hasattr(obj, "toolTip") else ""
                if tooltip:
                    obj.setAccessibleDescription(tooltip)

    @staticmethod
    def _guess_accessible_name(widget: QWidget) -> str:
        object_name = widget.objectName().strip() if widget.objectName() else ""
        if object_name:
            return object_name.replace("_", " ").strip()

        for attr in ("text", "title", "placeholderText"):
            if hasattr(widget, attr):
                candidate = getattr(widget, attr)
                if callable(candidate):
                    try:
                        value = str(candidate()).strip()
                    except TypeError:
                        continue
                else:
                    value = str(candidate).strip()

                if value:
                    return value

        tooltip = widget.toolTip().strip() if hasattr(widget, "toolTip") else ""
        return tooltip

    @classmethod
    def build_segmentation_description(
        cls,
        nucleus_count: int,
        channel_name: str,
        zoom_percent: float,
        slice_index: int,
    ) -> str:
        return (
            "Segmentation view: "
            f"{nucleus_count} nuclei detected, "
            f"showing channel {channel_name}, "
            f"zoom {zoom_percent:.0f}%, "
            f"slice {slice_index + 1}."
        )


def apply_accessible_names(widget_tree: QObject):
    """Module-level helper for compatibility with direct function usage."""
    AccessibilityManager.apply_accessible_names(widget_tree)
