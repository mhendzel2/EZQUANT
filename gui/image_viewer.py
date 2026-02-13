"""
Image viewer widget with multi-dimensional support
Displays images with Z-slice navigation, channel controls, and mask overlays
"""

import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QSlider, QCheckBox, QComboBox, QPushButton,
                               QSpinBox, QGroupBox, QScrollArea)
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QPainter
import pyqtgraph as pg
from typing import Optional, List, Tuple, Dict


class ImageViewer(QWidget):
    """
    Multi-dimensional image viewer with channel controls and mask overlay
    """
    
    # Signals
    slice_changed = Signal(int)  # Z-slice index
    channel_changed = Signal(list)  # List of visible channel indices
    nucleus_selected = Signal(int)  # Nucleus ID clicked
    view_changed = Signal(QRectF)  # View rectangle changed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Image data
        self.image_data: Optional[np.ndarray] = None  # Shape: (Z, C, Y, X) or (C, Y, X)
        self.mask_data: Optional[np.ndarray] = None  # Shape: (Z, Y, X) or (Y, X)
        self.metadata: Dict = {}
        
        # Display state
        self.current_slice = 0
        self.visible_channels: List[int] = []
        self.channel_colors: List[Tuple[int, int, int]] = []
        self.channel_names: List[str] = []
        self.dna_channel_index: Optional[int] = None
        
        # Mask overlay state
        self.mask_visible = True
        self.mask_opacity = 0.5
        self.selected_nucleus_id: Optional[int] = None
        
        # Display settings
        self.auto_contrast = True
        self.contrast_min: Dict[int, float] = {}
        self.contrast_max: Dict[int, float] = {}

        # Caches to minimize expensive redraws
        self._cached_image_key = None
        self._cached_mask_key = None
        self._cached_mask_base: Optional[np.ndarray] = None
        self._mask_lut: Optional[np.ndarray] = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Left: Image display
        self.image_widget = pg.GraphicsLayoutWidget()
        self.view_box = self.image_widget.addViewBox()
        self.view_box.setAspectLocked(True)
        
        # Image item
        self.img_item = pg.ImageItem()
        self.view_box.addItem(self.img_item)
        
        # Mask overlay item
        self.mask_item = pg.ImageItem()
        self.mask_item.setOpacity(self.mask_opacity)
        self.view_box.addItem(self.mask_item)
        
        # Enable mouse interactions
        self.img_item.setCompositionMode(QPainter.CompositionMode_SourceOver)
        self.view_box.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        
        layout.addWidget(self.image_widget, stretch=4)
        
        # Right: Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setAlignment(Qt.AlignTop)
        
        # Z-slice control
        slice_group = QGroupBox("Z-Slice")
        slice_layout = QVBoxLayout(slice_group)
        
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        
        self.slice_label = QLabel("Slice: 1 / 1")
        slice_layout.addWidget(self.slice_label)
        slice_layout.addWidget(self.slice_slider)
        
        controls_layout.addWidget(slice_group)
        
        # Channel controls
        channel_group = QGroupBox("Channels")
        self.channel_layout = QVBoxLayout(channel_group)
        
        # DNA channel selector
        dna_layout = QHBoxLayout()
        dna_layout.addWidget(QLabel("DNA Channel:"))
        self.dna_channel_combo = QComboBox()
        self.dna_channel_combo.currentIndexChanged.connect(self._on_dna_channel_changed)
        dna_layout.addWidget(self.dna_channel_combo)
        self.channel_layout.addLayout(dna_layout)
        
        # Channel checkboxes will be added dynamically
        self.channel_checkboxes: List[QCheckBox] = []
        self.channel_contrast_widgets: List[Tuple[QSpinBox, QSpinBox]] = []
        
        controls_layout.addWidget(channel_group)
        
        # Mask overlay controls
        mask_group = QGroupBox("Mask Overlay")
        mask_layout = QVBoxLayout(mask_group)
        
        self.mask_visible_check = QCheckBox("Show Masks")
        self.mask_visible_check.setChecked(True)
        self.mask_visible_check.toggled.connect(self._on_mask_visibility_changed)
        mask_layout.addWidget(self.mask_visible_check)
        
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Opacity:"))
        self.mask_opacity_slider = QSlider(Qt.Horizontal)
        self.mask_opacity_slider.setMinimum(0)
        self.mask_opacity_slider.setMaximum(100)
        self.mask_opacity_slider.setValue(50)
        self.mask_opacity_slider.valueChanged.connect(self._on_mask_opacity_changed)
        opacity_layout.addWidget(self.mask_opacity_slider)
        mask_layout.addLayout(opacity_layout)
        
        controls_layout.addWidget(mask_group)
        
        # View controls
        view_group = QGroupBox("View")
        view_layout = QVBoxLayout(view_group)
        
        fit_button = QPushButton("Fit to Window")
        fit_button.clicked.connect(self.fit_to_window)
        view_layout.addWidget(fit_button)
        
        reset_button = QPushButton("Reset Zoom")
        reset_button.clicked.connect(self.reset_zoom)
        view_layout.addWidget(reset_button)
        
        controls_layout.addWidget(view_group)
        
        controls_layout.addStretch()
        
        # Make controls scrollable
        scroll_area = QScrollArea()
        scroll_area.setWidget(controls_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(250)
        scroll_area.setMaximumWidth(350)
        
        layout.addWidget(scroll_area, stretch=1)
    
    def set_image(self, image: np.ndarray, metadata: Dict):
        """
        Set the image to display
        
        Args:
            image: Image array with shape (Z, C, Y, X) or (C, Y, X)
            metadata: Dictionary with image metadata
        """
        self.image_data = image
        self.metadata = metadata
        
        # Determine if 3D
        is_3d = image.ndim == 4
        
        # Get dimensions
        if is_3d:
            n_slices, n_channels, height, width = image.shape
        else:
            n_channels, height, width = image.shape
            n_slices = 1
        
        # Set up channel names and colors
        self.channel_names = metadata.get('channel_names', [f'Channel {i+1}' for i in range(n_channels)])
        self.channel_colors = self._get_default_colors(n_channels)
        
        # Update slice control
        self.slice_slider.setMaximum(max(0, n_slices - 1))
        self.slice_slider.setEnabled(n_slices > 1)
        self.current_slice = 0
        self._update_slice_label()
        
        # Update channel controls
        self._setup_channel_controls()
        
        # Initialize all channels as visible
        self.visible_channels = list(range(n_channels))
        for checkbox in self.channel_checkboxes:
            checkbox.setChecked(True)
        
        # Calculate initial contrast settings
        self._calculate_auto_contrast()

        self.selected_nucleus_id = None
        self._invalidate_image_cache()
        self._invalidate_mask_cache(clear_lut=False)
        
        # Display first slice
        self._update_display()
        self.fit_to_window()
    
    def set_mask(self, mask: np.ndarray):
        """
        Set the segmentation mask overlay
        
        Args:
            mask: Labeled mask array with shape (Z, Y, X) or (Y, X)
        """
        self.mask_data = mask
        self.selected_nucleus_id = None
        self._invalidate_mask_cache(clear_lut=True)
        if self.mask_visible:
            self._update_mask_overlay()
    
    def clear_mask(self):
        """Clear the mask overlay"""
        self.mask_data = None
        self.selected_nucleus_id = None
        self._invalidate_mask_cache(clear_lut=False)
        self.mask_item.clear()
    
    def clear(self):
        """Clear all image data and reset the viewer"""
        self.image_data = None
        self.mask_data = None
        self.current_slice = 0
        self.channel_names = []
        self.selected_nucleus_id = None
        
        # Clear display items
        self.image_item.clear()
        self.mask_item.clear()
        
        # Clear channel controls
        for checkbox in self.channel_checkboxes:
            checkbox.deleteLater()
        self.channel_checkboxes.clear()
        self.channel_contrast_widgets.clear()
        
        # Reset combos
        self.dna_channel_combo.clear()
        
        # Reset slice slider
        self.slice_slider.setRange(0, 0)
        self.slice_slider.setValue(0)
        self.slice_label.setText("Slice: 0/0")
    
    def highlight_nucleus(self, nucleus_id: int):
        """Highlight a specific nucleus and zoom to it"""
        self.selected_nucleus_id = nucleus_id
        if self.mask_data is not None and self.mask_visible:
            self._update_mask_overlay()
        
        # Zoom to nucleus (if mask available)
        if self.mask_data is not None:
            self._zoom_to_nucleus(nucleus_id)

    def _invalidate_image_cache(self):
        """Invalidate cached image composite."""
        self._cached_image_key = None

    def _invalidate_mask_cache(self, clear_lut: bool = False):
        """Invalidate cached mask overlays."""
        self._cached_mask_key = None
        self._cached_mask_base = None
        if clear_lut:
            self._mask_lut = None
    
    def _setup_channel_controls(self):
        """Set up channel control widgets"""
        # Clear existing controls
        for checkbox in self.channel_checkboxes:
            checkbox.deleteLater()
        self.channel_checkboxes.clear()
        self.channel_contrast_widgets.clear()
        
        # Update DNA channel combo
        self.dna_channel_combo.clear()
        self.dna_channel_combo.addItems(self.channel_names)
        if self.dna_channel_index is not None and self.dna_channel_index < len(self.channel_names):
            self.dna_channel_combo.setCurrentIndex(self.dna_channel_index)
        
        # Add channel checkboxes and contrast controls
        for i, name in enumerate(self.channel_names):
            # Checkbox
            checkbox = QCheckBox(name)
            checkbox.setChecked(True)
            checkbox.setStyleSheet(f"QCheckBox {{ color: rgb{self.channel_colors[i]}; }}")
            checkbox.toggled.connect(lambda checked, idx=i: self._on_channel_toggled(idx, checked))
            
            self.channel_checkboxes.append(checkbox)
            self.channel_layout.addWidget(checkbox)
            
            # Contrast controls (min/max)
            if not self.auto_contrast:
                contrast_layout = QHBoxLayout()
                contrast_layout.addWidget(QLabel("  Min:"))
                
                min_spin = QSpinBox()
                min_spin.setRange(0, 65535)
                min_spin.setValue(int(self.contrast_min.get(i, 0)))
                min_spin.valueChanged.connect(lambda val, idx=i: self._on_contrast_changed(idx, val, 'min'))
                contrast_layout.addWidget(min_spin)
                
                contrast_layout.addWidget(QLabel("Max:"))
                
                max_spin = QSpinBox()
                max_spin.setRange(0, 65535)
                max_spin.setValue(int(self.contrast_max.get(i, 255)))
                max_spin.valueChanged.connect(lambda val, idx=i: self._on_contrast_changed(idx, val, 'max'))
                contrast_layout.addWidget(max_spin)
                
                self.channel_layout.addLayout(contrast_layout)
                self.channel_contrast_widgets.append((min_spin, max_spin))
    
    def _calculate_auto_contrast(self):
        """Calculate automatic contrast settings based on image histogram"""
        if self.image_data is None:
            return
        
        is_3d = self.image_data.ndim == 4
        n_channels = self.image_data.shape[1 if is_3d else 0]
        max_samples = 1_000_000
        
        for i in range(n_channels):
            if is_3d:
                channel_data = self.image_data[:, i, :, :]
            else:
                channel_data = self.image_data[i, :, :]

            flat = channel_data.ravel()
            if flat.size > max_samples:
                step = max(1, flat.size // max_samples)
                sample = flat[::step]
            else:
                sample = flat

            # Calculate percentiles for auto-contrast
            self.contrast_min[i] = float(np.percentile(sample, 0.1))
            self.contrast_max[i] = float(np.percentile(sample, 99.9))
    
    def _update_display(self):
        """Update the displayed image and mask"""
        if self.image_data is None:
            return
        
        is_3d = self.image_data.ndim == 4
        
        # Get current slice
        if is_3d:
            slice_data = self.image_data[self.current_slice]
        else:
            slice_data = self.image_data

        visible_sorted = tuple(sorted(ch for ch in self.visible_channels if ch < slice_data.shape[0]))
        contrast_key = tuple(
            (ch, round(self.contrast_min.get(ch, 0.0), 6), round(self.contrast_max.get(ch, 0.0), 6))
            for ch in visible_sorted
        )
        image_key = (
            id(self.image_data),
            self.current_slice if is_3d else 0,
            visible_sorted,
            contrast_key
        )

        if self._cached_image_key != image_key:
            rgb_image = self._compose_rgb_image(slice_data, visible_sorted)
            self.img_item.setImage(rgb_image, autoLevels=False)
            self._cached_image_key = image_key
        
        # Update mask overlay
        if self.mask_data is not None and self.mask_visible:
            self._update_mask_overlay()
    
    def _compose_rgb_image(self, slice_data: np.ndarray, visible_channels: Tuple[int, ...]) -> np.ndarray:
        """Compose a channels-overlaid RGB image for current slice."""
        height, width = slice_data.shape[-2:]
        rgb_float = np.zeros((height, width, 3), dtype=np.float32)

        for ch_idx in visible_channels:
            channel_img = slice_data[ch_idx].astype(np.float32, copy=False)

            vmin = float(self.contrast_min.get(ch_idx, np.min(channel_img)))
            vmax = float(self.contrast_max.get(ch_idx, np.max(channel_img)))
            if vmax <= vmin:
                continue

            normalized = np.clip((channel_img - vmin) / (vmax - vmin), 0.0, 1.0)
            color = np.asarray(self.channel_colors[ch_idx], dtype=np.float32).reshape(1, 1, 3)
            np.maximum(rgb_float, normalized[..., None] * color, out=rgb_float)

        return rgb_float.astype(np.uint8, copy=False)
    
    def _update_mask_overlay(self):
        """Update the mask overlay visualization"""
        if self.mask_data is None:
            return
        
        is_3d = self.mask_data.ndim == 3
        
        # Get current slice mask
        if is_3d:
            mask_slice = self.mask_data[self.current_slice]
        else:
            mask_slice = self.mask_data

        mask_key = (id(self.mask_data), self.current_slice if is_3d else 0)
        if self._cached_mask_key != mask_key:
            self._cached_mask_base = self._create_colored_mask(mask_slice)
            self._cached_mask_key = mask_key

        if self._cached_mask_base is None:
            return

        colored_mask = self._cached_mask_base
        selected_id = self.selected_nucleus_id
        if selected_id is not None and selected_id > 0:
            if selected_id <= int(mask_slice.max()):
                selected_pixels = (mask_slice == selected_id)
                if np.any(selected_pixels):
                    colored_mask = colored_mask.copy()
                    colored_mask[selected_pixels] = np.array([255, 255, 0, 200], dtype=np.uint8)
        
        # Display mask
        self.mask_item.setImage(colored_mask, autoLevels=False)
        self.mask_item.setOpacity(self.mask_opacity)
    
    def _create_colored_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create a colored visualization of the labeled mask"""
        height, width = mask.shape
        
        max_id = int(mask.max())
        if max_id == 0:
            return np.zeros((height, width, 4), dtype=np.uint8)

        # Build/extend deterministic LUT once; avoid reseeding global RNG per redraw.
        if self._mask_lut is None or self._mask_lut.shape[0] <= max_id:
            lut = np.zeros((max_id + 1, 4), dtype=np.uint8)
            lut[0] = [0, 0, 0, 0]
            labels = np.arange(1, max_id + 1, dtype=np.uint32)
            lut[1:, 0] = ((labels * 37) % 251 + 4).astype(np.uint8)
            lut[1:, 1] = ((labels * 67) % 251 + 4).astype(np.uint8)
            lut[1:, 2] = ((labels * 97) % 251 + 4).astype(np.uint8)
            lut[1:, 3] = 150
            self._mask_lut = lut

        return self._mask_lut[mask]
    
    def _zoom_to_nucleus(self, nucleus_id: int):
        """Zoom the view to a specific nucleus"""
        if self.mask_data is None:
            return
        
        is_3d = self.mask_data.ndim == 3
        
        if is_3d:
            mask_slice = self.mask_data[self.current_slice]
        else:
            mask_slice = self.mask_data
        
        # Find bounding box
        coords = np.argwhere(mask_slice == nucleus_id)
        if len(coords) == 0:
            return
        
        min_row, min_col = coords.min(axis=0)
        max_row, max_col = coords.max(axis=0)
        
        # Add padding
        padding = 20
        min_row = max(0, min_row - padding)
        min_col = max(0, min_col - padding)
        max_row = min(mask_slice.shape[0] - 1, max_row + padding)
        max_col = min(mask_slice.shape[1] - 1, max_col + padding)
        
        # Set view
        self.view_box.setRange(
            xRange=(min_col, max_col),
            yRange=(min_row, max_row),
            padding=0
        )
    
    def _get_default_colors(self, n_channels: int) -> List[Tuple[int, int, int]]:
        """Get default colors for channels"""
        default_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 255, 255),# White
        ]
        
        colors = []
        for i in range(n_channels):
            if i < len(default_colors):
                colors.append(default_colors[i])
            else:
                # Deterministic high-contrast colors without mutating global RNG state
                colors.append((
                    int((37 * i) % 128 + 128),
                    int((67 * i) % 128 + 128),
                    int((97 * i) % 128 + 128),
                ))
        
        return colors
    
    def _update_slice_label(self):
        """Update the slice label text"""
        max_slice = self.slice_slider.maximum()
        current = self.current_slice + 1
        total = max_slice + 1
        self.slice_label.setText(f"Slice: {current} / {total}")
    
    # Event handlers
    def _on_slice_changed(self, value: int):
        """Handle slice slider change"""
        self.current_slice = value
        self._update_slice_label()
        self._invalidate_mask_cache(clear_lut=False)
        self._update_display()
        self.slice_changed.emit(value)
    
    def _on_channel_toggled(self, channel_idx: int, checked: bool):
        """Handle channel checkbox toggle"""
        if checked and channel_idx not in self.visible_channels:
            self.visible_channels.append(channel_idx)
        elif not checked and channel_idx in self.visible_channels:
            self.visible_channels.remove(channel_idx)
        
        self._invalidate_image_cache()
        self._update_display()
        self.channel_changed.emit(self.visible_channels)
    
    def _on_dna_channel_changed(self, index: int):
        """Handle DNA channel selection"""
        self.dna_channel_index = index
    
    def _on_contrast_changed(self, channel_idx: int, value: int, min_or_max: str):
        """Handle contrast adjustment"""
        if min_or_max == 'min':
            self.contrast_min[channel_idx] = float(value)
        else:
            self.contrast_max[channel_idx] = float(value)
        
        self._invalidate_image_cache()
        self._update_display()
    
    def _on_mask_visibility_changed(self, checked: bool):
        """Handle mask visibility toggle"""
        self.mask_visible = checked
        if checked:
            self._update_mask_overlay()
        else:
            self.mask_item.clear()
    
    def _on_mask_opacity_changed(self, value: int):
        """Handle mask opacity slider"""
        self.mask_opacity = value / 100.0
        self.mask_item.setOpacity(self.mask_opacity)
    
    def _on_mouse_clicked(self, event):
        """Handle mouse click on image"""
        if self.mask_data is None:
            return
        
        # Get click position
        pos = event.scenePos()
        mouse_point = self.view_box.mapSceneToView(pos)
        
        x = int(mouse_point.x())
        y = int(mouse_point.y())
        
        # Get mask at current slice
        is_3d = self.mask_data.ndim == 3
        if is_3d:
            mask_slice = self.mask_data[self.current_slice]
        else:
            mask_slice = self.mask_data
        
        # Check bounds
        if 0 <= y < mask_slice.shape[0] and 0 <= x < mask_slice.shape[1]:
            nucleus_id = int(mask_slice[y, x])
            if nucleus_id > 0:
                self.selected_nucleus_id = nucleus_id
                if self.mask_visible:
                    self._update_mask_overlay()
                self.nucleus_selected.emit(nucleus_id)
    
    def fit_to_window(self):
        """Fit the image to the window"""
        self.view_box.autoRange()
    
    def reset_zoom(self):
        """Reset zoom to 100%"""
        if self.image_data is not None:
            is_3d = self.image_data.ndim == 4
            if is_3d:
                height, width = self.image_data.shape[-2:]
            else:
                height, width = self.image_data.shape[-2:]
            
            self.view_box.setRange(
                xRange=(0, width),
                yRange=(0, height),
                padding=0
            )
    
    def get_dna_channel_index(self) -> Optional[int]:
        """Get the index of the designated DNA channel"""
        return self.dna_channel_index
    
    def get_current_slice(self) -> int:
        """Get the current Z-slice index"""
        return self.current_slice
