"""
Segmentation panel GUI for configuring and running segmentation
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                               QLabel, QComboBox, QPushButton, QDoubleSpinBox,
                               QCheckBox, QProgressBar, QTextEdit, QSpinBox,
                               QRadioButton, QButtonGroup)
from PySide6.QtCore import Qt, Signal
from typing import Optional, Dict
import numpy as np


class SegmentationPanel(QWidget):
    """
    GUI panel for segmentation configuration and execution
    """
    
    # Signals
    run_segmentation = Signal(dict)  # Emit segmentation parameters
    run_batch_segmentation = Signal(dict)  # Emit batch segmentation request
    auto_detect_diameter = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.current_image: Optional[np.ndarray] = None
        self.gpu_available = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface"""
        layout = QVBoxLayout(self)
        
        # Model selection
        model_group = QGroupBox("Segmentation Model")
        model_layout = QVBoxLayout(model_group)
        
        # Engine selection (Cellpose or SAM)
        engine_layout = QHBoxLayout()
        engine_layout.addWidget(QLabel("Engine:"))
        
        self.engine_group = QButtonGroup()
        self.cellpose_radio = QRadioButton("Cellpose")
        self.sam_radio = QRadioButton("SAM")
        self.cellpose_radio.setChecked(True)
        
        self.engine_group.addButton(self.cellpose_radio)
        self.engine_group.addButton(self.sam_radio)
        
        self.cellpose_radio.toggled.connect(self._on_engine_changed)
        
        engine_layout.addWidget(self.cellpose_radio)
        engine_layout.addWidget(self.sam_radio)
        engine_layout.addStretch()
        
        model_layout.addLayout(engine_layout)
        
        # Model type selection
        model_type_layout = QHBoxLayout()
        model_type_layout.addWidget(QLabel("Model:"))
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(['nuclei', 'cyto', 'cyto2', 'cyto3', 'cyto_sam'])
        model_type_layout.addWidget(self.model_combo)
        
        model_layout.addLayout(model_type_layout)
        
        layout.addWidget(model_group)
        
        # Cellpose parameters
        self.cellpose_params_group = QGroupBox("Cellpose Parameters")
        cellpose_layout = QVBoxLayout(self.cellpose_params_group)
        
        # Diameter
        diameter_layout = QHBoxLayout()
        diameter_layout.addWidget(QLabel("Diameter (pixels):"))
        
        self.diameter_spin = QDoubleSpinBox()
        self.diameter_spin.setRange(0, 500)
        self.diameter_spin.setValue(30)
        self.diameter_spin.setSingleStep(1)
        self.diameter_spin.setSpecialValueText("Auto")
        diameter_layout.addWidget(self.diameter_spin)
        
        self.auto_diameter_btn = QPushButton("Auto-Detect")
        self.auto_diameter_btn.clicked.connect(self.auto_detect_diameter.emit)
        diameter_layout.addWidget(self.auto_diameter_btn)
        
        cellpose_layout.addLayout(diameter_layout)
        
        # Flow threshold
        flow_layout = QHBoxLayout()
        flow_layout.addWidget(QLabel("Flow Threshold:"))
        
        self.flow_threshold_spin = QDoubleSpinBox()
        self.flow_threshold_spin.setRange(0.0, 3.0)
        self.flow_threshold_spin.setValue(0.4)
        self.flow_threshold_spin.setSingleStep(0.1)
        self.flow_threshold_spin.setToolTip("Higher values = fewer masks (more stringent)")
        flow_layout.addWidget(self.flow_threshold_spin)
        
        cellpose_layout.addLayout(flow_layout)
        
        # Cell probability threshold
        cellprob_layout = QHBoxLayout()
        cellprob_layout.addWidget(QLabel("Cell Prob Threshold:"))
        
        self.cellprob_threshold_spin = QDoubleSpinBox()
        self.cellprob_threshold_spin.setRange(-6.0, 6.0)
        self.cellprob_threshold_spin.setValue(0.0)
        self.cellprob_threshold_spin.setSingleStep(0.1)
        self.cellprob_threshold_spin.setToolTip("Higher values = fewer masks (more stringent)")
        cellprob_layout.addWidget(self.cellprob_threshold_spin)
        
        cellpose_layout.addLayout(cellprob_layout)
        
        # 3D segmentation
        self.do_3d_check = QCheckBox("3D Segmentation (volumetric)")
        self.do_3d_check.setToolTip("Use 3D segmentation for Z-stacks")
        self.do_3d_check.toggled.connect(self._on_3d_toggled)
        cellpose_layout.addWidget(self.do_3d_check)
        
        # 3D Backend selection (only shown when 3D is enabled)
        self.backend_3d_layout = QHBoxLayout()
        self.backend_3d_layout.addWidget(QLabel("3D Backend:"))
        
        self.backend_3d_combo = QComboBox()
        self.backend_3d_combo.addItems([
            'default (slice-by-slice)',
            'hybrid2d3d (2D + linking)',
            'true3d (placeholder)'
        ])
        self.backend_3d_combo.setToolTip(
            "Choose 3D segmentation backend:\n"
            "- default: Process each slice independently\n"
            "- hybrid2d3d: 2D segmentation + 3D instance linking\n"
            "- true3d: Full 3D volumetric (not yet implemented)"
        )
        self.backend_3d_layout.addWidget(self.backend_3d_combo)
        
        cellpose_layout.addLayout(self.backend_3d_layout)
        
        # Initially hide 3D backend selection
        self.backend_3d_combo.hide()
        for i in range(self.backend_3d_layout.count()):
            widget = self.backend_3d_layout.itemAt(i).widget()
            if widget:
                widget.hide()
        
        # Cellpose3 Restoration Mode
        restoration_layout = QHBoxLayout()
        restoration_layout.addWidget(QLabel("Restoration Mode:"))
        
        self.restoration_combo = QComboBox()
        self.restoration_combo.addItems([
            'none (disabled)',
            'auto (detect from image)',
            'denoise (for noisy images)',
            'deblur (for blurred images)'
        ])
        self.restoration_combo.setToolTip(
            "Cellpose3 restoration for degraded microscopy:\n"
            "- none: Standard segmentation\n"
            "- auto: Automatically detect and apply restoration\n"
            "- denoise: Reduce noise before segmentation\n"
            "- deblur: Sharpen blurred images\n"
            "\nRequires Cellpose 3.0+"
        )
        restoration_layout.addWidget(self.restoration_combo)
        
        cellpose_layout.addLayout(restoration_layout)
        
        # Channels
        channels_layout = QHBoxLayout()
        channels_layout.addWidget(QLabel("Channels:"))
        
        self.channel0_spin = QSpinBox()
        self.channel0_spin.setRange(0, 10)
        self.channel0_spin.setValue(0)
        self.channel0_spin.setPrefix("Cyto: ")
        channels_layout.addWidget(self.channel0_spin)
        
        self.channel1_spin = QSpinBox()
        self.channel1_spin.setRange(0, 10)
        self.channel1_spin.setValue(0)
        self.channel1_spin.setPrefix("Nucleus: ")
        channels_layout.addWidget(self.channel1_spin)
        
        cellpose_layout.addLayout(channels_layout)
        
        layout.addWidget(self.cellpose_params_group)
        
        # SAM parameters (initially hidden)
        self.sam_params_group = QGroupBox("SAM Parameters")
        sam_layout = QVBoxLayout(self.sam_params_group)
        
        # SAM model type
        sam_model_layout = QHBoxLayout()
        sam_model_layout.addWidget(QLabel("Model Size:"))
        
        self.sam_model_combo = QComboBox()
        self.sam_model_combo.addItems(['vit_h (2.5GB, best)', 'vit_l (1.2GB)', 'vit_b (375MB, fastest)'])
        sam_layout.addLayout(sam_model_layout)
        
        # SAM mode
        self.sam_automatic_check = QCheckBox("Automatic Mask Generation")
        self.sam_automatic_check.setChecked(True)
        self.sam_automatic_check.setToolTip("Automatically segment all objects")
        sam_layout.addWidget(self.sam_automatic_check)
        
        sam_layout.addWidget(QLabel("Note: SAM is best for interactive segmentation"))
        
        layout.addWidget(self.sam_params_group)
        self.sam_params_group.hide()
        
        # Run buttons
        run_buttons_layout = QHBoxLayout()
        
        self.run_button = QPushButton("Run Segmentation")
        self.run_button.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
        self.run_button.clicked.connect(self._on_run_clicked)
        run_buttons_layout.addWidget(self.run_button, stretch=2)
        
        self.batch_button = QPushButton("Apply to All Images")
        self.batch_button.setStyleSheet("QPushButton { padding: 8px; background-color: #4CAF50; color: white; }")
        self.batch_button.setToolTip("Apply current settings to all images in project")
        self.batch_button.clicked.connect(self._on_batch_clicked)
        run_buttons_layout.addWidget(self.batch_button, stretch=1)
        
        layout.addLayout(run_buttons_layout)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        # Results display
        results_group = QGroupBox("Segmentation Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        results_layout.addWidget(self.results_text)
        
        layout.addWidget(results_group)
        
        # History
        history_group = QGroupBox("Segmentation History")
        history_layout = QVBoxLayout(history_group)
        
        self.history_combo = QComboBox()
        self.history_combo.setToolTip("Previous segmentation runs")
        history_layout.addWidget(self.history_combo)
        
        history_buttons_layout = QHBoxLayout()
        self.load_history_btn = QPushButton("Load Selected")
        self.load_history_btn.clicked.connect(self._on_load_history)
        history_buttons_layout.addWidget(self.load_history_btn)
        
        self.clear_history_btn = QPushButton("Clear History")
        self.clear_history_btn.clicked.connect(self._on_clear_history)
        history_buttons_layout.addWidget(self.clear_history_btn)
        
        history_layout.addLayout(history_buttons_layout)
        
        layout.addWidget(history_group)
        
        layout.addStretch()
    
    def set_image(self, image: np.ndarray, metadata: Dict):
        """Set the current image for segmentation"""
        self.current_image = image
        
        # Enable 3D option if image is 3D
        is_3d = image.ndim == 4
        self.do_3d_check.setEnabled(is_3d)
        
        # Clear results
        self.results_text.clear()
    
    def set_gpu_available(self, available: bool):
        """Set GPU availability status"""
        self.gpu_available = available
        
        # Update UI to show GPU status
        if not available:
            self.cellpose_params_group.setTitle("Cellpose Parameters (CPU Mode - Slower)")
    
    def get_parameters(self) -> Dict:
        """Get current segmentation parameters"""
        if self.cellpose_radio.isChecked():
            # Parse restoration mode
            restoration_text = self.restoration_combo.currentText()
            restoration_mode = restoration_text.split()[0]  # Extract 'none', 'auto', etc.
            
            # Parse 3D backend
            backend_3d_text = self.backend_3d_combo.currentText()
            if 'hybrid2d3d' in backend_3d_text:
                use_3d_backend = 'hybrid2d3d'
            elif 'true3d' in backend_3d_text:
                use_3d_backend = 'true3d'
            else:
                use_3d_backend = 'default'
            
            return {
                'engine': 'cellpose',
                'model_name': self.model_combo.currentText(),
                'diameter': self.diameter_spin.value() if self.diameter_spin.value() > 0 else None,
                'flow_threshold': self.flow_threshold_spin.value(),
                'cellprob_threshold': self.cellprob_threshold_spin.value(),
                'do_3d': self.do_3d_check.isChecked(),
                'channels': [self.channel0_spin.value(), self.channel1_spin.value()],
                'restoration_mode': restoration_mode,
                'use_3d_backend': use_3d_backend,
            }
        else:
            sam_model = self.sam_model_combo.currentText().split()[0]  # Extract 'vit_h', etc.
            return {
                'engine': 'sam',
                'model_type': sam_model,
                'automatic': self.sam_automatic_check.isChecked(),
            }
    
    def set_parameters(self, params: Dict):
        """Load parameters into UI"""
        engine = params.get('engine', 'cellpose')
        
        if engine == 'cellpose':
            self.cellpose_radio.setChecked(True)
            self.model_combo.setCurrentText(params.get('model_name', 'nuclei'))
            
            diameter = params.get('diameter')
            self.diameter_spin.setValue(diameter if diameter is not None else 0)
            
            self.flow_threshold_spin.setValue(params.get('flow_threshold', 0.4))
            self.cellprob_threshold_spin.setValue(params.get('cellprob_threshold', 0.0))
            self.do_3d_check.setChecked(params.get('do_3d', False))
            
            channels = params.get('channels', [0, 0])
            self.channel0_spin.setValue(channels[0])
            self.channel1_spin.setValue(channels[1])
            
            # Set restoration mode
            restoration_mode = params.get('restoration_mode', 'none')
            for i in range(self.restoration_combo.count()):
                if restoration_mode in self.restoration_combo.itemText(i):
                    self.restoration_combo.setCurrentIndex(i)
                    break
            
            # Set 3D backend
            use_3d_backend = params.get('use_3d_backend', 'default')
            for i in range(self.backend_3d_combo.count()):
                if use_3d_backend in self.backend_3d_combo.itemText(i):
                    self.backend_3d_combo.setCurrentIndex(i)
                    break
        else:
            self.sam_radio.setChecked(True)
            model_type = params.get('model_type', 'vit_h')
            # Find matching combo item
            for i in range(self.sam_model_combo.count()):
                if model_type in self.sam_model_combo.itemText(i):
                    self.sam_model_combo.setCurrentIndex(i)
                    break
            
            self.sam_automatic_check.setChecked(params.get('automatic', True))
    
    def _on_3d_toggled(self, checked: bool):
        """Handle 3D checkbox toggle - show/hide 3D backend selection"""
        # Show/hide 3D backend selection
        for i in range(self.backend_3d_layout.count()):
            widget = self.backend_3d_layout.itemAt(i).widget()
            if widget:
                widget.setVisible(checked)
    
    
    def set_estimated_diameter(self, diameter: float):
        """Set the diameter from auto-detection"""
        self.diameter_spin.setValue(diameter)
        self.results_text.append(f"Auto-detected diameter: {diameter:.1f} pixels")
    
    def display_results(self, results: Dict):
        """Display segmentation results"""
        self.results_text.clear()
        
        text = f"<b>Segmentation Complete</b><br>"
        text += f"Model: {results.get('model_name', 'Unknown')}<br>"
        text += f"Nuclei detected: {results.get('nucleus_count', 0)}<br>"
        text += f"Median area: {results.get('median_area', 0):.1f} pixels<br>"
        text += f"CV of areas: {results.get('cv_area', 0):.1f}%<br>"
        text += f"Processing time: {results.get('processing_time', 0):.2f} seconds<br>"
        
        if results.get('diameter'):
            text += f"Diameter used: {results.get('diameter', 0):.1f} pixels<br>"
        
        self.results_text.setHtml(text)
    
    def add_to_history(self, params: Dict, results: Dict):
        """Add segmentation run to history"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        label = f"{timestamp} - {params.get('engine', 'unknown')} - {results.get('nucleus_count', 0)} nuclei"
        
        # Store params and results as item data
        self.history_combo.addItem(label)
        self.history_combo.setItemData(
            self.history_combo.count() - 1,
            {'params': params, 'results': results}
        )
    
    def set_running(self, running: bool):
        """Set UI state for running segmentation"""
        self.run_button.setEnabled(not running)
        self.batch_button.setEnabled(not running)
        self.cellpose_params_group.setEnabled(not running)
        self.sam_params_group.setEnabled(not running)
        
        if running:
            self.progress_bar.setRange(0, 0)  # Indeterminate
            self.progress_bar.show()
            self.results_text.setText("Running segmentation...")
        else:
            self.progress_bar.hide()
    
    def set_batch_progress(self, current: int, total: int):
        """Set progress for batch segmentation"""
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.results_text.setText(f"Processing image {current} of {total}...")
    
    def _on_engine_changed(self, checked: bool):
        """Handle engine radio button change"""
        if self.cellpose_radio.isChecked():
            self.cellpose_params_group.show()
            self.sam_params_group.hide()
            self.model_combo.clear()
            self.model_combo.addItems(['nuclei', 'cyto', 'cyto2', 'cyto3', 'cyto_sam'])
        else:
            self.cellpose_params_group.hide()
            self.sam_params_group.show()
    
    def _on_run_clicked(self):
        """Handle run button click"""
        if self.current_image is None:
            self.results_text.setText("No image loaded")
            return
        
        params = self.get_parameters()
        self.run_segmentation.emit(params)
    
    def _on_batch_clicked(self):
        """Handle batch segmentation button click"""
        params = self.get_parameters()
        self.run_batch_segmentation.emit(params)
    
    def _on_load_history(self):
        """Load parameters from history"""
        current_index = self.history_combo.currentIndex()
        if current_index >= 0:
            data = self.history_combo.itemData(current_index)
            if data:
                params = data.get('params', {})
                self.set_parameters(params)
                self.results_text.setText("Parameters loaded from history")
    
    def _on_clear_history(self):
        """Clear segmentation history"""
        self.history_combo.clear()
        self.results_text.setText("History cleared")
    
    def clear(self):
        """Clear the panel state"""
        self.current_image = None
        self.diameter_spin.setValue(30)
        self.flow_threshold_spin.setValue(0.4)
        self.cellprob_threshold_spin.setValue(0.0)
        self.results_text.clear()
        self.progress_bar.hide()
