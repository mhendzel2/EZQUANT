"""
Settings dialog for application configuration
"""

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
                               QWidget, QLabel, QSpinBox, QDoubleSpinBox,
                               QCheckBox, QComboBox, QGroupBox, QPushButton,
                               QLineEdit, QFileDialog, QFormLayout, QDialogButtonBox)
from PySide6.QtCore import Qt, QSettings
from pathlib import Path
from typing import Dict, Any

from gui.accessibility import AccessibilityManager, CVDPalette


class SettingsDialog(QDialog):
    """Dialog for managing application settings"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.settings = QSettings("NucleiSegApp", "EZQUANT")
        self.setWindowTitle("Settings")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        self._setup_ui()
        self._load_settings()
    
    def _setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Tab widget for different setting categories
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.tab_widget.addTab(self._create_general_tab(), "General")
        self.tab_widget.addTab(self._create_segmentation_tab(), "Segmentation")
        self.tab_widget.addTab(self._create_analysis_tab(), "Analysis")
        self.tab_widget.addTab(self._create_visualization_tab(), "Visualization")
        self.tab_widget.addTab(self._create_accessibility_tab(), "Accessibility")
        self.tab_widget.addTab(self._create_advanced_tab(), "Advanced")
        
        layout.addWidget(self.tab_widget)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults
        )
        button_box.accepted.connect(self._save_and_close)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._restore_defaults)
        
        layout.addWidget(button_box)
        AccessibilityManager.apply_accessible_names(self)
    
    def _create_general_tab(self) -> QWidget:
        """Create general settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Project settings
        project_group = QGroupBox("Project Settings")
        project_layout = QFormLayout()
        
        self.default_project_dir_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_project_dir)
        
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.default_project_dir_edit)
        dir_layout.addWidget(browse_btn)
        
        project_layout.addRow("Default Project Directory:", dir_layout)
        
        self.autosave_check = QCheckBox("Enable auto-save")
        project_layout.addRow(self.autosave_check)
        
        self.autosave_interval_spin = QSpinBox()
        self.autosave_interval_spin.setRange(1, 60)
        self.autosave_interval_spin.setSuffix(" minutes")
        project_layout.addRow("Auto-save Interval:", self.autosave_interval_spin)
        
        project_group.setLayout(project_layout)
        layout.addWidget(project_group)
        
        # UI settings
        ui_group = QGroupBox("User Interface")
        ui_layout = QFormLayout()
        
        self.show_gpu_dialog_check = QCheckBox("Show GPU detection dialog on startup")
        ui_layout.addRow(self.show_gpu_dialog_check)
        
        self.confirm_delete_check = QCheckBox("Confirm before deleting nuclei")
        ui_layout.addRow(self.confirm_delete_check)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["System Default", "Light", "Dark"])
        ui_layout.addRow("Theme:", self.theme_combo)
        
        ui_group.setLayout(ui_layout)
        layout.addWidget(ui_group)
        
        layout.addStretch()
        return widget
    
    def _create_segmentation_tab(self) -> QWidget:
        """Create segmentation settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Cellpose defaults
        cellpose_group = QGroupBox("Cellpose Default Parameters")
        cellpose_layout = QFormLayout()
        
        self.default_model_combo = QComboBox()
        self.default_model_combo.addItems(['nuclei', 'cyto', 'cyto2', 'cyto3', 'cyto_sam'])
        cellpose_layout.addRow("Default Model:", self.default_model_combo)
        
        self.default_diameter_spin = QDoubleSpinBox()
        self.default_diameter_spin.setRange(0, 500)
        self.default_diameter_spin.setSpecialValueText("Auto")
        cellpose_layout.addRow("Default Diameter:", self.default_diameter_spin)
        
        self.default_flow_threshold_spin = QDoubleSpinBox()
        self.default_flow_threshold_spin.setRange(0.0, 3.0)
        self.default_flow_threshold_spin.setSingleStep(0.1)
        cellpose_layout.addRow("Default Flow Threshold:", self.default_flow_threshold_spin)
        
        self.default_cellprob_spin = QDoubleSpinBox()
        self.default_cellprob_spin.setRange(-6.0, 6.0)
        self.default_cellprob_spin.setSingleStep(0.1)
        cellpose_layout.addRow("Default Cell Prob Threshold:", self.default_cellprob_spin)
        
        cellpose_group.setLayout(cellpose_layout)
        layout.addWidget(cellpose_group)
        
        # GPU settings
        gpu_group = QGroupBox("GPU Settings")
        gpu_layout = QFormLayout()
        
        self.use_gpu_check = QCheckBox("Use GPU for segmentation (if available)")
        gpu_layout.addRow(self.use_gpu_check)
        
        self.gpu_memory_spin = QSpinBox()
        self.gpu_memory_spin.setRange(1024, 32768)
        self.gpu_memory_spin.setSuffix(" MB")
        gpu_layout.addRow("GPU Memory Limit:", self.gpu_memory_spin)
        
        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)
        
        layout.addStretch()
        return widget
    
    def _create_analysis_tab(self) -> QWidget:
        """Create analysis settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Default measurements
        measurements_group = QGroupBox("Default Measurement Categories")
        measurements_layout = QVBoxLayout()
        
        self.basic_shape_default_check = QCheckBox("Basic Shape Measurements")
        measurements_layout.addWidget(self.basic_shape_default_check)
        
        self.advanced_morph_default_check = QCheckBox("Advanced Morphology")
        measurements_layout.addWidget(self.advanced_morph_default_check)
        
        self.intensity_default_check = QCheckBox("Intensity Statistics")
        measurements_layout.addWidget(self.intensity_default_check)
        
        self.cell_cycle_default_check = QCheckBox("Cell Cycle Phase Assignment")
        measurements_layout.addWidget(self.cell_cycle_default_check)
        
        measurements_group.setLayout(measurements_layout)
        layout.addWidget(measurements_group)
        
        # Analysis options
        analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QFormLayout()
        
        self.default_workflow_combo = QComboBox()
        self.default_workflow_combo.addItems(["2D Analysis", "3D Analysis"])
        analysis_layout.addRow("Default Workflow:", self.default_workflow_combo)
        
        self.outlier_detection_check = QCheckBox("Enable outlier detection")
        analysis_layout.addRow(self.outlier_detection_check)
        
        self.outlier_std_spin = QDoubleSpinBox()
        self.outlier_std_spin.setRange(1.0, 5.0)
        self.outlier_std_spin.setSingleStep(0.5)
        self.outlier_std_spin.setSuffix(" SD")
        analysis_layout.addRow("Outlier Threshold:", self.outlier_std_spin)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        layout.addStretch()
        return widget
    
    def _create_visualization_tab(self) -> QWidget:
        """Create visualization settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Plot settings
        plot_group = QGroupBox("Plot Settings")
        plot_layout = QFormLayout()
        
        self.default_plot_type_combo = QComboBox()
        self.default_plot_type_combo.addItems([
            "DNA Histogram",
            "Scatter: Area vs DNA",
            "Box Plots: Morphology",
            "Scatter Matrix",
            "Correlation Heatmap",
            "Phase Distribution"
        ])
        plot_layout.addRow("Default Plot Type:", self.default_plot_type_combo)
        
        self.plot_dpi_spin = QSpinBox()
        self.plot_dpi_spin.setRange(72, 600)
        self.plot_dpi_spin.setSuffix(" DPI")
        plot_layout.addRow("Export DPI:", self.plot_dpi_spin)
        
        self.show_grid_check = QCheckBox("Show grid lines by default")
        plot_layout.addRow(self.show_grid_check)
        
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)
        
        # Color scheme
        color_group = QGroupBox("Color Scheme")
        color_layout = QFormLayout()
        
        self.color_scheme_combo = QComboBox()
        self.color_scheme_combo.addItems([
            "Default",
            "Viridis",
            "Plasma",
            "Inferno",
            "Cividis",
            "RdBu",
            "Set1"
        ])
        color_layout.addRow("Default Color Scheme:", self.color_scheme_combo)
        
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)
        
        layout.addStretch()
        return widget
    
    def _create_advanced_tab(self) -> QWidget:
        """Create advanced settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Performance settings
        performance_group = QGroupBox("Performance")
        performance_layout = QFormLayout()
        
        self.max_threads_spin = QSpinBox()
        self.max_threads_spin.setRange(1, 32)
        self.max_threads_spin.setSpecialValueText("Auto")
        performance_layout.addRow("Max Threads:", self.max_threads_spin)
        
        self.cache_size_spin = QSpinBox()
        self.cache_size_spin.setRange(100, 10000)
        self.cache_size_spin.setSuffix(" MB")
        performance_layout.addRow("Cache Size:", self.cache_size_spin)
        
        performance_group.setLayout(performance_layout)
        layout.addWidget(performance_group)
        
        # Data storage
        storage_group = QGroupBox("Data Storage")
        storage_layout = QFormLayout()
        
        self.compression_check = QCheckBox("Compress saved projects")
        storage_layout.addRow(self.compression_check)
        
        self.keep_history_check = QCheckBox("Keep segmentation history")
        storage_layout.addRow(self.keep_history_check)
        
        self.max_history_spin = QSpinBox()
        self.max_history_spin.setRange(1, 50)
        self.max_history_spin.setSuffix(" entries")
        storage_layout.addRow("Max History Entries:", self.max_history_spin)
        
        storage_group.setLayout(storage_layout)
        layout.addWidget(storage_group)
        
        # Plugin settings
        plugin_group = QGroupBox("Plugins")
        plugin_layout = QFormLayout()
        
        self.auto_load_plugins_check = QCheckBox("Auto-load plugins on startup")
        plugin_layout.addRow(self.auto_load_plugins_check)
        
        self.plugin_dir_edit = QLineEdit()
        browse_plugin_btn = QPushButton("Browse...")
        browse_plugin_btn.clicked.connect(self._browse_plugin_dir)
        
        plugin_dir_layout = QHBoxLayout()
        plugin_dir_layout.addWidget(self.plugin_dir_edit)
        plugin_dir_layout.addWidget(browse_plugin_btn)
        
        plugin_layout.addRow("Plugin Directory:", plugin_dir_layout)
        
        plugin_group.setLayout(plugin_layout)
        layout.addWidget(plugin_group)
        
        layout.addStretch()
        return widget

    def _create_accessibility_tab(self) -> QWidget:
        """Create accessibility settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        palette_group = QGroupBox("Color & Contrast")
        palette_layout = QFormLayout()
        self.colorblind_safe_check = QCheckBox("Enable colorblind-safe palette")
        palette_layout.addRow(self.colorblind_safe_check)

        self.cvd_palette_combo = QComboBox()
        self.cvd_palette_combo.addItems(
            [CVDPalette.OKABE_ITO.value, CVDPalette.IBM.value, CVDPalette.DEFAULT.value]
        )
        palette_layout.addRow("Palette:", self.cvd_palette_combo)
        palette_group.setLayout(palette_layout)
        layout.addWidget(palette_group)

        ui_group = QGroupBox("Visual Scaling")
        ui_layout = QFormLayout()
        self.ui_scale_spin = QDoubleSpinBox()
        self.ui_scale_spin.setRange(1.0, 2.0)
        self.ui_scale_spin.setSingleStep(0.1)
        self.ui_scale_spin.setDecimals(1)
        self.ui_scale_spin.setSuffix("x")
        ui_layout.addRow("UI Scale:", self.ui_scale_spin)
        ui_group.setLayout(ui_layout)
        layout.addWidget(ui_group)

        input_group = QGroupBox("Motor & Input")
        input_layout = QFormLayout()
        self.sticky_keys_check = QCheckBox("Enable sticky keys support")
        input_layout.addRow(self.sticky_keys_check)

        self.double_click_speed_spin = QSpinBox()
        self.double_click_speed_spin.setRange(200, 1200)
        self.double_click_speed_spin.setSuffix(" ms")
        input_layout.addRow("Double-click interval:", self.double_click_speed_spin)

        self.drag_sensitivity_spin = QSpinBox()
        self.drag_sensitivity_spin.setRange(1, 30)
        input_layout.addRow("Drag sensitivity:", self.drag_sensitivity_spin)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        screen_reader_group = QGroupBox("Screen Reader")
        screen_reader_layout = QVBoxLayout()
        self.screen_reader_hints_check = QCheckBox("Enable screen reader hints")
        screen_reader_layout.addWidget(self.screen_reader_hints_check)
        screen_reader_group.setLayout(screen_reader_layout)
        layout.addWidget(screen_reader_group)

        layout.addStretch()
        return widget
    
    def _browse_project_dir(self):
        """Browse for default project directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Default Project Directory",
            self.default_project_dir_edit.text()
        )
        if directory:
            self.default_project_dir_edit.setText(directory)
    
    def _browse_plugin_dir(self):
        """Browse for plugin directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Plugin Directory",
            self.plugin_dir_edit.text()
        )
        if directory:
            self.plugin_dir_edit.setText(directory)
    
    def _load_settings(self):
        """Load settings from QSettings"""
        # General
        self.default_project_dir_edit.setText(
            self.settings.value("general/default_project_dir", str(Path.home() / "Documents" / "EZQUANT"))
        )
        self.autosave_check.setChecked(
            self.settings.value("general/autosave_enabled", True, type=bool)
        )
        self.autosave_interval_spin.setValue(
            self.settings.value("general/autosave_interval", 5, type=int)
        )
        self.show_gpu_dialog_check.setChecked(
            self.settings.value("general/show_gpu_dialog", True, type=bool)
        )
        self.confirm_delete_check.setChecked(
            self.settings.value("general/confirm_delete", True, type=bool)
        )
        self.theme_combo.setCurrentText(
            self.settings.value("general/theme", "System Default")
        )
        
        # Segmentation
        self.default_model_combo.setCurrentText(
            self.settings.value("segmentation/default_model", "nuclei")
        )
        self.default_diameter_spin.setValue(
            self.settings.value("segmentation/default_diameter", 30.0, type=float)
        )
        self.default_flow_threshold_spin.setValue(
            self.settings.value("segmentation/flow_threshold", 0.4, type=float)
        )
        self.default_cellprob_spin.setValue(
            self.settings.value("segmentation/cellprob_threshold", 0.0, type=float)
        )
        self.use_gpu_check.setChecked(
            self.settings.value("segmentation/use_gpu", True, type=bool)
        )
        self.gpu_memory_spin.setValue(
            self.settings.value("segmentation/gpu_memory_limit", 4096, type=int)
        )
        
        # Analysis
        self.basic_shape_default_check.setChecked(
            self.settings.value("analysis/basic_shape_default", True, type=bool)
        )
        self.advanced_morph_default_check.setChecked(
            self.settings.value("analysis/advanced_morph_default", True, type=bool)
        )
        self.intensity_default_check.setChecked(
            self.settings.value("analysis/intensity_default", True, type=bool)
        )
        self.cell_cycle_default_check.setChecked(
            self.settings.value("analysis/cell_cycle_default", False, type=bool)
        )
        self.default_workflow_combo.setCurrentText(
            self.settings.value("analysis/default_workflow", "2D Analysis")
        )
        self.outlier_detection_check.setChecked(
            self.settings.value("analysis/outlier_detection", True, type=bool)
        )
        self.outlier_std_spin.setValue(
            self.settings.value("analysis/outlier_threshold", 3.0, type=float)
        )
        
        # Visualization
        self.default_plot_type_combo.setCurrentText(
            self.settings.value("visualization/default_plot", "DNA Histogram")
        )
        self.plot_dpi_spin.setValue(
            self.settings.value("visualization/export_dpi", 300, type=int)
        )
        self.show_grid_check.setChecked(
            self.settings.value("visualization/show_grid", False, type=bool)
        )
        self.color_scheme_combo.setCurrentText(
            self.settings.value("visualization/color_scheme", "Default")
        )
        
        # Advanced
        self.max_threads_spin.setValue(
            self.settings.value("advanced/max_threads", 0, type=int)
        )
        self.cache_size_spin.setValue(
            self.settings.value("advanced/cache_size", 1000, type=int)
        )
        self.compression_check.setChecked(
            self.settings.value("advanced/compress_projects", True, type=bool)
        )
        self.keep_history_check.setChecked(
            self.settings.value("advanced/keep_history", True, type=bool)
        )
        self.max_history_spin.setValue(
            self.settings.value("advanced/max_history", 10, type=int)
        )
        self.auto_load_plugins_check.setChecked(
            self.settings.value("advanced/auto_load_plugins", True, type=bool)
        )

        # Get plugin directory relative to app
        from pathlib import Path
        default_plugin_dir = Path(__file__).parent.parent / "plugins"
        self.plugin_dir_edit.setText(
            self.settings.value("advanced/plugin_directory", str(default_plugin_dir))
        )

        # Accessibility
        self.colorblind_safe_check.setChecked(
            self.settings.value("accessibility/colorblind_safe", False, type=bool)
        )
        self.cvd_palette_combo.setCurrentText(
            self.settings.value("accessibility/palette", CVDPalette.OKABE_ITO.value)
        )
        self.ui_scale_spin.setValue(
            self.settings.value("accessibility/scale_factor", 1.0, type=float)
        )
        self.screen_reader_hints_check.setChecked(
            self.settings.value("accessibility/screen_reader_hints", True, type=bool)
        )
        self.sticky_keys_check.setChecked(
            self.settings.value("accessibility/sticky_keys", False, type=bool)
        )
        self.double_click_speed_spin.setValue(
            self.settings.value("accessibility/double_click_ms", 400, type=int)
        )
        self.drag_sensitivity_spin.setValue(
            self.settings.value("accessibility/drag_sensitivity", 10, type=int)
        )
    
    def _save_settings(self):
        """Save settings to QSettings"""
        # General
        self.settings.setValue("general/default_project_dir", self.default_project_dir_edit.text())
        self.settings.setValue("general/autosave_enabled", self.autosave_check.isChecked())
        self.settings.setValue("general/autosave_interval", self.autosave_interval_spin.value())
        self.settings.setValue("general/show_gpu_dialog", self.show_gpu_dialog_check.isChecked())
        self.settings.setValue("general/confirm_delete", self.confirm_delete_check.isChecked())
        self.settings.setValue("general/theme", self.theme_combo.currentText())
        
        # Segmentation
        self.settings.setValue("segmentation/default_model", self.default_model_combo.currentText())
        self.settings.setValue("segmentation/default_diameter", self.default_diameter_spin.value())
        self.settings.setValue("segmentation/flow_threshold", self.default_flow_threshold_spin.value())
        self.settings.setValue("segmentation/cellprob_threshold", self.default_cellprob_spin.value())
        self.settings.setValue("segmentation/use_gpu", self.use_gpu_check.isChecked())
        self.settings.setValue("segmentation/gpu_memory_limit", self.gpu_memory_spin.value())
        
        # Analysis
        self.settings.setValue("analysis/basic_shape_default", self.basic_shape_default_check.isChecked())
        self.settings.setValue("analysis/advanced_morph_default", self.advanced_morph_default_check.isChecked())
        self.settings.setValue("analysis/intensity_default", self.intensity_default_check.isChecked())
        self.settings.setValue("analysis/cell_cycle_default", self.cell_cycle_default_check.isChecked())
        self.settings.setValue("analysis/default_workflow", self.default_workflow_combo.currentText())
        self.settings.setValue("analysis/outlier_detection", self.outlier_detection_check.isChecked())
        self.settings.setValue("analysis/outlier_threshold", self.outlier_std_spin.value())
        
        # Visualization
        self.settings.setValue("visualization/default_plot", self.default_plot_type_combo.currentText())
        self.settings.setValue("visualization/export_dpi", self.plot_dpi_spin.value())
        self.settings.setValue("visualization/show_grid", self.show_grid_check.isChecked())
        self.settings.setValue("visualization/color_scheme", self.color_scheme_combo.currentText())
        
        # Advanced
        self.settings.setValue("advanced/max_threads", self.max_threads_spin.value())
        self.settings.setValue("advanced/cache_size", self.cache_size_spin.value())
        self.settings.setValue("advanced/compress_projects", self.compression_check.isChecked())
        self.settings.setValue("advanced/keep_history", self.keep_history_check.isChecked())
        self.settings.setValue("advanced/max_history", self.max_history_spin.value())
        self.settings.setValue("advanced/auto_load_plugins", self.auto_load_plugins_check.isChecked())
        self.settings.setValue("advanced/plugin_directory", self.plugin_dir_edit.text())

        # Accessibility
        self.settings.setValue("accessibility/colorblind_safe", self.colorblind_safe_check.isChecked())
        self.settings.setValue("accessibility/palette", self.cvd_palette_combo.currentText())
        self.settings.setValue("accessibility/scale_factor", self.ui_scale_spin.value())
        self.settings.setValue("accessibility/screen_reader_hints", self.screen_reader_hints_check.isChecked())
        self.settings.setValue("accessibility/sticky_keys", self.sticky_keys_check.isChecked())
        self.settings.setValue("accessibility/double_click_ms", self.double_click_speed_spin.value())
        self.settings.setValue("accessibility/drag_sensitivity", self.drag_sensitivity_spin.value())

        # Sync to disk
        self.settings.sync()

        AccessibilityManager.load_from_settings()
        AccessibilityManager.apply_ui_scale()
    
    def _restore_defaults(self):
        """Restore all settings to defaults"""
        self.settings.clear()
        self._load_settings()
    
    def _save_and_close(self):
        """Save settings and close dialog"""
        self._save_settings()
        self.accept()
    
    @staticmethod
    def get_setting(key: str, default: Any = None) -> Any:
        """
        Static method to get a setting value
        
        Args:
            key: Setting key (e.g., 'general/autosave_enabled')
            default: Default value if setting doesn't exist
            
        Returns:
            Setting value
        """
        settings = QSettings("NucleiSegApp", "EZQUANT")
        return settings.value(key, default)
    
    @staticmethod
    def set_setting(key: str, value: Any):
        """
        Static method to set a setting value
        
        Args:
            key: Setting key (e.g., 'general/autosave_enabled')
            value: Value to set
        """
        settings = QSettings("NucleiSegApp", "EZQUANT")
        settings.setValue(key, value)
        settings.sync()
