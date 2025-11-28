"""
Analysis panel for running measurements and managing analysis workflow
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QGroupBox, QCheckBox, QRadioButton,
                               QTableWidget, QTableWidgetItem, QHeaderView,
                               QProgressBar, QTextEdit, QSplitter, QFileDialog,
                               QButtonGroup, QListWidget, QListWidgetItem)
from PySide6.QtCore import Signal, Qt
import pandas as pd
import numpy as np
from typing import Optional, Dict, List


class AnalysisPanel(QWidget):
    """Panel for measurement configuration and execution"""
    
    run_measurements = Signal(dict)  # Emit configuration dict
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.measurements_df: Optional[pd.DataFrame] = None
        self.plugin_info: List[Dict] = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI layout"""
        # Main splitter for configuration and results
        splitter = QSplitter(Qt.Vertical)
        
        # === Configuration Section ===
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        
        # Workflow selection
        workflow_group = QGroupBox("Analysis Workflow")
        workflow_layout = QHBoxLayout()
        
        self.workflow_2d_radio = QRadioButton("2D Analysis")
        self.workflow_2d_radio.setChecked(True)
        self.workflow_2d_radio.setToolTip("Analyze each Z-slice independently")
        workflow_layout.addWidget(self.workflow_2d_radio)
        
        self.workflow_3d_radio = QRadioButton("3D Analysis")
        self.workflow_3d_radio.setToolTip("Analyze entire 3D volume")
        workflow_layout.addWidget(self.workflow_3d_radio)
        
        self.pool_check = QCheckBox("Pool all images by Group")
        self.pool_check.setToolTip("Combine measurements from all images in the project, grouped by their assigned group")
        workflow_layout.addWidget(self.pool_check)
        
        workflow_layout.addStretch()
        workflow_group.setLayout(workflow_layout)
        config_layout.addWidget(workflow_group)
        
        # Measurement categories
        categories_group = QGroupBox("Measurement Categories")
        categories_layout = QVBoxLayout()
        
        self.basic_shape_check = QCheckBox("Basic Shape (area, perimeter, circularity)")
        self.basic_shape_check.setChecked(True)
        categories_layout.addWidget(self.basic_shape_check)
        
        self.advanced_morph_check = QCheckBox("Advanced Morphology (eccentricity, solidity, axes)")
        self.advanced_morph_check.setChecked(True)
        categories_layout.addWidget(self.advanced_morph_check)
        
        self.intensity_check = QCheckBox("Intensity Statistics (mean, min, max, std, CV)")
        self.intensity_check.setChecked(True)
        categories_layout.addWidget(self.intensity_check)
        
        self.cell_cycle_check = QCheckBox("Cell Cycle Phase Assignment (DNA-based)")
        self.cell_cycle_check.setChecked(False)
        self.cell_cycle_check.setToolTip("Assign G1/S/G2M phases based on DNA intensity")
        categories_layout.addWidget(self.cell_cycle_check)
        
        categories_group.setLayout(categories_layout)
        config_layout.addWidget(categories_group)
        
        # Plugin selection
        plugins_group = QGroupBox("Custom Measurement Plugins")
        plugins_layout = QVBoxLayout()
        
        self.plugin_list = QListWidget()
        self.plugin_list.setMaximumHeight(100)
        self.plugin_list.setSelectionMode(QListWidget.MultiSelection)
        plugins_layout.addWidget(self.plugin_list)
        
        plugin_btn_layout = QHBoxLayout()
        
        self.refresh_plugins_btn = QPushButton("Refresh Plugins")
        self.refresh_plugins_btn.clicked.connect(self._refresh_plugins)
        plugin_btn_layout.addWidget(self.refresh_plugins_btn)
        
        self.manage_plugins_btn = QPushButton("Manage Plugins...")
        plugin_btn_layout.addWidget(self.manage_plugins_btn)
        
        plugin_btn_layout.addStretch()
        plugins_layout.addLayout(plugin_btn_layout)
        
        plugins_group.setLayout(plugins_layout)
        config_layout.addWidget(plugins_group)
        
        # Run button and progress
        run_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("Run Measurements")
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.run_btn.setMinimumHeight(40)
        run_layout.addWidget(self.run_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        run_layout.addWidget(self.progress_bar)
        
        config_layout.addLayout(run_layout)
        
        # === Results Section ===
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        
        # Results header
        results_header_layout = QHBoxLayout()
        results_header_layout.addWidget(QLabel("<b>Measurement Results</b>"))
        results_header_layout.addStretch()
        
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.clicked.connect(self._export_csv)
        results_header_layout.addWidget(self.export_csv_btn)
        
        self.export_excel_btn = QPushButton("Export Excel")
        self.export_excel_btn.clicked.connect(self._export_excel)
        results_header_layout.addWidget(self.export_excel_btn)
        
        results_layout.addLayout(results_header_layout)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        results_layout.addWidget(self.results_table)
        
        # Stats summary
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(100)
        results_layout.addWidget(self.stats_text)
        
        # Add widgets to splitter
        splitter.addWidget(config_widget)
        splitter.addWidget(results_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(splitter)
    
    def _refresh_plugins(self):
        """Refresh plugin list"""
        # This would call plugin loader to reload
        # For now, placeholder
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Refresh Plugins",
            "Plugin refresh functionality will be connected to PluginLoader"
        )
    
    def set_plugin_info(self, plugins: List[Dict]):
        """Set available plugins"""
        self.plugin_info = plugins
        
        self.plugin_list.clear()
        for plugin in plugins:
            item = QListWidgetItem(f"{plugin['name']} - {plugin['description']}")
            item.setData(Qt.UserRole, plugin)
            self.plugin_list.addItem(item)
    
    def _on_run_clicked(self):
        """Handle run measurements button click"""
        # Gather configuration
        config = {
            'is_3d': self.workflow_3d_radio.isChecked(),
            'enabled_categories': [],
            'enabled_plugins': [],
            'assign_phases': self.cell_cycle_check.isChecked()
        }
        
        # Gather enabled categories
        if self.basic_shape_check.isChecked():
            config['enabled_categories'].append('basic_shape')
        if self.advanced_morph_check.isChecked():
            config['enabled_categories'].append('advanced_morphology')
        if self.intensity_check.isChecked():
            config['enabled_categories'].append('intensity_stats')
        if self.cell_cycle_check.isChecked():
            config['enabled_categories'].append('cell_cycle')
        
        # Gather selected plugins
        for item in self.plugin_list.selectedItems():
            plugin_data = item.data(Qt.UserRole)
            if plugin_data:
                config['enabled_plugins'].append(plugin_data['file'])
        
        # Emit signal
        self.run_measurements.emit(config)
    
    def set_measurements(self, df: pd.DataFrame):
        """Display measurement results"""
        self.measurements_df = df
        
        if df is None or len(df) == 0:
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            self.stats_text.setHtml("<i>No measurements</i>")
            return
        
        # Populate table
        self.results_table.setRowCount(len(df))
        self.results_table.setColumnCount(len(df.columns))
        self.results_table.setHorizontalHeaderLabels(df.columns.tolist())
        
        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                value = df.iloc[i, j]
                
                # Format value
                if isinstance(value, (int, np.integer)):
                    text = str(value)
                elif isinstance(value, (float, np.floating)):
                    text = f"{value:.3f}"
                else:
                    text = str(value)
                
                item = QTableWidgetItem(text)
                self.results_table.setItem(i, j, item)
        
        # Update statistics summary
        self._update_stats_summary(df)
    
    def _update_stats_summary(self, df: pd.DataFrame):
        """Update statistics summary text"""
        html = "<h4>Summary Statistics</h4>"
        html += f"<p><b>Total nuclei:</b> {len(df)}</p>"
        
        # Phase distribution if available
        if 'phase' in df.columns:
            html += "<p><b>Cell Cycle Phases:</b><ul>"
            phase_counts = df['phase'].value_counts()
            for phase, count in phase_counts.items():
                pct = count / len(df) * 100
                html += f"<li>{phase}: {count} ({pct:.1f}%)</li>"
            html += "</ul></p>"
        
        # Key measurement stats
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            html += "<p><b>Key Measurements:</b><table border='1' cellpadding='3'>"
            html += "<tr><th>Measurement</th><th>Mean ± SD</th><th>Range</th></tr>"
            
            # Select a few key columns
            key_cols = []
            for col in ['area', 'circularity', 'eccentricity']:
                if col in numeric_cols:
                    key_cols.append(col)
            
            for col in numeric_cols:
                if 'dna' in col.lower() and 'mean_intensity' in col.lower():
                    key_cols.append(col)
                    break
            
            for col in key_cols[:5]:  # Limit to 5 rows
                mean = df[col].mean()
                std = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
                
                html += f"<tr><td><b>{col}</b></td>"
                html += f"<td>{mean:.2f} ± {std:.2f}</td>"
                html += f"<td>{min_val:.2f} - {max_val:.2f}</td></tr>"
            
            html += "</table></p>"
        
        self.stats_text.setHtml(html)
    
    def _export_csv(self):
        """Export measurements to CSV"""
        if self.measurements_df is None or len(self.measurements_df) == 0:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Data", "No measurements to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Measurements",
            "",
            "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                self.measurements_df.to_csv(file_path, index=False)
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Success", f"Exported to {file_path}")
            except Exception as e:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Export failed: {e}")
    
    def _export_excel(self):
        """Export measurements to Excel"""
        if self.measurements_df is None or len(self.measurements_df) == 0:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Data", "No measurements to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Measurements",
            "",
            "Excel Files (*.xlsx)"
        )
        
        if file_path:
            try:
                self.measurements_df.to_excel(file_path, index=False, engine='openpyxl')
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Success", f"Exported to {file_path}")
            except Exception as e:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Export failed: {e}")
    
    def set_progress(self, value: int):
        """Update progress bar"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(value)
        
        if value >= 100:
            self.progress_bar.setVisible(False)
