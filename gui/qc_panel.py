"""
Quality control panel for reviewing segmentation quality
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QScrollArea, QGridLayout, QGroupBox,
                               QSpinBox, QDoubleSpinBox, QCheckBox, QMessageBox,
                               QDialog, QDialogButtonBox, QTextEdit)
from PySide6.QtCore import Signal, Qt
from PySide6.QtWebEngineWidgets import QWebEngineView
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json


class QCPanel(QWidget):
    """Quality control panel for DNA intensity analysis"""
    
    nucleus_selected = Signal(int)  # Emit nucleus ID on selection
    request_resegmentation = Signal(dict)  # Emit suggested parameters
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.qc_results = None
        self.confirmed_errors = set()  # Track confirmed error nucleus IDs
        self.current_params = {}
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI layout"""
        layout = QVBoxLayout(self)
        
        # Controls section
        controls_group = QGroupBox("Analysis Controls")
        controls_layout = QHBoxLayout()
        
        # Number of phases
        controls_layout.addWidget(QLabel("Cell Cycle Phases:"))
        self.phases_spin = QSpinBox()
        self.phases_spin.setRange(2, 5)
        self.phases_spin.setValue(3)
        self.phases_spin.setToolTip("Number of cell cycle phases to detect")
        controls_layout.addWidget(self.phases_spin)
        
        # Percentile thresholds
        controls_layout.addWidget(QLabel("Outlier Percentiles:"))
        self.low_percentile_spin = QDoubleSpinBox()
        self.low_percentile_spin.setRange(0.1, 10.0)
        self.low_percentile_spin.setValue(5.0)
        self.low_percentile_spin.setSuffix("%")
        controls_layout.addWidget(self.low_percentile_spin)
        
        controls_layout.addWidget(QLabel("-"))
        
        self.high_percentile_spin = QDoubleSpinBox()
        self.high_percentile_spin.setRange(90.0, 99.9)
        self.high_percentile_spin.setValue(95.0)
        self.high_percentile_spin.setSuffix("%")
        controls_layout.addWidget(self.high_percentile_spin)
        
        # Run QC button
        self.run_qc_btn = QPushButton("Run QC Analysis")
        self.run_qc_btn.clicked.connect(self._on_run_qc_clicked)
        controls_layout.addWidget(self.run_qc_btn)
        
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Plot view
        self.plot_view = QWebEngineView()
        self.plot_view.setMinimumHeight(400)
        layout.addWidget(self.plot_view, stretch=2)
        
        # Flagged nuclei section
        flagged_group = QGroupBox("Flagged Nuclei")
        flagged_layout = QVBoxLayout()
        
        # Info label
        self.flagged_info_label = QLabel("No QC analysis run yet")
        flagged_layout.addWidget(self.flagged_info_label)
        
        # Scroll area for flagged nuclei
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        
        self.flagged_container = QWidget()
        self.flagged_grid = QGridLayout(self.flagged_container)
        scroll.setWidget(self.flagged_container)
        
        flagged_layout.addWidget(scroll)
        
        # Action buttons
        actions_layout = QHBoxLayout()
        
        self.clear_confirmations_btn = QPushButton("Clear Confirmations")
        self.clear_confirmations_btn.clicked.connect(self._clear_confirmations)
        actions_layout.addWidget(self.clear_confirmations_btn)
        
        self.suggest_params_btn = QPushButton("Get Parameter Suggestions")
        self.suggest_params_btn.clicked.connect(self._suggest_parameters)
        actions_layout.addWidget(self.suggest_params_btn)
        
        actions_layout.addStretch()
        flagged_layout.addLayout(actions_layout)
        
        flagged_group.setLayout(flagged_layout)
        layout.addWidget(flagged_group, stretch=1)
    
    def set_qc_results(self, results: dict, current_params: dict):
        """Display QC results"""
        self.qc_results = results
        self.current_params = current_params
        self.confirmed_errors.clear()
        
        # Update info label
        n_total = results.get('nucleus_count', 0)
        n_flagged = results.get('flagged_count', 0)
        pct_flagged = results.get('flagged_percentage', 0)
        
        self.flagged_info_label.setText(
            f"Total nuclei: {n_total} | Flagged: {n_flagged} ({pct_flagged:.1f}%)"
        )
        
        # Create histogram plot
        self._create_histogram_plot(results)
        
        # Display flagged nuclei
        self._display_flagged_nuclei(results.get('flagged_nuclei', []))
    
    def _create_histogram_plot(self, results: dict):
        """Create interactive histogram with phase boundaries"""
        intensities = results.get('intensities', np.array([]))
        
        if len(intensities) == 0:
            return
        
        # Create figure
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=intensities,
            nbinsx=50,
            name='DNA Intensity',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Add phase boundaries
        boundaries = results.get('phase_boundaries', [])
        for i, boundary in enumerate(boundaries):
            fig.add_vline(
                x=boundary,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Phase {i+1}|{i+2}",
                annotation_position="top"
            )
        
        # Add threshold lines
        low_threshold = results.get('low_threshold', 0)
        high_threshold = results.get('high_threshold', 0)
        
        fig.add_vline(
            x=low_threshold,
            line_dash="dot",
            line_color="orange",
            annotation_text="Low",
            annotation_position="bottom left"
        )
        
        fig.add_vline(
            x=high_threshold,
            line_dash="dot",
            line_color="orange",
            annotation_text="High",
            annotation_position="bottom right"
        )
        
        # Layout
        fig.update_layout(
            title="DNA Intensity Distribution",
            xaxis_title="Mean DNA Intensity",
            yaxis_title="Count",
            hovermode='x',
            showlegend=True,
            height=400
        )
        
        # Convert to HTML and display
        html = fig.to_html(include_plotlyjs='cdn')
        self.plot_view.setHtml(html)
    
    def _display_flagged_nuclei(self, flagged: list):
        """Display grid of flagged nuclei"""
        # Clear previous widgets
        for i in reversed(range(self.flagged_grid.count())):
            widget = self.flagged_grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        if not flagged:
            label = QLabel("No nuclei flagged")
            self.flagged_grid.addWidget(label, 0, 0)
            return
        
        # Create entry for each flagged nucleus
        for idx, nucleus_info in enumerate(flagged):
            nuc_id = nucleus_info['nucleus_id']
            intensity = nucleus_info['intensity']
            reason = nucleus_info['reason']
            z_score = nucleus_info['z_score']
            
            # Container widget
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(5, 5, 5, 5)
            
            # Info label
            info_text = f"ID: {nuc_id}\n"
            info_text += f"Intensity: {intensity:.1f}\n"
            info_text += f"Z-score: {z_score:.2f}\n"
            info_text += f"Reason: {reason.replace('_', ' ')}"
            
            info_label = QLabel(info_text)
            container_layout.addWidget(info_label)
            
            # Buttons
            btn_layout = QHBoxLayout()
            
            view_btn = QPushButton("View")
            view_btn.clicked.connect(lambda checked, nid=nuc_id: self._on_view_nucleus(nid))
            btn_layout.addWidget(view_btn)
            
            confirm_btn = QPushButton("Confirm Error")
            confirm_btn.setCheckable(True)
            confirm_btn.clicked.connect(lambda checked, nid=nuc_id: self._on_confirm_error(nid, checked))
            btn_layout.addWidget(confirm_btn)
            
            container_layout.addLayout(btn_layout)
            
            # Add to grid (3 columns)
            row = idx // 3
            col = idx % 3
            self.flagged_grid.addWidget(container, row, col)
    
    def _on_run_qc_clicked(self):
        """Handle run QC button click"""
        # This would trigger the QC analysis in the main window
        # For now, just a placeholder
        QMessageBox.information(
            self,
            "QC Analysis",
            "QC analysis should be triggered from segmentation results"
        )
    
    def _on_view_nucleus(self, nucleus_id: int):
        """Emit signal to view specific nucleus"""
        self.nucleus_selected.emit(nucleus_id)
    
    def _on_confirm_error(self, nucleus_id: int, confirmed: bool):
        """Track confirmed segmentation errors"""
        if confirmed:
            self.confirmed_errors.add(nucleus_id)
        else:
            self.confirmed_errors.discard(nucleus_id)
    
    def _clear_confirmations(self):
        """Clear all confirmed errors"""
        self.confirmed_errors.clear()
        
        # Reset all confirm buttons
        for i in range(self.flagged_grid.count()):
            container = self.flagged_grid.itemAt(i).widget()
            if container:
                # Find the confirm button
                for child in container.findChildren(QPushButton):
                    if child.text() == "Confirm Error":
                        child.setChecked(False)
    
    def _suggest_parameters(self):
        """Show parameter suggestion dialog"""
        if not self.qc_results:
            QMessageBox.warning(self, "No Results", "Run QC analysis first")
            return
        
        if not self.confirmed_errors:
            QMessageBox.information(
                self,
                "No Errors Confirmed",
                "Please confirm segmentation errors before requesting suggestions"
            )
            return
        
        # Calculate error rate
        n_total = self.qc_results.get('nucleus_count', 0)
        if n_total == 0:
            return
        
        error_rate = (len(self.confirmed_errors) / n_total) * 100
        
        # Show dialog
        dialog = ParameterSuggestionDialog(
            self.qc_results,
            self.current_params,
            error_rate,
            self
        )
        
        if dialog.exec() == QDialog.Accepted:
            # Emit signal with suggested parameters
            suggested = dialog.get_suggested_parameters()
            if suggested:
                self.request_resegmentation.emit(suggested)


class ParameterSuggestionDialog(QDialog):
    """Dialog showing parameter suggestions"""
    
    def __init__(self, qc_results: dict, current_params: dict, 
                 error_rate: float, parent=None):
        super().__init__(parent)
        self.qc_results = qc_results
        self.current_params = current_params
        self.error_rate = error_rate
        self.suggested_params = {}
        
        self.setWindowTitle("Parameter Suggestions")
        self.setMinimumWidth(500)
        
        self._setup_ui()
        self._generate_suggestions()
    
    def _setup_ui(self):
        """Setup dialog UI"""
        layout = QVBoxLayout(self)
        
        # Error rate info
        info_label = QLabel(
            f"<b>Confirmed Error Rate: {self.error_rate:.1f}%</b><br>"
            f"({len(self.current_params)} confirmed errors)"
        )
        layout.addWidget(info_label)
        
        # Suggestions text
        self.suggestions_text = QTextEdit()
        self.suggestions_text.setReadOnly(True)
        layout.addWidget(self.suggestions_text)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _generate_suggestions(self):
        """Generate parameter suggestions"""
        from core.quality_control import QualityControl
        
        qc = QualityControl()
        suggestions = qc.suggest_parameters(
            self.qc_results,
            self.current_params,
            self.error_rate
        )
        
        # Format suggestions text
        text = "<h3>Analysis Results</h3>"
        
        if suggestions['should_rerun']:
            text += "<p><b>Recommendation:</b> Re-run segmentation with adjusted parameters</p>"
            text += "<h4>Suggested Changes:</h4><ul>"
            
            for change in suggestions['changes']:
                text += f"<li>{change}</li>"
            
            text += "</ul>"
            
            text += "<h4>Current vs. Suggested Parameters:</h4>"
            text += "<table border='1' cellpadding='5'>"
            text += "<tr><th>Parameter</th><th>Current</th><th>Suggested</th></tr>"
            
            for key in ['diameter', 'flow_threshold', 'cellprob_threshold']:
                if key in self.current_params:
                    current = self.current_params[key]
                    suggested = suggestions['new_params'].get(key, current)
                    
                    # Highlight changed values
                    style = ' style="background-color: yellow;"' if current != suggested else ''
                    text += f"<tr{style}>"
                    text += f"<td><b>{key}</b></td>"
                    text += f"<td>{current}</td>"
                    text += f"<td>{suggested}</td>"
                    text += "</tr>"
            
            text += "</table>"
            
            self.suggested_params = suggestions['new_params']
        else:
            text += "<p>Error rate is below 5% threshold. No parameter changes recommended.</p>"
        
        self.suggestions_text.setHtml(text)
    
    def get_suggested_parameters(self) -> dict:
        """Return suggested parameters"""
        return self.suggested_params
