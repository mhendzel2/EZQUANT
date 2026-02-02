"""
Quality dashboard for project-wide metrics and QC summary
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
                               QGroupBox, QSplitter, QTextEdit)
from PySide6.QtCore import Signal, Qt
from PySide6.QtWebEngineWidgets import QWebEngineView
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, List
from datetime import datetime


class QualityDashboard(QWidget):
    """Dashboard showing project-wide quality metrics"""
    
    image_selected = Signal(str)  # Emit image name for viewing
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_data: List[Dict] = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup dashboard UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        title_label = QLabel("<h2>Quality Control Dashboard</h2>")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_data)
        header_layout.addWidget(self.refresh_btn)
        
        self.export_report_btn = QPushButton("Export Report")
        self.export_report_btn.clicked.connect(self._export_report)
        header_layout.addWidget(self.export_report_btn)
        
        layout.addLayout(header_layout)
        
        # Splitter for metrics and details
        splitter = QSplitter(Qt.Vertical)
        
        # === Summary Metrics ===
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        
        # Overall stats
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f0f0f0;")
        metrics_layout.addWidget(self.summary_label)
        
        # Trend plots
        self.trend_plot = QWebEngineView()
        self.trend_plot.setMinimumHeight(300)
        metrics_layout.addWidget(self.trend_plot)
        
        # === Image-level Details ===
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        
        details_layout.addWidget(QLabel("<b>Per-Image Quality Metrics</b>"))
        
        self.details_table = QTableWidget()
        self.details_table.setAlternatingRowColors(True)
        self.details_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.details_table.cellDoubleClicked.connect(self._on_image_double_clicked)
        details_layout.addWidget(self.details_table)
        
        # Add to splitter
        splitter.addWidget(metrics_widget)
        splitter.addWidget(details_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
    
    def set_project_data(self, project_data: List[Dict]):
        """
        Set project data for dashboard
        
        Args:
            project_data: List of dicts with keys:
                - image_name: str
                - n_nuclei: int
                - mean_area: float
                - cv_area: float
                - mean_dna_intensity: float
                - cv_dna_intensity: float
                - flagged_count: int
                - flagged_percentage: float
                - qc_pass: bool
                - timestamp: datetime
        """
        self.project_data = project_data
        self._update_display()
    
    def refresh_data(self):
        """Signal to refresh data from project - emits to main window"""
        # This signal is connected by the main window to trigger data refresh
        # The actual refresh logic is in the main window's _refresh_dashboard_data method
        # For standalone use, we can also re-run update display with existing data
        if self.project_data:
            self._update_display()
    
    def _update_display(self):
        """Update all dashboard components"""
        if not self.project_data:
            self.summary_label.setText("<i>No data available</i>")
            self.details_table.setRowCount(0)
            return
        
        # Update summary
        self._update_summary()
        
        # Update trend plots
        self._update_trend_plots()
        
        # Update details table
        self._update_details_table()
    
    def _update_summary(self):
        """Update summary statistics"""
        total_images = len(self.project_data)
        total_nuclei = sum(d.get('n_nuclei', 0) for d in self.project_data)
        
        qc_pass_count = sum(1 for d in self.project_data if d.get('qc_pass', False))
        qc_pass_rate = (qc_pass_count / total_images * 100) if total_images > 0 else 0
        
        mean_flagged_pct = np.mean([d.get('flagged_percentage', 0) for d in self.project_data])
        
        # Calculate overall CV
        all_areas = []
        all_dna = []
        for d in self.project_data:
            if 'mean_area' in d:
                all_areas.append(d['mean_area'])
            if 'mean_dna_intensity' in d:
                all_dna.append(d['mean_dna_intensity'])
        
        overall_area_cv = (np.std(all_areas) / np.mean(all_areas) * 100) if all_areas else 0
        overall_dna_cv = (np.std(all_dna) / np.mean(all_dna) * 100) if all_dna else 0
        
        summary_html = f"""
        <table border='0' cellpadding='5'>
        <tr>
            <td><b>Total Images:</b></td><td>{total_images}</td>
            <td style='padding-left: 20px'><b>Total Nuclei:</b></td><td>{total_nuclei}</td>
        </tr>
        <tr>
            <td><b>QC Pass Rate:</b></td><td style='color: {"green" if qc_pass_rate >= 80 else "orange" if qc_pass_rate >= 60 else "red"}'><b>{qc_pass_rate:.1f}%</b></td>
            <td style='padding-left: 20px'><b>Mean Flagged:</b></td><td>{mean_flagged_pct:.1f}%</td>
        </tr>
        <tr>
            <td><b>Area CV:</b></td><td>{overall_area_cv:.1f}%</td>
            <td style='padding-left: 20px'><b>DNA Intensity CV:</b></td><td>{overall_dna_cv:.1f}%</td>
        </tr>
        </table>
        """
        
        self.summary_label.setText(summary_html)
    
    def _update_trend_plots(self):
        """Update trend visualization"""
        if not self.project_data:
            return
        
        # Sort by timestamp if available
        sorted_data = sorted(self.project_data, 
                           key=lambda x: x.get('timestamp', datetime.now()))
        
        image_names = [d['image_name'] for d in sorted_data]
        n_nuclei = [d.get('n_nuclei', 0) for d in sorted_data]
        flagged_pct = [d.get('flagged_percentage', 0) for d in sorted_data]
        qc_pass = [d.get('qc_pass', False) for d in sorted_data]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Nucleus Count per Image', 'QC Flagged Percentage'),
            vertical_spacing=0.15
        )
        
        # Nucleus count
        colors = ['green' if p else 'red' for p in qc_pass]
        fig.add_trace(
            go.Bar(x=image_names, y=n_nuclei, marker_color=colors, name='Nucleus Count'),
            row=1, col=1
        )
        
        # Flagged percentage with threshold line
        fig.add_trace(
            go.Scatter(x=image_names, y=flagged_pct, mode='lines+markers',
                      name='Flagged %', marker=dict(size=8)),
            row=2, col=1
        )
        
        # Add 10% threshold line
        fig.add_hline(y=10, line_dash="dash", line_color="red", 
                     annotation_text="10% threshold",
                     row=2, col=1)
        
        fig.update_xaxes(title_text="Image", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Flagged %", row=2, col=1)
        
        fig.update_layout(
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        html = fig.to_html(include_plotlyjs='cdn')
        self.trend_plot.setHtml(html)
    
    def _update_details_table(self):
        """Update detailed metrics table"""
        self.details_table.setRowCount(len(self.project_data))
        self.details_table.setColumnCount(8)
        self.details_table.setHorizontalHeaderLabels([
            'Image', 'Nuclei', 'Mean Area', 'Area CV%', 
            'Mean DNA', 'DNA CV%', 'Flagged %', 'QC Status'
        ])
        
        for i, data in enumerate(self.project_data):
            # Image name
            name_item = QTableWidgetItem(data['image_name'])
            self.details_table.setItem(i, 0, name_item)
            
            # Nuclei count
            nuclei_item = QTableWidgetItem(str(data.get('n_nuclei', 0)))
            self.details_table.setItem(i, 1, nuclei_item)
            
            # Mean area
            area_item = QTableWidgetItem(f"{data.get('mean_area', 0):.1f}")
            self.details_table.setItem(i, 2, area_item)
            
            # Area CV
            area_cv_item = QTableWidgetItem(f"{data.get('cv_area', 0):.1f}")
            self.details_table.setItem(i, 3, area_cv_item)
            
            # Mean DNA
            dna_item = QTableWidgetItem(f"{data.get('mean_dna_intensity', 0):.1f}")
            self.details_table.setItem(i, 4, dna_item)
            
            # DNA CV
            dna_cv_item = QTableWidgetItem(f"{data.get('cv_dna_intensity', 0):.1f}")
            self.details_table.setItem(i, 5, dna_cv_item)
            
            # Flagged percentage
            flagged_pct = data.get('flagged_percentage', 0)
            flagged_item = QTableWidgetItem(f"{flagged_pct:.1f}")
            if flagged_pct > 10:
                flagged_item.setForeground(Qt.red)
            self.details_table.setItem(i, 6, flagged_item)
            
            # QC status
            qc_pass = data.get('qc_pass', False)
            status_item = QTableWidgetItem('PASS' if qc_pass else 'FAIL')
            status_item.setForeground(Qt.green if qc_pass else Qt.red)
            self.details_table.setItem(i, 7, status_item)
    
    def _on_image_double_clicked(self, row: int, col: int):
        """Handle double-click on image row"""
        if row < len(self.project_data):
            image_name = self.project_data[row]['image_name']
            self.image_selected.emit(image_name)
    
    def _export_report(self):
        """Export QC report to HTML"""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        
        if not self.project_data:
            QMessageBox.warning(self, "No Data", "No data to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export QC Report", "", "HTML Files (*.html)"
        )
        
        if file_path:
            try:
                self._generate_html_report(file_path)
                QMessageBox.information(self, "Success", f"Report exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {e}")
    
    def _generate_html_report(self, file_path: str):
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>QC Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .pass {{ color: green; font-weight: bold; }}
                .fail {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Quality Control Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary Statistics</h2>
            {self.summary_label.text()}
            
            <h2>Per-Image Details</h2>
            <table>
                <tr>
                    <th>Image</th><th>Nuclei</th><th>Mean Area</th><th>Area CV%</th>
                    <th>Mean DNA</th><th>DNA CV%</th><th>Flagged %</th><th>Status</th>
                </tr>
        """
        
        for data in self.project_data:
            qc_pass = data.get('qc_pass', False)
            status_class = 'pass' if qc_pass else 'fail'
            status_text = 'PASS' if qc_pass else 'FAIL'
            
            html += f"""
                <tr>
                    <td>{data['image_name']}</td>
                    <td>{data.get('n_nuclei', 0)}</td>
                    <td>{data.get('mean_area', 0):.1f}</td>
                    <td>{data.get('cv_area', 0):.1f}</td>
                    <td>{data.get('mean_dna_intensity', 0):.1f}</td>
                    <td>{data.get('cv_dna_intensity', 0):.1f}</td>
                    <td>{data.get('flagged_percentage', 0):.1f}</td>
                    <td class='{status_class}'>{status_text}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        with open(file_path, 'w') as f:
            f.write(html)
