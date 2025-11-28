"""
Visualization panel for displaying measurement results with Plotly
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QComboBox, QLabel, QGroupBox, QCheckBox,
                               QScrollArea, QSplitter)
from PySide6.QtCore import Signal, Qt, QObject, Slot
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Optional, Dict, List


class Bridge(QObject):
    """Bridge between JavaScript and Python"""
    nucleusClicked = Signal(int)
    
    @Slot(int)
    def on_click(self, nucleus_id):
        self.nucleusClicked.emit(nucleus_id)


class VisualizationPanel(QWidget):
    """Panel for interactive data visualization with Plotly"""
    
    nucleus_selected = Signal(int)  # Emit nucleus ID on plot selection
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.measurements_df: Optional[pd.DataFrame] = None
        self.current_plot_type = "histogram"
        self.selected_nucleus_id = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI layout"""
        layout = QVBoxLayout(self)
        
        # Controls section
        controls_group = QGroupBox("Plot Controls")
        controls_layout = QHBoxLayout()
        
        # Plot type selector
        controls_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "DNA Histogram",
            "Scatter: Area vs DNA",
            "Box Plots: Morphology",
            "Scatter Matrix",
            "Correlation Heatmap",
            "Phase Distribution"
        ])
        self.plot_type_combo.currentTextChanged.connect(self._on_plot_type_changed)
        controls_layout.addWidget(self.plot_type_combo)
        
        # X-axis selector
        controls_layout.addWidget(QLabel("X-axis:"))
        self.x_axis_combo = QComboBox()
        controls_layout.addWidget(self.x_axis_combo)
        
        # Y-axis selector
        controls_layout.addWidget(QLabel("Y-axis:"))
        self.y_axis_combo = QComboBox()
        controls_layout.addWidget(self.y_axis_combo)
        
        # Color selector
        controls_layout.addWidget(QLabel("Color by:"))
        self.color_combo = QComboBox()
        self.color_combo.addItem("None")
        controls_layout.addWidget(self.color_combo)
        
        # Update button
        self.update_plot_btn = QPushButton("Update Plot")
        self.update_plot_btn.clicked.connect(self._update_plot)
        controls_layout.addWidget(self.update_plot_btn)
        
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Plot view
        self.plot_view = QWebEngineView()
        self.plot_view.setMinimumHeight(500)
        layout.addWidget(self.plot_view)
        
        # Setup WebChannel
        self.bridge = Bridge()
        self.bridge.nucleusClicked.connect(self.nucleus_selected)
        self.channel = QWebChannel()
        self.channel.registerObject("bridge", self.bridge)
        self.plot_view.page().setWebChannel(self.channel)
        
        # Export controls
        export_layout = QHBoxLayout()
        
        self.export_png_btn = QPushButton("Export PNG")
        self.export_png_btn.clicked.connect(lambda: self._export_plot("png"))
        export_layout.addWidget(self.export_png_btn)
        
        self.export_svg_btn = QPushButton("Export SVG")
        self.export_svg_btn.clicked.connect(lambda: self._export_plot("svg"))
        export_layout.addWidget(self.export_svg_btn)
        
        self.export_html_btn = QPushButton("Export HTML")
        self.export_html_btn.clicked.connect(lambda: self._export_plot("html"))
        export_layout.addWidget(self.export_html_btn)
        
        export_layout.addStretch()
        
        self.stats_label = QLabel("")
        export_layout.addWidget(self.stats_label)
        
        layout.addLayout(export_layout)
    
    def set_measurements(self, df: pd.DataFrame):
        """Set measurements DataFrame and update UI"""
        self.measurements_df = df
        
        if df is None or len(df) == 0:
            self.stats_label.setText("No data")
            return
        
        # Update stats
        n_nuclei = len(df)
        self.stats_label.setText(f"Total nuclei: {n_nuclei}")
        
        # Update axis selectors with column names
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()
        self.color_combo.clear()
        
        self.x_axis_combo.addItems(numeric_columns)
        self.y_axis_combo.addItems(numeric_columns)
        
        self.color_combo.addItem("None")
        self.color_combo.addItems(df.columns.tolist())
        
        # Generate default plot
        self._create_default_plots()
    
    def _on_plot_type_changed(self, plot_type: str):
        """Handle plot type change"""
        self.current_plot_type = plot_type
        self._update_plot()
    
    def _update_plot(self):
        """Update plot based on current settings"""
        if self.measurements_df is None or len(self.measurements_df) == 0:
            return
        
        plot_type = self.plot_type_combo.currentText()
        
        if plot_type == "DNA Histogram":
            self._create_dna_histogram()
        elif plot_type == "Scatter: Area vs DNA":
            self._create_area_dna_scatter()
        elif plot_type == "Box Plots: Morphology":
            self._create_morphology_boxplots()
        elif plot_type == "Scatter Matrix":
            self._create_scatter_matrix()
        elif plot_type == "Correlation Heatmap":
            self._create_correlation_heatmap()
        elif plot_type == "Phase Distribution":
            self._create_phase_distribution()
    
    def _create_default_plots(self):
        """Create default visualization"""
        self._create_dna_histogram()
    
    def _create_dna_histogram(self):
        """Create DNA intensity histogram"""
        df = self.measurements_df
        
        # Find DNA intensity column
        dna_col = None
        for col in df.columns:
            if 'dna' in col.lower() and 'intensity' in col.lower():
                dna_col = col
                break
        
        if dna_col is None:
            # Try mean_intensity
            intensity_cols = [col for col in df.columns if 'mean_intensity' in col.lower()]
            if intensity_cols:
                dna_col = intensity_cols[0]
        
        if dna_col is None:
            return
        
        # Create histogram
        fig = go.Figure()
        
        # Color by phase if available
        if 'phase' in df.columns:
            for phase in df['phase'].unique():
                phase_data = df[df['phase'] == phase][dna_col]
                fig.add_trace(go.Histogram(
                    x=phase_data,
                    name=phase,
                    opacity=0.7,
                    nbinsx=50
                ))
        else:
            fig.add_trace(go.Histogram(
                x=df[dna_col],
                nbinsx=50,
                marker_color='lightblue',
                opacity=0.7
            ))
        
        fig.update_layout(
            title="DNA Intensity Distribution",
            xaxis_title=dna_col,
            yaxis_title="Count",
            hovermode='x',
            height=500,
            showlegend='phase' in df.columns
        )
        
        # Add click callback
        html = self._add_plotly_callbacks(fig)
        self.plot_view.setHtml(html)
    
    def _create_area_dna_scatter(self):
        """Create scatter plot of area vs DNA intensity"""
        df = self.measurements_df
        
        # Find area and DNA columns
        area_col = 'area' if 'area' in df.columns else 'volume'
        
        dna_col = None
        for col in df.columns:
            if 'dna' in col.lower() and 'mean_intensity' in col.lower():
                dna_col = col
                break
        
        if dna_col is None:
            return
        
        # Color by phase if available
        color = 'phase' if 'phase' in df.columns else None
        
        fig = px.scatter(
            df,
            x=area_col,
            y=dna_col,
            color=color,
            hover_data=['nucleus_id'],
            title=f"{dna_col} vs {area_col}",
            labels={area_col: area_col, dna_col: dna_col}
        )
        
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(height=500)
        
        html = self._add_plotly_callbacks(fig)
        self.plot_view.setHtml(html)
    
    def _create_morphology_boxplots(self):
        """Create box plots for morphology measurements"""
        df = self.measurements_df
        
        # Select morphology columns
        morph_cols = []
        for col in ['area', 'perimeter', 'circularity', 'eccentricity', 
                    'solidity', 'aspect_ratio']:
            if col in df.columns:
                morph_cols.append(col)
        
        if not morph_cols:
            return
        
        # Create subplots
        n_cols = min(3, len(morph_cols))
        n_rows = (len(morph_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=morph_cols
        )
        
        for i, col in enumerate(morph_cols):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            if 'phase' in df.columns:
                for phase in df['phase'].unique():
                    phase_data = df[df['phase'] == phase][col]
                    fig.add_trace(
                        go.Box(y=phase_data, name=f"{col}_{phase}"),
                        row=row, col=col_idx
                    )
            else:
                fig.add_trace(
                    go.Box(y=df[col], name=col),
                    row=row, col=col_idx
                )
        
        fig.update_layout(
            title="Morphology Distributions",
            showlegend=False,
            height=300 * n_rows
        )
        
        html = fig.to_html(include_plotlyjs='cdn')
        self.plot_view.setHtml(html)
    
    def _create_scatter_matrix(self):
        """Create scatter matrix for key measurements"""
        df = self.measurements_df
        
        # Select key columns
        key_cols = []
        for col in ['area', 'circularity', 'eccentricity']:
            if col in df.columns:
                key_cols.append(col)
        
        # Add DNA intensity
        for col in df.columns:
            if 'dna' in col.lower() and 'mean_intensity' in col.lower():
                key_cols.append(col)
                break
        
        if len(key_cols) < 2:
            return
        
        color = 'phase' if 'phase' in df.columns else None
        
        fig = px.scatter_matrix(
            df,
            dimensions=key_cols,
            color=color,
            title="Scatter Matrix of Key Measurements"
        )
        
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        fig.update_layout(height=800)
        
        html = fig.to_html(include_plotlyjs='cdn')
        self.plot_view.setHtml(html)
    
    def _create_correlation_heatmap(self):
        """Create correlation heatmap"""
        df = self.measurements_df
        
        # Get numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Remove nucleus_id
        if 'nucleus_id' in numeric_df.columns:
            numeric_df = numeric_df.drop('nucleus_id', axis=1)
        
        # Calculate correlation
        corr = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 8}
        ))
        
        fig.update_layout(
            title="Correlation Heatmap",
            height=600,
            width=600
        )
        
        html = fig.to_html(include_plotlyjs='cdn')
        self.plot_view.setHtml(html)
    
    def _create_phase_distribution(self):
        """Create phase distribution plot"""
        df = self.measurements_df
        
        if 'phase' not in df.columns:
            return
        
        # Count phases
        phase_counts = df['phase'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(x=phase_counts.index, y=phase_counts.values)
        ])
        
        fig.update_layout(
            title="Cell Cycle Phase Distribution",
            xaxis_title="Phase",
            yaxis_title="Count",
            height=400
        )
        
        html = fig.to_html(include_plotlyjs='cdn')
        self.plot_view.setHtml(html)
    
    def _add_plotly_callbacks(self, fig: go.Figure) -> str:
        """Add JavaScript callbacks for bidirectional linking"""
        html = fig.to_html(include_plotlyjs='cdn')
        
        # Add click event handler
        callback_js = """
        <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
        <script>
        new QWebChannel(qt.webChannelTransport, function(channel) {
            window.bridge = channel.objects.bridge;
        });

        var plot = document.getElementsByClassName('plotly')[0];
        plot.on('plotly_click', function(data){
            if (data.points.length > 0) {
                var point = data.points[0];
                // Try to get nucleus_id from customdata or pointIndex
                var nucleus_id = point.customdata ? point.customdata[0] : point.pointIndex + 1;
                console.log('Clicked nucleus:', nucleus_id);
                
                if (window.bridge) {
                    window.bridge.on_click(nucleus_id);
                }
            }
        });
        </script>
        """
        
        html = html.replace('</body>', callback_js + '</body>')
        return html
    
    def _export_plot(self, format: str):
        """Export current plot"""
        # This would trigger a file save dialog and export
        # Placeholder for now
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Export",
            f"Export to {format.upper()} not yet implemented.\n"
            f"Use browser right-click -> Save Image for now."
        )
    
    def highlight_nucleus(self, nucleus_id: int):
        """Highlight a specific nucleus in the plot"""
        self.selected_nucleus_id = nucleus_id
        # Would update plot to highlight the selected nucleus
        # This requires re-rendering with modified marker sizes/colors
