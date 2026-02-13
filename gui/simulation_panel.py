"""
Simulation Panel for GUI integration.

Provides a user interface for running FRAP/SPT simulations and parameter inference.
"""

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
        QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
        QComboBox, QTextEdit, QTabWidget, QFormLayout
    )
    from PySide6.QtCore import Signal, QThread
except ImportError:
    # Fallback if PySide6 not available
    print("PySide6 not available - simulation panel GUI disabled")
    QWidget = object

import numpy as np
from typing import Optional

from core.simulation import (
    Particle3DSimulator,
    DiffusionParameters,
    CompartmentGeometry,
    FRAPSimulator,
    FRAPParameters,
    SPTSimulator,
    SPTParameters
)


class SimulationWorker(QThread):
    """Worker thread for running simulations."""
    
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, sim_type, params):
        super().__init__()
        self.sim_type = sim_type
        self.params = params
    
    def run(self):
        """Run simulation in background thread."""
        try:
            self.progress.emit("Initializing simulation...")
            
            if self.sim_type == 'frap':
                result = self._run_frap()
            elif self.sim_type == 'spt':
                result = self._run_spt()
            elif self.sim_type == 'diffusion':
                result = self._run_diffusion()
            else:
                raise ValueError(f"Unknown simulation type: {self.sim_type}")
            
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def _run_frap(self):
        """Run FRAP simulation."""
        self.progress.emit("Running FRAP simulation...")
        
        diffusion_params = DiffusionParameters(**self.params['diffusion'])
        frap_params = FRAPParameters(**self.params['frap'])
        
        geometry = CompartmentGeometry() if self.params.get('use_geometry', True) else None
        
        simulator = FRAPSimulator(diffusion_params, frap_params, geometry)
        result = simulator.simulate_frap_experiment(
            permeability=self.params.get('permeability', 0.0)
        )
        
        self.progress.emit("FRAP simulation complete")
        return result
    
    def _run_spt(self):
        """Run SPT simulation."""
        self.progress.emit("Running SPT simulation...")
        
        diffusion_params = DiffusionParameters(**self.params['diffusion'])
        spt_params = SPTParameters(**self.params['spt'])
        
        geometry = CompartmentGeometry() if self.params.get('use_geometry', True) else None
        
        simulator = SPTSimulator(diffusion_params, spt_params, geometry)
        result = simulator.simulate_tracks()
        
        self.progress.emit("SPT simulation complete")
        return result
    
    def _run_diffusion(self):
        """Run basic diffusion simulation."""
        self.progress.emit("Running diffusion simulation...")
        
        diffusion_params = DiffusionParameters(**self.params['diffusion'])
        geometry = CompartmentGeometry() if self.params.get('use_geometry', True) else None
        
        simulator = Particle3DSimulator(diffusion_params, geometry)
        
        initial_pos = np.array([0.0, 0.0, 0.0])
        positions, times, metadata = simulator.simulate_trajectory(
            initial_pos,
            permeability=self.params.get('permeability', 0.0)
        )
        
        self.progress.emit("Diffusion simulation complete")
        
        return {
            'positions': positions,
            'times': times,
            'metadata': metadata
        }


class SimulationPanel(QWidget):
    """GUI panel for particle diffusion simulation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize user interface."""
        layout = QVBoxLayout()
        
        # Tabs for different simulation types
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_frap_tab(), "FRAP Simulation")
        self.tabs.addTab(self._create_spt_tab(), "SPT Simulation")
        self.tabs.addTab(self._create_inference_tab(), "Parameter Inference")
        
        layout.addWidget(self.tabs)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("Run Simulation")
        self.run_btn.clicked.connect(self.run_simulation)
        button_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_simulation)
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
        
        # Status/output
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMaximumHeight(150)
        layout.addWidget(QLabel("Output:"))
        layout.addWidget(self.output_text)
        
        self.setLayout(layout)
    
    def _create_frap_tab(self):
        """Create FRAP simulation tab."""
        widget = QWidget()
        layout = QFormLayout()
        
        # Diffusion parameters
        layout.addRow(QLabel("<b>Diffusion Parameters</b>"))
        
        self.frap_D = QDoubleSpinBox()
        self.frap_D.setRange(0.01, 100.0)
        self.frap_D.setValue(1.0)
        self.frap_D.setSuffix(" µm²/s")
        layout.addRow("Diffusion Coefficient:", self.frap_D)
        
        self.frap_alpha = QDoubleSpinBox()
        self.frap_alpha.setRange(0.3, 2.0)
        self.frap_alpha.setValue(1.0)
        self.frap_alpha.setSingleStep(0.1)
        layout.addRow("Anomalous Exponent (α):", self.frap_alpha)
        
        # FRAP parameters
        layout.addRow(QLabel("<b>FRAP Parameters</b>"))
        
        self.frap_radius = QDoubleSpinBox()
        self.frap_radius.setRange(0.1, 10.0)
        self.frap_radius.setValue(1.0)
        self.frap_radius.setSuffix(" µm")
        layout.addRow("Bleach Radius:", self.frap_radius)
        
        self.frap_depth = QDoubleSpinBox()
        self.frap_depth.setRange(0.1, 1.0)
        self.frap_depth.setValue(0.8)
        self.frap_depth.setSingleStep(0.1)
        layout.addRow("Bleach Depth:", self.frap_depth)
        
        self.frap_time = QDoubleSpinBox()
        self.frap_time.setRange(1.0, 300.0)
        self.frap_time.setValue(60.0)
        self.frap_time.setSuffix(" s")
        layout.addRow("Recovery Time:", self.frap_time)
        
        self.frap_particles = QSpinBox()
        self.frap_particles.setRange(10, 10000)
        self.frap_particles.setValue(1000)
        layout.addRow("Number of Particles:", self.frap_particles)
        
        widget.setLayout(layout)
        return widget
    
    def _create_spt_tab(self):
        """Create SPT simulation tab."""
        widget = QWidget()
        layout = QFormLayout()
        
        # Diffusion parameters
        layout.addRow(QLabel("<b>Diffusion Parameters</b>"))
        
        self.spt_D = QDoubleSpinBox()
        self.spt_D.setRange(0.01, 100.0)
        self.spt_D.setValue(1.0)
        self.spt_D.setSuffix(" µm²/s")
        layout.addRow("Diffusion Coefficient:", self.spt_D)
        
        self.spt_alpha = QDoubleSpinBox()
        self.spt_alpha.setRange(0.3, 2.0)
        self.spt_alpha.setValue(1.0)
        self.spt_alpha.setSingleStep(0.1)
        layout.addRow("Anomalous Exponent (α):", self.spt_alpha)
        
        # SPT parameters
        layout.addRow(QLabel("<b>SPT Parameters</b>"))
        
        self.spt_tracks = QSpinBox()
        self.spt_tracks.setRange(1, 1000)
        self.spt_tracks.setValue(100)
        layout.addRow("Number of Tracks:", self.spt_tracks)
        
        self.spt_length = QSpinBox()
        self.spt_length.setRange(10, 1000)
        self.spt_length.setValue(100)
        layout.addRow("Track Length (frames):", self.spt_length)
        
        self.spt_interval = QDoubleSpinBox()
        self.spt_interval.setRange(0.001, 1.0)
        self.spt_interval.setValue(0.033)
        self.spt_interval.setSuffix(" s")
        self.spt_interval.setDecimals(3)
        layout.addRow("Frame Interval:", self.spt_interval)
        
        widget.setLayout(layout)
        return widget
    
    def _create_inference_tab(self):
        """Create parameter inference tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        info_label = QLabel(
            "Parameter inference allows you to estimate diffusion parameters\n"
            "from experimental FRAP or SPT data.\n\n"
            "Load experimental data and run inference to obtain posterior\n"
            "distributions on D, α, permeability, and binding rates."
        )
        layout.addWidget(info_label)
        
        load_btn = QPushButton("Load Experimental Data")
        layout.addWidget(load_btn)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def run_simulation(self):
        """Run selected simulation."""
        current_tab = self.tabs.currentIndex()
        
        if current_tab == 0:  # FRAP
            self._run_frap_simulation()
        elif current_tab == 1:  # SPT
            self._run_spt_simulation()
        elif current_tab == 2:  # Inference
            self.output_text.append("Inference not yet implemented in GUI")
    
    def _run_frap_simulation(self):
        """Run FRAP simulation with current parameters."""
        params = {
            'diffusion': {
                'D': self.frap_D.value(),
                'alpha': self.frap_alpha.value(),
                'dt': 0.01,
                'n_steps': 10000
            },
            'frap': {
                'bleach_center': np.array([0.0, 0.0, 0.0]),
                'bleach_radius': self.frap_radius.value(),
                'bleach_depth': self.frap_depth.value(),
                'post_bleach_time': self.frap_time.value(),
                'n_particles': self.frap_particles.value(),
                'imaging_interval': 0.5
            },
            'use_geometry': True,
            'permeability': 0.1
        }
        
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.output_text.clear()
        
        self.worker = SimulationWorker('frap', params)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_frap_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def _run_spt_simulation(self):
        """Run SPT simulation with current parameters."""
        params = {
            'diffusion': {
                'D': self.spt_D.value(),
                'alpha': self.spt_alpha.value(),
                'dt': 0.001,
                'n_steps': 10000
            },
            'spt': {
                'n_trajectories': self.spt_tracks.value(),
                'trajectory_length': self.spt_length.value(),
                'frame_interval': self.spt_interval.value(),
                'localization_precision': 0.020
            },
            'use_geometry': True
        }
        
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.output_text.clear()
        
        self.worker = SimulationWorker('spt', params)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_spt_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def stop_simulation(self):
        """Stop running simulation."""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.output_text.append("Simulation stopped by user")
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def on_progress(self, message):
        """Handle progress updates."""
        self.output_text.append(message)
    
    def on_frap_finished(self, result):
        """Handle FRAP simulation completion."""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Display results
        fit_params = result.get('fit_params', {})
        
        output = "\n=== FRAP Simulation Results ===\n"
        output += f"Recovery points: {len(result.get('recovery', []))}\n"
        output += f"Half-time: {fit_params.get('tau', float('nan')):.3f} s\n"
        output += f"Mobile fraction: {fit_params.get('mobile_fraction', float('nan')):.2%}\n"
        output += f"Fitted D: {fit_params.get('D_fit', float('nan')):.3f} µm²/s\n"
        
        self.output_text.append(output)
    
    def on_spt_finished(self, result):
        """Handle SPT simulation completion."""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Display results
        analysis = result.get('analysis', {})
        
        output = "\n=== SPT Simulation Results ===\n"
        output += f"Tracks generated: {analysis.get('n_tracks', 0)}\n"
        output += f"Tracks analyzed: {analysis.get('n_analyzed', 0)}\n"
        output += f"Mean track length: {analysis.get('mean_track_length', 0):.1f} frames\n"
        output += f"Estimated D: {analysis.get('D_mean', float('nan')):.3f} ± "
        output += f"{analysis.get('D_std', float('nan')):.3f} µm²/s\n"
        output += f"Estimated α: {analysis.get('alpha_mean', float('nan')):.3f} ± "
        output += f"{analysis.get('alpha_std', float('nan')):.3f}\n"
        
        self.output_text.append(output)
    
    def on_error(self, error_msg):
        """Handle simulation errors."""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.output_text.append(f"\nERROR: {error_msg}")


# Example standalone usage
if __name__ == '__main__':
    import sys
    try:
        from PySide6.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        panel = SimulationPanel()
        panel.setWindowTitle("Particle Diffusion Simulation")
        panel.resize(600, 800)
        panel.show()
        sys.exit(app.exec())
    except ImportError:
        print("PySide6 required for GUI. Install with: pip install PySide6")
