"""
Batch processing for high-throughput analysis
"""

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QListWidget, QListWidgetItem, QProgressBar,
                               QTextEdit, QFileDialog, QMessageBox, QCheckBox,
                               QGroupBox, QComboBox, QSpinBox)
from PySide6.QtCore import QThread, Signal, Qt
import os
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime


class BatchProcessingWorker(QThread):
    """Worker thread for batch processing"""
    
    progress = Signal(int, int, str)  # current, total, message
    image_complete = Signal(str, dict)  # image_name, results
    finished = Signal(list)  # List of all results
    error = Signal(str, str)  # image_name, error_message
    
    def __init__(self, file_paths: List[str], config: Dict):
        super().__init__()
        self.file_paths = file_paths
        self.config = config
        self.should_stop = False
    
    def run(self):
        """Run batch processing"""
        results = []
        
        for i, file_path in enumerate(self.file_paths):
            if self.should_stop:
                break
            
            image_name = Path(file_path).name
            self.progress.emit(i + 1, len(self.file_paths), f"Processing {image_name}...")
            
            try:
                result = self._process_image(file_path)
                results.append(result)
                self.image_complete.emit(image_name, result)
            except Exception as e:
                error_msg = str(e)
                self.error.emit(image_name, error_msg)
                results.append({
                    'image_name': image_name,
                    'status': 'error',
                    'error': error_msg
                })
        
        self.finished.emit(results)
    
    def _process_image(self, file_path: str) -> Dict:
        """Process single image"""
        from core.image_io import TIFFLoader
        from core.segmentation import SegmentationEngine
        from core.quality_control import QualityControl
        from core.measurements import MeasurementEngine
        
        image_name = Path(file_path).name
        result = {
            'image_name': image_name,
            'file_path': file_path,
            'status': 'processing',
            'timestamp': datetime.now()
        }
        
        # Load image
        loader = TIFFLoader()
        image_data, metadata = loader.load_tiff(file_path)
        result['image_shape'] = image_data.shape
        
        # Segment
        seg_engine = SegmentationEngine()
        seg_params = self.config.get('segmentation_params', {})
        
        # Prepare image for segmentation
        if image_data.ndim == 4:
            # Use DNA channel or first channel
            dna_idx = self.config.get('dna_channel', 0)
            seg_image = image_data[:, dna_idx, :, :]
        else:
            seg_image = image_data
        
        if self.config.get('method') == 'cellpose':
            masks, stats = seg_engine.segment_cellpose(
                seg_image,
                **seg_params
            )
        else:
            masks, stats = seg_engine.segment_sam(seg_image)
        
        result['n_nuclei'] = stats['n_objects']
        result['mean_area'] = stats.get('mean_area', 0)
        result['median_area'] = stats.get('median_area', 0)
        
        # QC analysis
        if self.config.get('run_qc', True):
            qc = QualityControl()
            qc_results = qc.analyze_dna_intensity(masks, seg_image)
            
            result['flagged_count'] = qc_results.get('flagged_count', 0)
            result['flagged_percentage'] = qc_results.get('flagged_percentage', 0)
            result['mean_dna_intensity'] = qc_results.get('mean_intensity', 0)
            result['cv_dna_intensity'] = qc_results.get('cv_intensity', 0)
            
            # QC pass/fail (< 10% flagged)
            result['qc_pass'] = result['flagged_percentage'] < 10
        
        # Measurements
        if self.config.get('run_measurements', False):
            measurement_engine = MeasurementEngine()
            measurement_engine.set_enabled_categories(
                self.config.get('measurement_categories', ['basic_shape'])
            )
            
            # Prepare intensity images
            intensity_images = {}
            if image_data.ndim == 4:
                for c in range(image_data.shape[1]):
                    intensity_images[f'channel_{c}'] = image_data[:, c, :, :]
            else:
                intensity_images['channel_0'] = image_data
            
            measurements_df = measurement_engine.extract_measurements(
                masks,
                intensity_images,
                is_3d=self.config.get('is_3d', False)
            )
            
            result['measurements'] = measurements_df
            result['n_measurements'] = len(measurements_df)
        
        result['status'] = 'complete'
        return result
    
    def stop(self):
        """Stop processing"""
        self.should_stop = True


class BatchProcessingDialog(QDialog):
    """Dialog for batch processing configuration and execution"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.file_paths: List[str] = []
        self.worker: Optional[BatchProcessingWorker] = None
        self.results: List[Dict] = []
        
        self.setWindowTitle("Batch Processing")
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup dialog UI"""
        layout = QVBoxLayout(self)
        
        # === File Selection ===
        file_group = QGroupBox("Input Files")
        file_layout = QVBoxLayout()
        
        file_btn_layout = QHBoxLayout()
        
        self.add_files_btn = QPushButton("Add Files...")
        self.add_files_btn.clicked.connect(self._add_files)
        file_btn_layout.addWidget(self.add_files_btn)
        
        self.add_folder_btn = QPushButton("Add Folder...")
        self.add_folder_btn.clicked.connect(self._add_folder)
        file_btn_layout.addWidget(self.add_folder_btn)
        
        self.clear_files_btn = QPushButton("Clear")
        self.clear_files_btn.clicked.connect(self._clear_files)
        file_btn_layout.addWidget(self.clear_files_btn)
        
        file_btn_layout.addStretch()
        file_layout.addLayout(file_btn_layout)
        
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(150)
        file_layout.addWidget(self.file_list)
        
        self.file_count_label = QLabel("0 files selected")
        file_layout.addWidget(self.file_count_label)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # === Processing Configuration ===
        config_group = QGroupBox("Processing Configuration")
        config_layout = QVBoxLayout()
        
        # Segmentation method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Segmentation Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["cellpose", "sam"])
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()
        config_layout.addLayout(method_layout)
        
        # Cellpose model
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Cellpose Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["nuclei", "cyto", "cyto2", "cyto3"])
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        config_layout.addLayout(model_layout)
        
        # DNA channel
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("DNA Channel Index:"))
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(0, 10)
        self.channel_spin.setValue(0)
        channel_layout.addWidget(self.channel_spin)
        channel_layout.addStretch()
        config_layout.addLayout(channel_layout)
        
        # Options
        self.qc_check = QCheckBox("Run QC Analysis")
        self.qc_check.setChecked(True)
        config_layout.addWidget(self.qc_check)
        
        self.measurements_check = QCheckBox("Extract Measurements")
        self.measurements_check.setChecked(False)
        config_layout.addWidget(self.measurements_check)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # === Progress ===
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        progress_layout.addWidget(self.log_text)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # === Control Buttons ===
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.clicked.connect(self._start_processing)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_processing)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)
        
        button_layout.addStretch()
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def _add_files(self):
        """Add files to batch"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select TIFF Files",
            "",
            "TIFF Files (*.tif *.tiff);;All Files (*.*)"
        )
        
        if files:
            self.file_paths.extend(files)
            self._update_file_list()
    
    def _add_folder(self):
        """Add all TIFF files from folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        
        if folder:
            folder_path = Path(folder)
            tiff_files = list(folder_path.glob("*.tif")) + list(folder_path.glob("*.tiff"))
            self.file_paths.extend([str(f) for f in tiff_files])
            self._update_file_list()
    
    def _clear_files(self):
        """Clear file list"""
        self.file_paths.clear()
        self._update_file_list()
    
    def _update_file_list(self):
        """Update file list display"""
        self.file_list.clear()
        for path in self.file_paths:
            self.file_list.addItem(Path(path).name)
        
        self.file_count_label.setText(f"{len(self.file_paths)} files selected")
    
    def _start_processing(self):
        """Start batch processing"""
        if not self.file_paths:
            QMessageBox.warning(self, "No Files", "Please add files to process")
            return
        
        # Gather configuration
        config = {
            'method': self.method_combo.currentText(),
            'segmentation_params': {
                'model_type': self.model_combo.currentText(),
                'diameter': None,  # Auto-detect
                'channels': [0, 0]
            },
            'dna_channel': self.channel_spin.value(),
            'run_qc': self.qc_check.isChecked(),
            'run_measurements': self.measurements_check.isChecked(),
            'measurement_categories': ['basic_shape', 'intensity_stats'],
            'is_3d': False
        }
        
        # Create and start worker
        self.worker = BatchProcessingWorker(self.file_paths, config)
        self.worker.progress.connect(self._on_progress)
        self.worker.image_complete.connect(self._on_image_complete)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.add_files_btn.setEnabled(False)
        self.add_folder_btn.setEnabled(False)
        self.clear_files_btn.setEnabled(False)
        
        self.results.clear()
        self.log_text.clear()
        self.log_text.append(f"Started batch processing at {datetime.now().strftime('%H:%M:%S')}")
        self.log_text.append(f"Processing {len(self.file_paths)} files...\n")
        
        self.worker.start()
    
    def _stop_processing(self):
        """Stop batch processing"""
        if self.worker:
            self.worker.stop()
            self.log_text.append("\nStopping processing...")
    
    def _on_progress(self, current: int, total: int, message: str):
        """Handle progress update"""
        self.progress_label.setText(f"{current}/{total}: {message}")
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
    
    def _on_image_complete(self, image_name: str, result: Dict):
        """Handle image completion"""
        self.results.append(result)
        
        status = result.get('status', 'unknown')
        n_nuclei = result.get('n_nuclei', 0)
        qc_pass = result.get('qc_pass', None)
        
        log_msg = f"✓ {image_name}: {n_nuclei} nuclei"
        if qc_pass is not None:
            log_msg += f", QC: {'PASS' if qc_pass else 'FAIL'}"
        
        self.log_text.append(log_msg)
    
    def _on_error(self, image_name: str, error: str):
        """Handle processing error"""
        self.log_text.append(f"✗ {image_name}: ERROR - {error}")
    
    def _on_finished(self, results: List[Dict]):
        """Handle completion"""
        self.results = results
        
        # Update UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.add_files_btn.setEnabled(True)
        self.add_folder_btn.setEnabled(True)
        self.clear_files_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        self.progress_label.setText("Complete")
        self.progress_bar.setValue(self.progress_bar.maximum())
        
        # Summary
        successful = sum(1 for r in results if r.get('status') == 'complete')
        failed = len(results) - successful
        
        self.log_text.append(f"\n{'='*50}")
        self.log_text.append(f"Batch processing complete at {datetime.now().strftime('%H:%M:%S')}")
        self.log_text.append(f"Successful: {successful}/{len(results)}")
        if failed > 0:
            self.log_text.append(f"Failed: {failed}")
        
        QMessageBox.information(
            self,
            "Complete",
            f"Batch processing complete!\n\nSuccessful: {successful}\nFailed: {failed}"
        )
    
    def _export_results(self):
        """Export batch results"""
        if not self.results:
            QMessageBox.warning(self, "No Results", "No results to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        
        if file_path:
            try:
                # Convert results to DataFrame
                export_data = []
                for result in self.results:
                    row = {
                        'image_name': result.get('image_name'),
                        'status': result.get('status'),
                        'n_nuclei': result.get('n_nuclei', 0),
                        'mean_area': result.get('mean_area', 0),
                        'median_area': result.get('median_area', 0),
                        'flagged_count': result.get('flagged_count', 0),
                        'flagged_percentage': result.get('flagged_percentage', 0),
                        'mean_dna_intensity': result.get('mean_dna_intensity', 0),
                        'cv_dna_intensity': result.get('cv_dna_intensity', 0),
                        'qc_pass': result.get('qc_pass', False)
                    }
                    export_data.append(row)
                
                df = pd.DataFrame(export_data)
                
                if file_path.endswith('.xlsx'):
                    df.to_excel(file_path, index=False, engine='openpyxl')
                else:
                    df.to_csv(file_path, index=False)
                
                QMessageBox.information(self, "Success", f"Results exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {e}")
    
    def get_results(self) -> List[Dict]:
        """Get processing results"""
        return self.results
