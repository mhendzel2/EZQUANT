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

from workers.segmentation_worker import BatchSegmentationWorker
from core.project_data import ImageData


class BatchProcessingDialog(QDialog):
    """Dialog for batch processing configuration and execution"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.file_paths: List[str] = []
        self.worker: Optional[BatchSegmentationWorker] = None
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
        parameters = {
            'engine': self.method_combo.currentText(),
            'model_name': self.model_combo.currentText(),
            'model_type': 'vit_h', # Default for SAM if selected
            'diameter': None,  # Auto-detect
            'channels': [0, 0], # Need to handle DNA channel properly
            'dna_channel': self.channel_spin.value(),
            'do_3d': False,
            'automatic': True # For SAM
        }
        
        # Prepare images data for worker
        images_data = []
        for i, file_path in enumerate(self.file_paths):
            # Create minimal ImageData
            img_data = ImageData(
                path=file_path,
                filename=Path(file_path).name,
                added_date=datetime.now().isoformat(),
                channels=[]
            )
            images_data.append((i, file_path, img_data))
        
        # Create and start worker
        self.worker = BatchSegmentationWorker(images_data, parameters)
        self.worker.progress.connect(self._on_progress)
        self.worker.image_finished.connect(self._on_image_finished)
        self.worker.error.connect(self._on_error)
        self.worker.all_finished.connect(self._on_all_finished)
        
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
            self.worker.cancel()
            self.log_text.append("\nStopping processing...")
    
    def _on_progress(self, current: int, total: int):
        """Handle progress update"""
        self.progress_label.setText(f"Processing image {current}/{total}")
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
    
    def _on_image_finished(self, index: int, masks: np.ndarray, info: Dict):
        """Handle image completion"""
        # Reconstruct result dict for compatibility
        image_name = self.file_paths[index]
        image_name = Path(image_name).name
        
        result = info.copy()
        result['image_name'] = image_name
        result['status'] = 'complete'
        result['n_nuclei'] = info.get('nucleus_count', 0)
        
        self.results.append(result)
        
        n_nuclei = result.get('n_nuclei', 0)
        self.log_text.append(f"✓ {image_name}: {n_nuclei} nuclei")
    
    def _on_error(self, index: int, error: str):
        """Handle processing error"""
        image_name = Path(self.file_paths[index]).name
        self.log_text.append(f"✗ {image_name}: ERROR - {error}")
    
    def _on_all_finished(self, summary: Dict):
        """Handle completion"""
        # Update UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.add_files_btn.setEnabled(True)
        self.add_folder_btn.setEnabled(True)
        self.clear_files_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        self.progress_label.setText("Complete")
        self.progress_bar.setValue(self.progress_bar.maximum())
        
        successful = summary.get('successful', 0)
        failed = summary.get('failed', 0)
        
        self.log_text.append(f"\n{'='*50}")
        self.log_text.append(f"Batch processing complete at {datetime.now().strftime('%H:%M:%S')}")
        self.log_text.append(f"Successful: {successful}/{summary.get('total', 0)}")
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
