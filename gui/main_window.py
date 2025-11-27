"""
Main window for the Nuclei Segmentation Application
"""

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QTabWidget, QMenuBar, QMenu, QToolBar, QStatusBar,
                               QFileDialog, QMessageBox, QDockWidget, QLabel,
                               QSplitter)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import pandas as pd
import datetime

from core.project_data import Project, ImageData
from core.image_io import TIFFLoader
from core.plugin_loader import PluginLoader
from gui.image_viewer import ImageViewer
from gui.segmentation_panel import SegmentationPanel
from gui.analysis_panel import AnalysisPanel
from gui.visualization_panel import VisualizationPanel
from gui.settings_dialog import SettingsDialog
from gui.project_panel import ProjectPanel
from workers.segmentation_worker import SegmentationWorker, DiameterEstimationWorker, BatchSegmentationWorker
from workers.measurement_worker import MeasurementWorker


class MainWindow(QMainWindow):
    """Main application window"""
    
    # Signals
    project_changed = Signal()
    image_loaded = Signal(int)  # image index
    
    def __init__(self, gpu_available: bool = False, gpu_info: str = ""):
        super().__init__()
        
        self.gpu_available = gpu_available
        self.gpu_info = gpu_info
        
        # Project management
        self.project: Optional[Project] = None
        self.current_image_index: Optional[int] = None
        
        # Plugin loader
        self.plugin_loader = PluginLoader()
        self.plugin_loader.load_all_plugins()
        
        # Current measurements
        self.current_measurements: Optional[pd.DataFrame] = None
        self.current_masks: Optional[np.ndarray] = None
        self.current_intensity_images: Optional[Dict[str, np.ndarray]] = None
        
        # Auto-save timer
        self.autosave_timer = QTimer()
        self.autosave_timer.timeout.connect(self._autosave)
        
        # Load settings and configure autosave
        autosave_enabled = SettingsDialog.get_setting("general/autosave_enabled", True)
        autosave_interval = SettingsDialog.get_setting("general/autosave_interval", 5)
        if autosave_enabled:
            self.autosave_timer.start(autosave_interval * 60000)  # Convert to milliseconds
        
        self.setup_ui()
        self.create_actions()
        self.create_menus()
        self.create_toolbar()
        self.create_statusbar()
        
        # Start with new project
        self.new_project()
        
        self.setWindowTitle("Nuclei Segmentation & Analysis")
        self.resize(1400, 900)
    
    def setup_ui(self):
        """Set up the main UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Project panel
        self.project_panel = ProjectPanel()
        self.project_panel.setMinimumWidth(200)
        self.project_panel.setMaximumWidth(400)
        splitter.addWidget(self.project_panel)
        
        # Connect project panel signals
        self.project_panel.add_images_requested.connect(self.import_tiff)
        self.project_panel.image_selected.connect(self._on_project_image_selected)
        self.project_panel.remove_image_requested.connect(self._on_remove_image)
        
        # Right side: Main tab widget
        self.tab_widget = QTabWidget()
        splitter.addWidget(self.tab_widget)
        
        # Set initial splitter sizes
        splitter.setSizes([250, 1150])
        
        layout.addWidget(splitter)
        
        # Create tabs
        self.segmentation_tab = self._create_segmentation_tab()
        self.tab_widget.addTab(self.segmentation_tab, "Segmentation")
        
        self.analysis_tab = self._create_analysis_tab()
        self.tab_widget.addTab(self.analysis_tab, "Analysis")
        
        self.visualization_tab = self._create_visualization_tab()
        self.tab_widget.addTab(self.visualization_tab, "Visualization")
        
        # Workers
        self.segmentation_worker: Optional[SegmentationWorker] = None
        self.diameter_worker: Optional[DiameterEstimationWorker] = None
        self.measurement_worker: Optional[MeasurementWorker] = None
    
    def _create_segmentation_tab(self) -> QWidget:
        """Create the segmentation tab with image viewer and controls"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Left: Image viewer
        self.image_viewer = ImageViewer()
        layout.addWidget(self.image_viewer, stretch=3)
        
        # Right: Segmentation panel
        self.segmentation_panel = SegmentationPanel()
        self.segmentation_panel.set_gpu_available(self.gpu_available)
        
        # Connect signals
        self.segmentation_panel.run_segmentation.connect(self._on_run_segmentation)
        self.segmentation_panel.run_batch_segmentation.connect(self._on_run_batch_segmentation)
        self.segmentation_panel.auto_detect_diameter.connect(self._on_auto_detect_diameter)
        self.image_viewer.nucleus_selected.connect(self._on_nucleus_selected)
        
        layout.addWidget(self.segmentation_panel, stretch=1)
        
        return tab
        layout.addWidget(self.segmentation_panel, stretch=1)
        
        return tab
    
    def _create_analysis_tab(self) -> QWidget:
        """Create the analysis tab for measurements"""
        self.analysis_panel = AnalysisPanel()
        
        # Set plugin info
        plugin_info = self.plugin_loader.get_all_plugin_info()
        self.analysis_panel.set_plugin_info(plugin_info)
        
        # Connect signals
        self.analysis_panel.run_measurements.connect(self._on_run_measurements)
        self.analysis_panel.manage_plugins_btn.clicked.connect(self.show_plugin_manager)
        
        return self.analysis_panel
    
    def _create_visualization_tab(self) -> QWidget:
        """Create the visualization tab for plots"""
        self.visualization_panel = VisualizationPanel()
        
        # Connect signals
        self.visualization_panel.nucleus_selected.connect(self._on_nucleus_selected)
        
        return self.visualization_panel
    
    def create_actions(self):
        """Create menu and toolbar actions"""
        # File actions
        self.new_action = QAction("&New Project", self)
        self.new_action.setShortcut(QKeySequence.New)
        self.new_action.setStatusTip("Create a new project")
        self.new_action.triggered.connect(self.new_project)
        
        self.open_action = QAction("&Open Project...", self)
        self.open_action.setShortcut(QKeySequence.Open)
        self.open_action.setStatusTip("Open an existing project")
        self.open_action.triggered.connect(self.open_project)
        
        self.save_action = QAction("&Save Project", self)
        self.save_action.setShortcut(QKeySequence.Save)
        self.save_action.setStatusTip("Save the current project")
        self.save_action.triggered.connect(self.save_project)
        
        self.save_as_action = QAction("Save Project &As...", self)
        self.save_as_action.setShortcut(QKeySequence.SaveAs)
        self.save_as_action.setStatusTip("Save the project with a new name")
        self.save_as_action.triggered.connect(self.save_project_as)
        
        self.import_action = QAction("&Import TIFF...", self)
        self.import_action.setShortcut(QKeySequence("Ctrl+I"))
        self.import_action.setStatusTip("Import a TIFF image file")
        self.import_action.triggered.connect(self.import_tiff)
        
        self.export_action = QAction("&Export Measurements...", self)
        self.export_action.setShortcut(QKeySequence("Ctrl+E"))
        self.export_action.setStatusTip("Export measurements to CSV or Excel")
        self.export_action.triggered.connect(self.export_measurements)
        
        self.batch_action = QAction("&Batch Process...", self)
        self.batch_action.setStatusTip("Process multiple images")
        self.batch_action.triggered.connect(self.batch_process)
        
        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut(QKeySequence.Quit)
        self.exit_action.setStatusTip("Exit the application")
        self.exit_action.triggered.connect(self.close)
        
        # Edit actions
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.setEnabled(False)
        
        self.redo_action = QAction("&Redo", self)
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.redo_action.setEnabled(False)
        
        # View actions
        self.quality_dashboard_action = QAction("Quality &Dashboard", self)
        self.quality_dashboard_action.setStatusTip("View project quality metrics")
        self.quality_dashboard_action.triggered.connect(self.show_quality_dashboard)
        
        # Tools actions
        self.plugin_manager_action = QAction("&Plugin Manager", self)
        self.plugin_manager_action.setStatusTip("Manage measurement plugins")
        self.plugin_manager_action.triggered.connect(self.show_plugin_manager)
        
        self.settings_action = QAction("&Settings...", self)
        self.settings_action.setStatusTip("Application settings")
        self.settings_action.triggered.connect(self.show_settings)
        
        # Help actions
        self.help_action = QAction("&Documentation", self)
        self.help_action.setShortcut(QKeySequence.HelpContents)
        self.help_action.triggered.connect(self.show_help)
        
        self.about_action = QAction("&About", self)
        self.about_action.triggered.connect(self.show_about)
    
    def create_menus(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.new_action)
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.import_action)
        file_menu.addAction(self.export_action)
        file_menu.addAction(self.batch_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.quality_dashboard_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        tools_menu.addAction(self.plugin_manager_action)
        tools_menu.addAction(self.settings_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.help_action)
        help_menu.addSeparator()
        help_menu.addAction(self.about_action)
    
    def create_toolbar(self):
        """Create toolbar"""
        toolbar = self.addToolBar("Main Toolbar")
        toolbar.setMovable(False)
        
        toolbar.addAction(self.new_action)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.save_action)
        toolbar.addSeparator()
        toolbar.addAction(self.import_action)
        toolbar.addAction(self.export_action)
        toolbar.addSeparator()
        toolbar.addAction(self.undo_action)
        toolbar.addAction(self.redo_action)
    
    def create_statusbar(self):
        """Create status bar"""
        self.statusBar().showMessage("Ready")
        
        # Add GPU info to status bar
        gpu_label = QLabel(f"  {self.gpu_info}  ")
        self.statusBar().addPermanentWidget(gpu_label)
    
    # Project management methods
    def new_project(self):
        """Create a new project"""
        if self.project and self._check_unsaved_changes():
            return
        
        self.project = Project()
        self.current_image_index = None
        self.project_panel.clear_images()
        self.project_changed.emit()
        self.statusBar().showMessage("New project created", 3000)
    
    def open_project(self):
        """Open an existing project"""
        if self.project and self._check_unsaved_changes():
            return
        
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "Project Files (*.nsa *.json *.db);;All Files (*)"
        )
        
        if filepath:
            try:
                self.project = Project(filepath)
                self.current_image_index = None
                
                # Populate project panel
                self.project_panel.clear_images()
                for img_data in self.project.images:
                    has_seg = img_data.current_segmentation_id is not None
                    self.project_panel.add_image(img_data.filename, has_segmentation=has_seg)
                
                self.project_changed.emit()
                self.statusBar().showMessage(f"Opened project: {filepath}", 3000)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Opening Project",
                    f"Could not open project:\n{str(e)}"
                )
    
    def save_project(self):
        """Save the current project"""
        if not self.project:
            return
        
        if not self.project.project_path:
            self.save_project_as()
            return
        
        try:
            self.project.save()
            self.statusBar().showMessage(
                f"Project saved: {self.project.project_path}", 3000
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving Project",
                f"Could not save project:\n{str(e)}"
            )
    
    def save_project_as(self):
        """Save the project with a new name"""
        if not self.project:
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            "",
            "Project Files (*.nsa);;All Files (*)"
        )
        
        if filepath:
            if not filepath.endswith('.nsa'):
                filepath += '.nsa'
            
            try:
                self.project.save(filepath)
                self.statusBar().showMessage(f"Project saved: {filepath}", 3000)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Saving Project",
                    f"Could not save project:\n{str(e)}"
                )
    
    def import_tiff(self):
        """Add TIFF image files or Metamorph .nd files to project"""
        if not self.project:
            self.new_project()
        
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Image Files to Project",
            "",
            "Image Files (*.tif *.tiff *.nd *.vsi *.lif);;TIFF Files (*.tif *.tiff);;Metamorph ND Files (*.nd);;Olympus VSI Files (*.vsi);;Leica LIF Files (*.lif);;All Files (*)"
        )
        
        if filepaths:
            # Add multiple files to project
            added_count = 0
            for filepath in filepaths:
                try:
                    # Check file type and route to appropriate handler
                    if filepath.lower().endswith('.nd'):
                        self._add_metamorph_nd_to_project(filepath)
                    elif filepath.lower().endswith('.vsi'):
                        self._add_vsi_to_project(filepath)
                    elif filepath.lower().endswith('.lif'):
                        self._add_lif_to_project(filepath)
                    else:
                        self._add_single_image_to_project(filepath)
                    added_count += 1
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Error Adding Image",
                        f"Could not add {Path(filepath).name}:\n{str(e)}"
                    )
            
            if added_count > 0:
                self.statusBar().showMessage(
                    f"Added {added_count} image(s) to project", 3000
                )
                # Select the first added image
                if self.project_panel.get_image_count() > 0 and self.current_image_index is None:
                    self.project_panel.set_selected_index(len(self.project.images) - added_count)
    
    def _import_single_image(self, filepath: str):
        """Import a single TIFF image"""
        try:
            # Load image
            image, metadata = TIFFLoader.load_tiff(filepath)
            
            # Create image data entry
            img_data = ImageData(
                path=filepath,
                filename=Path(filepath).name,
                added_date=str(Path(filepath).stat().st_mtime),
                channels=metadata.get('channel_names', []),
                shape=metadata.get('final_shape'),
                dtype=metadata.get('dtype'),
                pixel_size=metadata.get('pixel_size'),
                bit_depth=metadata.get('bit_depth', 8)
            )
            
            # Add to project
            img_index = self.project.add_image(img_data)
            self.current_image_index = img_index
            
            # Display in viewer
            self.image_viewer.set_image(image, metadata)
            self.segmentation_panel.set_image(image, metadata)
            
            self.image_loaded.emit(img_index)
            self.statusBar().showMessage(
                f"Imported: {img_data.filename} "
                f"({img_data.shape}, {img_data.bit_depth}-bit)", 3000
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Importing TIFF",
                f"Could not import TIFF file:\n{str(e)}"
            )
    
    def _import_metamorph_nd(self, nd_filepath: str):
        """Import Metamorph .nd file series"""
        try:
            # Get info about the .nd file
            nd_info = TIFFLoader.get_metamorph_info(nd_filepath)
            
            n_stages = nd_info['n_stages']
            n_timepoints = nd_info['n_timepoints']
            
            # Ask user how to import
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QRadioButton, QButtonGroup
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Import Metamorph Series")
            layout = QVBoxLayout(dialog)
            
            layout.addWidget(QLabel(f"<b>Metamorph Series Detected</b>"))
            layout.addWidget(QLabel(f"Stages: {n_stages}"))
            layout.addWidget(QLabel(f"Timepoints: {n_timepoints}"))
            layout.addWidget(QLabel(f"Total files: {nd_info['n_files']}"))
            layout.addWidget(QLabel(""))
            layout.addWidget(QLabel("How would you like to import this series?"))
            
            button_group = QButtonGroup(dialog)
            
            first_only_radio = QRadioButton("Import first stage/timepoint only")
            first_only_radio.setChecked(True)
            button_group.addButton(first_only_radio, 0)
            layout.addWidget(first_only_radio)
            
            all_stages_radio = QRadioButton(f"Import all stages as separate images ({n_stages} images)")
            button_group.addButton(all_stages_radio, 1)
            layout.addWidget(all_stages_radio)
            
            all_series_radio = QRadioButton(f"Import entire series ({n_stages * n_timepoints} images)")
            button_group.addButton(all_series_radio, 2)
            layout.addWidget(all_series_radio)
            
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            
            if dialog.exec() == QDialog.Accepted:
                choice = button_group.checkedId()
                
                if choice == 0:
                    # Import first only
                    image, metadata = TIFFLoader.load_metamorph_nd(nd_filepath, stage=0, timepoint=0)
                    
                    img_data = ImageData(
                        path=nd_filepath,
                        filename=Path(nd_filepath).name,
                        added_date=str(Path(nd_filepath).stat().st_mtime),
                        channels=metadata.get('channel_names', []),
                        shape=metadata.get('final_shape'),
                        dtype=metadata.get('dtype'),
                        pixel_size=metadata.get('pixel_size'),
                        bit_depth=metadata.get('bit_depth', 8)
                    )
                    
                    img_index = self.project.add_image(img_data)
                    self.current_image_index = img_index
                    
                    self.image_viewer.set_image(image, metadata)
                    self.segmentation_panel.set_image(image, metadata)
                    self.image_loaded.emit(img_index)
                    
                    self.statusBar().showMessage(f"Imported: {img_data.filename} (first position)", 3000)
                
                elif choice in [1, 2]:
                    # Import multiple
                    self._import_metamorph_batch(nd_filepath, import_all=(choice == 2))
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Importing Metamorph Series",
                f"Could not import .nd file:\n{str(e)}"
            )
    
    def _import_metamorph_batch(self, nd_filepath: str, import_all: bool = False):
        """Import multiple images from Metamorph series"""
        from core.metamorph_nd import MetamorphNDFile
        
        try:
            nd_file = MetamorphNDFile(nd_filepath)
            
            stages = sorted(set([f['stage'] for f in nd_file.file_list]))
            timepoints = sorted(set([f['timepoint'] for f in nd_file.file_list]))
            
            # Progress dialog
            from PySide6.QtWidgets import QProgressDialog
            
            if import_all:
                total = len(stages) * len(timepoints)
                progress_text = "Importing all stage positions and timepoints..."
            else:
                total = len(stages)
                progress_text = "Importing all stage positions..."
                timepoints = [0]  # Only first timepoint
            
            progress = QProgressDialog(progress_text, "Cancel", 0, total, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            
            count = 0
            first_index = None
            
            for stage in stages:
                for timepoint in timepoints:
                    if progress.wasCanceled():
                        break
                    
                    try:
                        image, metadata = nd_file.build_stack(stage=stage, timepoint=timepoint)
                        
                        # Create descriptive filename
                        stage_name = metadata.get('stage_name', f'Stage_{stage}')
                        if import_all and len(timepoints) > 1:
                            filename = f"{Path(nd_filepath).stem}_{stage_name}_t{timepoint:03d}"
                        else:
                            filename = f"{Path(nd_filepath).stem}_{stage_name}"
                        
                        img_data = ImageData(
                            path=nd_filepath,
                            filename=filename,
                            added_date=str(Path(nd_filepath).stat().st_mtime),
                            channels=metadata.get('channel_names', []),
                            shape=metadata.get('final_shape'),
                            dtype=metadata.get('dtype'),
                            pixel_size=metadata.get('pixel_size'),
                            bit_depth=metadata.get('bit_depth', 8)
                        )
                        
                        img_index = self.project.add_image(img_data)
                        
                        if first_index is None:
                            first_index = img_index
                        
                        count += 1
                        progress.setValue(count)
                        
                    except Exception as e:
                        print(f"Warning: Could not import stage {stage}, timepoint {timepoint}: {e}")
                
                if progress.wasCanceled():
                    break
            
            progress.close()
            
            # Load first image
            if first_index is not None:
                self.current_image_index = first_index
                img_data = self.project.get_image(first_index)
                
                image, metadata = TIFFLoader.load_metamorph_nd(nd_filepath, stage=0, timepoint=0)
                self.image_viewer.set_image(image, metadata)
                self.segmentation_panel.set_image(image, metadata)
                self.image_loaded.emit(first_index)
            
            self.statusBar().showMessage(f"Imported {count} images from Metamorph series", 5000)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Batch import failed:\n{str(e)}"
            )
    
    def _import_vsi(self, vsi_filepath: str):
        """Import Olympus VSI file with scene selection"""
        try:
            # Get VSI file info
            info = TIFFLoader.get_vsi_info(vsi_filepath)
            
            scenes = info.get('scenes', [])
            
            if len(scenes) == 0:
                QMessageBox.warning(self, "Import Error", "No scenes found in VSI file.")
                return
            elif len(scenes) == 1:
                # Only one scene, load it directly
                self._import_vsi_scene(vsi_filepath, 0)
            else:
                # Multiple scenes, show selection dialog
                from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QDialogButtonBox, QLabel
                
                dialog = QDialog(self)
                dialog.setWindowTitle("Select Scene to Import")
                dialog.setMinimumWidth(400)
                dialog.setMinimumHeight(300)
                
                layout = QVBoxLayout()
                
                label = QLabel(f"VSI file contains {len(scenes)} scenes. Select one to import:")
                layout.addWidget(label)
                
                list_widget = QListWidget()
                for i, scene in enumerate(scenes):
                    name = scene.get('name', f'Scene {i+1}')
                    shape = scene.get('shape', 'unknown')
                    n_channels = scene.get('n_channels', '?')
                    n_slices = scene.get('n_slices', '?')
                    list_widget.addItem(f"{name} - Shape: {shape}, Channels: {n_channels}, Z-slices: {n_slices}")
                layout.addWidget(list_widget)
                
                buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                buttons.accepted.connect(dialog.accept)
                buttons.rejected.connect(dialog.reject)
                layout.addWidget(buttons)
                
                dialog.setLayout(layout)
                
                if dialog.exec() == QDialog.Accepted and list_widget.currentRow() >= 0:
                    scene_idx = list_widget.currentRow()
                    self._import_vsi_scene(vsi_filepath, scene_idx)
        
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import VSI file:\n{str(e)}")
    
    def _import_vsi_scene(self, vsi_filepath: str, scene: int):
        """Import a specific scene from VSI file"""
        try:
            # Load image
            image, metadata = TIFFLoader.load_vsi(vsi_filepath, scene=scene)
            
            # Create unique filename with scene info
            base_name = Path(vsi_filepath).stem
            scene_name = metadata.get('current_scene', scene)
            filename = f"{base_name}_scene{scene}_{scene_name}"
            
            # Create image data entry
            img_data = ImageData(
                path=vsi_filepath,
                filename=filename,
                added_date=str(Path(vsi_filepath).stat().st_mtime),
                channels=metadata.get('channel_names', []),
                z_slices=metadata.get('n_slices', 1),
                is_3d=metadata.get('is_3d', False),
                bit_depth=metadata.get('bit_depth', 8)
            )
            
            img_data.raw_image = image
            img_data.metadata = metadata
            
            # Add to project
            img_id = self.project.add_image(img_data)
            
            # Display
            self.display_image(img_id)
            
            self.statusBar().showMessage(f"Loaded VSI scene: {filename}", 5000)
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to load VSI scene:\n{str(e)}")
    
    def _import_lif(self, lif_filepath: str):
        """Import Leica LIF file with scene selection"""
        try:
            # Get LIF file info
            info = TIFFLoader.get_lif_info(lif_filepath)
            
            scenes = info.get('scenes', [])
            
            if len(scenes) == 0:
                QMessageBox.warning(self, "Import Error", "No scenes found in LIF file.")
                return
            elif len(scenes) == 1:
                # Only one scene, load it directly
                self._import_lif_scene(lif_filepath, 0)
            else:
                # Multiple scenes, show selection dialog
                from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QDialogButtonBox, QLabel
                
                dialog = QDialog(self)
                dialog.setWindowTitle("Select Scene to Import")
                dialog.setMinimumWidth(400)
                dialog.setMinimumHeight(300)
                
                layout = QVBoxLayout()
                
                label = QLabel(f"LIF file contains {len(scenes)} scenes. Select one to import:")
                layout.addWidget(label)
                
                list_widget = QListWidget()
                for i, scene in enumerate(scenes):
                    name = scene.get('name', f'Scene {i+1}')
                    shape = scene.get('shape', 'unknown')
                    n_channels = scene.get('n_channels', '?')
                    n_slices = scene.get('n_slices', '?')
                    list_widget.addItem(f"{name} - Shape: {shape}, Channels: {n_channels}, Z-slices: {n_slices}")
                layout.addWidget(list_widget)
                
                buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                buttons.accepted.connect(dialog.accept)
                buttons.rejected.connect(dialog.reject)
                layout.addWidget(buttons)
                
                dialog.setLayout(layout)
                
                if dialog.exec() == QDialog.Accepted and list_widget.currentRow() >= 0:
                    scene_idx = list_widget.currentRow()
                    self._import_lif_scene(lif_filepath, scene_idx)
        
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import LIF file:\n{str(e)}")
    
    def _import_lif_scene(self, lif_filepath: str, scene: int):
        """Import a specific scene from LIF file"""
        try:
            # Load image
            image, metadata = TIFFLoader.load_lif(lif_filepath, scene=scene)
            
            # Create unique filename with scene info
            base_name = Path(lif_filepath).stem
            scene_name = metadata.get('current_scene', scene)
            filename = f"{base_name}_scene{scene}_{scene_name}"
            
            # Create image data entry
            img_data = ImageData(
                path=lif_filepath,
                filename=filename,
                added_date=str(Path(lif_filepath).stat().st_mtime),
                channels=metadata.get('channel_names', []),
                z_slices=metadata.get('n_slices', 1),
                is_3d=metadata.get('is_3d', False),
                bit_depth=metadata.get('bit_depth', 8)
            )
            
            img_data.raw_image = image
            img_data.metadata = metadata
            
            # Add to project
            img_id = self.project.add_image(img_data)
            
            # Display
            self.display_image(img_id)
            
            self.statusBar().showMessage(f"Loaded LIF scene: {filename}", 5000)
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to load LIF scene:\n{str(e)}")
    
    def _add_single_image_to_project(self, filepath: str):
        """Add a single TIFF image to project without loading it"""
        try:
            # Load metadata only (quick)
            image, metadata = TIFFLoader.load_tiff(filepath)
            
            # Create image data entry
            img_data = ImageData(
                path=filepath,
                filename=Path(filepath).name,
                added_date=datetime.datetime.now().isoformat(),
                channels=metadata.get('channel_names', []),
                shape=metadata.get('final_shape'),
                dtype=metadata.get('dtype'),
                pixel_size=metadata.get('pixel_size'),
                bit_depth=metadata.get('bit_depth', 8)
            )
            
            # Add to project
            img_index = self.project.add_image(img_data)
            
            # Add to project panel UI
            self.project_panel.add_image(img_data.filename, has_segmentation=False)
            
            return img_index
            
        except Exception as e:
            raise Exception(f"Could not add image: {str(e)}")
    
    def _add_metamorph_nd_to_project(self, nd_filepath: str):
        """Add Metamorph .nd file to project"""
        # For now, just add first stage/timepoint
        # TODO: Implement dialog for user selection
        try:
            image, metadata = TIFFLoader.load_metamorph_nd(nd_filepath, stage=0, timepoint=0)
            
            img_data = ImageData(
                path=nd_filepath,
                filename=Path(nd_filepath).name,
                added_date=datetime.datetime.now().isoformat(),
                channels=metadata.get('channel_names', []),
                shape=metadata.get('final_shape'),
                dtype=metadata.get('dtype'),
                pixel_size=metadata.get('pixel_size'),
                bit_depth=metadata.get('bit_depth', 8)
            )
            
            img_index = self.project.add_image(img_data)
            self.project_panel.add_image(img_data.filename, has_segmentation=False)
            
            return img_index
        except Exception as e:
            raise Exception(f"Could not add Metamorph ND: {str(e)}")
    
    def _add_vsi_to_project(self, vsi_filepath: str):
        """Add VSI file to project (first scene)"""
        try:
            image, metadata = TIFFLoader.load_vsi(vsi_filepath, scene=0)
            
            img_data = ImageData(
                path=vsi_filepath,
                filename=Path(vsi_filepath).name,
                added_date=datetime.datetime.now().isoformat(),
                channels=metadata.get('channel_names', []),
                shape=metadata.get('final_shape'),
                dtype=metadata.get('dtype'),
                pixel_size=metadata.get('pixel_size'),
                bit_depth=metadata.get('bit_depth', 8)
            )
            
            img_index = self.project.add_image(img_data)
            self.project_panel.add_image(img_data.filename, has_segmentation=False)
            
            return img_index
        except Exception as e:
            raise Exception(f"Could not add VSI: {str(e)}")
    
    def _add_lif_to_project(self, lif_filepath: str):
        """Add LIF file to project (first scene)"""
        try:
            image, metadata = TIFFLoader.load_lif(lif_filepath, scene=0)
            
            img_data = ImageData(
                path=lif_filepath,
                filename=Path(lif_filepath).name,
                added_date=datetime.datetime.now().isoformat(),
                channels=metadata.get('channel_names', []),
                shape=metadata.get('final_shape'),
                dtype=metadata.get('dtype'),
                pixel_size=metadata.get('pixel_size'),
                bit_depth=metadata.get('bit_depth', 8)
            )
            
            img_index = self.project.add_image(img_data)
            self.project_panel.add_image(img_data.filename, has_segmentation=False)
            
            return img_index
        except Exception as e:
            raise Exception(f"Could not add LIF: {str(e)}")
    
    def _on_project_image_selected(self, index: int):
        """Load and display selected image from project panel"""
        if not self.project or index < 0:
            return
        
        try:
            img_data = self.project.get_image(index)
            if not img_data:
                return
            
            # Load the actual image data
            filepath = img_data.path
            
            if filepath.lower().endswith('.nd'):
                image, metadata = TIFFLoader.load_metamorph_nd(filepath, stage=0, timepoint=0)
            elif filepath.lower().endswith('.vsi'):
                image, metadata = TIFFLoader.load_vsi(filepath, scene=0)
            elif filepath.lower().endswith('.lif'):
                image, metadata = TIFFLoader.load_lif(filepath, scene=0)
            else:
                image, metadata = TIFFLoader.load_tiff(filepath)
            
            # Update current image
            self.current_image_index = index
            
            # Display in viewer
            self.image_viewer.set_image(image, metadata)
            self.segmentation_panel.set_image(image, metadata)
            
            # Load existing segmentation if available
            if img_data.current_segmentation_id is not None:
                # TODO: Load segmentation from project
                pass
            
            self.image_loaded.emit(index)
            self.statusBar().showMessage(
                f"Loaded: {img_data.filename}", 3000
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Image",
                f"Could not load image:\n{str(e)}"
            )
    
    def _on_remove_image(self, index: int):
        """Remove image from project"""
        if not self.project:
            return
        
        self.project.remove_image(index)
        self.project_panel.remove_image(index)
        
        # Clear viewer if this was the current image
        if self.current_image_index == index:
            self.current_image_index = None
            self.image_viewer.clear()
        elif self.current_image_index is not None and self.current_image_index > index:
            self.current_image_index -= 1
        
        self.statusBar().showMessage("Image removed from project", 2000)
    
    def export_measurements(self):
        """Export measurements to file"""
        QMessageBox.information(
            self,
            "Export Measurements",
            "Export functionality will be implemented soon."
        )
    
    def batch_process(self):
        """Batch process multiple images"""
        QMessageBox.information(
            self,
            "Batch Processing",
            "Batch processing functionality will be implemented soon."
        )
    
    def show_quality_dashboard(self):
        """Show quality metrics dashboard"""
        QMessageBox.information(
            self,
            "Quality Dashboard",
            "Quality dashboard will be implemented soon."
        )
    
    def show_plugin_manager(self):
        """Show plugin manager dialog"""
        QMessageBox.information(
            self,
            "Plugin Manager",
            "Plugin manager will be implemented soon."
        )
    
    def show_settings(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self)
        if dialog.exec():
            # Settings were saved, apply any immediate changes
            self._apply_settings()
    
    def _apply_settings(self):
        """Apply settings that require immediate action"""
        # Update autosave timer
        autosave_enabled = SettingsDialog.get_setting("general/autosave_enabled", True)
        autosave_interval = SettingsDialog.get_setting("general/autosave_interval", 5)
        
        if autosave_enabled:
            self.autosave_timer.start(autosave_interval * 60000)  # Convert to milliseconds
        else:
            self.autosave_timer.stop()
        
        # Reload plugins if plugin directory changed
        plugin_dir = SettingsDialog.get_setting("advanced/plugin_directory")
        if plugin_dir and str(self.plugin_loader.plugin_directory) != plugin_dir:
            self.plugin_loader.plugin_directory = Path(plugin_dir)
            self.plugin_loader.reload_plugins()
            
            # Update analysis panel with new plugins
            plugin_info = self.plugin_loader.get_all_plugin_info()
            self.analysis_panel.set_plugin_info(plugin_info)
    
    def show_help(self):
        """Show documentation"""
        QMessageBox.information(
            self,
            "Documentation",
            "Documentation will be available soon.\n\n"
            "For now, please refer to README.md"
        )
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Nuclei Segmentation & Analysis",
            "<h2>Nuclei Segmentation & Analysis</h2>"
            "<p>Version 1.0.0</p>"
            "<p>A desktop application for nuclei segmentation and morphometric analysis.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Cellpose 4 and SAM integration</li>"
            "<li>Cell cycle-aware quality control</li>"
            "<li>Comprehensive morphometric measurements</li>"
            "<li>Interactive visualization</li>"
            "</ul>"
            f"<p><b>GPU Status:</b> {self.gpu_info}</p>"
        )
    
    def _check_unsaved_changes(self) -> bool:
        """
        Check if there are unsaved changes and prompt user
        Returns True if action should be cancelled
        """
        # TODO: Implement proper change tracking
        return False
    
    def _on_run_segmentation(self, parameters: Dict):
        """Handle segmentation run request"""
        if self.current_image_index is None:
            QMessageBox.warning(self, "No Image", "Please import an image first.")
            return
        
        # Get current image data
        img_data = self.project.get_image(self.current_image_index)
        if img_data is None:
            return
        
        # Load image
        try:
            image, metadata = TIFFLoader.load_tiff(img_data.path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load image: {e}")
            return
        
        # Create and start worker
        self.segmentation_worker = SegmentationWorker(
            image=image,
            parameters=parameters,
            gpu_available=self.gpu_available
        )
    def _on_segmentation_finished(self, masks: np.ndarray, results: Dict):
        """Handle segmentation completion"""
        # Update UI
        self.segmentation_panel.set_running(False)
        self.segmentation_panel.display_results(results)
        
        # Get current parameters
        params = self.segmentation_panel.get_parameters()
        self.segmentation_panel.add_to_history(params, results)
        
        # Display mask overlay
        self.image_viewer.set_mask(masks)
        
        # Store masks for analysis
        self.current_masks = masks
        
        # Store in project
        if self.current_image_index is not None:
            img_data = self.project.get_image(self.current_image_index)
            if img_data:
                # Add to segmentation history
                import datetime
                seg_record = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'parameters': params,
                    'results': results
                }
                img_data.segmentation_history.append(seg_record)
                
                # Mark as segmented in project panel
                self.project_panel.set_image_segmented(self.current_image_index, True)
                
                # TODO: Store mask data (need to handle large arrays)
        
        self.statusBar().showMessage("Segmentation complete - Ready for analysis", 5000)
    
    def _on_segmentation_error(self, error_message: str):
        """Handle segmentation error"""
        self.segmentation_panel.set_running(False)
        QMessageBox.critical(self, "Segmentation Error", error_message)
        self.statusBar().showMessage("Segmentation failed", 5000)
    
    def _on_run_batch_segmentation(self, parameters: Dict):
        """Handle batch segmentation request"""
        if not self.project or len(self.project.images) == 0:
            QMessageBox.warning(
                self,
                "No Images",
                "Please add images to the project first."
            )
            return
        
        # Ask for confirmation
        from PySide6.QtWidgets import QDialogButtonBox
        
        total_images = len(self.project.images)
        unsegmented_count = sum(1 for img in self.project.images if img.current_segmentation_id is None)
        
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle("Batch Segmentation")
        msg.setText(f"Apply current segmentation settings to all {total_images} images in the project?")
        msg.setInformativeText(
            f"Segmented: {total_images - unsegmented_count}\n"
            f"Not segmented: {unsegmented_count}\n\n"
            "This will segment all images, including those already segmented.\n"
            "This may take a while depending on the number of images."
        )
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msg.setDefaultButton(QMessageBox.StandardButton.Yes)
        
        if msg.exec() != QMessageBox.StandardButton.Yes:
            return
        
        # Prepare image data for batch processing
        images_data = []
        for i, img_data in enumerate(self.project.images):
            images_data.append((i, img_data.path, img_data))
        
        # Create and start batch worker
        self.batch_worker = BatchSegmentationWorker(
            images_data=images_data,
            parameters=parameters,
            gpu_available=self.gpu_available
        )
        
        # Connect signals
        self.batch_worker.progress.connect(self._on_batch_progress)
        self.batch_worker.image_finished.connect(self._on_batch_image_finished)
        self.batch_worker.all_finished.connect(self._on_batch_all_finished)
        self.batch_worker.error.connect(self._on_batch_image_error)
        self.batch_worker.status.connect(self.statusBar().showMessage)
        
        # Update UI
        self.segmentation_panel.set_running(True)
        self.statusBar().showMessage("Starting batch segmentation...", 3000)
        
        # Start processing
        self.batch_worker.start()
    
    def _on_batch_progress(self, current: int, total: int):
        """Handle batch segmentation progress"""
        self.segmentation_panel.set_batch_progress(current, total)
    
    def _on_batch_image_finished(self, img_index: int, masks: np.ndarray, results: Dict):
        """Handle completion of single image in batch"""
        # Store results in project
        img_data = self.project.get_image(img_index)
        if img_data:
            # Add to segmentation history
            import datetime
            seg_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'parameters': self.segmentation_panel.get_parameters(),
                'results': results
            }
            img_data.segmentation_history.append(seg_record)
            
            # Mark as segmented
            if img_data.current_segmentation_id is None:
                img_data.current_segmentation_id = len(img_data.segmentation_history) - 1
            
            # Update project panel
            self.project_panel.set_image_segmented(img_index, True)
        
        # If this is the currently displayed image, update the view
        if self.current_image_index == img_index:
            self.current_masks = masks
            self.image_viewer.set_mask(masks)
    
    def _on_batch_image_error(self, img_index: int, error_msg: str):
        """Handle error for single image in batch"""
        print(f"Error processing image {img_index}: {error_msg}")
    
    def _on_batch_all_finished(self, summary: Dict):
        """Handle completion of all batch segmentation"""
        self.segmentation_panel.set_running(False)
        
        # Show summary dialog
        total = summary.get('total', 0)
        successful = summary.get('successful', 0)
        failed = summary.get('failed', 0)
        total_nuclei = summary.get('total_nuclei', 0)
        cancelled = summary.get('cancelled', False)
        
        if cancelled:
            msg_title = "Batch Segmentation Cancelled"
            msg_text = f"Processed {successful} of {total} images before cancellation."
        else:
            msg_title = "Batch Segmentation Complete"
            msg_text = f"Successfully segmented {successful} of {total} images."
        
        details = f"Successful: {successful}\nFailed: {failed}\nTotal nuclei detected: {total_nuclei}"
        
        QMessageBox.information(
            self,
            msg_title,
            f"{msg_text}\n\n{details}"
        )
        
        self.statusBar().showMessage(
            f"Batch segmentation complete: {successful}/{total} successful", 5000
        )
    
    def _on_auto_detect_diameter(self):
        """Handle auto-detect diameter request"""
        if self.current_image_index is None:
            return
        
        img_data = self.project.get_image(self.current_image_index)
        if img_data is None:
            return
        
        try:
            image, metadata = TIFFLoader.load_tiff(img_data.path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load image: {e}")
            return
        
        # Create and start worker
        self.diameter_worker = DiameterEstimationWorker(
            image=image,
            gpu_available=self.gpu_available
        )
        
        self.diameter_worker.finished.connect(self.segmentation_panel.set_estimated_diameter)
        self.diameter_worker.error.connect(lambda msg: QMessageBox.warning(self, "Error", msg))
        self.diameter_worker.status.connect(self.statusBar().showMessage)
        
        self.diameter_worker.start()
    
    def _on_nucleus_selected(self, nucleus_id: int):
        """Handle nucleus selection in image viewer"""
        self.statusBar().showMessage(f"Selected nucleus ID: {nucleus_id}", 3000)
    
    def _on_run_measurements(self, config: Dict):
        """Handle run measurements request"""
        if self.current_masks is None:
            QMessageBox.warning(
                self,
                "No Segmentation",
                "Please run segmentation first before extracting measurements."
            )
            return
        
        if self.current_image_index is None:
            return
        
        # Get current image data
        img_data = self.project.get_image(self.current_image_index)
        if img_data is None:
            return
        
        # Load image to get intensity data
        try:
            image, metadata = TIFFLoader.load_tiff(img_data.path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load image: {e}")
            return
        
        # Prepare intensity images dict
        self.current_intensity_images = {}
        
        # Get channel names from metadata
        channel_names = metadata.get('channel_names', [])
        
        # Handle 3D vs 2D
        is_3d = image.ndim == 4
        
        if is_3d:
            # For 3D, we'll use max projection for now or full 3D depending on config
            if config.get('is_3d', False):
                # Use full 3D data
                for i, ch_name in enumerate(channel_names):
                    self.current_intensity_images[ch_name] = image[:, i, :, :]
            else:
                # Use max projection for 2D analysis
                for i, ch_name in enumerate(channel_names):
                    self.current_intensity_images[ch_name] = np.max(image[:, i, :, :], axis=0)
        else:
            # 2D image
            for i, ch_name in enumerate(channel_names):
                self.current_intensity_images[ch_name] = image[i, :, :]
        
        # Get DNA channel from image viewer
        dna_channel_index = self.image_viewer.get_dna_channel_index()
        if dna_channel_index is not None and dna_channel_index < len(channel_names):
            config['dna_channel'] = channel_names[dna_channel_index]
        
        # Create and start measurement worker
        self.measurement_worker = MeasurementWorker(
            masks=self.current_masks,
            intensity_images=self.current_intensity_images,
            config=config,
            plugin_loader=self.plugin_loader
        )
        
        # Connect signals
        self.measurement_worker.finished.connect(self._on_measurements_finished)
        self.measurement_worker.error.connect(self._on_measurements_error)
        self.measurement_worker.progress.connect(self.analysis_panel.set_progress)
        self.measurement_worker.status.connect(self.statusBar().showMessage)
        
        # Start worker
        self.measurement_worker.start()
        self.statusBar().showMessage("Running measurements...", 0)
    
    def _on_measurements_finished(self, measurements_df: pd.DataFrame):
        """Handle measurements completion"""
        self.current_measurements = measurements_df
        
        # Update analysis panel
        self.analysis_panel.set_measurements(measurements_df)
        
        # Update visualization panel
        self.visualization_panel.set_measurements(measurements_df)
        
        # Store in project
        if self.current_image_index is not None:
            img_data = self.project.get_image(self.current_image_index)
            if img_data:
                # Store measurements (simplified - in full version might serialize to file)
                img_data.measurements = measurements_df.to_dict()
        
        self.statusBar().showMessage(
            f"Measurements complete: {len(measurements_df)} nuclei analyzed", 5000
        )
        
        # Automatically switch to visualization tab
        self.tab_widget.setCurrentWidget(self.visualization_tab)
    
    def _on_measurements_error(self, error_message: str):
        """Handle measurement error"""
        QMessageBox.critical(self, "Measurement Error", error_message)
        self.statusBar().showMessage("Measurement failed", 5000)
    
    def _autosave(self):
        """Auto-save the current project"""
        autosave_enabled = SettingsDialog.get_setting("general/autosave_enabled", True)
        if self.project and self.project.project_path and autosave_enabled:
            try:
                self.project.save()
                print("Auto-saved project")
            except Exception as e:
                print(f"Auto-save failed: {e}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.project and self._check_unsaved_changes():
            event.ignore()
            return
        
        # Close project database if open
        if self.project:
            self.project.close()
        
        event.accept()
