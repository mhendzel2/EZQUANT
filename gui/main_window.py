"""
Main window for the Nuclei Segmentation Application
"""

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QTabWidget, QMenuBar, QMenu, QToolBar, QStatusBar,
                               QFileDialog, QMessageBox, QDockWidget, QLabel,
                               QSplitter, QInputDialog, QProgressDialog, QDialogButtonBox, QApplication)
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
from core.policy_engine import PolicyEngine
from gui.image_viewer import ImageViewer
from gui.segmentation_panel import SegmentationPanel
from gui.analysis_panel import AnalysisPanel
from gui.visualization_panel import VisualizationPanel
from gui.settings_dialog import SettingsDialog
from gui.project_panel import ProjectPanel
from gui.accessibility import AccessibilityManager
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
        
        # Dirty flag for tracking unsaved changes
        self.dirty = False
        
        # Plugin loader
        self.plugin_loader = PluginLoader()
        self.plugin_loader.load_all_plugins()

        # Policy engine (role-based access control)
        self.policy_engine = PolicyEngine()

        # Accessibility manager
        self.accessibility = AccessibilityManager.instance()
        
        # Quality dashboard reference
        self.quality_dashboard = None
        
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
        self.analysis_panel.refresh_plugins_requested.connect(self._on_refresh_plugins)
        
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

        self.import_folder_action = QAction("Import &Folder...", self)
        self.import_folder_action.setShortcut(QKeySequence("Ctrl+Shift+I"))
        self.import_folder_action.setStatusTip("Import all images from a folder")
        self.import_folder_action.triggered.connect(self.import_folder)
        
        self.export_action = QAction("&Export Measurements...", self)
        self.export_action.setShortcut(QKeySequence("Ctrl+E"))
        self.export_action.setStatusTip("Export measurements to CSV or Excel")
        self.export_action.triggered.connect(self.export_measurements)
        
        self.batch_action = QAction("&Batch Process...", self)
        self.batch_action.setStatusTip("Process multiple images")
        self.batch_action.triggered.connect(self.batch_process)
        
        self.close_image_action = QAction("Close &Image", self)
        self.close_image_action.setShortcut(QKeySequence("Ctrl+W"))
        self.close_image_action.setStatusTip("Close the current image")
        self.close_image_action.triggered.connect(self.close_current_image)
        
        self.close_project_action = QAction("&Close Project", self)
        self.close_project_action.setShortcut(QKeySequence("Ctrl+Shift+W"))
        self.close_project_action.setStatusTip("Close the current project")
        self.close_project_action.triggered.connect(self.close_project)
        
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
        file_menu.addAction(self.import_folder_action)
        file_menu.addAction(self.export_action)
        file_menu.addAction(self.batch_action)
        file_menu.addSeparator()
        file_menu.addAction(self.close_image_action)
        file_menu.addAction(self.close_project_action)
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
        toolbar.addAction(self.import_folder_action)
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

        # Role indicator
        self._role_label = QLabel(f"  Role: {self.policy_engine.current_role}  ")
        self._role_label.setToolTip("Current user role. Change in Settings → Role.")
        self.statusBar().addPermanentWidget(self._role_label)
    
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
    
    def close_project(self):
        """Close the current project"""
        if self.project and self._check_unsaved_changes():
            return
        
        # Clear everything
        self.project = None
        self.current_image_index = None
        self.current_masks = None
        self.current_intensity_images = None
        self.current_measurements = None
        
        # Clear UI
        self.project_panel.clear_images()
        self.image_viewer.clear()
        self.segmentation_panel.clear()
        self.analysis_panel.clear_measurements()
        self.visualization_panel.clear()
        
        # Update window title
        self.setWindowTitle("Nuclei Segmentation & Analysis")
        self.dirty = False
        
        self.project_changed.emit()
        self.statusBar().showMessage("Project closed", 3000)
    
    def close_current_image(self):
        """Close the currently displayed image"""
        if self.current_image_index is None:
            self.statusBar().showMessage("No image to close", 2000)
            return
        
        # Remove from project
        if self.project:
            self.project.remove_image(self.current_image_index)
            self.project_panel.remove_image(self.current_image_index)
        
        # Clear current state
        self.current_image_index = None
        self.current_masks = None
        self.current_intensity_images = None
        
        # Clear viewer
        self.image_viewer.clear()
        self.segmentation_panel.clear()
        
        self.statusBar().showMessage("Image closed", 2000)
    
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
            self.dirty = False
            # Remove unsaved indicator from title
            title = self.windowTitle()
            if title.endswith(' *'):
                self.setWindowTitle(title[:-2])
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
    
    def _import_single_image(self, filepath: str, group: str = "Default", update_view: bool = True):
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
                group=group,
                shape=metadata.get('final_shape'),
                dtype=metadata.get('dtype'),
                pixel_size=metadata.get('pixel_size'),
                bit_depth=metadata.get('bit_depth', 8)
            )
            
            # Add to project
            img_index = self.project.add_image(img_data)
            
            if update_view:
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
    
    def _import_metamorph_nd(self, nd_filepath: str, group: str = "Default"):
        """Import Metamorph .nd file series - automatically imports all channels"""
        try:
            # Load the .nd file with all channels automatically
            image, metadata = TIFFLoader.load_metamorph_nd(nd_filepath, stage=0, timepoint=0)
            
            # Create image data entry
            img_data = ImageData(
                path=nd_filepath,
                filename=Path(nd_filepath).name,
                added_date=str(Path(nd_filepath).stat().st_mtime),
                channels=metadata.get('channel_names', []),
                group=group,
                shape=metadata.get('final_shape'),
                dtype=metadata.get('dtype'),
                pixel_size=metadata.get('pixel_size'),
                bit_depth=metadata.get('bit_depth', 8)
            )
            
            # Add to project
            img_index = self.project.add_image(img_data)
            self.current_image_index = img_index
            
            # Update project panel
            self.project_panel.add_image(img_data.filename, has_segmentation=False)
            
            # Display in viewer
            self.image_viewer.set_image(image, metadata)
            self.segmentation_panel.set_image(image, metadata)
            self.image_loaded.emit(img_index)
            
            # Log channel info
            n_channels = metadata.get('n_channels', 1)
            channel_names = metadata.get('channel_names', [])
            channel_info = f"{n_channels} channel(s)"
            if channel_names:
                channel_info += f": {', '.join(channel_names[:3])}"
                if len(channel_names) > 3:
                    channel_info += f"..."
            
            self.statusBar().showMessage(
                f"Imported: {img_data.filename} ({channel_info})", 3000
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Importing Metamorph Series",
                f"Could not import .nd file:\n{str(e)}"
            )
    
    def _import_metamorph_batch(self, nd_filepath: str, import_all: bool = False, group: str = "Default"):
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
                            group=group,
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
                self._load_segmentation_mask(img_data)
            else:
                # Clear any previous masks
                self.current_masks = None
                self.image_viewer.clear_mask()
            
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
    
    def _load_segmentation_mask(self, img_data):
        """Load segmentation mask from saved file"""
        try:
            # Check if mask path is stored
            mask_path = getattr(img_data, 'mask_path', None)
            
            if mask_path is None and self.project and self.project.project_path:
                # Try to find mask file based on naming convention
                from pathlib import Path
                project_dir = Path(self.project.project_path).parent
                masks_dir = project_dir / "masks"
                mask_filename = f"{Path(img_data.filename).stem}_mask.npz"
                potential_path = masks_dir / mask_filename
                
                if potential_path.exists():
                    mask_path = str(potential_path)
            
            if mask_path and Path(mask_path).exists():
                data = np.load(mask_path)
                self.current_masks = data['masks']
                self.image_viewer.set_mask(self.current_masks)
                self.statusBar().showMessage(
                    f"Loaded existing segmentation for {img_data.filename}", 3000
                )
            else:
                # No mask file found
                self.current_masks = None
                
        except Exception as e:
            print(f"Warning: Could not load mask file: {e}")
            self.current_masks = None
    
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
        """Export measurements to file (CSV or Excel)"""
        if not self.project:
            QMessageBox.warning(self, "No Project", "Please create or open a project first.")
            return
        
        # Get aggregated measurements from project
        measurements_df = self.project.get_aggregated_measurements()
        
        if measurements_df is None or len(measurements_df) == 0:
            QMessageBox.warning(
                self,
                "No Measurements",
                "No measurements available to export.\n"
                "Please run segmentation and analysis first."
            )
            return
        
        # Open file dialog for save location
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Measurements",
            "measurements.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )
        
        if not filepath:
            return
        
        try:
            if filepath.endswith('.xlsx') or 'Excel' in selected_filter:
                if not filepath.endswith('.xlsx'):
                    filepath += '.xlsx'
                measurements_df.to_excel(filepath, index=False, engine='openpyxl')
            else:
                if not filepath.endswith('.csv'):
                    filepath += '.csv'
                measurements_df.to_csv(filepath, index=False)
            
            QMessageBox.information(
                self,
                "Export Successful",
                f"Measurements exported to:\n{filepath}\n\n"
                f"Total rows: {len(measurements_df)}\n"
                f"Columns: {len(measurements_df.columns)}"
            )
            self.statusBar().showMessage(f"Exported measurements to {filepath}", 5000)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Could not export measurements:\n{str(e)}"
            )

    def batch_process(self):
        """Launch batch processing dialog"""
        from gui.batch_processing import BatchProcessingDialog
        
        dialog = BatchProcessingDialog(self)
        
        # Pre-populate with project images if available
        if self.project and len(self.project.images) > 0:
            file_paths = [img.path for img in self.project.images]
            dialog.file_paths = file_paths
            dialog._update_file_list()
        
        if dialog.exec():
            # Get results and update project
            results = dialog.get_results()
            if results:
                self.dirty = True
                self.statusBar().showMessage(
                    f"Batch processing complete: {len(results)} images processed", 5000
                )

    def show_quality_dashboard(self):
        """Show quality metrics dashboard in a new tab"""
        from gui.quality_dashboard import QualityDashboard
        
        # Check if dashboard tab already exists
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Dashboard":
                self.tab_widget.setCurrentIndex(i)
                # Refresh data
                if self.quality_dashboard:
                    self._refresh_dashboard_data()
                return
        
        # Create new dashboard tab
        self.quality_dashboard = QualityDashboard()
        self.quality_dashboard.image_selected.connect(self._on_dashboard_image_selected)
        
        self.tab_widget.addTab(self.quality_dashboard, "Dashboard")
        self.tab_widget.setCurrentWidget(self.quality_dashboard)
        
        # Populate with project data
        self._refresh_dashboard_data()
    
    def _refresh_dashboard_data(self):
        """Refresh quality dashboard with current project data"""
        if not self.quality_dashboard or not self.project:
            return
        
        project_data = []
        for img_data in self.project.images:
            # Compile QC metrics for each image
            data_entry = {
                'image_name': img_data.filename,
                'n_nuclei': 0,
                'mean_area': 0.0,
                'cv_area': 0.0,
                'mean_dna_intensity': 0.0,
                'cv_dna_intensity': 0.0,
                'flagged_count': len(img_data.qc_flags),
                'flagged_percentage': 0.0,
                'qc_pass': True,
                'timestamp': img_data.added_date
            }
            
            # If measurements exist, compute stats
            if img_data.measurements_df is not None:
                df = pd.DataFrame(img_data.measurements_df) if isinstance(img_data.measurements_df, dict) else img_data.measurements_df
                if len(df) > 0:
                    data_entry['n_nuclei'] = len(df)
                    
                    if 'area' in df.columns:
                        data_entry['mean_area'] = df['area'].mean()
                        data_entry['cv_area'] = (df['area'].std() / df['area'].mean() * 100) if df['area'].mean() > 0 else 0
                    
                    # Find DNA intensity column
                    dna_cols = [c for c in df.columns if 'dna' in c.lower() and 'intensity' in c.lower()]
                    if dna_cols:
                        dna_col = dna_cols[0]
                        data_entry['mean_dna_intensity'] = df[dna_col].mean()
                        data_entry['cv_dna_intensity'] = (df[dna_col].std() / df[dna_col].mean() * 100) if df[dna_col].mean() > 0 else 0
                    
                    data_entry['flagged_percentage'] = (data_entry['flagged_count'] / data_entry['n_nuclei'] * 100) if data_entry['n_nuclei'] > 0 else 0
                    data_entry['qc_pass'] = data_entry['flagged_percentage'] < 10
            
            project_data.append(data_entry)
        
        self.quality_dashboard.set_project_data(project_data)
    
    def _on_dashboard_image_selected(self, image_name: str):
        """Handle image selection from dashboard"""
        if not self.project:
            return
        
        # Find image index by name
        for i, img_data in enumerate(self.project.images):
            if img_data.filename == image_name:
                self.project_panel.set_selected_index(i)
                self._on_project_image_selected(i)
                self.tab_widget.setCurrentIndex(0)  # Switch to segmentation tab
                break

    def show_plugin_manager(self):
        """Show plugin manager dialog"""
        from gui.plugin_manager_dialog import PluginManagerDialog
        
        dialog = PluginManagerDialog(self.plugin_loader, self)
        if dialog.exec():
            # Reload plugins if changes were made
            self.plugin_loader.reload_plugins()
            
            # Update analysis panel with new plugins
            plugin_info = self.plugin_loader.get_all_plugin_info()
            self.analysis_panel.set_plugin_info(plugin_info)
            
            self.statusBar().showMessage("Plugins reloaded", 3000)
    
    def _on_refresh_plugins(self):
        """Handle plugin refresh request from analysis panel"""
        # Reload plugins
        count = self.plugin_loader.reload_plugins()
        
        # Update analysis panel with new plugins
        plugin_info = self.plugin_loader.get_all_plugin_info()
        self.analysis_panel.set_plugin_info(plugin_info)
        
        self.statusBar().showMessage(f"Reloaded {count} plugins", 3000)
    
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
        """Show documentation in default browser"""
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        
        # Check for local documentation first
        docs_path = Path(__file__).parent.parent / "docs" / "QUICK_START.md"
        readme_path = Path(__file__).parent.parent / "README.md"
        
        if docs_path.exists():
            # Open local documentation
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(docs_path)))
            self.statusBar().showMessage("Opening documentation...", 3000)
        elif readme_path.exists():
            # Fallback to README
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(readme_path)))
            self.statusBar().showMessage("Opening README...", 3000)
        else:
            QMessageBox.information(
                self,
                "Documentation",
                "Documentation files not found.\n\n"
                "For help, please visit the project repository or\n"
                "refer to the included documentation files."
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
        if not self.dirty:
            return False
        
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes.\nDo you want to save before continuing?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            self.save_project()
            return False  # Continue with action
        elif reply == QMessageBox.No:
            return False  # Discard changes, continue with action
        else:
            return True  # Cancel the action
    
    def _mark_dirty(self):
        """Mark the project as having unsaved changes"""
        self.dirty = True
        # Update window title to indicate unsaved changes
        title = self.windowTitle()
        if not title.endswith('*'):
            self.setWindowTitle(title + ' *')
    
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
                seg_record = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'parameters': params,
                    'results': results
                }
                img_data.segmentation_history.append(seg_record)
                img_data.current_segmentation_id = len(img_data.segmentation_history) - 1
                
                # Mark as segmented in project panel
                self.project_panel.set_image_segmented(self.current_image_index, True)
                
                # Mark project as dirty (unsaved changes)
                self._mark_dirty()
                
                # Persist mask data to project directory
                self._save_segmentation_mask(img_data, masks)
        
        self.statusBar().showMessage("Segmentation complete - Ready for analysis", 5000)
    
    def _save_segmentation_mask(self, img_data, masks: np.ndarray):
        """Save segmentation mask to project directory"""
        if not self.project or not self.project.project_path:
            return
        
        try:
            from pathlib import Path
            
            # Create masks directory in project folder
            project_dir = Path(self.project.project_path).parent
            masks_dir = project_dir / "masks"
            masks_dir.mkdir(exist_ok=True)
            
            # Generate mask filename based on image
            mask_filename = f"{Path(img_data.filename).stem}_mask.npz"
            mask_path = masks_dir / mask_filename
            
            # Save mask as compressed numpy array
            np.savez_compressed(str(mask_path), masks=masks)
            
            # Store mask path in image data for later retrieval
            if not hasattr(img_data, 'mask_path'):
                img_data.mask_path = str(mask_path)
        except Exception as e:
            print(f"Warning: Could not save mask file: {e}")
    
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
        
        # Get current model name from segmentation panel for better diameter estimation
        model_name = self.segmentation_panel.model_combo.currentText() if hasattr(self.segmentation_panel, 'model_combo') else 'nuclei'
        
        # Create and start worker
        self.diameter_worker = DiameterEstimationWorker(
            image=image,
            gpu_available=self.gpu_available,
            model_name=model_name
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
    
    def import_folder(self):
        """Import all supported images from a folder"""
        if not self.project:
            self.new_project()
            
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Import",
            ""
        )
        
        if not folder_path:
            return
            
        # Ask for group name
        group, ok = QInputDialog.getText(
            self, 
            "Group Name", 
            "Enter group/treatment name for these images:",
            text="Default"
        )
        
        if not ok:
            return
            
        # Find all supported files
        supported_extensions = ['.tif', '.tiff', '.nd', '.vsi', '.lif']
        files_to_import = []
        
        folder = Path(folder_path)
        for ext in supported_extensions:
            files_to_import.extend(list(folder.glob(f"*{ext}")))
            files_to_import.extend(list(folder.glob(f"*{ext.upper()}")))
            
        # Remove duplicates (case insensitivity might cause duplicates on Windows)
        files_to_import = list(set(files_to_import))
        files_to_import.sort()
        
        if not files_to_import:
            QMessageBox.information(self, "No Images Found", f"No supported images found in {folder_path}")
            return
            
        # Progress dialog
        progress = QProgressDialog("Importing images...", "Cancel", 0, len(files_to_import), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show() # Ensure dialog is shown
        
        count = 0
        for i, filepath in enumerate(files_to_import):
            if progress.wasCanceled():
                break
                
            progress.setLabelText(f"Importing {filepath.name}...")
            QApplication.processEvents() # Keep UI responsive
            
            try:
                str_path = str(filepath)
                if str_path.lower().endswith('.nd'):
                    # For ND files, we might want to skip the interactive dialog in batch mode
                    # But for now, let's just call the method which might show dialogs
                    # Ideally we would refactor to avoid dialogs in batch mode
                    self._import_metamorph_nd(str_path, group=group)
                elif str_path.lower().endswith('.vsi'):
                    self._import_vsi(str_path) # TODO: Add group support to VSI
                elif str_path.lower().endswith('.lif'):
                    self._import_lif(str_path) # TODO: Add group support to LIF
                else:
                    self._import_single_image(str_path, group=group, update_view=False)
                
                count += 1
            except Exception as e:
                print(f"Error importing {filepath.name}: {e}")
                
            progress.setValue(i + 1)
            
        progress.close()
        
        # Load the last imported image into view if any
        if count > 0 and self.project and len(self.project.images) > 0:
            last_idx = len(self.project.images) - 1
            img_data = self.project.get_image(last_idx)
            if img_data:
                try:
                    image, metadata = TIFFLoader.load_tiff(img_data.path)
                    self.current_image_index = last_idx
                    self.image_viewer.set_image(image, metadata)
                    self.segmentation_panel.set_image(image, metadata)
                    self.image_loaded.emit(last_idx)
                except Exception as e:
                    print(f"Error loading last image: {e}")

        self.statusBar().showMessage(f"Imported {count} files from folder", 5000)
        self.statusBar().showMessage(f"Imported {count} files from folder", 5000)
