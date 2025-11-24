# Integration Guide for Tasks 7-11

This document describes how to integrate the newly created components (Quality Control, Manual Correction, Measurements, Plugin System, Visualization) into the main application.

## Overview of New Components

### Task 7: Quality Control System
**Files Created:**
- `core/quality_control.py` - QualityControl class for DNA intensity analysis
- `gui/qc_panel.py` - QCPanel widget for interactive QC review

**Key Features:**
- DNA intensity histogram with GMM-based cell cycle phase detection
- Outlier flagging (default: 5th-95th percentile)
- Visual flagged nucleus gallery with confirm/reject buttons
- Automatic parameter suggestion when >5% confirmed errors
- Manual phase boundary adjustment with draggable lines

### Task 8: Manual Correction Tools
**Files Created:**
- `gui/manual_correction_tools.py` - Correction tools with undo/redo

**Key Features:**
- **SplitCommand**: Draw line to split nucleus (watershed-based)
- **MergeCommand**: Multi-select nuclei with Ctrl+click
- **DeleteCommand**: Remove nucleus
- **AddCommand**: Draw polygon to add nucleus
- **EditCommand** pattern with 50-operation undo stack
- Edit log tracking for reproducibility

### Task 9: Measurement Engine
**Files Created:**
- `core/measurements.py` - MeasurementEngine for quantification

**Key Features:**
- 2D measurements: area, perimeter, circularity, equivalent diameter
- 3D measurements: volume, surface area, sphericity
- Advanced morphology: eccentricity, solidity, aspect ratio, orientation
- Intensity stats: mean, min, max, median, std, integrated density, CV
- Cell cycle phase assignment (G1/S/G2M) using GMM on DNA intensity
- Plugin execution support

### Task 10: Plugin System
**Files Created:**
- `core/plugin_loader.py` - PluginLoader for dynamic plugin discovery

**Key Features:**
- Automatic plugin discovery in `plugins/` directory
- Dynamic loading with error handling
- Plugin metadata extraction (name, description, version)
- Reload functionality for development
- Plugin execution with error isolation

### Task 11: Visualization Tab
**Files Created:**
- `gui/visualization_panel.py` - Interactive Plotly-based visualization
- `gui/analysis_panel.py` - Measurement configuration and results

**Key Features:**
- DNA intensity histogram with phase coloring
- Area vs DNA scatter plots
- Morphology box plots
- Scatter matrix for multi-variate analysis
- Correlation heatmap
- Phase distribution bar chart
- Export to PNG/SVG/HTML
- Bidirectional linking (plot click → image viewer highlight)

---

## Integration Steps

### Step 1: Update Main Window to Include New Tabs

**File:** `gui/main_window.py`

Add imports:
```python
from gui.qc_panel import QCPanel
from gui.analysis_panel import AnalysisPanel
from gui.visualization_panel import VisualizationPanel
from gui.manual_correction_tools import CorrectionToolbar, ManualCorrectionManager
from core.quality_control import QualityControl
from core.measurements import MeasurementEngine
from core.plugin_loader import PluginLoader
```

In `__init__`:
```python
# Initialize new components
self.qc = QualityControl()
self.measurement_engine = MeasurementEngine()
self.plugin_loader = PluginLoader()
self.plugin_loader.load_all_plugins()
self.correction_manager = None  # Created when mask is loaded
```

Update `_create_analysis_tab`:
```python
def _create_analysis_tab(self):
    """Create analysis tab with measurements"""
    self.analysis_panel = AnalysisPanel()
    
    # Connect signals
    self.analysis_panel.run_measurements.connect(self._on_run_measurements)
    
    # Set available plugins
    plugin_info = self.plugin_loader.get_all_plugin_info()
    self.analysis_panel.set_plugin_info(plugin_info)
    
    return self.analysis_panel
```

Update `_create_visualization_tab`:
```python
def _create_visualization_tab(self):
    """Create visualization tab"""
    self.viz_panel = VisualizationPanel()
    
    # Connect bidirectional linking
    self.viz_panel.nucleus_selected.connect(self._on_nucleus_selected_from_plot)
    
    return self.viz_panel
```

### Step 2: Add QC Panel to Segmentation Tab

In `_create_segmentation_tab`, add QC panel below segmentation results:

```python
def _create_segmentation_tab(self):
    # ... existing code ...
    
    # Add QC panel
    self.qc_panel = QCPanel()
    self.qc_panel.nucleus_selected.connect(self._on_nucleus_selected_from_qc)
    self.qc_panel.request_resegmentation.connect(self._on_qc_resegmentation_requested)
    
    # Add to layout (below image viewer)
    segmentation_layout.addWidget(self.qc_panel)
    
    return seg_tab
```

### Step 3: Add Manual Correction Tools

After segmentation is complete, enable correction tools:

```python
def _on_segmentation_finished(self, masks, stats, elapsed_time):
    # ... existing code ...
    
    # Initialize correction manager
    self.correction_manager = ManualCorrectionManager(masks)
    
    # Add correction toolbar to segmentation tab
    if not hasattr(self, 'correction_toolbar'):
        self.correction_toolbar = CorrectionToolbar()
        self.correction_toolbar.tool_changed.connect(self._on_correction_tool_changed)
        self.correction_toolbar.undo_requested.connect(self._on_undo_correction)
        self.correction_toolbar.redo_requested.connect(self._on_redo_correction)
        
        # Insert toolbar above image viewer
        seg_layout = self.segmentation_tab.layout()
        seg_layout.insertWidget(0, self.correction_toolbar)
    
    # Enable toolbar
    self.correction_toolbar.setEnabled(True)
```

### Step 4: Implement Correction Tool Handlers

```python
def _on_correction_tool_changed(self, tool: str):
    """Handle tool change"""
    # Update image viewer interaction mode
    self.image_viewer.set_interaction_mode(tool)
    
def _on_undo_correction(self):
    """Undo last edit"""
    if self.correction_manager:
        new_mask = self.correction_manager.undo()
        if new_mask is not None:
            self.image_viewer.set_mask(new_mask)
            self.correction_toolbar.update_edit_count(
                self.correction_manager.get_edit_count()
            )

def _on_redo_correction(self):
    """Redo last undone edit"""
    if self.correction_manager:
        new_mask = self.correction_manager.redo()
        if new_mask is not None:
            self.image_viewer.set_mask(new_mask)
            self.correction_toolbar.update_edit_count(
                self.correction_manager.get_edit_count()
            )
```

### Step 5: Implement QC Workflow

```python
def _run_qc_analysis(self):
    """Run QC analysis on current segmentation"""
    if self.current_masks is None or self.current_image is None:
        return
    
    # Get DNA channel
    dna_channel_idx = self.image_viewer.get_dna_channel()
    if dna_channel_idx < 0 or dna_channel_idx >= self.current_image.shape[1]:
        QMessageBox.warning(self, "No DNA Channel", "Please select DNA channel")
        return
    
    dna_image = self.current_image[:, dna_channel_idx, :, :]
    
    # Get QC parameters
    n_phases = self.qc_panel.phases_spin.value()
    low_pct = self.qc_panel.low_percentile_spin.value()
    high_pct = self.qc_panel.high_percentile_spin.value()
    
    # Run analysis
    qc_results = self.qc.analyze_dna_intensity(
        self.current_masks,
        dna_image,
        n_phases=n_phases,
        percentile_threshold=(low_pct, high_pct)
    )
    
    # Get current segmentation parameters
    current_params = self.segmentation_panel.get_parameters()
    
    # Display results
    self.qc_panel.set_qc_results(qc_results, current_params)

def _on_qc_resegmentation_requested(self, suggested_params: dict):
    """Handle QC-suggested re-segmentation"""
    reply = QMessageBox.question(
        self,
        "Re-run Segmentation",
        "QC analysis suggests re-running segmentation with adjusted parameters.\n"
        "Would you like to apply these changes?",
        QMessageBox.Yes | QMessageBox.No
    )
    
    if reply == QMessageBox.Yes:
        # Update parameters
        self.segmentation_panel.set_parameters(suggested_params)
        
        # Re-run segmentation
        self._on_run_segmentation()
```

### Step 6: Implement Measurement Workflow

```python
def _on_run_measurements(self, config: dict):
    """Run measurements with given configuration"""
    if self.current_masks is None or self.current_image is None:
        QMessageBox.warning(self, "No Data", "Load image and run segmentation first")
        return
    
    # Use corrected mask if available
    masks = self.correction_manager.get_current_mask() if self.correction_manager else self.current_masks
    
    # Prepare intensity images dict
    intensity_images = {}
    n_channels = self.current_image.shape[1] if self.current_image.ndim == 4 else 1
    
    for i in range(n_channels):
        if self.current_image.ndim == 4:
            img = self.current_image[:, i, :, :]
        else:
            img = self.current_image
        
        channel_name = f"channel_{i}"
        # Use DNA for first channel if available
        if i == self.image_viewer.get_dna_channel():
            channel_name = "dna"
        
        intensity_images[channel_name] = img
    
    # Configure measurement engine
    self.measurement_engine.set_enabled_categories(config['enabled_categories'])
    
    # Extract measurements
    df = self.measurement_engine.extract_measurements(
        masks,
        intensity_images,
        is_3d=config['is_3d'],
        dna_channel='dna',
        assign_phases=config['assign_phases']
    )
    
    # Execute plugins if selected
    if config['enabled_plugins']:
        plugin_instances = [
            self.plugin_loader.get_plugin_instance(p) 
            for p in config['enabled_plugins']
        ]
        plugin_instances = [p for p in plugin_instances if p is not None]
        
        plugin_df = self.measurement_engine.execute_plugins(
            masks,
            intensity_images,
            plugin_instances
        )
        
        # Merge with main measurements
        df = pd.merge(df, plugin_df, on='nucleus_id', how='left')
    
    # Store measurements
    self.current_measurements = df
    
    # Display in analysis panel
    self.analysis_panel.set_measurements(df)
    
    # Display in visualization panel
    self.viz_panel.set_measurements(df)
    
    # Save to project
    if self.current_project and self.current_image_name:
        image_data = self.current_project.get_image(self.current_image_name)
        if image_data:
            image_data.measurements_df = df
            self.current_project.save()
```

### Step 7: Implement Bidirectional Linking

```python
def _on_nucleus_selected_from_plot(self, nucleus_id: int):
    """Handle nucleus selection from visualization"""
    # Highlight in image viewer
    self.image_viewer.highlight_nucleus(nucleus_id)
    
    # Switch to segmentation tab
    self.tab_widget.setCurrentIndex(0)

def _on_nucleus_selected_from_qc(self, nucleus_id: int):
    """Handle nucleus selection from QC panel"""
    self.image_viewer.highlight_nucleus(nucleus_id)
```

### Step 8: Add Menu Items

In `_create_menu_bar`:

```python
# Tools menu additions
tools_menu = menubar.addMenu("Tools")

qc_action = tools_menu.addAction("Run QC Analysis")
qc_action.triggered.connect(self._run_qc_analysis)

measure_action = tools_menu.addAction("Run Measurements")
measure_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(1))  # Switch to analysis tab

reload_plugins_action = tools_menu.addAction("Reload Plugins")
reload_plugins_action.triggered.connect(self._reload_plugins)

def _reload_plugins(self):
    """Reload plugins"""
    count = self.plugin_loader.reload_plugins()
    plugin_info = self.plugin_loader.get_all_plugin_info()
    self.analysis_panel.set_plugin_info(plugin_info)
    
    QMessageBox.information(self, "Plugins Reloaded", f"Loaded {count} plugins")
```

---

## Testing Checklist

### Quality Control
- [ ] DNA histogram displays correctly
- [ ] Phase boundaries are visible
- [ ] Outliers are flagged
- [ ] Confirmed error tracking works
- [ ] Parameter suggestions are generated
- [ ] Clicking "View" highlights nucleus in image viewer

### Manual Correction
- [ ] Split tool divides nucleus correctly
- [ ] Merge tool combines selected nuclei
- [ ] Delete removes nucleus
- [ ] Add creates new nucleus from ROI
- [ ] Undo/redo stack works (up to 50 operations)
- [ ] Edit counter updates correctly

### Measurements
- [ ] 2D measurements extract correctly
- [ ] 3D measurements work for volume data
- [ ] Intensity stats calculate for each channel
- [ ] Cell cycle phases assigned correctly
- [ ] Results table displays all measurements
- [ ] Export CSV/Excel functions work

### Plugin System
- [ ] Plugins auto-discovered from plugins/ folder
- [ ] Example plugins load correctly
- [ ] Plugin execution produces results
- [ ] Plugin errors are caught and displayed
- [ ] Reload functionality works

### Visualization
- [ ] All plot types render correctly
- [ ] Plotly interactivity works (zoom, pan, hover)
- [ ] Phase coloring appears in plots
- [ ] Clicking plot point highlights nucleus
- [ ] Export functionality works
- [ ] Axis selectors update plot

---

## Known Issues and TODOs

1. **QWebChannel Integration**: Bidirectional linking from plot to image viewer requires QWebChannel implementation for JavaScript-Qt communication

2. **Type Hints**: Some linter errors for GMM `.flatten()` and `.shape` - these are numpy array methods and work at runtime

3. **3D Visualization**: Currently only 2D plots; 3D surface plots for volumetric data would enhance analysis

4. **Real-time Correction Preview**: Draw tools currently work on mask array; integrating with PyQtGraph overlay would improve UX

5. **Plugin Hot-reload**: File watcher for automatic plugin reload on file save

6. **Batch Processing**: Extend to process multiple images with same pipeline

---

## Performance Considerations

1. **Large Datasets**: For >1000 nuclei, consider:
   - Pagination in results table
   - Downsampling for scatter plots
   - SQLite backend for measurements

2. **3D Data**: Volume rendering and 3D measurements are computationally expensive:
   - Show progress bars
   - Use worker threads
   - Consider GPU acceleration for watershed

3. **Plugin Execution**: Plugins run sequentially:
   - Future: parallelize with multiprocessing
   - Set timeout for long-running plugins

---

## Architecture Summary

```
Main Window
├── Segmentation Tab
│   ├── Image Viewer (multi-channel, Z-slices, mask overlay)
│   ├── Segmentation Panel (Cellpose/SAM controls)
│   ├── Correction Toolbar (split/merge/delete/add tools)
│   └── QC Panel (DNA histogram, flagged nuclei)
│
├── Analysis Tab
│   ├── Workflow Selection (2D/3D)
│   ├── Category Checkboxes (shape, morphology, intensity, cell cycle)
│   ├── Plugin Selection List
│   └── Results Table + Summary Stats
│
└── Visualization Tab
    ├── Plot Controls (type, axes, color)
    ├── Plotly View (interactive plots)
    └── Export Buttons (PNG/SVG/HTML)

Core Modules
├── quality_control.py (QualityControl class)
├── measurements.py (MeasurementEngine class)
├── plugin_loader.py (PluginLoader class)
├── segmentation.py (SegmentationEngine class)
├── image_io.py (TIFFLoader class)
└── project_data.py (Project, ImageData classes)
```

---

## Next Steps

After integration:

1. **Task 12**: QC Dashboard with aggregate statistics across multiple images
2. **Task 13**: Batch processing pipeline for high-throughput analysis
3. **Task 14**: PyInstaller build for portable Windows installer
4. **Testing**: Comprehensive unit and integration tests
5. **Documentation**: User manual with screenshots and workflows
