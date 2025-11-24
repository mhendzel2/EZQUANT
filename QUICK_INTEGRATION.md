# Quick Integration Checklist - Tasks 7-11

This is a condensed checklist for integrating the new components into `main_window.py`.

## Prerequisites
- All previous tasks (1-6) completed
- Working Import → View → Segment workflow

---

## Step-by-Step Integration

### 1. Add Imports (top of main_window.py)

```python
from gui.qc_panel import QCPanel
from gui.analysis_panel import AnalysisPanel
from gui.visualization_panel import VisualizationPanel
from gui.manual_correction_tools import CorrectionToolbar, ManualCorrectionManager
from core.quality_control import QualityControl
from core.measurements import MeasurementEngine
from core.plugin_loader import PluginLoader
```

---

### 2. Initialize Components (in __init__)

```python
# Add to __init__ after existing initialization
self.qc = QualityControl()
self.measurement_engine = MeasurementEngine()
self.plugin_loader = PluginLoader()
self.plugin_loader.load_all_plugins()
self.correction_manager = None
self.current_measurements = None
```

---

### 3. Update Tab Creation

**Replace _create_analysis_tab:**
```python
def _create_analysis_tab(self):
    self.analysis_panel = AnalysisPanel()
    self.analysis_panel.run_measurements.connect(self._on_run_measurements)
    plugin_info = self.plugin_loader.get_all_plugin_info()
    self.analysis_panel.set_plugin_info(plugin_info)
    return self.analysis_panel
```

**Replace _create_visualization_tab:**
```python
def _create_visualization_tab(self):
    self.viz_panel = VisualizationPanel()
    self.viz_panel.nucleus_selected.connect(self._on_nucleus_selected_from_plot)
    return self.viz_panel
```

---

### 4. Add QC Panel to Segmentation Tab

**In _create_segmentation_tab, before return:**
```python
# Add QC panel
self.qc_panel = QCPanel()
self.qc_panel.nucleus_selected.connect(self._on_nucleus_selected_from_qc)
self.qc_panel.request_resegmentation.connect(self._on_qc_resegmentation_requested)
left_layout.addWidget(self.qc_panel)
```

---

### 5. Add Manual Correction Support

**In _on_segmentation_finished, after displaying mask:**
```python
# Initialize correction manager
self.correction_manager = ManualCorrectionManager(masks)

# Add correction toolbar if not exists
if not hasattr(self, 'correction_toolbar'):
    self.correction_toolbar = CorrectionToolbar()
    self.correction_toolbar.tool_changed.connect(self._on_correction_tool_changed)
    self.correction_toolbar.undo_requested.connect(self._on_undo_correction)
    self.correction_toolbar.redo_requested.connect(self._on_redo_correction)
    
    # Insert at top of segmentation tab
    seg_widget = self.tab_widget.widget(0)
    seg_layout = seg_widget.layout()
    seg_layout.insertWidget(0, self.correction_toolbar)

self.correction_toolbar.setEnabled(True)
self.correction_toolbar.update_edit_count(0)

# Auto-run QC
self._run_qc_analysis()
```

---

### 6. Add New Methods

**Copy-paste these methods to main_window.py:**

```python
def _on_correction_tool_changed(self, tool: str):
    """Handle correction tool change"""
    # Future: Update image viewer interaction mode
    pass

def _on_undo_correction(self):
    """Undo last correction"""
    if self.correction_manager:
        new_mask = self.correction_manager.undo()
        if new_mask is not None:
            self.image_viewer.set_mask(new_mask)
            self.correction_toolbar.update_edit_count(
                self.correction_manager.get_edit_count()
            )
            self.correction_toolbar.set_undo_enabled(
                self.correction_manager.can_undo()
            )
            self.correction_toolbar.set_redo_enabled(
                self.correction_manager.can_redo()
            )

def _on_redo_correction(self):
    """Redo last correction"""
    if self.correction_manager:
        new_mask = self.correction_manager.redo()
        if new_mask is not None:
            self.image_viewer.set_mask(new_mask)
            self.correction_toolbar.update_edit_count(
                self.correction_manager.get_edit_count()
            )
            self.correction_toolbar.set_undo_enabled(
                self.correction_manager.can_undo()
            )
            self.correction_toolbar.set_redo_enabled(
                self.correction_manager.can_redo()
            )

def _run_qc_analysis(self):
    """Run QC analysis"""
    if self.current_masks is None or self.current_image is None:
        return
    
    dna_channel_idx = self.image_viewer.dna_channel_combo.currentIndex()
    if dna_channel_idx < 0:
        return
    
    if self.current_image.ndim == 4:
        dna_image = self.current_image[:, dna_channel_idx, :, :]
    else:
        dna_image = self.current_image
    
    n_phases = self.qc_panel.phases_spin.value()
    low_pct = self.qc_panel.low_percentile_spin.value()
    high_pct = self.qc_panel.high_percentile_spin.value()
    
    qc_results = self.qc.analyze_dna_intensity(
        self.current_masks,
        dna_image,
        n_phases=n_phases,
        percentile_threshold=(low_pct, high_pct)
    )
    
    current_params = self.segmentation_panel.get_parameters()
    self.qc_panel.set_qc_results(qc_results, current_params)

def _on_qc_resegmentation_requested(self, suggested_params: dict):
    """Handle QC suggested re-segmentation"""
    from PySide6.QtWidgets import QMessageBox
    reply = QMessageBox.question(
        self,
        "Re-run Segmentation",
        "Apply suggested parameter changes and re-run segmentation?",
        QMessageBox.Yes | QMessageBox.No
    )
    
    if reply == QMessageBox.Yes:
        self.segmentation_panel.set_parameters(suggested_params)
        self._on_run_segmentation()

def _on_nucleus_selected_from_qc(self, nucleus_id: int):
    """Handle nucleus selection from QC panel"""
    self.image_viewer.highlight_nucleus(nucleus_id)

def _on_nucleus_selected_from_plot(self, nucleus_id: int):
    """Handle nucleus selection from plot"""
    self.image_viewer.highlight_nucleus(nucleus_id)
    self.tab_widget.setCurrentIndex(0)  # Switch to segmentation tab

def _on_run_measurements(self, config: dict):
    """Run measurements with config"""
    from PySide6.QtWidgets import QMessageBox
    
    if self.current_masks is None or self.current_image is None:
        QMessageBox.warning(self, "No Data", "Load image and run segmentation first")
        return
    
    # Use corrected mask if available
    masks = (self.correction_manager.get_current_mask() 
             if self.correction_manager 
             else self.current_masks)
    
    # Prepare intensity images
    intensity_images = {}
    dna_idx = self.image_viewer.dna_channel_combo.currentIndex()
    
    if self.current_image.ndim == 4:
        n_channels = self.current_image.shape[1]
        for i in range(n_channels):
            name = "dna" if i == dna_idx else f"channel_{i}"
            intensity_images[name] = self.current_image[:, i, :, :]
    else:
        intensity_images["dna"] = self.current_image
    
    # Configure and run
    self.measurement_engine.set_enabled_categories(config['enabled_categories'])
    
    df = self.measurement_engine.extract_measurements(
        masks,
        intensity_images,
        is_3d=config['is_3d'],
        dna_channel='dna',
        assign_phases=config['assign_phases']
    )
    
    # Execute plugins
    if config['enabled_plugins']:
        plugin_instances = [
            self.plugin_loader.get_plugin_instance(p) 
            for p in config['enabled_plugins']
        ]
        plugin_instances = [p for p in plugin_instances if p]
        
        plugin_df = self.measurement_engine.execute_plugins(
            masks, intensity_images, plugin_instances
        )
        
        if not plugin_df.empty:
            df = pd.merge(df, plugin_df, on='nucleus_id', how='left')
    
    # Store and display
    self.current_measurements = df
    self.analysis_panel.set_measurements(df)
    self.viz_panel.set_measurements(df)
    
    # Save to project
    if self.current_project and self.current_image_name:
        image_data = self.current_project.get_image(self.current_image_name)
        if image_data:
            image_data.measurements_df = df
            self.current_project.save()
    
    QMessageBox.information(self, "Success", f"Extracted measurements for {len(df)} nuclei")
```

---

### 7. Add Menu Items

**In _create_menu_bar, add to Tools menu:**
```python
tools_menu = menubar.addMenu("Tools")

qc_action = tools_menu.addAction("Run QC Analysis")
qc_action.triggered.connect(self._run_qc_analysis)

measure_action = tools_menu.addAction("Run Measurements")
measure_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(1))

reload_plugins_action = tools_menu.addAction("Reload Plugins")
reload_plugins_action.triggered.connect(self._reload_plugins)

def _reload_plugins(self):
    from PySide6.QtWidgets import QMessageBox
    count = self.plugin_loader.reload_plugins()
    plugin_info = self.plugin_loader.get_all_plugin_info()
    self.analysis_panel.set_plugin_info(plugin_info)
    QMessageBox.information(self, "Success", f"Loaded {count} plugins")
```

---

## Verification

After integration, test this workflow:

1. **Load TIFF** → File → Import TIFF
2. **Segment** → Click "Run Segmentation"
3. **QC** → Should auto-run, showing histogram with flagged nuclei
4. **Correct** → Use split/merge/delete tools, test undo/redo
5. **Measure** → Switch to Analysis tab, select categories, click "Run Measurements"
6. **Visualize** → Switch to Visualization tab, explore plots
7. **Export** → Click "Export CSV" in Analysis tab

---

## Troubleshooting

**Import errors:** Run `pip install -r requirements_updated.txt`

**QC panel not showing:** Check that `left_layout.addWidget(self.qc_panel)` is in correct layout

**Measurements fail:** Verify DNA channel is selected in image viewer

**Plugins not loading:** Check plugins/ folder exists and contains .py files

**Plot not displaying:** Ensure `PySide6-WebEngine` is installed

---

## Done!

All tasks 7-11 are now integrated. The application has a complete workflow:

**Import → View → Segment → QC → Correct → Measure → Visualize → Export**
