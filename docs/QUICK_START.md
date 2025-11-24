# Quick Start Guide - Nuclei Segmentation Application

Get started with the Nuclei Segmentation App in 5 minutes.

---

## First Launch

### Starting the Application

**Windows:**
- Double-click **Nuclei Segmentation App** on Desktop, or
- Start Menu → **Nuclei Segmentation App**

**From Source:**
```bash
python main.py
```

### Initial Setup

The application will:
1. Detect available GPU (if NVIDIA GPU with CUDA is present)
2. Display main window with empty project
3. Show status bar with GPU information

---

## Basic Workflow

### 1. Create a New Project

**File → New Project**
- Enter project name (e.g., "MyExperiment")
- Choose storage location
- Project file (`.json` or `.db`) will be created automatically

### 2. Load Your First Image

**File → Open Image** or drag-and-drop TIFF file
- Supported: 2D/3D TIFF, grayscale or multichannel
- Bit depth: 8-bit or 16-bit
- Image will appear in the viewer

### 3. Navigate the Image

**Segmentation Tab - Image Viewer:**
- **Zoom:** Mouse wheel or +/- buttons
- **Pan:** Click and drag
- **Z-slice:** Slider (for 3D images)
- **Channel:** Dropdown (for multichannel images)
- **Reset View:** Double-click image

### 4. Run Segmentation

**Segmentation Panel:**

**Quick Start (Cellpose):**
```
1. Model: nuclei
2. Diameter: 30 (or click "Estimate")
3. Flow threshold: 0.4
4. Click "Run Segmentation"
```

Wait 10-60 seconds depending on image size and GPU availability.

**Advanced (SAM):**
```
1. Switch to SAM tab
2. Load checkpoint file (sam_vit_h_4b8939.pth)
3. Click "Auto Segment"
```

### 5. Review Quality Control

**QC Panel (right side):**
- View **DNA intensity histogram** with phase boundaries
- Check **flagged nuclei** (highlighted in red)
- Review **QC statistics** (mean area, CV, pass rate)
- Accept suggestions or adjust parameters

### 6. Manual Corrections (if needed)

**Correction Toolbar:**
- **Split:** Draw line across touching nuclei
- **Merge:** Select multiple nuclei, click merge
- **Delete:** Click nucleus to remove
- **Add:** Draw polygon to add missing nucleus
- **Undo/Redo:** Ctrl+Z / Ctrl+Y (up to 50 operations)

### 7. Extract Measurements

**Analysis Tab:**
```
1. Select measurement mode:
   - 2D workflow (area, perimeter, circularity)
   - 3D workflow (volume, surface area, sphericity)
2. Choose channels for intensity measurements
3. Click "Extract Measurements"
4. View results in table
```

### 8. Visualize Results

**Visualization Tab:**

**Chart Types:**
- **Histogram:** Distribution of any metric
- **Scatter:** Relationship between two metrics
- **Box Plot:** Compare groups
- **Scatter Matrix:** All pairwise relationships
- **Correlation Heatmap:** Correlation matrix

**Interactive Features:**
- Click points to highlight nuclei in viewer
- Zoom, pan, export charts as PNG/SVG

### 9. Export Results

**File → Export:**
- **CSV:** Simple table format
- **Excel:** Formatted spreadsheet
- **Images:** Segmentation masks, overlays
- **Report:** HTML with QC summary

---

## Common Tasks

### Batch Process Multiple Images

**Tools → Batch Processing:**
```
1. Click "Add Files" to select multiple TIFFs
2. Configure segmentation parameters
3. Enable QC filtering (optional)
4. Click "Start Processing"
5. Results saved to output folder
```

### View Quality Dashboard

**Dashboard Tab:**
- See project-wide metrics across all images
- Trend plots (nucleus count over time)
- Pass/fail summary
- Export HTML report

### Load Plugins

**Tools → Plugin Manager:**
```
1. Browse available plugins
2. Enable/disable plugins
3. Plugins add custom measurements
4. Results appear in Analysis tab
```

### Save Your Work

**File → Save Project:**
- All images, masks, and measurements saved
- Reopen later with **File → Open Project**

---

## Keyboard Shortcuts

### General
- `Ctrl+N` - New Project
- `Ctrl+O` - Open Image
- `Ctrl+S` - Save Project
- `Ctrl+Q` - Quit

### Image Viewer
- `+` / `-` - Zoom in/out
- `Arrow Keys` - Pan
- `Home` - Reset view
- `Ctrl+Wheel` - Fine zoom

### Corrections
- `Ctrl+Z` - Undo
- `Ctrl+Y` - Redo
- `Delete` - Delete selected nucleus
- `Esc` - Cancel current operation

### Analysis
- `Ctrl+M` - Extract Measurements
- `Ctrl+E` - Export Results

---

## Tips for Best Results

### Image Preparation
✓ Use high-quality TIFF files (not JPG/PNG)  
✓ Ensure proper contrast (nuclear stain clearly visible)  
✓ Avoid oversaturation (pixel values not at max)  
✓ Remove out-of-focus slices (for 3D)

### Segmentation Parameters
✓ **Diameter**: Should match average nucleus size in pixels  
   - Too small → over-segmentation (nuclei split)  
   - Too large → under-segmentation (nuclei merged)  
✓ **Flow threshold**: Lower = more sensitive (more detections)  
✓ **Use GPU** when available (5-10× faster)

### Quality Control
✓ Review flagged nuclei (may indicate:)  
   - Debris or artifacts  
   - Mitotic cells  
   - Apoptotic cells  
✓ Adjust QC thresholds in settings if needed  
✓ Use manual corrections for critical analyses

### Measurements
✓ **2D workflow** for:  
   - Single Z-plane images  
   - Maximum intensity projections  
   - When Z-resolution is poor  

✓ **3D workflow** for:  
   - Z-stacks with good axial resolution  
   - Volume measurements needed  
   - When analyzing 3D morphology

### Performance
✓ Close other GPU applications when segmenting  
✓ Batch processing is faster than manual iteration  
✓ Save project periodically (auto-save every 5 minutes)  
✓ Use SQLite storage for large projects (>500 MB)

---

## Troubleshooting

### Application Won't Launch
**Problem:** Double-clicking does nothing

**Solutions:**
1. Check Windows Event Viewer for errors
2. Run from Command Prompt to see error messages:
   ```
   "C:\Program Files\NucleiSegmentationApp\NucleiSegmentationApp.exe"
   ```
3. Reinstall application
4. Check system requirements (Windows 10+, 8GB RAM minimum)

### Segmentation Errors
**Problem:** "CUDA out of memory" or crashes

**Solutions:**
1. Reduce image size (crop or downsample)
2. Switch to CPU mode (Settings → Preferences)
3. Close other applications
4. Restart application
5. Update GPU drivers

### GPU Not Detected
**Problem:** "No GPU available, using CPU"

**Solutions:**
1. Install NVIDIA CUDA Toolkit 11.8+
2. Update GPU drivers
3. Verify GPU in Task Manager → Performance
4. Restart application

### Import Errors
**Problem:** "Cannot load TIFF file"

**Solutions:**
1. Verify file is valid TIFF (open in ImageJ/Fiji)
2. Check file isn't corrupted
3. Try exporting from ImageJ as uncompressed TIFF
4. Ensure file isn't open in another application

### Slow Performance
**Problem:** Application is sluggish

**Solutions:**
1. Enable GPU acceleration (Settings → Preferences)
2. Reduce image dimensions if very large (>4K)
3. Close unused tabs
4. Increase RAM allocation (Settings → Performance)
5. Save and restart application

---

## Example Workflow: HeLa Cells

**Sample data workflow for DAPI-stained HeLa nuclei:**

1. **Load Image:** HeLa_DAPI.tif (2048×2048, 16-bit)

2. **Segmentation:**
   - Model: `nuclei`
   - Diameter: `35` pixels
   - Flow threshold: `0.4`
   - GPU: Enabled
   - Time: ~5 seconds

3. **QC Results:**
   - 247 nuclei detected
   - 12 flagged (outliers)
   - Mean area: 850 px²
   - Pass rate: 95%

4. **Corrections:**
   - Split 3 touching nuclei (Split tool)
   - Delete 2 debris objects (Delete tool)
   - Final count: 248 nuclei

5. **Measurements:**
   - Mode: 2D workflow
   - Channels: DAPI only
   - Metrics: Area, perimeter, circularity, mean intensity, integrated intensity
   - Time: ~2 seconds

6. **Visualization:**
   - Histogram: Nucleus area distribution
   - Scatter: Area vs. mean intensity
   - Result: Two populations visible (G1 and G2/M phases)

7. **Export:**
   - Format: Excel
   - File: HeLa_measurements.xlsx
   - Includes: All metrics + QC flags

**Total time:** ~3 minutes (including manual review)

---

## Next Steps

### Learn More
- **Full Documentation:** See `docs/` folder
- **Plugin Development:** `docs/PLUGIN_DEVELOPMENT.md`
- **Advanced Features:** `docs/INTEGRATION_GUIDE_TASKS_7_11.md`

### Get Help
- **Issues:** GitHub Issues (https://github.com/yourlab/nuclei-segmentation/issues)
- **Email:** support@yourlab.edu
- **Tutorials:** YouTube channel (link)

### Contribute
- Report bugs on GitHub
- Submit feature requests
- Share your plugins
- Contribute code (pull requests welcome)

---

## System Requirements

### Minimum
- **OS:** Windows 10 (64-bit)
- **RAM:** 8 GB
- **Storage:** 5 GB free space
- **Display:** 1920×1080 resolution

### Recommended
- **OS:** Windows 11 (64-bit)
- **RAM:** 16 GB+
- **GPU:** NVIDIA GPU with 6GB+ VRAM (for GPU acceleration)
- **Storage:** 10 GB free space (SSD preferred)
- **Display:** 2560×1440 resolution or higher

### For Large 3D Images
- **RAM:** 32 GB+
- **GPU:** NVIDIA RTX 3060 or better (8GB+ VRAM)
- **Storage:** 50 GB+ free space

---

**Happy Segmenting! 🔬**

*Last Updated: November 22, 2025*  
*Version: 1.0.0*
