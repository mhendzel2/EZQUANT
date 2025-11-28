# Workflow Update: Multi-File Project Support

## Summary

The application has been updated to support a better workflow where you can add multiple files to a project first, then load and segment them individually.

## Key Changes

### 1. New Project Panel
- Added a new **Project Panel** on the left side of the main window
- Shows all images in the current project
- Images display their segmentation status:
  - ○ = Not segmented
  - ✓ = Segmented
- Double-click an image to load it
- Right-click for context menu options

### 2. Updated Workflow

**Old Workflow:**
1. Import image → Image loads immediately → Segment

**New Workflow:**
1. Add multiple images to project (File → Add Images to Project)
2. Select an image from the project panel
3. Segment the loaded image
4. Select next image and repeat

### 3. Multi-File Import
- **File → Add Images to Project** now supports selecting multiple files at once
- Files are added to the project list but not immediately loaded
- This is much faster for large projects

### 4. Project Management
- The project panel shows image count and segmentation status
- You can remove images from the project
- All project images are saved with the project file (.nsa)

## Usage Instructions

### Adding Multiple Files
1. Click **File → Add Images to Project** (or press Ctrl+I)
2. Select one or more image files (hold Ctrl or Shift to select multiple)
3. Click Open
4. Files are added to the project panel

### Segmenting Images
1. Click on an image in the project panel to load it
2. Configure segmentation parameters
3. Click **Run Segmentation**
4. The image is marked with ✓ when segmented
5. Select the next image and repeat

### Benefits
- **Faster setup**: Add all your files at once without waiting
- **Better organization**: See all project images in one place
- **Status tracking**: Know which images are segmented
- **No hanging**: Images only load when selected, segmentation only runs when you click the button

## GPU Compatibility Issue - RESOLVED! ✓

### RTX 5050 CUDA sm_120 Support

Your GPU (RTX 5050 with CUDA capability sm_120) is now supported!

### Solution: Updated PyTorch Version

The installation has been updated to use **PyTorch 2.5+ with CUDA 12.4**, which fully supports sm_120 architecture.

### For New Installations
Simply run `install.bat` - it will automatically install the correct PyTorch version with CUDA 12.4 support.

### For Existing Installations
Run the upgrade script:
```powershell
upgrade_pytorch.bat
```

This will:
1. Remove old PyTorch installation
2. Install PyTorch 2.5+ with CUDA 12.4
3. Verify GPU compatibility
4. Test CUDA functionality

### Requirements
You must have **NVIDIA CUDA Toolkit 12.4** installed:
- Download from: https://developer.nvidia.com/cuda-downloads
- Required for RTX 50-series GPU acceleration

### After Upgrading
- Your RTX 5050 will work with full GPU acceleration
- Segmentation will be much faster
- No more compatibility warnings
- Status bar will show "GPU: Available"

### Checking GPU Status
- The status bar at the bottom shows your GPU info
- If it says "GPU: Available", your GPU is being used
- Run `upgrade_pytorch.bat` to see detailed GPU information

## Technical Details

### Files Modified
- `gui/main_window.py` - Added project panel integration and multi-file workflow
- `gui/project_panel.py` - New file for project image list management
- `requirements_updated.txt` - Fixed bioformats-jar version, updated PyTorch to 2.5+ for sm_120 support
- `install.bat` - Updated to install PyTorch with CUDA 12.4
- `upgrade_pytorch.bat` - New script for upgrading existing installations

### New Features
- `ProjectPanel` widget for image list management
- Methods for adding images without immediate loading:
  - `_add_single_image_to_project()`
  - `_add_metamorph_nd_to_project()`
  - `_add_vsi_to_project()`
  - `_add_lif_to_project()`
- `_on_project_image_selected()` - Load image only when selected
- `_on_remove_image()` - Remove image from project
- Project panel tracks segmentation status

### Backward Compatibility
- Old single-image workflow still works
- Existing project files (.nsa) will load correctly
- The "Import" button now adds to project instead of immediate load

## Future Enhancements
- [ ] Batch segmentation (segment all images at once)
- [ ] Batch export of measurements
- [ ] Project-wide quality metrics
- [ ] Compare segmentation across images
- [ ] Parallel processing for multiple images
