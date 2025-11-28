# Quick Start: Multi-File Workflow

## Problem Solved
✓ No more hanging during file import  
✓ Add multiple files quickly  
✓ Segment images one at a time on demand  

## How to Use

### Step 1: Add Images to Project
1. Click **File → Add Images to Project** (or press Ctrl+I)
2. Select multiple files (use Ctrl+Click or Shift+Click)
3. Click **Open**

**Result:** Files are added to the project panel (left side), but not loaded yet.

### Step 2: Load an Image
1. **Single-click** an image in the project panel to select it
2. **Double-click** or press Enter to load it into the viewer

**Result:** The selected image loads and displays in the main viewer.

### Step 3: Segment the Image
1. Configure segmentation settings (diameter, thresholds, etc.)
2. Click **Run Segmentation**
3. Wait for completion

**Result:** The image is segmented and marked with a ✓ in the project panel.

### Step 4: Repeat
1. Select the next image in the project panel
2. Segment it
3. Continue through your project

## Project Panel Icons
- **○** = Image not segmented yet
- **✓** = Image has been segmented

## Tips

### Removing Images
- Right-click on an image → **Remove from Project**
- Or select image and click **Remove** button
- This doesn't delete the file, just removes it from the project

### Saving Your Project
- **File → Save Project** (Ctrl+S)
- Saves all images and their segmentation history
- Next time you open the project, all images will be there

### Multi-File Selection
When adding images, you can:
- **Ctrl+Click**: Select individual files
- **Shift+Click**: Select a range of files
- **Ctrl+A**: Select all files in folder

## What Changed?

**Before:**
- Import image → Image loads immediately → Hangs if large

**Now:**
- Add images to project (fast, no loading)
- Select image when ready (loads on demand)
- Segment when you want (controlled timing)

## GPU Issue Note

If you see this warning:
```
NVIDIA GeForce RTX 5050 with CUDA capability sm_120 is not compatible...
```

**What it means:** Your GPU is too new for the current PyTorch version.

**What happens:** Segmentation runs on CPU (slower but works).

**Solution:** Wait for PyTorch update or use nightly builds (see WORKFLOW_UPDATE.md for details).

## Need Help?

- Status bar (bottom) shows current operation
- Error messages provide details if something goes wrong
- Check WORKFLOW_UPDATE.md for technical details
