# Batch Segmentation Feature

## Overview

You can now apply segmentation settings to all images in your project at once using the **"Apply to All Images"** button.

## How to Use

### Step 1: Add Images to Project
1. Click **File → Add Images to Project** (Ctrl+I)
2. Select multiple image files
3. All images appear in the project panel (left side)

### Step 2: Configure Segmentation Settings
1. Load one image from the project panel (double-click)
2. Configure your segmentation parameters:
   - Model type (nuclei, cyto, etc.)
   - Diameter
   - Flow threshold
   - Cell probability threshold
   - 3D options
   - Channels

### Step 3: Test on Current Image (Optional but Recommended)
1. Click **"Run Segmentation"** on the current image
2. Verify the results look good
3. Adjust parameters if needed

### Step 4: Apply to All Images
1. Click **"Apply to All Images"** button (green button next to "Run Segmentation")
2. Confirm the batch operation in the dialog
3. Wait for processing to complete

## What Happens During Batch Processing

1. **Progress Display**: The progress bar shows "Processing image X of Y..."
2. **Automatic Saving**: Each segmented image is:
   - Marked with ✓ in the project panel
   - Saved in the project with segmentation history
   - Updated if currently displayed
3. **Summary Report**: When complete, you'll see:
   - Number of successfully segmented images
   - Number of failed images (if any)
   - Total nuclei detected across all images

## Features

### ✓ Benefits
- **Fast Setup**: Configure parameters once, apply to all
- **Consistent Results**: Same settings applied to all images
- **Progress Tracking**: Real-time progress updates
- **Background Processing**: UI remains responsive (though slower on CPU)
- **Error Handling**: Failed images don't stop the batch
- **Automatic Status Update**: Project panel shows segmentation status

### ✓ Smart Handling
- **Re-segmentation**: Already segmented images are re-segmented with new settings
- **Current Image**: If viewing an image being processed, the display updates automatically
- **Project Saving**: All results saved in the project

## UI Elements

### Buttons
- **Run Segmentation** (Blue): Segment only the current image
- **Apply to All Images** (Green): Batch segment all project images

### Progress Indicators
- Progress bar shows: "Processing image X of Y..."
- Status bar shows current operation
- Project panel icons update as images complete (○ → ✓)

## Tips

### Best Practices
1. **Test First**: Always test parameters on one image before batch processing
2. **Save Project**: Save your project before batch processing (just in case)
3. **Monitor Progress**: Watch for any error messages during processing
4. **Check Results**: After batch completion, spot-check a few images

### Performance Notes
- **CPU Mode**: If your GPU isn't compatible, processing will be slower
  - Small projects (<10 images): Few minutes
  - Medium projects (10-50 images): 10-30 minutes
  - Large projects (>50 images): Could take hours
- **GPU Mode**: Much faster when GPU is available
  - Can process 1-2 images per minute

### Troubleshooting

**"No Images" Warning**
- Solution: Add images to the project first

**Processing Seems Stuck**
- Check the progress bar - it should be updating
- CPU mode is slow, give it time
- Check status bar for current operation

**Some Images Failed**
- Error messages are logged but don't stop the batch
- Check individual images that failed
- Common causes: corrupted files, incompatible formats

**Want to Cancel**
- Currently processing will complete
- Future feature: Cancel button

## Example Workflow

```
1. Add 20 images to project
   → All show ○ (not segmented)

2. Load first image, configure:
   - Model: nuclei
   - Diameter: 30 pixels
   - Flow threshold: 0.4
   - Cell prob: 0.0

3. Test on first image
   → Looks good!

4. Click "Apply to All Images"
   → Confirm dialog → Yes

5. Watch progress:
   → "Processing image 1 of 20..."
   → "Processing image 2 of 20..."
   → ...
   → "Processing image 20 of 20..."

6. Summary dialog:
   → "Successfully segmented 20 of 20 images"
   → "Total nuclei detected: 3,456"

7. All images now show ✓ in project panel
```

## Advanced

### Re-running Batch Segmentation
You can run batch segmentation multiple times with different settings:
- Previous results are overwritten
- Segmentation history is kept
- Latest settings become active

### Selective Batch Processing
Currently processes ALL images in project.
Future enhancement: Select specific images to process.

## Keyboard Shortcuts
- **Ctrl+I**: Add images to project
- **Ctrl+S**: Save project

## Next Steps After Batch Segmentation
1. Export measurements for all images
2. Compare results across images
3. Generate project-wide statistics
4. Quality control review

---

**Note**: This feature works best when all images in your project are similar (same specimen type, staining, magnification, etc.) and can use the same segmentation parameters.
