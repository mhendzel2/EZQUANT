# True 3D Segmentation and Active Learning Guide

## Overview

This document describes the enhanced segmentation capabilities added to EZQUANT, including:

1. **True 3D Segmentation Backends** - Replace slice-by-slice processing with volumetric instance consistency
2. **Cellpose3 Restoration Mode** - Improved segmentation for degraded microscopy images
3. **Active Learning Loop** - Continuous model improvement through user corrections

---

## 1. True 3D Segmentation

### Background

Traditional "3D segmentation" often processes each z-slice independently and can produce inconsistent instance IDs across slices. True 3D segmentation maintains instance consistency throughout the volume.

### Available Backends

#### 1.1 Default (Slice-by-Slice)
- **Use case**: Quick processing, simple z-stacks
- **Pros**: Fast, works with any 2D model
- **Cons**: No instance consistency across slices
- **How to use**: Select "default (slice-by-slice)" in 3D Backend dropdown

#### 1.2 Hybrid 2D+3D
- **Use case**: General-purpose 3D segmentation with good accuracy
- **Pros**: 
  - Leverages robust 2D models (Cellpose)
  - Maintains instance IDs across slices via overlap-based linking
  - Lower memory requirements than full 3D models
  - Good for anisotropic data (poor z-resolution)
- **Cons**: May miss some instance boundaries if overlap is small
- **How to use**: 
  1. Enable "3D Segmentation (volumetric)" checkbox
  2. Select "hybrid2d3d (2D + linking)" in 3D Backend dropdown
  3. Run segmentation as normal

**Algorithm**: 
- Runs 2D Cellpose on each slice independently
- Links instances across consecutive slices using IoU-based matching
- Uses Hungarian algorithm for optimal assignment
- Creates consistent 3D instance IDs

**Parameters** (in config):
- `min_overlap_ratio`: Minimum IoU to link instances (default: 0.3)
- `max_distance_z`: Maximum z-distance to search for matches (default: 2 slices)

#### 1.3 True 3D (Placeholder)
- **Status**: Not yet implemented
- **Future**: Will support 3D U-Net, nnU-Net, SAM3-derived models
- **Use case**: Maximum accuracy for isotropic high-quality volumes

### Anisotropy Handling

For microscopy data where z-resolution is worse than xy-resolution:

```python
from core.segmentation_3d import Anisotropy3DPreprocessor

# Specify voxel sizes in microns (z, y, x)
preprocessor = Anisotropy3DPreprocessor(voxel_size=(2.0, 0.5, 0.5))

# Resample to isotropic
volume_iso, zoom_factors = preprocessor.resample_to_isotropic(volume)

# Run segmentation on isotropic volume
masks_iso = segment(volume_iso)

# Resample masks back to original space
masks = preprocessor.resample_masks_to_original(masks_iso, volume.shape)
```

---

## 2. Cellpose3 Restoration Mode

### Background

Cellpose3 includes built-in restoration capabilities for noisy or blurred images, common in undergraduate lab settings. This reduces the need for complex preprocessing.

### Restoration Modes

#### 2.1 None (Disabled)
- Standard Cellpose segmentation
- Use when image quality is good

#### 2.2 Auto (Recommended)
- Automatically detects image quality issues
- Applies denoising if SNR < 10
- Applies deblurring if blur score > 0.1
- **Best choice for most users**

#### 2.3 Denoise
- Explicitly apply denoising restoration
- Use for high-noise images (low SNR)
- Examples: Low light imaging, high gain settings

#### 2.4 Deblur
- Explicitly apply deblurring restoration
- Use for out-of-focus or motion-blurred images

### How to Use in GUI

1. Open Segmentation panel
2. Under "Cellpose Parameters", find "Restoration Mode" dropdown
3. Select desired mode (recommend 'auto')
4. Run segmentation normally

### How to Use Programmatically

```python
from core.cellpose3_restoration import Cellpose3RestorationEngine

engine = Cellpose3RestorationEngine(gpu_available=True)

masks, info = engine.segment_with_restoration(
    image=image,
    model_name='nuclei',
    restoration_mode='auto',  # or 'denoise', 'deblur', 'none'
    diameter=30
)

# Check restoration metrics
print(f"SNR: {info['restoration_metrics']['snr']:.2f}")
print(f"Blur score: {info['restoration_metrics']['blur_score']:.4f}")
```

### Requirements

- Cellpose 3.0 or higher: `pip install cellpose>=3.0`
- If using older Cellpose, restoration mode will be silently disabled

### Image Quality Metrics

The system estimates:
- **SNR (Signal-to-Noise Ratio)**: mean/std of pixel intensities
  - High SNR (>10): Good quality
  - Low SNR (<10): Noisy image, denoising recommended
- **Blur Score**: Based on Laplacian variance
  - Low score (<0.1): Sharp image
  - High score (>0.1): Blurred image, deblurring recommended

---

## 3. Active Learning Loop

### Concept

The active learning system:
1. Identifies uncertain nuclei that may benefit from correction
2. Tracks user corrections
3. Suggests when to retrain the model adapter
4. Fine-tunes a lightweight adapter on correction data

### Components

#### 3.1 Uncertainty Estimation

Identifies nuclei that the model is uncertain about:

```python
from core.active_learning import ActiveLearningManager

manager = ActiveLearningManager()

# After segmentation
uncertain_nuclei = manager.identify_uncertain_nuclei(
    masks=masks,
    flows=flows,
    cellprob=cellprob,
    top_n=10  # Get 10 most uncertain
)

# uncertain_nuclei is a list of nucleus IDs to review
```

**Uncertainty signals**:
- Low flow magnitude at boundaries
- Low cell probability
- Irregular shapes (low solidity)

#### 3.2 Correction Tracking

Records all manual corrections for model improvement:

```python
# When user corrects a nucleus
manager.record_correction(
    image_id='sample_001',
    nucleus_id=42,
    original_mask=original_mask,
    corrected_mask=corrected_mask,
    correction_type='split',  # or 'merge', 'delete', 'add', 'modify'
    metadata={'user': 'lab_student_1'}
)

# Corrections are automatically saved to disk
```

#### 3.3 Model Adapter Fine-Tuning

**Status**: Infrastructure in place, training not yet implemented

When implemented, will:
- Fine-tune a lightweight adapter layer on top of base Cellpose model
- Require minimum 20 corrections before training
- Run nightly or on-demand
- Maintain model versions for reproducibility

```python
# Check if retraining is recommended
if manager.should_retrain():
    results = manager.trigger_retraining()
    print(f"Training status: {results['status']}")
```

### Data Storage

Corrections are stored in:
- `./active_learning/corrections/corrections.json` - Metadata
- `./active_learning/corrections/corr_*.npz` - Mask pairs

### Best Practices

1. **Start with good data**: Get initial segmentation working well
2. **Focus corrections**: Prioritize uncertain nuclei shown by the system
3. **Be consistent**: Multiple users should agree on correction standards
4. **Avoid bad corrections**: System tracks but doesn't validate corrections
5. **Regular retraining**: Retrain after every 20-50 corrections
6. **Version tracking**: Keep track of which model version produced which results

---

## 4. Evaluation Metrics

### 3D Metrics

```python
from core.segmentation_3d import compute_3d_iou, compute_3d_dice, compute_split_merge_errors

# Compare segmentation to ground truth
iou = compute_3d_iou(pred_mask, gt_mask)
dice = compute_3d_dice(pred_mask, gt_mask)

# Count split and merge errors
splits, merges = compute_split_merge_errors(pred_labels, gt_labels)

print(f"3D IoU: {iou:.3f}")
print(f"3D Dice: {dice:.3f}")
print(f"Split errors: {splits}, Merge errors: {merges}")
```

### Volumetric Stability

Run segmentation multiple times and measure consistency:

```python
import numpy as np

results = []
for i in range(5):
    masks, _ = engine.segment_cellpose(volume, diameter=30)
    volume_measure = np.sum(masks > 0)
    results.append(volume_measure)

# Coefficient of variation
cv = np.std(results) / np.mean(results) * 100
print(f"Volume measurement CV: {cv:.2f}%")
# Lower CV = more stable/reproducible
```

---

## 5. Quick Start Examples

### Example 1: True 3D Segmentation

```python
from core.segmentation import SegmentationEngine

engine = SegmentationEngine(gpu_available=True)

# Enable hybrid 2D+3D backend
masks, info = engine.segment_cellpose(
    image=volume_3d,
    model_name='nuclei',
    diameter=25,
    do_3d=True,
    use_3d_backend='hybrid2d3d'
)

print(f"Detected {info['nucleus_count']} nuclei with 3D consistency")
```

### Example 2: Restoration Mode

```python
# Automatic restoration
masks, info = engine.segment_cellpose(
    image=noisy_volume,
    model_name='nuclei',
    restoration_mode='auto'
)

if 'restoration_metrics' in info:
    print(f"Image SNR: {info['restoration_metrics']['snr']:.2f}")
    print(f"Applied restoration: {info['restoration_mode']}")
```

### Example 3: Active Learning

```python
from core.active_learning import ActiveLearningManager

# Initialize
manager = ActiveLearningManager(min_corrections_for_training=20)

# After segmentation
uncertain = manager.identify_uncertain_nuclei(masks, flows, top_n=5)
print(f"Review these nuclei: {uncertain}")

# After user correction
manager.record_correction(
    image_id='test_001',
    nucleus_id=uncertain[0],
    original_mask=masks == uncertain[0],
    corrected_mask=corrected_mask,
    correction_type='split'
)

# Check if ready to retrain
if manager.should_retrain():
    print("Ready for model retraining!")
```

---

## 6. Troubleshooting

### Issue: "Cellpose3 restoration not available"

**Solution**: Upgrade Cellpose
```bash
pip install --upgrade cellpose>=3.0
```

### Issue: Hybrid 2D+3D creates too many instances

**Possible causes**:
- `min_overlap_ratio` too high
- Z-slices have little overlap between nuclei

**Solutions**:
- Reduce `min_overlap_ratio` to 0.2 or lower
- Increase `max_distance_z` to search further
- Use anisotropic preprocessing if z-resolution is poor

### Issue: Active learning suggests no uncertain nuclei

**Possible causes**:
- Segmentation is very confident (good!)
- Not enough variability in dataset

**Solutions**:
- This is often a good sign - model is working well
- Manually review some nuclei anyway to build correction dataset
- Try on more challenging images

### Issue: Out of memory with 3D segmentation

**Solutions**:
- Use hybrid 2D+3D instead of true 3D
- Process smaller volumes at a time
- Reduce image resolution
- Use CPU instead of GPU if GPU memory limited

---

## 7. Future Enhancements

### Planned Features

1. **True 3D U-Net Backend**
   - Full volumetric segmentation
   - Requires training on 3D datasets
   - Target: Q2 2026

2. **SAM3 Integration**
   - Foundation model for 3D medical imaging
   - Zero-shot 3D segmentation
   - Target: Q3 2026

3. **Model Adapter Training**
   - Complete LoRA-style adapter implementation
   - Distributed training support
   - Target: Q2 2026

4. **Active Learning GUI**
   - Visual uncertainty heatmaps
   - One-click correction workflow
   - Training progress dashboard
   - Target: Q1 2026

### Contributing

To contribute to these features:
1. See `core/segmentation_3d.py` - True3DBackend class (placeholder)
2. See `core/active_learning.py` - ModelAdapter class (placeholder)
3. Submit PRs with unit tests
4. Document new features in this guide

---

## 8. References

[1] Cellpose3: Stringer et al., "Cellpose 3.0: One-click image restoration for improved cellular segmentation" (2024)

[2] 3D U-Net: Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation" (2016)

[3] nnU-Net: Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation" (2021)

[4] Active Learning: Settles, "Active Learning Literature Survey" (2009)

[5] Hungarian Algorithm: Kuhn, "The Hungarian method for the assignment problem" (1955)

---

## 9. API Reference

### Core Classes

- `Anisotropy3DPreprocessor` - Handle anisotropic voxel sizes
- `Hybrid2D3DBackend` - 2D segmentation + 3D linking
- `True3DBackend` - Placeholder for volumetric networks
- `Cellpose3RestorationEngine` - Restoration-enhanced segmentation
- `UncertaintyEstimator` - Identify uncertain predictions
- `CorrectionTracker` - Record and manage corrections
- `ModelAdapter` - Placeholder for adapter fine-tuning
- `ActiveLearningManager` - Coordinate active learning workflow

### Key Functions

- `compute_3d_iou(mask1, mask2)` - 3D Intersection over Union
- `compute_3d_dice(mask1, mask2)` - 3D Dice coefficient
- `compute_split_merge_errors(pred, gt)` - Count segmentation errors
- `should_use_restoration(image)` - Recommend restoration mode

See inline documentation for detailed API specifications.
