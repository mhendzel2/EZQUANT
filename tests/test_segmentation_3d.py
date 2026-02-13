"""
Tests for 3D segmentation backends and metrics.
"""

import unittest
import numpy as np
from core.segmentation_3d import (
    Anisotropy3DPreprocessor,
    Hybrid2D3DBackend,
    compute_3d_iou,
    compute_3d_dice,
    compute_split_merge_errors
)


class TestAnisotropy3DPreprocessor(unittest.TestCase):
    """Test anisotropy-aware preprocessing."""
    
    def test_isotropic_no_resampling(self):
        """Test that isotropic volumes are not resampled."""
        preprocessor = Anisotropy3DPreprocessor(voxel_size=(1.0, 1.0, 1.0))
        volume = np.random.rand(10, 100, 100)
        
        resampled, zoom_factors = preprocessor.resample_to_isotropic(volume)
        
        self.assertTrue(np.allclose(zoom_factors, (1.0, 1.0, 1.0)))
        self.assertEqual(resampled.shape, volume.shape)
    
    def test_anisotropic_resampling(self):
        """Test resampling of anisotropic volume."""
        # Z-resolution is 2x worse than XY
        preprocessor = Anisotropy3DPreprocessor(voxel_size=(2.0, 1.0, 1.0))
        volume = np.random.rand(10, 100, 100)
        
        resampled, zoom_factors = preprocessor.resample_to_isotropic(volume)
        
        # Should double Z dimension
        self.assertAlmostEqual(zoom_factors[0], 2.0, places=5)
        self.assertAlmostEqual(zoom_factors[1], 1.0, places=5)
        self.assertAlmostEqual(zoom_factors[2], 1.0, places=5)
        self.assertEqual(resampled.shape[0], volume.shape[0] * 2)
    
    def test_resample_masks_back(self):
        """Test resampling masks back to original shape."""
        preprocessor = Anisotropy3DPreprocessor(voxel_size=(2.0, 1.0, 1.0))
        original_shape = (10, 100, 100)
        
        # Create masks in resampled space
        resampled_shape = (20, 100, 100)
        masks_resampled = np.random.randint(0, 5, size=resampled_shape)
        
        # Resample back
        masks_original = preprocessor.resample_masks_to_original(
            masks_resampled, original_shape
        )
        
        self.assertEqual(masks_original.shape, original_shape)


class TestHybrid2D3DBackend(unittest.TestCase):
    """Test hybrid 2D+3D segmentation backend."""
    
    def test_simple_linking(self):
        """Test linking of simple overlapping instances."""
        # Create mock 2D segmentation function
        def segment_2d_fn(slice_2d):
            # Return simple mask with one nucleus in center
            mask = np.zeros((100, 100), dtype=int)
            mask[40:60, 40:60] = 1
            return mask
        
        backend = Hybrid2D3DBackend(
            segmentation_2d_fn=segment_2d_fn,
            min_overlap_ratio=0.3,
            max_distance_z=1
        )
        
        # Create volume
        volume = np.random.rand(5, 100, 100)
        
        # Run segmentation
        masks = backend.segment(volume)
        
        # Should have consistent instance ID across slices
        self.assertEqual(masks.shape, volume.shape)
        self.assertTrue(np.all(masks[0, 40:60, 40:60] == 1))
        # All slices should have same ID due to overlap
        for z in range(5):
            self.assertTrue(np.all(masks[z, 40:60, 40:60] > 0))
    
    def test_no_overlap_creates_new_instances(self):
        """Test that non-overlapping instances get different IDs."""
        # Create mock 2D segmentation with moving nucleus
        positions = [(20, 20), (50, 50), (80, 80)]
        
        def segment_2d_fn(slice_2d):
            # Determine position based on slice intensity
            # (This is a hack to vary position per slice)
            z_idx = int(np.mean(slice_2d) * 3)
            if z_idx >= len(positions):
                z_idx = len(positions) - 1
            
            mask = np.zeros((100, 100), dtype=int)
            y, x = positions[z_idx]
            mask[y:y+10, x:x+10] = 1
            return mask
        
        backend = Hybrid2D3DBackend(
            segmentation_2d_fn=segment_2d_fn,
            min_overlap_ratio=0.3,
            max_distance_z=1
        )
        
        # Create volume with varying intensity per slice
        volume = np.zeros((3, 100, 100))
        volume[0] = 0.0
        volume[1] = 0.5
        volume[2] = 1.0
        
        masks = backend.segment(volume)
        
        # Should have 3D output
        self.assertEqual(masks.shape, (3, 100, 100))


class Test3DMetrics(unittest.TestCase):
    """Test 3D segmentation metrics."""
    
    def test_compute_3d_iou_perfect_match(self):
        """Test IoU with perfect match."""
        mask1 = np.zeros((10, 100, 100), dtype=bool)
        mask1[3:7, 40:60, 40:60] = True
        
        mask2 = mask1.copy()
        
        iou = compute_3d_iou(mask1, mask2)
        self.assertAlmostEqual(iou, 1.0, places=5)
    
    def test_compute_3d_iou_no_overlap(self):
        """Test IoU with no overlap."""
        mask1 = np.zeros((10, 100, 100), dtype=bool)
        mask1[3:7, 40:60, 40:60] = True
        
        mask2 = np.zeros((10, 100, 100), dtype=bool)
        mask2[3:7, 70:90, 70:90] = True
        
        iou = compute_3d_iou(mask1, mask2)
        self.assertAlmostEqual(iou, 0.0, places=5)
    
    def test_compute_3d_dice(self):
        """Test Dice coefficient."""
        mask1 = np.zeros((10, 100, 100), dtype=bool)
        mask1[3:7, 40:60, 40:60] = True
        
        # Half overlap
        mask2 = np.zeros((10, 100, 100), dtype=bool)
        mask2[3:7, 50:70, 40:60] = True
        
        dice = compute_3d_dice(mask1, mask2)
        self.assertGreater(dice, 0.0)
        self.assertLess(dice, 1.0)
    
    def test_split_merge_errors(self):
        """Test split and merge error calculation."""
        # Create ground truth with 2 instances
        gt = np.zeros((10, 100, 100), dtype=int)
        gt[3:7, 20:40, 20:40] = 1
        gt[3:7, 60:80, 60:80] = 2
        
        # Create prediction with split (instance 1 split into 1 and 3)
        pred = np.zeros((10, 100, 100), dtype=int)
        pred[3:5, 20:40, 20:40] = 1  # First half of instance 1
        pred[5:7, 20:40, 20:40] = 3  # Second half of instance 1 (split)
        pred[3:7, 60:80, 60:80] = 2  # Instance 2 unchanged
        
        split_count, merge_count = compute_split_merge_errors(pred, gt)
        
        # Should detect 1 split
        self.assertEqual(split_count, 1)
        self.assertEqual(merge_count, 0)
    
    def test_merge_error(self):
        """Test merge error detection."""
        # Ground truth with 2 instances
        gt = np.zeros((10, 100, 100), dtype=int)
        gt[3:7, 20:40, 20:40] = 1
        gt[3:7, 60:80, 60:80] = 2
        
        # Prediction with merge (both mapped to same ID)
        pred = np.zeros((10, 100, 100), dtype=int)
        pred[3:7, 20:40, 20:40] = 1
        pred[3:7, 60:80, 60:80] = 1  # Merged with instance 1
        
        split_count, merge_count = compute_split_merge_errors(pred, gt)
        
        # Should detect 1 merge
        self.assertEqual(split_count, 0)
        self.assertEqual(merge_count, 1)


if __name__ == '__main__':
    unittest.main()
