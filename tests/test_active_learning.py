"""
Tests for active learning components.
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from core.active_learning import (
    UncertaintyEstimator,
    CorrectionTracker,
    ActiveLearningManager
)


class TestUncertaintyEstimator(unittest.TestCase):
    """Test uncertainty estimation."""
    
    def test_shape_irregularity(self):
        """Test shape irregularity computation."""
        # Create regular circular mask
        y, x = np.ogrid[-50:50, -50:50]
        regular_mask = (x**2 + y**2 <= 20**2)
        
        irregularity = UncertaintyEstimator._compute_shape_irregularity(regular_mask)
        
        # Regular circle should have low irregularity
        self.assertLess(irregularity, 0.3)
    
    def test_select_top_uncertain(self):
        """Test selection of top uncertain nuclei."""
        uncertainties = {
            1: 0.9,  # Most uncertain
            2: 0.3,
            3: 0.7,
            4: 0.5,
            5: 0.1   # Least uncertain
        }
        
        top_3 = UncertaintyEstimator.select_top_uncertain(uncertainties, n=3)
        
        self.assertEqual(len(top_3), 3)
        self.assertEqual(top_3[0], 1)  # ID 1 should be first (highest uncertainty)
        self.assertEqual(top_3[1], 3)  # ID 3 should be second
        self.assertEqual(top_3[2], 4)  # ID 4 should be third


class TestCorrectionTracker(unittest.TestCase):
    """Test correction tracking."""
    
    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.tracker = CorrectionTracker(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_add_correction(self):
        """Test adding a correction."""
        # Create mock masks
        original = np.zeros((100, 100), dtype=bool)
        original[40:60, 40:60] = True
        
        corrected = np.zeros((100, 100), dtype=bool)
        corrected[40:65, 40:65] = True  # Slightly larger
        
        # Add correction
        self.tracker.add_correction(
            image_id='test_image_001',
            nucleus_id=1,
            original_mask=original,
            corrected_mask=corrected,
            correction_type='modify',
            metadata={'user': 'test_user'}
        )
        
        # Check that correction was recorded
        self.assertEqual(len(self.tracker.corrections), 1)
        
        correction = self.tracker.corrections[0]
        self.assertEqual(correction['image_id'], 'test_image_001')
        self.assertEqual(correction['nucleus_id'], 1)
        self.assertEqual(correction['correction_type'], 'modify')
        self.assertIn('metrics', correction)
        self.assertIn('iou', correction['metrics'])
    
    def test_save_and_load_corrections(self):
        """Test saving and loading corrections."""
        # Add a correction
        original = np.zeros((50, 50), dtype=bool)
        original[20:30, 20:30] = True
        corrected = original.copy()
        
        self.tracker.add_correction(
            image_id='test_image_002',
            nucleus_id=2,
            original_mask=original,
            corrected_mask=corrected,
            correction_type='delete'
        )
        
        # Create new tracker with same path
        new_tracker = CorrectionTracker(storage_path=self.temp_dir)
        
        # Should load the correction
        self.assertEqual(len(new_tracker.corrections), 1)
        self.assertEqual(new_tracker.corrections[0]['nucleus_id'], 2)
    
    def test_correction_metrics(self):
        """Test correction metrics computation."""
        # Create masks with known overlap
        original = np.zeros((100, 100), dtype=bool)
        original[40:60, 40:60] = True  # 20x20 = 400 pixels
        
        corrected = np.zeros((100, 100), dtype=bool)
        corrected[40:70, 40:70] = True  # 30x30 = 900 pixels
        
        metrics = self.tracker._compute_correction_metrics(original, corrected)
        
        # Check that metrics are computed
        self.assertIn('iou', metrics)
        self.assertIn('dice', metrics)
        self.assertIn('area_change_ratio', metrics)
        
        # IoU should be intersection/union = 400/900
        expected_iou = 400.0 / 900.0
        self.assertAlmostEqual(metrics['iou'], expected_iou, places=5)
        
        # Area change ratio should be 900/400
        expected_ratio = 900.0 / 400.0
        self.assertAlmostEqual(metrics['area_change_ratio'], expected_ratio, places=5)


class TestActiveLearningManager(unittest.TestCase):
    """Test active learning manager."""
    
    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.manager = ActiveLearningManager(
            storage_path=self.temp_dir,
            min_corrections_for_training=3
        )
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_should_retrain_insufficient_corrections(self):
        """Test that retraining is not recommended with few corrections."""
        # Initially should not retrain
        self.assertFalse(self.manager.should_retrain())
    
    def test_should_retrain_sufficient_corrections(self):
        """Test that retraining is recommended with enough corrections."""
        # Add minimum number of corrections
        for i in range(3):
            original = np.zeros((50, 50), dtype=bool)
            original[20:30, 20:30] = True
            corrected = original.copy()
            
            self.manager.record_correction(
                image_id=f'test_image_{i}',
                nucleus_id=i,
                original_mask=original,
                corrected_mask=corrected,
                correction_type='modify'
            )
        
        # Should now recommend retraining
        self.assertTrue(self.manager.should_retrain())
    
    def test_identify_uncertain_nuclei(self):
        """Test identification of uncertain nuclei."""
        # Create mock masks and flows
        masks = np.zeros((100, 100), dtype=int)
        masks[20:40, 20:40] = 1
        masks[60:80, 60:80] = 2
        
        # Mock flows (not really used in simple test)
        flows = [np.random.rand(100, 100)]
        
        uncertain_ids = self.manager.identify_uncertain_nuclei(
            masks=masks,
            flows=flows,
            cellprob=None,
            top_n=2
        )
        
        # Should return list of nucleus IDs
        self.assertIsInstance(uncertain_ids, list)
        self.assertLessEqual(len(uncertain_ids), 2)


if __name__ == '__main__':
    unittest.main()
