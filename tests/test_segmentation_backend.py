import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys

# Mock aicssegmentation and segmenter_model_zoo before importing backend
sys.modules["aicssegmentation"] = MagicMock()
sys.modules["aicssegmentation.structure_wrapper"] = MagicMock()
sys.modules["segmenter_model_zoo"] = MagicMock()
sys.modules["segmenter_model_zoo.zoo"] = MagicMock()

from core.segmentation_backend import ClassicSegmenterBackend, MlSegmenterBackend, get_segmenter_backend

class TestClassicSegmenterBackend(unittest.TestCase):
    def setUp(self):
        self.backend = ClassicSegmenterBackend()
        # Mock the registry to avoid import errors during test
        self.backend._registry = {
            "TEST_STRUCT": "mock_module"
        }

    @patch("importlib.import_module")
    def test_segment_classic(self, mock_import):
        # Setup mock wrapper
        mock_wrapper_module = MagicMock()
        mock_workflow_wrapper = MagicMock()
        mock_wrapper_module.Workflow_Wrapper.return_value = mock_workflow_wrapper
        mock_import.return_value = mock_wrapper_module
        
        # Mock segment_image return
        input_vol = np.zeros((10, 100, 100), dtype=np.uint8)
        expected_mask = np.zeros((10, 100, 100), dtype=np.uint8)
        expected_mask[5, 50, 50] = 1
        mock_workflow_wrapper.segment_image.return_value = expected_mask
        
        # Run segment
        result = self.backend.segment(input_vol, structure_id="TEST_STRUCT")
        
        # Verify
        self.assertTrue(np.array_equal(result, expected_mask))
        mock_workflow_wrapper.segment_image.assert_called_once()

class TestMlSegmenterBackend(unittest.TestCase):
    def setUp(self):
        # Mock supports_segmenter_ml to return True
        with patch("core.segmentation_backend.supports_segmenter_ml", return_value=True):
            self.backend = MlSegmenterBackend()

    @patch("core.segmentation_backend.supports_segmenter_ml", return_value=True)
    def test_segment_ml(self, mock_supports):
        # Mock ModelZoo
        mock_model = MagicMock()
        self.backend.ModelZoo.load_model.return_value = mock_model
        
        # Mock predict return
        input_vol = np.zeros((10, 100, 100), dtype=np.float32)
        expected_prediction = np.zeros((10, 100, 100), dtype=np.float32)
        expected_prediction[5, 50, 50] = 1.0
        mock_model.predict.return_value = expected_prediction
        
        # Run segment
        result = self.backend.segment(input_vol, workflow_id="TEST_MODEL")
        
        # Verify
        self.assertTrue(np.array_equal(result, expected_prediction.astype(np.uint8)))
        self.backend.ModelZoo.load_model.assert_called_with("TEST_MODEL")
        mock_model.predict.assert_called_once()

if __name__ == "__main__":
    unittest.main()
