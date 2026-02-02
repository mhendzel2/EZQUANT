import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import core.image_io
from core.image_io import TIFFLoader
import PIL.Image

class TestMemoryFeatures(unittest.TestCase):
    def test_pil_limit_disabled(self):
        """Test that PIL image limit is disabled"""
        # Ensure imports trigger the logic
        import main
        self.assertIsNone(PIL.Image.MAX_IMAGE_PIXELS)

    @patch('core.image_io.tifffile.TiffFile')
    @patch('pathlib.Path.exists')
    def test_large_file_memmap(self, mock_exists, mock_tifffile):
        """Test that large files are loaded using memmap"""
        mock_exists.return_value = True

        # Setup mock
        mock_tif = MagicMock()
        mock_tifffile.return_value.__enter__.return_value = mock_tif

        # Mock series to report large size (> 4GB)
        mock_series = MagicMock()
        mock_series.size = 5 * 1024**3 # 5 billion pixels
        mock_series.dtype.itemsize = 1
        mock_series.shape = (100, 100)
        mock_series.axes = 'YX'

        mock_tif.series = [mock_series]

        # Mock asarray to return a dummy array
        mock_tif.asarray.return_value = np.zeros((100, 100))

        # Mock metadata
        mock_tif.is_imagej = False
        mock_tif.is_ome = False
        mock_tif.pages = []
        mock_tif.imagej_metadata = {}
        mock_tif.ome_metadata = None
        mock_tif.filename = "test.tif"

        # Call load_tiff
        TIFFLoader.load_tiff("test.tif")

        # Verify asarray was called with out='memmap'
        mock_tif.asarray.assert_called_with(out='memmap')

    @patch('core.image_io.tifffile.TiffFile')
    @patch('pathlib.Path.exists')
    def test_normal_file_loading(self, mock_exists, mock_tifffile):
        """Test that normal files are loaded normally"""
        mock_exists.return_value = True

        # Setup mock
        mock_tif = MagicMock()
        mock_tifffile.return_value.__enter__.return_value = mock_tif

        # Mock series to report small size
        mock_series = MagicMock()
        mock_series.size = 100 * 100
        mock_series.dtype.itemsize = 1
        mock_series.shape = (100, 100)
        mock_series.axes = 'YX'

        mock_tif.series = [mock_series]

        # Mock asarray
        mock_tif.asarray.return_value = np.zeros((100, 100))

        # Mock metadata
        mock_tif.is_imagej = False
        mock_tif.is_ome = False
        mock_tif.pages = []
        mock_tif.imagej_metadata = {}
        mock_tif.ome_metadata = None
        mock_tif.filename = "test.tif"

        # Call load_tiff
        TIFFLoader.load_tiff("test.tif")

        # Verify asarray was called without out='memmap'
        call_args = mock_tif.asarray.call_args
        self.assertNotEqual(call_args.kwargs.get('out'), 'memmap')

if __name__ == '__main__':
    unittest.main()
