"""
Bioformats I/O module for loading VSI (Olympus) and LIF (Leica) files
Uses aicsimageio library for reading proprietary microscopy formats
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional


class BioformatsLoader:
    """
    Handles loading of proprietary microscopy formats using aicsimageio
    Supports VSI (Olympus), LIF (Leica), CZI (Zeiss), and other formats
    """
    
    @staticmethod
    def load_vsi(filepath: str, scene: int = 0, timepoint: int = 0) -> Tuple[np.ndarray, Dict]:
        """
        Load Olympus VSI file
        
        Args:
            filepath: Path to .vsi file
            scene: Scene/series index to load (default: 0)
            timepoint: Timepoint to load (default: 0)
            
        Returns:
            tuple: (image_array, metadata_dict)
                image_array: numpy array with shape (Z, C, Y, X) or (C, Y, X)
                metadata_dict: dictionary with image properties
        """
        try:
            from aicsimageio import AICSImage
            from aicsimageio.readers import OmeZarrReader
        except ImportError:
            raise ImportError(
                "aicsimageio is required to read VSI files. "
                "Install with: pip install aicsimageio"
            )
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load VSI file
        img = AICSImage(filepath)
        
        # Get available scenes
        n_scenes = img.scenes
        if scene >= len(n_scenes):
            scene = 0
        
        # Set scene if multiple available
        if len(n_scenes) > 1:
            img.set_scene(n_scenes[scene])
        
        # Get image dimensions
        metadata = {
            'filename': filepath.name,
            'format': 'VSI (Olympus)',
            'scenes': n_scenes,
            'current_scene': scene,
            'shape': img.shape,
            'dims': img.dims.order,
            'n_channels': img.dims.C,
            'n_slices': img.dims.Z,
            'n_timepoints': img.dims.T,
            'is_3d': img.dims.Z > 1,
            'channel_names': img.channel_names if hasattr(img, 'channel_names') else [],
            'dtype': str(img.dtype),
            'bit_depth': 8 if img.dtype == np.uint8 else 16,
            'pixel_size': None,
        }
        
        # Extract physical pixel size
        if hasattr(img, 'physical_pixel_sizes'):
            pps = img.physical_pixel_sizes
            if pps.X is not None:
                metadata['pixel_size'] = pps.X  # in micrometers
            metadata['pixel_size_z'] = pps.Z if pps.Z is not None else None
        
        # Get image data (TZCYX format from aicsimageio)
        # Select specific timepoint if multiple
        if img.dims.T > 1:
            # Get single timepoint
            data = img.get_image_dask_data("TZCYX", T=timepoint)
            # Convert to numpy and remove T dimension
            image = np.array(data[0])  # (Z, C, Y, X)
        else:
            data = img.get_image_dask_data("ZCYX")
            image = np.array(data)  # (Z, C, Y, X)
        
        # If single Z slice, remove Z dimension to get (C, Y, X)
        if metadata['n_slices'] == 1:
            image = image[0]  # (C, Y, X)
        
        # Generate channel names if not available
        if not metadata['channel_names']:
            metadata['channel_names'] = [f'Channel {i+1}' for i in range(metadata['n_channels'])]
        
        metadata['final_shape'] = image.shape
        
        return image, metadata
    
    @staticmethod
    def load_lif(filepath: str, scene: int = 0, timepoint: int = 0) -> Tuple[np.ndarray, Dict]:
        """
        Load Leica LIF file
        
        Args:
            filepath: Path to .lif file
            scene: Scene/series index to load (default: 0)
            timepoint: Timepoint to load (default: 0)
            
        Returns:
            tuple: (image_array, metadata_dict)
                image_array: numpy array with shape (Z, C, Y, X) or (C, Y, X)
                metadata_dict: dictionary with image properties
        """
        try:
            from aicsimageio import AICSImage
        except ImportError:
            raise ImportError(
                "aicsimageio is required to read LIF files. "
                "Install with: pip install aicsimageio"
            )
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load LIF file
        img = AICSImage(filepath)
        
        # Get available scenes (LIF files often contain multiple images)
        n_scenes = img.scenes
        if scene >= len(n_scenes):
            scene = 0
        
        # Set scene if multiple available
        if len(n_scenes) > 1:
            img.set_scene(n_scenes[scene])
        
        # Get image dimensions
        metadata = {
            'filename': filepath.name,
            'format': 'LIF (Leica)',
            'scenes': n_scenes,
            'current_scene': scene,
            'shape': img.shape,
            'dims': img.dims.order,
            'n_channels': img.dims.C,
            'n_slices': img.dims.Z,
            'n_timepoints': img.dims.T,
            'is_3d': img.dims.Z > 1,
            'channel_names': img.channel_names if hasattr(img, 'channel_names') else [],
            'dtype': str(img.dtype),
            'bit_depth': 8 if img.dtype == np.uint8 else 16,
            'pixel_size': None,
        }
        
        # Extract physical pixel size
        if hasattr(img, 'physical_pixel_sizes'):
            pps = img.physical_pixel_sizes
            if pps.X is not None:
                metadata['pixel_size'] = pps.X  # in micrometers
            metadata['pixel_size_z'] = pps.Z if pps.Z is not None else None
        
        # Get image data (TZCYX format from aicsimageio)
        # Select specific timepoint if multiple
        if img.dims.T > 1:
            # Get single timepoint
            data = img.get_image_dask_data("TZCYX", T=timepoint)
            # Convert to numpy and remove T dimension
            image = np.array(data[0])  # (Z, C, Y, X)
        else:
            data = img.get_image_dask_data("ZCYX")
            image = np.array(data)  # (Z, C, Y, X)
        
        # If single Z slice, remove Z dimension to get (C, Y, X)
        if metadata['n_slices'] == 1:
            image = image[0]  # (C, Y, X)
        
        # Generate channel names if not available
        if not metadata['channel_names']:
            metadata['channel_names'] = [f'Channel {i+1}' for i in range(metadata['n_channels'])]
        
        metadata['final_shape'] = image.shape
        
        return image, metadata
    
    @staticmethod
    def get_vsi_info(filepath: str) -> Dict:
        """
        Get metadata from VSI file without loading full image data
        
        Args:
            filepath: Path to .vsi file
            
        Returns:
            Dictionary with metadata including available scenes
        """
        try:
            from aicsimageio import AICSImage
        except ImportError:
            raise ImportError(
                "aicsimageio is required to read VSI files. "
                "Install with: pip install aicsimageio"
            )
        
        img = AICSImage(filepath)
        
        scenes_info = []
        for scene_name in img.scenes:
            img.set_scene(scene_name)
            scenes_info.append({
                'name': scene_name,
                'shape': img.shape,
                'dims': img.dims.order,
                'n_channels': img.dims.C,
                'n_slices': img.dims.Z,
                'n_timepoints': img.dims.T,
                'channel_names': img.channel_names if hasattr(img, 'channel_names') else []
            })
        
        return {
            'format': 'VSI (Olympus)',
            'n_scenes': len(img.scenes),
            'scenes': scenes_info
        }
    
    @staticmethod
    def get_lif_info(filepath: str) -> Dict:
        """
        Get metadata from LIF file without loading full image data
        
        Args:
            filepath: Path to .lif file
            
        Returns:
            Dictionary with metadata including available scenes
        """
        try:
            from aicsimageio import AICSImage
        except ImportError:
            raise ImportError(
                "aicsimageio is required to read LIF files. "
                "Install with: pip install aicsimageio"
            )
        
        img = AICSImage(filepath)
        
        scenes_info = []
        for scene_name in img.scenes:
            img.set_scene(scene_name)
            scenes_info.append({
                'name': scene_name,
                'shape': img.shape,
                'dims': img.dims.order,
                'n_channels': img.dims.C,
                'n_slices': img.dims.Z,
                'n_timepoints': img.dims.T,
                'channel_names': img.channel_names if hasattr(img, 'channel_names') else []
            })
        
        return {
            'format': 'LIF (Leica)',
            'n_scenes': len(img.scenes),
            'scenes': scenes_info
        }
    
    @staticmethod
    def load_bioformats(filepath: str, scene: int = 0, timepoint: int = 0) -> Tuple[np.ndarray, Dict]:
        """
        Load any bioformats-supported file (auto-detects format)
        
        Args:
            filepath: Path to microscopy file
            scene: Scene/series index to load (default: 0)
            timepoint: Timepoint to load (default: 0)
            
        Returns:
            tuple: (image_array, metadata_dict)
        """
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()
        
        if suffix == '.vsi':
            return BioformatsLoader.load_vsi(str(filepath), scene, timepoint)
        elif suffix == '.lif':
            return BioformatsLoader.load_lif(str(filepath), scene, timepoint)
        else:
            # Try generic aicsimageio loading
            try:
                from aicsimageio import AICSImage
            except ImportError:
                raise ImportError(
                    "aicsimageio is required to read this file format. "
                    "Install with: pip install aicsimageio"
                )
            
            img = AICSImage(filepath)
            
            # Get available scenes
            n_scenes = img.scenes
            if scene >= len(n_scenes):
                scene = 0
            
            if len(n_scenes) > 1:
                img.set_scene(n_scenes[scene])
            
            # Get metadata
            metadata = {
                'filename': filepath.name,
                'format': suffix.upper()[1:],
                'scenes': n_scenes,
                'current_scene': scene,
                'shape': img.shape,
                'dims': img.dims.order,
                'n_channels': img.dims.C,
                'n_slices': img.dims.Z,
                'n_timepoints': img.dims.T,
                'is_3d': img.dims.Z > 1,
                'channel_names': img.channel_names if hasattr(img, 'channel_names') else [],
                'dtype': str(img.dtype),
                'bit_depth': 8 if img.dtype == np.uint8 else 16,
                'pixel_size': None,
            }
            
            # Extract physical pixel size
            if hasattr(img, 'physical_pixel_sizes'):
                pps = img.physical_pixel_sizes
                if pps.X is not None:
                    metadata['pixel_size'] = pps.X
                metadata['pixel_size_z'] = pps.Z if pps.Z is not None else None
            
            # Get image data
            if img.dims.T > 1:
                data = img.get_image_dask_data("TZCYX", T=timepoint)
                image = np.array(data[0])
            else:
                data = img.get_image_dask_data("ZCYX")
                image = np.array(data)
            
            # If single Z slice, remove Z dimension
            if metadata['n_slices'] == 1:
                image = image[0]
            
            # Generate channel names if not available
            if not metadata['channel_names']:
                metadata['channel_names'] = [f'Channel {i+1}' for i in range(metadata['n_channels'])]
            
            metadata['final_shape'] = image.shape
            
            return image, metadata
