"""
Image I/O module for loading TIFF files with tifffile
Supports 2D/3D multichannel TIFF files in 8-bit and 16-bit formats
"""

import numpy as np
import tifffile
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import PIL.Image

# Disable DecompressionBombError for large images
PIL.Image.MAX_IMAGE_PIXELS = None


class TIFFLoader:
    """
    Handles loading of single and multiframe TIFF files
    Supports 2D multichannel and 3D multichannel images
    Also supports Metamorph .nd files that reference multiple TIFF files
    """

    @staticmethod
    def load_tiff(filepath: str) -> Tuple[np.ndarray, Dict]:
        """
        Load a TIFF file or Metamorph .nd file and extract metadata

        Args:
            filepath: Path to TIFF or .nd file

        Returns:
            tuple: (image_array, metadata_dict)
                image_array: numpy array with shape (Z, C, Y, X) or (C, Y, X)
                metadata_dict: dictionary with image properties
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Check if this is a Metamorph .nd file
        if filepath.suffix.lower() == '.nd':
            return TIFFLoader.load_metamorph_nd(str(filepath))

        # Check if this is a VSI (Olympus) file
        if filepath.suffix.lower() == '.vsi':
            return TIFFLoader.load_vsi(str(filepath))

        # Check if this is a LIF (Leica) file
        if filepath.suffix.lower() == '.lif':
            return TIFFLoader.load_lif(str(filepath))

        # Load regular TIFF image
        with tifffile.TiffFile(filepath) as tif:
            # Check for very large image (e.g. > 4GB)
            total_bytes = 0
            if hasattr(tif, 'series') and len(tif.series) > 0:
                total_bytes = tif.series[0].size * tif.series[0].dtype.itemsize

            # Use memmap if larger than 4GB (or if allocation fails)
            try:
                if total_bytes > 4 * 1024**3:
                    print(f"Large image detected ({total_bytes / 1024**3:.2f} GB). Using memory mapping.")
                    image = tif.asarray(out='memmap')
                else:
                    image = tif.asarray()
            except MemoryError:
                print("MemoryError encountered. Falling back to memory mapping.")
                image = tif.asarray(out='memmap')

            # Extract metadata
            metadata = TIFFLoader._extract_metadata(tif)

        # Normalize dimensions to (Z, C, Y, X) or (C, Y, X)
        image, metadata = TIFFLoader._normalize_dimensions(image, metadata)

        return image, metadata

    @staticmethod
    def _extract_metadata(tif: tifffile.TiffFile) -> Dict:
        """Extract metadata from TiffFile object"""
        metadata = {
            'filename': Path(tif.filename).name if hasattr(tif, 'filename') else 'unknown',
            'shape': tif.series[0].shape,
            'dtype': str(tif.series[0].dtype),
            'axes': tif.series[0].axes if hasattr(tif.series[0], 'axes') else None,
            'is_imagej': tif.is_imagej,
            'is_ome': tif.is_ome,
            'pixel_size': None,
            'channel_names': [],
            'bit_depth': 8,
        }

        # Get bit depth
        if tif.pages:
            page = tif.pages[0]
            if hasattr(page, 'bitspersample'):
                metadata['bit_depth'] = page.bitspersample
            elif tif.series[0].dtype == np.uint8:
                metadata['bit_depth'] = 8
            elif tif.series[0].dtype == np.uint16:
                metadata['bit_depth'] = 16

        # Try to extract pixel size
        if tif.imagej_metadata:
            # ImageJ metadata
            ij_meta = tif.imagej_metadata
            if 'XResolution' in ij_meta:
                metadata['pixel_size'] = 1.0 / ij_meta['XResolution']
            if 'channels' in ij_meta:
                metadata['n_channels'] = ij_meta['channels']
            if 'slices' in ij_meta:
                metadata['n_slices'] = ij_meta['slices']

        # OME-TIFF metadata
        if tif.is_ome and tif.ome_metadata:
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)

                # Extract pixel size
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                pixels = root.find('.//ome:Pixels', ns)
                if pixels is not None:
                    if 'PhysicalSizeX' in pixels.attrib:
                        metadata['pixel_size'] = float(pixels.attrib['PhysicalSizeX'])

                    # Extract channel names
                    channels = pixels.findall('.//ome:Channel', ns)
                    metadata['channel_names'] = [
                        ch.attrib.get('Name', f'Channel {i}')
                        for i, ch in enumerate(channels)
                    ]
            except Exception as e:
                print(f"Warning: Could not parse OME metadata: {e}")

        return metadata

    @staticmethod
    def _normalize_dimensions(image: np.ndarray, metadata: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Normalize image dimensions to (Z, C, Y, X) or (C, Y, X)

        Args:
            image: Raw image array
            metadata: Metadata dictionary

        Returns:
            tuple: (normalized_image, updated_metadata)
        """
        original_shape = image.shape
        axes = metadata.get('axes', '')

        # Try to infer dimensions from axes string
        if axes:
            axes = axes.upper()

            # Build dimension mapping
            dim_map = {}
            for i, ax in enumerate(axes):
                if ax in 'TZCYX':
                    dim_map[ax] = i

            # Reorder to (T, Z, C, Y, X) then remove T if present
            new_order = []
            for ax in 'TZCYX':
                if ax in dim_map:
                    new_order.append(dim_map[ax])

            if len(new_order) > 0:
                # Create all dimensions list
                all_dims = list(range(len(original_shape)))
                remaining = [d for d in all_dims if d not in new_order]
                new_order.extend(remaining)

                image = np.transpose(image, new_order[:len(original_shape)])

                # Remove time dimension if present
                if 'T' in dim_map and image.shape[0] == 1:
                    image = image[0]

        # Handle common cases by shape analysis
        ndim = image.ndim

        if ndim == 2:
            # Single channel 2D image: (Y, X) -> (1, Y, X)
            image = image[np.newaxis, ...]
            metadata['n_channels'] = 1
            metadata['n_slices'] = 1
            metadata['is_3d'] = False

        elif ndim == 3:
            # Could be (C, Y, X) or (Z, Y, X) or (Y, X, C)
            min_dim_idx = np.argmin(image.shape)
            min_dim_size = image.shape[min_dim_idx]

            # Check metadata first
            if metadata.get('n_slices', 0) > 1 and metadata.get('n_channels', 0) <= 1:
                 # Explicitly Z-stack
                 image = image[:, np.newaxis, :, :]
                 metadata['is_3d'] = True
            elif metadata.get('n_channels', 0) > 1:
                 # Explicitly channels
                 if min_dim_idx == 2:
                     image = np.transpose(image, (2, 0, 1))
                 metadata['n_slices'] = 1
                 metadata['is_3d'] = False

            # Heuristics
            elif min_dim_size <= 4 and image.dtype == np.uint8:
                # Likely RGB/RGBA
                if min_dim_idx == 0:
                    # (C, Y, X)
                    metadata['n_channels'] = image.shape[0]
                    metadata['n_slices'] = 1
                    metadata['is_3d'] = False
                elif min_dim_idx == 2:
                    # (Y, X, C) -> (C, Y, X)
                    image = np.transpose(image, (2, 0, 1))
                    metadata['n_channels'] = image.shape[0]
                    metadata['n_slices'] = 1
                    metadata['is_3d'] = False
                else:
                    # Weird case, assume Z
                    image = image[:, np.newaxis, :, :]
                    metadata['n_channels'] = 1
                    metadata['n_slices'] = image.shape[0]
                    metadata['is_3d'] = True
            else:
                # Assume (Z, Y, X) single channel 3D
                # Convert to (Z, C, Y, X) with C=1
                image = image[:, np.newaxis, :, :]
                metadata['n_channels'] = 1
                metadata['n_slices'] = image.shape[0]
                metadata['is_3d'] = True

        elif ndim == 4:
            # Could be (Z, C, Y, X) or (C, Z, Y, X) or others
            # Heuristic: channels dimension is usually smaller
            if image.shape[1] <= 16:
                # Likely (Z, C, Y, X) - correct format
                metadata['n_slices'] = image.shape[0]
                metadata['n_channels'] = image.shape[1]
                metadata['is_3d'] = True
            elif image.shape[0] <= 16:
                # Likely (C, Z, Y, X) -> transpose to (Z, C, Y, X)
                image = np.transpose(image, (1, 0, 2, 3))
                metadata['n_slices'] = image.shape[0]
                metadata['n_channels'] = image.shape[1]
                metadata['is_3d'] = True
            else:
                # Ambiguous, assume (Z, C, Y, X)
                metadata['n_slices'] = image.shape[0]
                metadata['n_channels'] = image.shape[1]
                metadata['is_3d'] = True

        elif ndim == 5:
            # Probably (T, Z, C, Y, X) - take first timepoint
            image = image[0]
            metadata['n_slices'] = image.shape[0]
            metadata['n_channels'] = image.shape[1]
            metadata['is_3d'] = True

        # Generate channel names if not present
        if not metadata.get('channel_names'):
            n_channels = metadata.get('n_channels', image.shape[-3] if image.ndim == 4 else image.shape[0])
            metadata['channel_names'] = [f'Channel {i+1}' for i in range(n_channels)]

        # Update final shape
        metadata['final_shape'] = image.shape

        return image, metadata

    @staticmethod
    def load_metamorph_nd(nd_filepath: str, stage: int = 0, timepoint: int = 0) -> Tuple[np.ndarray, Dict]:
        """
        Load Metamorph .nd file series as a single stack

        Args:
            nd_filepath: Path to .nd file
            stage: Stage position to load (default: 0)
            timepoint: Timepoint to load (default: 0)

        Returns:
            tuple: (image_array, metadata_dict)
        """
        from core.metamorph_nd import MetamorphNDFile

        nd_file = MetamorphNDFile(nd_filepath)
        image, metadata = nd_file.build_stack(stage=stage, timepoint=timepoint)

        # Add nd file path to metadata
        metadata['nd_file'] = nd_filepath
        metadata['is_metamorph_series'] = True

        return image, metadata

    @staticmethod
    def load_metamorph_nd_all(nd_filepath: str) -> List[Tuple[np.ndarray, Dict]]:
        """
        Load all stacks from Metamorph .nd file

        Args:
            nd_filepath: Path to .nd file

        Returns:
            List of (image_array, metadata_dict) tuples
        """
        from core.metamorph_nd import MetamorphNDFile

        nd_file = MetamorphNDFile(nd_filepath)
        return nd_file.get_all_stacks()

    @staticmethod
    def get_metamorph_info(nd_filepath: str) -> Dict:
        """
        Get metadata from Metamorph .nd file without loading images

        Args:
            nd_filepath: Path to .nd file

        Returns:
            Dictionary with metadata
        """
        from core.metamorph_nd import MetamorphNDFile

        nd_file = MetamorphNDFile(nd_filepath)
        return {
            'metadata': nd_file.metadata,
            'n_files': len(nd_file.file_list),
            'n_stages': len(nd_file.group_by_stage()),
            'n_timepoints': len(nd_file.group_by_timepoint()),
            'file_list': nd_file.file_list
        }

    @staticmethod
    def load_vsi(vsi_filepath: str, scene: int = 0, timepoint: int = 0) -> Tuple[np.ndarray, Dict]:
        """
        Load Olympus VSI file

        Args:
            vsi_filepath: Path to .vsi file
            scene: Scene/series index to load (default: 0)
            timepoint: Timepoint to load (default: 0)

        Returns:
            tuple: (image_array, metadata_dict)
        """
        from core.bioformats_io import BioformatsLoader

        image, metadata = BioformatsLoader.load_vsi(vsi_filepath, scene=scene, timepoint=timepoint)

        # Add file path to metadata
        metadata['vsi_file'] = vsi_filepath
        metadata['is_vsi_series'] = True

        return image, metadata

    @staticmethod
    def load_lif(lif_filepath: str, scene: int = 0, timepoint: int = 0) -> Tuple[np.ndarray, Dict]:
        """
        Load Leica LIF file

        Args:
            lif_filepath: Path to .lif file
            scene: Scene/series index to load (default: 0)
            timepoint: Timepoint to load (default: 0)

        Returns:
            tuple: (image_array, metadata_dict)
        """
        from core.bioformats_io import BioformatsLoader

        image, metadata = BioformatsLoader.load_lif(lif_filepath, scene=scene, timepoint=timepoint)

        # Add file path to metadata
        metadata['lif_file'] = lif_filepath
        metadata['is_lif_series'] = True

        return image, metadata

    @staticmethod
    def get_vsi_info(vsi_filepath: str) -> Dict:
        """
        Get metadata from VSI file without loading images

        Args:
            vsi_filepath: Path to .vsi file

        Returns:
            Dictionary with metadata including available scenes
        """
        from core.bioformats_io import BioformatsLoader

        return BioformatsLoader.get_vsi_info(vsi_filepath)

    @staticmethod
    def get_lif_info(lif_filepath: str) -> Dict:
        """
        Get metadata from LIF file without loading images

        Args:
            lif_filepath: Path to .lif file

        Returns:
            Dictionary with metadata including available scenes
        """
        from core.bioformats_io import BioformatsLoader

        return BioformatsLoader.get_lif_info(lif_filepath)

    @staticmethod
    def save_tiff(filepath: str, image: np.ndarray, metadata: Optional[Dict] = None,
                  compress: bool = True):
        """
        Save image as TIFF file

        Args:
            filepath: Output file path
            image: Image array
            metadata: Optional metadata dictionary
            compress: Whether to use LZW compression
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Prepare ImageJ-compatible metadata
        imagej_metadata = {}

        if metadata:
            if 'n_channels' in metadata:
                imagej_metadata['channels'] = metadata['n_channels']
            if 'n_slices' in metadata:
                imagej_metadata['slices'] = metadata['n_slices']
            if 'pixel_size' in metadata and metadata['pixel_size']:
                imagej_metadata['unit'] = 'um'
                imagej_metadata['spacing'] = metadata['pixel_size']

        # Determine axes based on dimensions
        if image.ndim == 3:
            axes = 'CYX'
        elif image.ndim == 4:
            axes = 'ZCYX'
        else:
            axes = None

        # Save
        tifffile.imwrite(
            filepath,
            image,
            photometric='minisblack',
            metadata=imagej_metadata if imagej_metadata else None,
            imagej=True if imagej_metadata else False,
            compression='lzw' if compress else None,
        )

    @staticmethod
    def save_labeled_mask(filepath: str, mask: np.ndarray, lut: Optional[np.ndarray] = None):
        """
        Save labeled segmentation mask with optional LUT (color lookup table)

        Args:
            filepath: Output file path
            mask: Labeled mask array (integer labels)
            lut: Optional color lookup table (N x 3 RGB values)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert to 16-bit if needed (supports more labels)
        if mask.max() > 255:
            mask = mask.astype(np.uint16)
        else:
            mask = mask.astype(np.uint8)

        # Create ImageJ-compatible LUT if provided
        if lut is not None:
            # ImageJ expects 256 or 65536 entry LUT
            n_colors = 256 if mask.dtype == np.uint8 else 65536
            imagej_lut = np.zeros((n_colors, 3), dtype=np.uint8)
            imagej_lut[:len(lut)] = lut

            tifffile.imwrite(
                filepath,
                mask,
                photometric='palette',
                colormap=imagej_lut,
                imagej=True,
                metadata={'mode': 'color'}
            )
        else:
            tifffile.imwrite(
                filepath,
                mask,
                photometric='minisblack',
                imagej=True
            )


def get_default_channel_colors(n_channels: int) -> List[Tuple[int, int, int]]:
    """
    Get default RGB colors for visualizing channels

    Args:
        n_channels: Number of channels

    Returns:
        List of (R, G, B) tuples
    """
    default_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 255, 255),# White
    ]

    # Extend if more channels
    colors = []
    for i in range(n_channels):
        if i < len(default_colors):
            colors.append(default_colors[i])
        else:
            # Generate random-ish colors
            np.random.seed(i)
            colors.append(tuple(np.random.randint(128, 256, 3).tolist()))

    return colors
