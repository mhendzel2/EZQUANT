"""
Metamorph .nd file parser for loading image series
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np


class MetamorphNDFile:
    """Parser for Metamorph .nd files"""
    
    def __init__(self, nd_filepath: str):
        """
        Initialize Metamorph ND file parser
        
        Args:
            nd_filepath: Path to .nd file
        """
        self.nd_filepath = Path(nd_filepath)
        self.base_dir = self.nd_filepath.parent
        
        if not self.nd_filepath.exists():
            raise FileNotFoundError(f"ND file not found: {nd_filepath}")
        
        self.metadata = {}
        self.file_list = []
        
        self._parse_nd_file()
    
    def _parse_nd_file(self):
        """Parse the .nd file to extract metadata and file list"""
        with open(self.nd_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Parse metadata
        self.metadata = {
            'description': self._extract_value(content, 'Description'),
            'n_dimensions': self._extract_int(content, 'NDInfoFile'),
            'n_stages': self._extract_int(content, 'NStagePositions'),
            'n_wavelengths': self._extract_int(content, 'NWavelengths'),
            'n_zsteps': self._extract_int(content, 'NZSteps'),
            'n_timepoints': self._extract_int(content, 'NTimePoints'),
            'do_stage': self._extract_bool(content, 'DoStage'),
            'do_wavelength': self._extract_bool(content, 'DoWave'),
            'do_z': self._extract_bool(content, 'DoZ'),
            'do_timeseries': self._extract_bool(content, 'DoTimeSeries'),
        }
        
        # Extract wavelength names
        wavelength_names = []
        for i in range(self.metadata.get('n_wavelengths', 0)):
            name = self._extract_value(content, f'WaveName{i+1}')
            if name:
                wavelength_names.append(name)
            else:
                wavelength_names.append(f'Wavelength_{i+1}')
        
        self.metadata['wavelength_names'] = wavelength_names
        
        # Extract stage position names
        stage_names = []
        for i in range(self.metadata.get('n_stages', 0)):
            name = self._extract_value(content, f'Stage{i+1}')
            if name:
                stage_names.append(name)
            else:
                stage_names.append(f'Stage_{i+1}')
        
        self.metadata['stage_names'] = stage_names

        # Extract base name/prefix (key names vary across MetaMorph versions)
        self.metadata['base_name'] = (
            self._extract_value(content, 'BaseName')
            or self._extract_value(content, 'FileName')
            or self._extract_value(content, 'Filename')
            or self._extract_value(content, 'Prefix')
            or self.nd_filepath.stem
        )

        # Build file list (prefer discovering existing files; fallback to convention-based generation)
        self._build_file_list()
    
    def _extract_value(self, content: str, key: str) -> Optional[str]:
        """Extract string value for key"""
        pattern = rf'"{key}",\s*"([^"]*)"'
        match = re.search(pattern, content)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_int(self, content: str, key: str) -> int:
        """Extract integer value for key"""
        pattern = rf'"{key}",\s*(\d+)'
        match = re.search(pattern, content)
        if match:
            return int(match.group(1))
        return 0
    
    def _extract_bool(self, content: str, key: str) -> bool:
        """Extract boolean value for key"""
        pattern = rf'"{key}",\s*(TRUE|FALSE)'
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).upper() == 'TRUE'
        return False
    
    def _build_file_list(self):
        """Build list of image files based on metadata"""
        # Prefer discovering files by scanning directory; this is the most robust.
        discovered = self._discover_files_from_directory()
        if discovered:
            self.file_list = discovered
            return

        # Fallback: generate filenames based on common MetaMorph naming conventions
        base_pattern = (self.metadata.get('base_name') or '').strip()
        if base_pattern.lower().endswith('.tif') or base_pattern.lower().endswith('.tiff'):
            base_pattern = Path(base_pattern).stem
        if not base_pattern:
            base_pattern = self.nd_filepath.stem
        
        # Generate all file combinations
        n_stages = max(1, self.metadata.get('n_stages', 1))
        n_wavelengths = max(1, self.metadata.get('n_wavelengths', 1))
        n_zsteps = max(1, self.metadata.get('n_zsteps', 1))
        n_timepoints = max(1, self.metadata.get('n_timepoints', 1))
        
        do_stage = self.metadata.get('do_stage', False)
        do_wavelength = self.metadata.get('do_wavelength', False)
        do_z = self.metadata.get('do_z', False)
        do_timeseries = self.metadata.get('do_timeseries', False)
        
        # Build file list
        for t in range(n_timepoints if do_timeseries else 1):
            for s in range(n_stages if do_stage else 1):
                for w in range(n_wavelengths if do_wavelength else 1):
                    for z in range(n_zsteps if do_z else 1):
                        filename = self._construct_filename(
                            base_pattern, s, w, z, t,
                            do_stage, do_wavelength, do_z, do_timeseries
                        )
                        
                        filepath = self.base_dir / filename
                        
                        if filepath.exists():
                            self.file_list.append({
                                'filepath': filepath,
                                'stage': s,
                                'wavelength': w,
                                'z': z,
                                'timepoint': t,
                                'stage_name': self.metadata['stage_names'][s] if s < len(self.metadata['stage_names']) else f'Stage_{s}',
                                'wavelength_name': self.metadata['wavelength_names'][w] if w < len(self.metadata['wavelength_names']) else f'Wavelength_{w}'
                            })

    def _discover_files_from_directory(self) -> List[Dict]:
        """Discover TIFFs referenced by this ND by scanning the ND directory.

        MetaMorph ND files don't always provide a single reliable filename template.
        This method scans for TIFF/TIFF files and parses common tokens:
        - t### (timepoint, 1-based)
        - s#   (stage/position, 1-based)
        - w#   (wavelength/channel, 1-based)
        - z### (z slice, 1-based)

        Returns:
            List[Dict] with the same schema as self.file_list.
        """
        base_candidates = []
        base_name = (self.metadata.get('base_name') or '').strip()
        if base_name:
            base_candidates.append(Path(base_name).stem)
        base_candidates.append(self.nd_filepath.stem)
        base_candidates = [b for b in dict.fromkeys(base_candidates) if b]

        tif_files = list(self.base_dir.glob('*.tif')) + list(self.base_dir.glob('*.tiff'))
        if not tif_files:
            return []

        def parse_tokens(stem: str) -> Tuple[int, int, int, int]:
            # Defaults are 0-based indices
            stage_idx = 0
            wave_idx = 0
            z_idx = 0
            time_idx = 0

            # Parse token occurrences like t001, s1, w2, z003 anywhere in the stem
            for key, value in re.findall(r'(?i)([tswz])(\d+)', stem):
                key = key.lower()
                number = int(value)
                if key == 't':
                    time_idx = max(0, number - 1)
                elif key == 's':
                    stage_idx = max(0, number - 1)
                elif key == 'w':
                    wave_idx = max(0, number - 1)
                elif key == 'z':
                    z_idx = max(0, number - 1)

            return stage_idx, wave_idx, z_idx, time_idx

        discovered: List[Dict] = []
        for path in tif_files:
            stem = path.stem

            # If we have a plausible base name, prefer files that start with it.
            if base_candidates and not any(stem.startswith(c) for c in base_candidates):
                continue

            stage_idx, wave_idx, z_idx, time_idx = parse_tokens(stem)

            discovered.append({
                'filepath': path,
                'stage': stage_idx,
                'wavelength': wave_idx,
                'z': z_idx,
                'timepoint': time_idx,
                'stage_name': self.metadata['stage_names'][stage_idx] if stage_idx < len(self.metadata['stage_names']) else f'Stage_{stage_idx}',
                'wavelength_name': self.metadata['wavelength_names'][wave_idx] if wave_idx < len(self.metadata['wavelength_names']) else f'Wavelength_{wave_idx}'
            })

        # Sort for stable downstream stacking
        discovered.sort(key=lambda x: (x['timepoint'], x['stage'], x['z'], x['wavelength']))
        return discovered
    
    def _construct_filename(self, base: str, stage: int, wave: int, z: int, time: int,
                           do_stage: bool, do_wave: bool, do_z: bool, do_time: bool) -> str:
        """
        Construct filename based on Metamorph naming convention
        
        Common patterns:
        - base_s1_w1.tif (stage, wavelength)
        - base_w1s1.tif (wavelength, stage)
        - base_s1_w1_z001.tif (stage, wavelength, z)
        - base_t001_s1_w1.tif (time, stage, wavelength)
        """
        parts = [base]
        
        # Time is usually first if present
        if do_time:
            parts.append(f't{time+1:03d}')
        
        # Stage position
        if do_stage:
            parts.append(f's{stage+1}')
        
        # Wavelength
        if do_wave:
            parts.append(f'w{wave+1}')
        
        # Z position
        if do_z:
            parts.append(f'z{z+1:03d}')
        
        filename = '_'.join(parts) + '.tif'
        
        return filename
    
    def get_file_list(self) -> List[Dict]:
        """Get list of all image files with metadata"""
        return self.file_list
    
    def get_files_by_stage(self, stage_index: int) -> List[Dict]:
        """Get all files for a specific stage position"""
        return [f for f in self.file_list if f['stage'] == stage_index]
    
    def get_files_by_timepoint(self, timepoint: int) -> List[Dict]:
        """Get all files for a specific timepoint"""
        return [f for f in self.file_list if f['timepoint'] == timepoint]
    
    def get_files_by_wavelength(self, wavelength: int) -> List[Dict]:
        """Get all files for a specific wavelength"""
        return [f for f in self.file_list if f['wavelength'] == wavelength]
    
    def group_by_stage(self) -> Dict[int, List[Dict]]:
        """Group files by stage position"""
        grouped = {}
        for file_info in self.file_list:
            stage = file_info['stage']
            if stage not in grouped:
                grouped[stage] = []
            grouped[stage].append(file_info)
        return grouped
    
    def group_by_timepoint(self) -> Dict[int, List[Dict]]:
        """Group files by timepoint"""
        grouped = {}
        for file_info in self.file_list:
            timepoint = file_info['timepoint']
            if timepoint not in grouped:
                grouped[timepoint] = []
            grouped[timepoint].append(file_info)
        return grouped
    
    def build_stack(self, stage: int = 0, timepoint: int = 0) -> Tuple[np.ndarray, Dict]:
        """
        Build a multichannel Z-stack from individual files
        
        Args:
            stage: Stage position index
            timepoint: Timepoint index
            
        Returns:
            tuple: (image_array, metadata)
                image_array: shape (Z, C, Y, X) or (C, Y, X)
                metadata: dict with image properties
        """
        from core.image_io import TIFFLoader
        
        # Get files for this stage and timepoint
        files = [f for f in self.file_list 
                if f['stage'] == stage and f['timepoint'] == timepoint]
        
        if not files:
            raise ValueError(f"No files found for stage {stage}, timepoint {timepoint}")
        
        # Sort by z and wavelength
        files = sorted(files, key=lambda x: (x['z'], x['wavelength']))
        
        # Determine dimensions
        n_z = max([f['z'] for f in files]) + 1
        n_wavelengths = max([f['wavelength'] for f in files]) + 1
        
        # Load first image to get spatial dimensions
        first_img, first_meta = TIFFLoader.load_tiff(str(files[0]['filepath']))
        
        # Get Y, X dimensions from first image - always use last two dimensions
        # After _normalize_dimensions, images are either (C, Y, X) or (Z, C, Y, X)
        # In both cases, the last two dimensions are (Y, X)
        height, width = first_img.shape[-2], first_img.shape[-1]
        
        # Initialize output array
        if n_z > 1:
            # 3D stack
            output = np.zeros((n_z, n_wavelengths, height, width), dtype=first_img.dtype)
        else:
            # 2D multi-wavelength
            output = np.zeros((n_wavelengths, height, width), dtype=first_img.dtype)
        
        # Load all files
        for file_info in files:
            img, _ = TIFFLoader.load_tiff(str(file_info['filepath']))
            
            # Extract single 2D plane from the loaded image
            # After TIFFLoader, images are (C, Y, X) or (Z, C, Y, X)
            if img.ndim == 2:
                # Already 2D
                plane = img
            elif img.ndim == 3:
                # (C, Y, X) - take first channel
                plane = img[0]
            elif img.ndim == 4:
                # (Z, C, Y, X) - take middle Z slice, first channel
                z_mid = img.shape[0] // 2
                plane = img[z_mid, 0]
            else:
                # Unexpected, take slice
                plane = img.reshape(img.shape[-2:])
            
            # Verify dimensions match
            if plane.shape != (height, width):
                # Try to handle size mismatch by cropping or padding
                if plane.shape[0] >= height and plane.shape[1] >= width:
                    # Crop to expected size
                    plane = plane[:height, :width]
                else:
                    # Resize if significantly different
                    import warnings
                    warnings.warn(
                        f"Image size mismatch: expected ({height}, {width}), "
                        f"got {plane.shape}. Resizing to match."
                    )
                    from skimage.transform import resize
                    plane = resize(plane, (height, width), preserve_range=True).astype(img.dtype)
            
            z_idx = file_info['z']
            w_idx = file_info['wavelength']
            
            if n_z > 1:
                output[z_idx, w_idx] = plane
            else:
                output[w_idx] = plane
        
        # Build metadata
        metadata = {
            'filename': self.nd_filepath.name,
            'stage': stage,
            'stage_name': self.metadata['stage_names'][stage] if stage < len(self.metadata['stage_names']) else f'Stage_{stage}',
            'timepoint': timepoint,
            'n_channels': n_wavelengths,
            'n_slices': n_z,
            'is_3d': n_z > 1,
            'channel_names': self.metadata['wavelength_names'][:n_wavelengths],
            'final_shape': output.shape,
            'dtype': str(output.dtype),
            'bit_depth': 16 if output.dtype == np.uint16 else 8,
        }
        
        return output, metadata
    
    def get_all_stacks(self) -> List[Tuple[np.ndarray, Dict]]:
        """
        Build all stacks for all stage positions and timepoints
        
        Returns:
            List of (image_array, metadata) tuples
        """
        stacks = []
        
        # Get unique combinations
        stages = sorted(set([f['stage'] for f in self.file_list]))
        timepoints = sorted(set([f['timepoint'] for f in self.file_list]))
        
        for stage in stages:
            for timepoint in timepoints:
                try:
                    stack, metadata = self.build_stack(stage, timepoint)
                    stacks.append((stack, metadata))
                except Exception as e:
                    print(f"Warning: Could not build stack for stage {stage}, timepoint {timepoint}: {e}")
        
        return stacks
