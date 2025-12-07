"""
Measurement extraction engine for quantifying segmented nuclei
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from skimage.measure import regionprops, regionprops_table
from sklearn.mixture import GaussianMixture
import warnings


class MeasurementEngine:
    """
    Extract morphometric and intensity measurements from segmented nuclei
    """
    
    def __init__(self, pixel_size: Optional[Tuple[float, ...]] = None):
        """
        Initialize MeasurementEngine
        
        Args:
            pixel_size: Tuple of pixel dimensions in physical units (e.g., micrometers).
                       For 2D: (pixel_size_y, pixel_size_x) or (pixel_size,) for isotropic
                       For 3D: (pixel_size_z, pixel_size_y, pixel_size_x)
                       If None, measurements are in pixels/voxels
        """
        self.measurement_categories = {
            'basic_shape': ['area', 'perimeter', 'circularity', 'equivalent_diameter'],
            'advanced_morphology': ['eccentricity', 'solidity', 'major_axis_length', 
                                   'minor_axis_length', 'orientation'],
            'intensity_stats': ['mean_intensity', 'min_intensity', 'max_intensity',
                               'median_intensity', 'std_intensity', 'integrated_density', 'cv'],
            'cell_cycle': ['dna_intensity', 'phase']
        }
        
        self.enabled_categories = set(self.measurement_categories.keys())
        self.is_3d = False
        self.pixel_size = pixel_size
        self._pixel_area = None  # Computed based on dimensionality
        self._pixel_volume = None
        self._pixel_length = None  # For linear measurements
    
    def extract_measurements(self,
                            masks: np.ndarray,
                            intensity_images: Optional[Dict[str, np.ndarray]] = None,
                            is_3d: bool = False,
                            dna_channel: Optional[str] = None,
                            assign_phases: bool = False,
                            pixel_size: Optional[Tuple[float, ...]] = None) -> pd.DataFrame:
        """
        Extract all enabled measurements from segmented masks
        
        Args:
            masks: Labeled segmentation masks (2D or 3D)
            intensity_images: Dict of channel_name -> intensity image
            is_3d: Whether to compute 3D measurements
            dna_channel: Name of DNA channel for cell cycle analysis
            assign_phases: Whether to assign cell cycle phases
            pixel_size: Override pixel size for this extraction.
                       For 2D: (pixel_size_y, pixel_size_x) or (pixel_size,) for isotropic
                       For 3D: (pixel_size_z, pixel_size_y, pixel_size_x)
        
        Returns:
            DataFrame with measurements for each nucleus
        """
        self.is_3d = is_3d
        
        # Update pixel size if provided
        if pixel_size is not None:
            self.pixel_size = pixel_size
        
        # Compute scaling factors based on pixel size and dimensionality
        self._compute_scaling_factors(is_3d)
        
        # Initialize results
        measurements = []
        
        # Get region properties
        if intensity_images:
            # Use first channel for morphology
            intensity_img = list(intensity_images.values())[0]
        else:
            intensity_img = None
        
        regions = regionprops(masks, intensity_image=intensity_img)
        
        if len(regions) == 0:
            return pd.DataFrame()
        
        # Extract measurements for each nucleus
        for region in regions:
            nucleus_measurements = {'nucleus_id': region.label}
            
            # Basic shape measurements
            if 'basic_shape' in self.enabled_categories:
                nucleus_measurements.update(self._measure_basic_shape(region))
            
            # Advanced morphology
            if 'advanced_morphology' in self.enabled_categories:
                nucleus_measurements.update(self._measure_advanced_morphology(region))
            
            # Intensity statistics for each channel
            if 'intensity_stats' in self.enabled_categories and intensity_images:
                for channel_name, img in intensity_images.items():
                    channel_stats = self._measure_intensity_stats(region, img, masks)
                    # Prefix with channel name
                    for key, value in channel_stats.items():
                        nucleus_measurements[f"{channel_name}_{key}"] = value
            
            measurements.append(nucleus_measurements)
        
        # Create DataFrame
        df = pd.DataFrame(measurements)
        
        # Cell cycle phase assignment
        if 'cell_cycle' in self.enabled_categories and assign_phases and dna_channel:
            if dna_channel in intensity_images:
                df = self._assign_cell_cycle_phases(df, dna_channel)
        
        return df
    
    def _measure_basic_shape(self, region) -> Dict[str, float]:
        """Measure basic shape properties with optional physical unit scaling"""
        measurements = {}
        
        # Get scaling factors (default to 1.0 if not set)
        pixel_area = self._pixel_area if self._pixel_area else 1.0
        pixel_volume = self._pixel_volume if self._pixel_volume else 1.0
        pixel_length = self._pixel_length if self._pixel_length else 1.0
        
        if self.is_3d:
            # 3D measurements
            volume_voxels = float(region.area)  # In 3D, area is volume in voxels
            measurements['volume'] = volume_voxels * pixel_volume
            
            # Surface area using marching cubes approximation if scipy available
            try:
                from skimage.measure import marching_cubes, mesh_surface_area
                
                # Get the binary mask for this region
                binary_mask = region.image.astype(np.uint8)
                if binary_mask.sum() > 10:  # Need enough voxels for marching cubes
                    # Pad the mask to ensure closed surface
                    padded = np.pad(binary_mask, 1, mode='constant', constant_values=0)
                    try:
                        verts, faces, _, _ = marching_cubes(padded, level=0.5)
                        
                        # Scale vertices by pixel size if anisotropic
                        if self.pixel_size is not None and len(self.pixel_size) >= 3:
                            verts[:, 0] *= self.pixel_size[0]  # Z
                            verts[:, 1] *= self.pixel_size[1]  # Y  
                            verts[:, 2] *= self.pixel_size[2]  # X
                        elif self.pixel_size is not None and len(self.pixel_size) == 1:
                            verts *= self.pixel_size[0]
                        
                        surface_area = mesh_surface_area(verts, faces)
                        measurements['surface_area'] = float(surface_area)
                    except (RuntimeError, ValueError):
                        # Fallback to bounding box approximation
                        bbox = region.bbox
                        dx = (bbox[3] - bbox[0]) * (self.pixel_size[0] if self.pixel_size and len(self.pixel_size) >= 3 else pixel_length)
                        dy = (bbox[4] - bbox[1]) * (self.pixel_size[1] if self.pixel_size and len(self.pixel_size) >= 3 else pixel_length)
                        dz = (bbox[5] - bbox[2]) * (self.pixel_size[2] if self.pixel_size and len(self.pixel_size) >= 3 else pixel_length)
                        measurements['surface_area'] = float(2 * (dx*dy + dy*dz + dz*dx))
                else:
                    # Very small object - use bbox approximation
                    bbox = region.bbox
                    dx = (bbox[3] - bbox[0]) * pixel_length
                    dy = (bbox[4] - bbox[1]) * pixel_length
                    dz = (bbox[5] - bbox[2]) * pixel_length
                    measurements['surface_area'] = float(2 * (dx*dy + dy*dz + dz*dx))
                    
            except ImportError:
                # Fallback: bbox surface area approximation
                bbox = region.bbox
                dx = (bbox[3] - bbox[0]) * pixel_length
                dy = (bbox[4] - bbox[1]) * pixel_length
                dz = (bbox[5] - bbox[2]) * pixel_length
                measurements['surface_area'] = float(2 * (dx*dy + dy*dz + dz*dx))
            
            # Sphericity (how sphere-like the object is)
            if measurements['surface_area'] > 0:
                measurements['sphericity'] = float(
                    (np.pi ** (1/3) * (6 * measurements['volume']) ** (2/3)) /
                    measurements['surface_area']
                )
            else:
                measurements['sphericity'] = 0.0
            
            # Equivalent sphere diameter
            measurements['equivalent_diameter'] = float(
                2 * (3 * measurements['volume'] / (4 * np.pi)) ** (1/3)
            )
            
            # Surface-to-volume ratio
            if measurements['volume'] > 0:
                measurements['surface_to_volume_ratio'] = float(
                    measurements['surface_area'] / measurements['volume']
                )
            else:
                measurements['surface_to_volume_ratio'] = 0.0
                
        else:
            # 2D measurements
            area_pixels = float(region.area)
            measurements['area'] = area_pixels * pixel_area
            measurements['perimeter'] = float(region.perimeter) * pixel_length
            
            # Circularity (4π*area/perimeter²) - dimensionless
            if region.perimeter > 0:
                measurements['circularity'] = float(
                    4 * np.pi * region.area / (region.perimeter ** 2)
                )
            else:
                measurements['circularity'] = 0.0
            
            measurements['equivalent_diameter'] = float(region.equivalent_diameter) * pixel_length
        
        return measurements
    
    def _measure_advanced_morphology(self, region) -> Dict[str, float]:
        """Measure advanced morphological features"""
        measurements = {}
        
        if self.is_3d:
            # 3D measurements - some properties are not available or meaningful in 3D
            # Use try/except to safely access properties that may not exist
            try:
                # Solidity works for both 2D and 3D
                measurements['solidity'] = float(region.solidity)
            except (AttributeError, ValueError):
                measurements['solidity'] = 0.0
            
            # For 3D, compute axis lengths from inertia tensor if available
            try:
                # In 3D, major/minor axis lengths come from the moments
                if hasattr(region, 'inertia_tensor_eigvals'):
                    eigvals = region.inertia_tensor_eigvals
                    if len(eigvals) >= 3:
                        # Principal axis lengths approximation from eigenvalues
                        sorted_eigvals = sorted(eigvals, reverse=True)
                        # Convert moments to approximate axis lengths
                        measurements['major_axis_length'] = float(np.sqrt(sorted_eigvals[0]) * 4)
                        measurements['minor_axis_length'] = float(np.sqrt(sorted_eigvals[-1]) * 4)
                        measurements['intermediate_axis_length'] = float(np.sqrt(sorted_eigvals[1]) * 4)
                    else:
                        measurements['major_axis_length'] = 0.0
                        measurements['minor_axis_length'] = 0.0
                        measurements['intermediate_axis_length'] = 0.0
                else:
                    measurements['major_axis_length'] = 0.0
                    measurements['minor_axis_length'] = 0.0
                    measurements['intermediate_axis_length'] = 0.0
            except (AttributeError, ValueError, TypeError):
                measurements['major_axis_length'] = 0.0
                measurements['minor_axis_length'] = 0.0
                measurements['intermediate_axis_length'] = 0.0
            
            # Aspect ratio for 3D (major/minor)
            if measurements['minor_axis_length'] > 0:
                measurements['aspect_ratio'] = float(
                    measurements['major_axis_length'] / measurements['minor_axis_length']
                )
            else:
                measurements['aspect_ratio'] = 0.0
            
            # Extent works for both 2D and 3D
            try:
                measurements['extent'] = float(region.extent)
            except (AttributeError, ValueError):
                measurements['extent'] = 0.0
            
            # Eccentricity is not directly available for 3D in skimage
            # We can approximate using axis ratios
            if measurements['major_axis_length'] > 0:
                ratio = measurements['minor_axis_length'] / measurements['major_axis_length']
                measurements['eccentricity'] = float(np.sqrt(1 - ratio**2))
            else:
                measurements['eccentricity'] = 0.0
                
        else:
            # 2D measurements - these work reliably for 2D
            try:
                measurements['eccentricity'] = float(region.eccentricity)
            except (AttributeError, ValueError):
                measurements['eccentricity'] = 0.0
                
            try:
                measurements['solidity'] = float(region.solidity)
            except (AttributeError, ValueError):
                measurements['solidity'] = 0.0
                
            try:
                measurements['major_axis_length'] = float(region.major_axis_length)
            except (AttributeError, ValueError):
                measurements['major_axis_length'] = 0.0
                
            try:
                measurements['minor_axis_length'] = float(region.minor_axis_length)
            except (AttributeError, ValueError):
                measurements['minor_axis_length'] = 0.0
            
            # Aspect ratio
            if measurements['minor_axis_length'] > 0:
                measurements['aspect_ratio'] = float(
                    measurements['major_axis_length'] / measurements['minor_axis_length']
                )
            else:
                measurements['aspect_ratio'] = 0.0
            
            # Orientation (2D only)
            try:
                measurements['orientation'] = float(region.orientation)
            except (AttributeError, ValueError):
                measurements['orientation'] = 0.0
            
            # Extent (area/bbox_area)
            try:
                measurements['extent'] = float(region.extent)
            except (AttributeError, ValueError):
                measurements['extent'] = 0.0
        
        return measurements
    
    def _measure_intensity_stats(self,
                                 region,
                                 intensity_image: np.ndarray,
                                 masks: np.ndarray) -> Dict[str, float]:
        """Measure intensity statistics within nucleus"""
        measurements = {}
        
        # Get intensities within this nucleus
        mask = (masks == region.label)
        intensities = intensity_image[mask]
        
        if len(intensities) == 0:
            return {
                'mean_intensity': 0.0,
                'min_intensity': 0.0,
                'max_intensity': 0.0,
                'median_intensity': 0.0,
                'std_intensity': 0.0,
                'integrated_density': 0.0,
                'cv': 0.0
            }
        
        measurements['mean_intensity'] = float(np.mean(intensities))
        measurements['min_intensity'] = float(np.min(intensities))
        measurements['max_intensity'] = float(np.max(intensities))
        measurements['median_intensity'] = float(np.median(intensities))
        measurements['std_intensity'] = float(np.std(intensities))
        
        # Integrated density (sum of intensities)
        measurements['integrated_density'] = float(np.sum(intensities))
        
        # Coefficient of variation (CV = std/mean * 100)
        if measurements['mean_intensity'] > 0:
            measurements['cv'] = float(
                measurements['std_intensity'] / measurements['mean_intensity'] * 100
            )
        else:
            measurements['cv'] = 0.0
        
        return measurements
    
    def _assign_cell_cycle_phases(self,
                                  df: pd.DataFrame,
                                  dna_channel: str) -> pd.DataFrame:
        """
        Assign cell cycle phases based on DNA intensity
        
        Args:
            df: DataFrame with measurements
            dna_channel: Name of DNA channel
        
        Returns:
            DataFrame with added 'phase' column
        """
        intensity_col = f"{dna_channel}_mean_intensity"
        
        if intensity_col not in df.columns:
            warnings.warn(f"Column {intensity_col} not found for cell cycle analysis")
            return df
        
        intensities = df[intensity_col].values
        
        if len(intensities) < 10:  # Need enough data
            df['phase'] = 'Unknown'
            return df
        
        try:
            # Fit 3-component GMM (G1, S, G2/M)
            gmm = GaussianMixture(
                n_components=3,
                covariance_type='full',
                random_state=42
            )
            
            intensities_reshaped = intensities.reshape(-1, 1)
            gmm.fit(intensities_reshaped)
            
            # Predict phases
            phase_labels = gmm.predict(intensities_reshaped)
            
            # Get mean intensities for each phase
            means = gmm.means_.flatten()
            
            # Sort phases by intensity (G1 < S < G2/M)
            sorted_indices = np.argsort(means)
            phase_map = {sorted_indices[0]: 'G1',
                        sorted_indices[1]: 'S',
                        sorted_indices[2]: 'G2/M'}
            
            # Map to phase names
            df['phase'] = [phase_map[label] for label in phase_labels]
            
            # Also store raw DNA intensity for reference
            df['dna_intensity'] = intensities
            
        except Exception as e:
            warnings.warn(f"Cell cycle phase assignment failed: {e}")
            df['phase'] = 'Unknown'
            df['dna_intensity'] = intensities
        
        return df
    
    def set_enabled_categories(self, categories: List[str]):
        """Enable specific measurement categories"""
        self.enabled_categories = set(categories)
    
    def set_pixel_size(self, pixel_size: Optional[Tuple[float, ...]]):
        """
        Set pixel size for physical unit conversions
        
        Args:
            pixel_size: Tuple of pixel dimensions in physical units.
                       For 2D: (pixel_size_y, pixel_size_x) or (pixel_size,) for isotropic
                       For 3D: (pixel_size_z, pixel_size_y, pixel_size_x)
                       If None, measurements are in pixels/voxels
        """
        self.pixel_size = pixel_size
    
    def _compute_scaling_factors(self, is_3d: bool):
        """
        Compute scaling factors for converting pixels to physical units
        
        Args:
            is_3d: Whether 3D scaling is needed
        """
        if self.pixel_size is None:
            # No scaling - measurements in pixels/voxels
            self._pixel_area = 1.0
            self._pixel_volume = 1.0
            self._pixel_length = 1.0
            return
        
        ps = self.pixel_size
        
        if is_3d:
            if len(ps) == 1:
                # Isotropic 3D
                self._pixel_volume = ps[0] ** 3
                self._pixel_area = ps[0] ** 2  # For surface area
                self._pixel_length = ps[0]
            elif len(ps) >= 3:
                # Anisotropic 3D: (z, y, x)
                self._pixel_volume = ps[0] * ps[1] * ps[2]
                self._pixel_area = ps[1] * ps[2]  # XY area for surface approximation
                self._pixel_length = (ps[1] + ps[2]) / 2  # Average XY for linear
            else:
                # Fallback: treat as 2D with Z=1
                self._pixel_volume = ps[0] * ps[1] if len(ps) == 2 else ps[0]
                self._pixel_area = ps[0] * ps[1] if len(ps) == 2 else ps[0] ** 2
                self._pixel_length = ps[0]
        else:
            if len(ps) == 1:
                # Isotropic 2D
                self._pixel_area = ps[0] ** 2
                self._pixel_length = ps[0]
            elif len(ps) >= 2:
                # Anisotropic 2D: (y, x)
                self._pixel_area = ps[0] * ps[1]
                self._pixel_length = (ps[0] + ps[1]) / 2
            else:
                self._pixel_area = 1.0
                self._pixel_length = 1.0
            self._pixel_volume = 1.0  # Not used for 2D
    
    def get_measurement_units(self) -> Dict[str, str]:
        """
        Get units for each measurement based on pixel_size
        
        Returns:
            Dict mapping measurement name to unit string
        """
        if self.pixel_size is None:
            area_unit = "pixels²" if not self.is_3d else "voxels"
            length_unit = "pixels"
            volume_unit = "voxels"
        else:
            # Assume pixel_size is in micrometers (most common)
            area_unit = "µm²"
            length_unit = "µm"
            volume_unit = "µm³"
        
        return {
            'area': area_unit,
            'perimeter': length_unit,
            'equivalent_diameter': length_unit,
            'volume': volume_unit,
            'surface_area': area_unit,
            'major_axis_length': length_unit,
            'minor_axis_length': length_unit,
        }
    
    def get_available_categories(self) -> Dict[str, List[str]]:
        """Get dictionary of available measurement categories"""
        return self.measurement_categories.copy()
    
    def execute_plugins(self,
                       masks: np.ndarray,
                       intensity_images: Optional[Dict[str, np.ndarray]],
                       plugins: List,
                       metadata: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute custom measurement plugins
        
        Args:
            masks: Labeled segmentation masks
            intensity_images: Dict of channel_name -> intensity image
            plugins: List of MeasurementPlugin instances
            metadata: Optional metadata dict
        
        Returns:
            DataFrame with plugin measurements
        """
        if not plugins:
            return pd.DataFrame()
        
        regions = regionprops(masks)
        
        if len(regions) == 0:
            return pd.DataFrame()
        
        measurements = []
        
        for region in regions:
            nucleus_measurements = {'nucleus_id': region.label}
            
            # Execute each plugin
            for plugin in plugins:
                try:
                    plugin_results = plugin.measure(region, intensity_images, metadata)
                    
                    # Prefix with plugin name
                    plugin_name = plugin.get_name().lower().replace(' ', '_')
                    for key, value in plugin_results.items():
                        nucleus_measurements[f"{plugin_name}_{key}"] = value
                
                except Exception as e:
                    warnings.warn(f"Plugin {plugin.get_name()} failed: {e}")
            
            measurements.append(nucleus_measurements)
        
        return pd.DataFrame(measurements)
