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
    
    def __init__(self):
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
    
    def extract_measurements(self,
                            masks: np.ndarray,
                            intensity_images: Optional[Dict[str, np.ndarray]] = None,
                            is_3d: bool = False,
                            dna_channel: Optional[str] = None,
                            assign_phases: bool = False) -> pd.DataFrame:
        """
        Extract all enabled measurements from segmented masks
        
        Args:
            masks: Labeled segmentation masks (2D or 3D)
            intensity_images: Dict of channel_name -> intensity image
            is_3d: Whether to compute 3D measurements
            dna_channel: Name of DNA channel for cell cycle analysis
            assign_phases: Whether to assign cell cycle phases
        
        Returns:
            DataFrame with measurements for each nucleus
        """
        self.is_3d = is_3d
        
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
        """Measure basic shape properties"""
        measurements = {}
        
        if self.is_3d:
            # 3D measurements
            measurements['volume'] = float(region.area)  # In 3D, area is volume
            
            # Surface area approximation
            # For now, use bbox surface area as approximation
            bbox = region.bbox
            dx = bbox[3] - bbox[0]
            dy = bbox[4] - bbox[1]
            dz = bbox[5] - bbox[2]
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
        else:
            # 2D measurements
            measurements['area'] = float(region.area)
            measurements['perimeter'] = float(region.perimeter)
            
            # Circularity (4π*area/perimeter²)
            if region.perimeter > 0:
                measurements['circularity'] = float(
                    4 * np.pi * region.area / (region.perimeter ** 2)
                )
            else:
                measurements['circularity'] = 0.0
            
            measurements['equivalent_diameter'] = float(region.equivalent_diameter)
        
        return measurements
    
    def _measure_advanced_morphology(self, region) -> Dict[str, float]:
        """Measure advanced morphological features"""
        measurements = {}
        
        # These work for both 2D and 3D
        measurements['eccentricity'] = float(region.eccentricity)
        measurements['solidity'] = float(region.solidity)
        measurements['major_axis_length'] = float(region.major_axis_length)
        measurements['minor_axis_length'] = float(region.minor_axis_length)
        
        # Aspect ratio
        if region.minor_axis_length > 0:
            measurements['aspect_ratio'] = float(
                region.major_axis_length / region.minor_axis_length
            )
        else:
            measurements['aspect_ratio'] = 0.0
        
        # Orientation (2D only)
        if not self.is_3d:
            measurements['orientation'] = float(region.orientation)
        
        # Extent (area/bbox_area)
        measurements['extent'] = float(region.extent)
        
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
