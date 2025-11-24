"""
Example plugin: Radial intensity profile analysis
"""

import numpy as np
from typing import Dict, Any

from plugins.plugin_template import MeasurementPlugin


class RadialProfilePlugin(MeasurementPlugin):
    """
    Calculate radial intensity distribution from nucleus center
    """
    
    def get_name(self) -> str:
        return "Radial Intensity Profile"
    
    def get_description(self) -> str:
        return "Measures intensity distribution from center to periphery (5 concentric zones)"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def measure(self, 
                region: Any, 
                intensity_images: Dict[str, np.ndarray],
                metadata: Dict[str, Any]) -> Dict[str, float]:
        """Calculate radial intensity profile"""
        
        if not intensity_images:
            return {}
        
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            print("Warning: scipy not available for radial profile")
            return {}
        
        results = {}
        
        # Get nucleus mask and calculate distance from edge
        min_row, min_col, max_row, max_col = region.bbox
        mask = region.image
        
        # Calculate distance from edge (distance transform)
        distances = distance_transform_edt(mask)
        max_dist = distances.max()
        
        if max_dist < 2:  # Too small
            return {}
        
        # Define number of zones
        n_zones = 5
        
        for channel_name, image in intensity_images.items():
            # Extract region
            region_image = image[min_row:max_row, min_col:max_col]
            
            # Calculate mean intensity in each zone
            for i in range(n_zones):
                # Define zone boundaries
                inner = (i / n_zones) * max_dist
                outer = ((i + 1) / n_zones) * max_dist
                
                # Create zone mask
                zone_mask = (distances >= inner) & (distances < outer)
                
                if np.any(zone_mask):
                    mean_intensity = float(np.mean(region_image[zone_mask]))
                    results[f"{channel_name}_zone_{i+1}_mean"] = mean_intensity
            
            # Calculate center-to-edge gradient
            if n_zones >= 2:
                zone1_key = f"{channel_name}_zone_1_mean"
                zone5_key = f"{channel_name}_zone_{n_zones}_mean"
                
                if zone1_key in results and zone5_key in results:
                    gradient = results[zone1_key] - results[zone5_key]
                    results[f"{channel_name}_center_edge_gradient"] = gradient
        
        return results


# Register plugin
AVAILABLE_PLUGINS = [
    RadialProfilePlugin,
]
