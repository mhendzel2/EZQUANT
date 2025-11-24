"""
Plugin template for custom measurements in Nuclei Segmentation Application

This template shows how to create a custom measurement plugin.
Copy this file, rename it, and implement your custom measurements.
"""

from typing import Dict, Any
import numpy as np
from skimage.measure import regionprops


class MeasurementPlugin:
    """
    Base class for measurement plugins
    All plugins must inherit from this class and implement the required methods
    """
    
    def get_name(self) -> str:
        """
        Return the name of the plugin
        This will be displayed in the plugin manager and measurements table
        
        Returns:
            str: Plugin name
        """
        raise NotImplementedError("Must implement get_name()")
    
    def get_description(self) -> str:
        """
        Return a description of what the plugin measures
        
        Returns:
            str: Plugin description
        """
        raise NotImplementedError("Must implement get_description()")
    
    def get_version(self) -> str:
        """
        Return the plugin version
        
        Returns:
            str: Version string (e.g., "1.0.0")
        """
        return "1.0.0"
    
    def measure(self, 
                region: Any, 
                intensity_images: Dict[str, np.ndarray],
                metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform custom measurements on a single nucleus
        
        Args:
            region: scikit-image regionprops region object
                Access properties like:
                - region.area: nucleus area in pixels
                - region.centroid: (y, x) coordinates of center
                - region.bbox: bounding box coordinates
                - region.coords: array of pixel coordinates
                - region.image: binary mask of the region
            
            intensity_images: Dictionary of intensity images by channel name
                Keys are channel names (e.g., "DAPI", "Channel 1")
                Values are numpy arrays of the full image
                Extract intensity for this nucleus:
                    pixels = intensity_images['DAPI'][region.coords[:, 0], region.coords[:, 1]]
            
            metadata: Dictionary with image metadata
                - 'pixel_size': Physical size of pixels (micrometers)
                - 'bit_depth': Image bit depth (8 or 16)
                - 'is_3d': Boolean indicating if image is 3D
                - 'slice_index': Current slice number (for 2D analysis of 3D images)
                - 'analysis_mode': "2D" or "3D"
        
        Returns:
            Dict[str, float]: Dictionary of measurement names and values
                Example: {"custom_metric": 123.45, "another_metric": 67.89}
        """
        raise NotImplementedError("Must implement measure()")


# ============================================================================
# EXAMPLE PLUGIN IMPLEMENTATIONS
# ============================================================================

class TemplatePlugin(MeasurementPlugin):
    """
    Template plugin - demonstrates the basic structure
    Calculates a simple custom metric: area-to-perimeter ratio
    """
    
    def get_name(self) -> str:
        return "Template Plugin"
    
    def get_description(self) -> str:
        return "Example plugin calculating area-to-perimeter ratio"
    
    def measure(self, 
                region: Any, 
                intensity_images: Dict[str, np.ndarray],
                metadata: Dict[str, Any]) -> Dict[str, float]:
        """Calculate custom metrics"""
        
        # Example 1: Use region properties
        area = region.area
        perimeter = region.perimeter
        
        if perimeter > 0:
            ratio = area / perimeter
        else:
            ratio = 0.0
        
        # Example 2: Custom calculation from intensity
        # Get first available channel
        if intensity_images:
            channel_name = list(intensity_images.keys())[0]
            channel_image = intensity_images[channel_name]
            
            # Extract pixels for this nucleus
            nucleus_pixels = channel_image[region.coords[:, 0], region.coords[:, 1]]
            
            # Calculate custom metric (e.g., coefficient of variation)
            mean_intensity = np.mean(nucleus_pixels)
            std_intensity = np.std(nucleus_pixels)
            cv = (std_intensity / mean_intensity * 100) if mean_intensity > 0 else 0
        else:
            cv = 0.0
        
        # Example 3: Use metadata
        pixel_size = metadata.get('pixel_size', 1.0)
        area_um2 = area * (pixel_size ** 2)
        
        # Return measurements as dictionary
        return {
            "area_perimeter_ratio": ratio,
            "intensity_cv_percent": cv,
            "area_micrometers_squared": area_um2
        }


class IntensityGradientPlugin(MeasurementPlugin):
    """
    Example: Calculate intensity gradient (edge vs center)
    Useful for detecting nuclear periphery staining patterns
    """
    
    def get_name(self) -> str:
        return "Intensity Gradient"
    
    def get_description(self) -> str:
        return "Measures intensity difference between nucleus edge and center"
    
    def measure(self, 
                region: Any, 
                intensity_images: Dict[str, np.ndarray],
                metadata: Dict[str, Any]) -> Dict[str, float]:
        
        if not intensity_images:
            return {}
        
        results = {}
        
        for channel_name, channel_image in intensity_images.items():
            # Get nucleus mask
            mask = region.image  # Binary mask in region's bounding box
            
            # Create edge and center masks
            from scipy.ndimage import binary_erosion
            eroded = binary_erosion(mask, iterations=2)
            edge_mask = mask & ~eroded
            center_mask = eroded
            
            if np.any(edge_mask) and np.any(center_mask):
                # Get bounding box
                min_row, min_col, max_row, max_col = region.bbox
                
                # Extract intensity
                region_image = channel_image[min_row:max_row, min_col:max_col]
                
                edge_intensity = np.mean(region_image[edge_mask])
                center_intensity = np.mean(region_image[center_mask])
                
                gradient = edge_intensity - center_intensity
                
                results[f"{channel_name}_edge_mean"] = edge_intensity
                results[f"{channel_name}_center_mean"] = center_intensity
                results[f"{channel_name}_gradient"] = gradient
        
        return results


# ============================================================================
# PLUGIN REGISTRATION
# ============================================================================

# Add your plugin class here to make it available to the application
AVAILABLE_PLUGINS = [
    TemplatePlugin,
    IntensityGradientPlugin,
]
