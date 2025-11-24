"""
Example plugin: Texture analysis using GLCM
"""

import numpy as np
from typing import Dict, Any

from plugins.plugin_template import MeasurementPlugin


class TextureAnalysisPlugin(MeasurementPlugin):
    """
    Calculate texture features using Gray Level Co-occurrence Matrix (GLCM)
    """
    
    def get_name(self) -> str:
        return "GLCM Texture Features"
    
    def get_description(self) -> str:
        return "Calculates texture metrics (contrast, homogeneity, energy, correlation) using GLCM"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def measure(self, 
                region: Any, 
                intensity_images: Dict[str, np.ndarray],
                metadata: Dict[str, Any]) -> Dict[str, float]:
        """Calculate GLCM texture features"""
        
        if not intensity_images:
            return {}
        
        try:
            from skimage.feature import graycomatrix, graycoprops
        except ImportError:
            print("Warning: scikit-image not available for texture analysis")
            return {}
        
        results = {}
        
        # Use DNA channel if designated, otherwise first channel
        dna_channel = metadata.get('dna_channel')
        if dna_channel and dna_channel in intensity_images:
            channels_to_process = {dna_channel: intensity_images[dna_channel]}
        else:
            # Process first channel only to avoid too many measurements
            first_channel = list(intensity_images.keys())[0]
            channels_to_process = {first_channel: intensity_images[first_channel]}
        
        for channel_name, image in channels_to_process.items():
            # Extract nucleus region
            min_row, min_col, max_row, max_col = region.bbox
            nucleus_image = image[min_row:max_row, min_col:max_col].copy()
            nucleus_mask = region.image
            
            # Mask out non-nucleus pixels
            nucleus_image[~nucleus_mask] = 0
            
            # Skip if too small
            if nucleus_image.size < 25:  # At least 5x5
                continue
            
            # Normalize to 8-bit for GLCM
            if nucleus_image.max() > 0:
                nucleus_image = ((nucleus_image / nucleus_image.max()) * 255).astype(np.uint8)
            else:
                continue
            
            try:
                # Calculate GLCM at multiple angles
                distances = [1]
                angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
                
                glcm = graycomatrix(
                    nucleus_image,
                    distances=distances,
                    angles=angles,
                    levels=256,
                    symmetric=True,
                    normed=True
                )
                
                # Extract properties and average over angles
                contrast = float(graycoprops(glcm, 'contrast').mean())
                dissimilarity = float(graycoprops(glcm, 'dissimilarity').mean())
                homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
                energy = float(graycoprops(glcm, 'energy').mean())
                correlation = float(graycoprops(glcm, 'correlation').mean())
                
                # Add to results with channel prefix
                prefix = f"{channel_name}_" if len(intensity_images) > 1 else ""
                results[f"{prefix}texture_contrast"] = contrast
                results[f"{prefix}texture_dissimilarity"] = dissimilarity
                results[f"{prefix}texture_homogeneity"] = homogeneity
                results[f"{prefix}texture_energy"] = energy
                results[f"{prefix}texture_correlation"] = correlation
                
            except Exception as e:
                print(f"Error calculating GLCM for {channel_name}: {e}")
                continue
        
        return results


# Register plugin
AVAILABLE_PLUGINS = [
    TextureAnalysisPlugin,
]
