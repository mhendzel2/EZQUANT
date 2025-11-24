# Plugin Development Guide

## Overview

The Nuclei Segmentation & Analysis application supports custom measurement plugins that allow you to extend the built-in measurement capabilities. Plugins are Python scripts that implement specific measurements on segmented nuclei.

## Getting Started

### Plugin Location

Place your plugin files in the `plugins/` directory of the application. The application will automatically discover and load all valid plugins at startup.

### Basic Plugin Structure

Every plugin must:
1. Import the `MeasurementPlugin` base class
2. Create a class that inherits from `MeasurementPlugin`
3. Implement the required methods
4. Add the class to the `AVAILABLE_PLUGINS` list

## Required Methods

### `get_name()` → str

Returns the display name of your plugin. This will appear in:
- Plugin Manager dialog
- Measurement settings
- Exported data column headers

```python
def get_name(self) -> str:
    return "My Custom Plugin"
```

### `get_description()` → str

Returns a brief description of what your plugin measures.

```python
def get_description(self) -> str:
    return "Calculates custom texture features using GLCM"
```

### `get_version()` → str (optional)

Returns the version string of your plugin. Defaults to "1.0.0" if not implemented.

```python
def get_version(self) -> str:
    return "2.1.0"
```

### `measure(region, intensity_images, metadata)` → Dict[str, float]

The core measurement method called for each segmented nucleus.

**Parameters:**

- **`region`**: scikit-image `RegionProperties` object
  - `region.area`: Number of pixels in nucleus
  - `region.perimeter`: Perimeter length in pixels
  - `region.centroid`: (row, col) coordinates of center
  - `region.bbox`: (min_row, min_col, max_row, max_col)
  - `region.coords`: Nx2 array of (row, col) coordinates
  - `region.image`: Binary mask of nucleus (cropped to bounding box)
  - `region.intensity_image`: Intensity values (if provided)
  - [Full list of properties](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)

- **`intensity_images`**: Dict[str, np.ndarray]
  - Keys: Channel names (e.g., "DAPI", "GFP", "Channel 1")
  - Values: Full intensity image arrays
  - Extract nucleus pixels: `pixels = image[region.coords[:, 0], region.coords[:, 1]]`

- **`metadata`**: Dict[str, Any]
  - `'pixel_size'`: Physical pixel size in micrometers (float or None)
  - `'bit_depth'`: Image bit depth (8 or 16)
  - `'is_3d'`: Boolean indicating 3D image
  - `'slice_index'`: Current slice number (for 2D analysis)
  - `'analysis_mode'`: "2D" or "3D"
  - `'dna_channel'`: Name of the DNA channel (if designated)

**Returns:**

Dictionary mapping measurement names (str) to values (float). All values must be numeric.

```python
return {
    "custom_metric_1": 123.45,
    "custom_metric_2": 67.89,
}
```

## Example Plugins

### Example 1: Simple Calculation

```python
class AspectRatioPlugin(MeasurementPlugin):
    def get_name(self) -> str:
        return "Aspect Ratio"
    
    def get_description(self) -> str:
        return "Calculates nucleus aspect ratio (major/minor axis)"
    
    def measure(self, region, intensity_images, metadata):
        major = region.axis_major_length
        minor = region.axis_minor_length
        
        if minor > 0:
            aspect_ratio = major / minor
        else:
            aspect_ratio = 0.0
        
        return {"aspect_ratio": aspect_ratio}
```

### Example 2: Intensity-Based Measurement

```python
class IntensityRangePlugin(MeasurementPlugin):
    def get_name(self) -> str:
        return "Intensity Range"
    
    def get_description(self) -> str:
        return "Calculates intensity range (max - min) for each channel"
    
    def measure(self, region, intensity_images, metadata):
        results = {}
        
        for channel_name, image in intensity_images.items():
            # Extract nucleus pixels
            pixels = image[region.coords[:, 0], region.coords[:, 1]]
            
            # Calculate range
            intensity_range = np.max(pixels) - np.min(pixels)
            
            results[f"{channel_name}_range"] = float(intensity_range)
        
        return results
```

### Example 3: Texture Analysis

```python
from skimage.feature import graycomatrix, graycoprops

class TexturePlugin(MeasurementPlugin):
    def get_name(self) -> str:
        return "GLCM Texture"
    
    def get_description(self) -> str:
        return "Gray Level Co-occurrence Matrix texture features"
    
    def measure(self, region, intensity_images, metadata):
        if not intensity_images:
            return {}
        
        results = {}
        
        # Use first channel or DNA channel
        dna_channel = metadata.get('dna_channel')
        if dna_channel and dna_channel in intensity_images:
            channel_name = dna_channel
        else:
            channel_name = list(intensity_images.keys())[0]
        
        image = intensity_images[channel_name]
        
        # Extract nucleus region
        min_row, min_col, max_row, max_col = region.bbox
        nucleus_image = image[min_row:max_row, min_col:max_col]
        nucleus_mask = region.image
        
        # Mask out non-nucleus pixels
        nucleus_image = nucleus_image.copy()
        nucleus_image[~nucleus_mask] = 0
        
        # Convert to 8-bit for GLCM
        if nucleus_image.max() > 0:
            nucleus_image = ((nucleus_image / nucleus_image.max()) * 255).astype(np.uint8)
        
        # Calculate GLCM
        glcm = graycomatrix(
            nucleus_image, 
            distances=[1], 
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )
        
        # Extract texture properties
        results['contrast'] = float(graycoprops(glcm, 'contrast').mean())
        results['dissimilarity'] = float(graycoprops(glcm, 'dissimilarity').mean())
        results['homogeneity'] = float(graycoprops(glcm, 'homogeneity').mean())
        results['energy'] = float(graycoprops(glcm, 'energy').mean())
        results['correlation'] = float(graycoprops(glcm, 'correlation').mean())
        
        return results
```

### Example 4: Radial Profile

```python
from scipy.ndimage import distance_transform_edt

class RadialProfilePlugin(MeasurementPlugin):
    def get_name(self) -> str:
        return "Radial Intensity Profile"
    
    def get_description(self) -> str:
        return "Measures intensity as function of distance from centroid"
    
    def measure(self, region, intensity_images, metadata):
        if not intensity_images:
            return {}
        
        results = {}
        
        # Get nucleus mask and centroid
        min_row, min_col, max_row, max_col = region.bbox
        mask = region.image
        
        # Calculate distance transform (distance from edge)
        distances = distance_transform_edt(mask)
        
        for channel_name, image in intensity_images.items():
            # Extract region
            region_image = image[min_row:max_row, min_col:max_col]
            
            # Calculate mean intensity in concentric zones
            max_dist = distances.max()
            n_bins = 5
            
            for i in range(n_bins):
                # Define ring
                inner = (i / n_bins) * max_dist
                outer = ((i + 1) / n_bins) * max_dist
                
                ring_mask = (distances >= inner) & (distances < outer)
                
                if np.any(ring_mask):
                    mean_intensity = np.mean(region_image[ring_mask])
                    results[f"{channel_name}_ring_{i+1}"] = float(mean_intensity)
        
        return results
```

## Best Practices

### 1. Error Handling

Always handle edge cases and potential errors:

```python
def measure(self, region, intensity_images, metadata):
    try:
        # Your measurement code
        result = calculate_something(region)
        return {"my_metric": float(result)}
    except Exception as e:
        print(f"Error in {self.get_name()}: {e}")
        return {}  # Return empty dict on error
```

### 2. Type Conversion

Always convert results to Python float:

```python
# Good
return {"metric": float(numpy_value)}

# Bad (may cause serialization errors)
return {"metric": numpy_value}
```

### 3. Channel-Specific Measurements

Prefix channel-specific measurements with channel name:

```python
for channel_name, image in intensity_images.items():
    value = calculate(image, region)
    results[f"{channel_name}_my_metric"] = value
```

### 4. Physical Units

Use metadata to convert pixel measurements to physical units:

```python
pixel_size = metadata.get('pixel_size', 1.0)
area_pixels = region.area
area_um2 = area_pixels * (pixel_size ** 2)
```

### 5. Performance

For expensive calculations:
- Cache results when possible
- Limit computation to necessary pixels
- Consider downsampling large regions

```python
# Extract only nucleus pixels
coords = region.coords
pixels = image[coords[:, 0], coords[:, 1]]
# Much faster than processing entire bounding box
```

## Debugging Plugins

### Enable Debug Output

Add print statements to debug:

```python
def measure(self, region, intensity_images, metadata):
    print(f"Processing nucleus {region.label}")
    print(f"Available channels: {list(intensity_images.keys())}")
    print(f"Metadata: {metadata}")
    # ... your code
```

### Test Standalone

Test your plugin independently:

```python
if __name__ == "__main__":
    from skimage import data, measure
    import numpy as np
    
    # Load test image
    image = data.coins()
    
    # Create fake segmentation
    from skimage.filters import threshold_otsu
    thresh = threshold_otsu(image)
    binary = image > thresh
    labeled = measure.label(binary)
    
    # Test plugin
    plugin = YourPlugin()
    regions = measure.regionprops(labeled, intensity_image=image)
    
    for region in regions[:3]:  # Test first 3
        result = plugin.measure(
            region,
            {'test_channel': image},
            {'pixel_size': 1.0, 'bit_depth': 8}
        )
        print(f"Nucleus {region.label}: {result}")
```

## Common Issues

### Issue: Plugin not appearing

**Solution:** Ensure your plugin class is added to `AVAILABLE_PLUGINS`:

```python
AVAILABLE_PLUGINS = [
    MyPlugin,  # Add here
]
```

### Issue: Import errors

**Solution:** Install required packages or use try/except:

```python
try:
    from special_library import special_function
    SPECIAL_AVAILABLE = True
except ImportError:
    SPECIAL_AVAILABLE = False
    print("special_library not available")

def measure(self, region, intensity_images, metadata):
    if not SPECIAL_AVAILABLE:
        return {}
    # ... use special_function
```

### Issue: Measurements not in export

**Solution:** Check that returned dictionary uses string keys and float values:

```python
# Correct
return {"metric": float(value)}

# Wrong
return {0: value}  # Key must be string
return {"metric": value}  # Value should be float
```

## Plugin Distribution

To share your plugin with others:

1. Create a single `.py` file with your plugin
2. Document the plugin purpose and requirements
3. List any additional dependencies
4. Provide example usage

Users can simply copy the file to their `plugins/` directory.

## API Reference

### Available Libraries

These libraries are pre-installed with the application:

- `numpy`: Array operations
- `scipy`: Scientific computing
- `scikit-image`: Image processing
- `pandas`: Data manipulation (for advanced plugins)

### Accessing Built-in Measurements

You can access standard regionprops measurements:

```python
# Geometric
area = region.area
perimeter = region.perimeter
eccentricity = region.eccentricity
solidity = region.solidity

# Axes
major_axis = region.axis_major_length
minor_axis = region.axis_minor_length
orientation = region.orientation

# Bounding
bbox = region.bbox
centroid = region.centroid
```

## Support

For questions or issues with plugin development:
- Check the example plugins in `plugins/examples/`
- Review this documentation
- Contact application support

## Version History

- **1.0.0** (2025-11-22): Initial plugin system release
