# VSI and LIF File Support

## Overview
EZQUANT now supports importing proprietary microscopy file formats:
- **VSI** - Olympus cellSens/OlyVIA format
- **LIF** - Leica Image File format

## Installation

The VSI and LIF support requires the `aicsimageio` library. To install:

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Install aicsimageio with all format support
pip install "aicsimageio[all]"

# OR install minimal version (may have limited format support)
pip install aicsimageio
```

## Usage

### Importing VSI Files
1. Go to **File > Import Image**
2. Select file type filter: **Olympus VSI Files (*.vsi)**
3. Choose your .vsi file
4. If the file contains multiple scenes, a dialog will appear to select which scene to import
5. The selected scene will be loaded into EZQUANT

### Importing LIF Files
1. Go to **File > Import Image**
2. Select file type filter: **Leica LIF Files (*.lif)**
3. Choose your .lif file
4. If the file contains multiple images/series, a dialog will appear to select which one to import
5. The selected image will be loaded into EZQUANT

## Features

### Multi-Scene Support
Both VSI and LIF files can contain multiple scenes or image series. EZQUANT:
- Automatically detects all available scenes
- Shows scene selection dialog if multiple scenes exist
- Displays metadata including:
  - Scene name
  - Image dimensions (shape)
  - Number of channels
  - Number of Z-slices
  - Number of timepoints

### Metadata Extraction
The loader extracts comprehensive metadata including:
- Physical pixel size (micrometers)
- Channel names
- Z-stack information
- Bit depth
- Image dimensions

### Format Support via aicsimageio
The `aicsimageio` library provides support for many additional formats:
- CZI (Zeiss)
- ND2 (Nikon)
- OIB/OIF (Olympus FluoView)
- DV (DeltaVision)
- And many more OME-compatible formats

## Technical Details

### File Structure
- **core/bioformats_io.py** - BioformatsLoader class with VSI/LIF support
- **core/image_io.py** - Integrated VSI/LIF loading methods into TIFFLoader
- **gui/main_window.py** - Updated import dialog and scene selection UI

### Data Format
Loaded images are normalized to:
- 3D images: `(Z, C, Y, X)` - Z-slices, Channels, Height, Width
- 2D images: `(C, Y, X)` - Channels, Height, Width

### Dependencies
- `aicsimageio>=4.14.0` - Main library for bioformats
- `dask` - Lazy loading of large images
- `fsspec` - File system abstraction
- `xarray` - Multi-dimensional labeled arrays
- `zarr` - Chunked array storage

## Troubleshooting

### Import Error: "aicsimageio could not be resolved"
**Solution**: Install aicsimageio:
```bash
pip install "aicsimageio[all]"
```

### Error: "No scenes found in VSI/LIF file"
**Cause**: File may be corrupted or not a valid VSI/LIF file
**Solution**: Verify file integrity and format

### Slow Loading
**Cause**: Large multi-scene files or high-resolution images
**Solution**: This is normal. aicsimageio uses lazy loading to manage memory efficiently.

### Missing Channel Names
**Cause**: Metadata may not contain channel names
**Solution**: EZQUANT automatically generates generic channel names (Channel 1, Channel 2, etc.)

## Limitations

1. **Timepoints**: Currently only loads the first timepoint from time-series data
2. **Scene Selection**: Must select scenes one at a time (cannot batch import all scenes yet)
3. **Memory**: Very large images may require significant RAM

## Future Enhancements

Planned improvements:
- Batch import of all scenes from multi-scene files
- Timepoint selection for time-series data
- Support for additional proprietary formats (CZI, ND2)
- Lazy loading UI indicator for large files

## Example Usage

```python
# Programmatic usage
from core.image_io import TIFFLoader

# Load VSI file
image, metadata = TIFFLoader.load_vsi("sample.vsi", scene=0, timepoint=0)

# Get VSI file info without loading
info = TIFFLoader.get_vsi_info("sample.vsi")
print(f"Found {info['n_scenes']} scenes")

# Load LIF file
image, metadata = TIFFLoader.load_lif("sample.lif", scene=0)

# Get LIF file info
info = TIFFLoader.get_lif_info("sample.lif")
for scene in info['scenes']:
    print(f"Scene: {scene['name']}, Channels: {scene['n_channels']}")
```

## References

- [aicsimageio Documentation](https://allencellmodeling.github.io/aicsimageio/)
- [Olympus VSI Format](https://www.olympus-lifescience.com/en/software/cellsens/)
- [Leica LIF Format](https://www.leica-microsystems.com/products/microscope-software/p/leica-las-x-ls/)
