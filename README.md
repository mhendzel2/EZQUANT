# Nuclei Segmentation & Analysis Application

A desktop application for undergraduate students to perform nuclei segmentation and morphometric analysis using advanced deep learning models (Cellpose 4, SAM).

## Features

- **Advanced Segmentation**: Cellpose 4 and Segment Anything Model (SAM) integration
- **Multi-dimensional Support**: 2D/3D multichannel TIFF files (8-bit, 16-bit), plus MetaMorph ND (`.nd`) acquisition spec files that reference TIFF series
- **Quality Control**: Cell cycle-aware DNA intensity analysis with outlier detection
- **Manual Correction**: Split, merge, delete, and add nuclei tools
- **Comprehensive Measurements**: Morphometric and intensity analysis (toggleable)
- **Cell Cycle Analysis**: Optional phase assignment (G1/S/G2M) with manual boundary adjustment
- **Interactive Visualization**: Plotly-based charts with bidirectional selection
- **Project Management**: Save/load projects with data aggregation across images
- **Plugin System**: Extensible architecture for custom measurements
- **Batch Processing**: Process multiple images with consistent parameters

## System Requirements

### Minimum
- Windows 10 or newer
- 8 GB RAM
- 10 GB free disk space
- CPU: Multi-core processor (Intel i5 or AMD equivalent)

### Recommended
- Windows 11
- 16 GB RAM
- 20 GB free disk space
- GPU: NVIDIA GPU with 8 GB+ VRAM
- CUDA Toolkit 11.8 or newer

## Installation

### From Installer (Recommended)
1. Download the installer: `NucleiSegmentationApp_Setup.exe`
2. Run the installer and follow the prompts
3. Launch from Start Menu or desktop shortcut

### From Source
1. Install Python 3.9 or newer
2. Clone or download this repository
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python main.py
   ```

## Quick Start

1. **Create a Project**: File → New Project
2. **Import Image**: File → Import TIFF, select your image file
3. **Select DNA Channel**: Choose the channel containing DNA/nuclear staining
4. **Run Segmentation**: Choose Cellpose or SAM model, adjust parameters, click "Run Segmentation"
5. **Quality Control**: Review flagged nuclei, adjust cell cycle boundaries if needed
6. **Manual Corrections** (optional): Use split/merge/delete tools to refine segmentation
7. **Analyze**: Choose 2D or 3D workflow, select measurements, run analysis
8. **Visualize**: Explore interactive plots in the Visualization tab
9. **Export**: File → Export → Measurements (CSV/Excel)

## Plugin Development

The application supports custom measurement plugins. See `plugins/plugin_template.py` and `docs/PLUGIN_DEVELOPMENT.md` for details.

## License

[To be determined]

## Citation

If you use this software in your research, please cite:
- Cellpose: Stringer et al., Nature Methods (2021)
- SAM: Kirillov et al., ICCV (2023)

## Support

For issues, questions, or feature requests, please contact [contact information].
