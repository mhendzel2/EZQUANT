# Allen Cell & Structure Segmenter Integration

This project now integrates the [Allen Cell & Structure Segmenter](https://www.allencell.org/segmenter.html) for 3D intracellular structure segmentation.

## Overview

The integration supports two backends:
1.  **Classic**: Uses `aicssegmentation` classic image processing workflows (filters, thresholding).
2.  **ML**: Uses `segmenter_model_zoo` and `aicsmlsegment` for deep learning-based segmentation.

## Usage

### Python API

You can use the `SegmentationEngine` to run Allen Segmenter workflows.

```python
from core.segmentation import SegmentationEngine
import numpy as np

# Initialize engine
engine = SegmentationEngine(gpu_available=True)

# Load your image (Z, Y, X)
image = ... 

# Run Classic Segmentation
masks, info = engine.segment_allen(
    image,
    mode='classic',
    structure_id='LAMP1'
)

# Run ML Segmentation
masks, info = engine.segment_allen(
    image,
    mode='ml',
    workflow_id='DNA_MEM_instance_basic'
)
```

### Configuration

The `segment_allen` method accepts a `config` dictionary to override parameters.

```python
config = {
    "parameters": {
        "scaling_param": 1.0,
        "cutoff": 0.5
    }
}
```

## Requirements

*   **Classic**: `aicssegmentation`
*   **ML**: `segmenter_model_zoo`, `aicsmlsegment`, `torch` (with CUDA support for GPU acceleration)

## Limitations

*   The ML models are trained on specific cell lines and imaging conditions. Performance may vary on other datasets.
*   ML segmentation requires a CUDA-capable GPU.

## References

*   [Allen Cell & Structure Segmenter](https://www.allencell.org/segmenter.html)
*   [aicssegmentation Documentation](https://allencell.github.io/aics-segmentation/)
*   [segmenter_model_zoo](https://github.com/AllenCell/segmenter_model_zoo)
