from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects


def _to_2d(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim >= 3:
        return image[0]
    raise ValueError("Unsupported image dimensions")


def segment_nuclei(image: np.ndarray, expected_diameter_px: int = 20) -> np.ndarray:
    img2d = _to_2d(image).astype(float)
    thr = threshold_otsu(img2d)
    mask = img2d > thr
    min_size = max(int(np.pi * (expected_diameter_px / 4) ** 2), 8)
    mask = remove_small_objects(mask, min_size=min_size)
    labels, _ = ndi.label(mask)
    return labels.astype(np.int32)
