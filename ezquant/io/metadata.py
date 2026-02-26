from __future__ import annotations

import numpy as np


def parse_basic_metadata(arr: np.ndarray, tif_metadata: dict | None = None) -> dict:
    tif_metadata = tif_metadata or {}
    pixel_size_um = tif_metadata.get("pixel_size_um")
    time_interval_s = tif_metadata.get("time_interval_s")
    channel_names = tif_metadata.get("channel_names", ["ch0"])
    bit_depth = int(arr.dtype.itemsize * 8)
    dimensions = tuple(int(v) for v in arr.shape)
    return {
        "pixel_size_um": pixel_size_um,
        "time_interval_s": time_interval_s,
        "channel_names": channel_names,
        "bit_depth": bit_depth,
        "dimensions": dimensions,
        "dtype": str(arr.dtype),
    }
