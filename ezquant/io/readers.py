from __future__ import annotations

import hashlib
from pathlib import Path

import tifffile

from ezquant.io.metadata import parse_basic_metadata


def detect_reader(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".tif", ".tiff", ".ome.tif", ".ome.tiff"}:
        return "tifffile"
    return "unknown"


def read_image_with_metadata(path: str):
    with tifffile.TiffFile(path) as tf:
        arr = tf.asarray()
        md = {}
        ome = tf.ome_metadata
        if ome:
            md["ome_metadata_present"] = True
        meta = parse_basic_metadata(arr, md)
    return arr, meta


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
