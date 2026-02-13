from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParsedMetadata:
    pixel_size_um: float | None
    time_interval_s: float | None
    channel_names: list[str]
    bit_depth: int
    dimensions: tuple[int, ...]
