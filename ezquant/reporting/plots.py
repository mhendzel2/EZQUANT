from __future__ import annotations

from pathlib import Path


def write_placeholder_plot(output_dir: str, name: str) -> str:
    path = Path(output_dir) / f"{name}.txt"
    path.write_text("plot placeholder", encoding="utf-8")
    return str(path)
