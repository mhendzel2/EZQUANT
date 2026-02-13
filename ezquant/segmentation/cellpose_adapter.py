from __future__ import annotations


class CellposeAdapter:
    """Optional dependency adapter placeholder."""

    def __init__(self) -> None:
        try:
            import cellpose  # noqa: F401
        except Exception as exc:
            raise RuntimeError("cellpose is not installed") from exc
