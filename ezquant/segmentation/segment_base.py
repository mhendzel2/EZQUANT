from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SegmenterBase(ABC):
    @abstractmethod
    def segment(self, image: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError
