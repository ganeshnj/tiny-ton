"""GPU memory buffer abstraction."""

from __future__ import annotations

import numpy as np


class Buffer:
    """Wraps a numpy array as a GPU-addressable buffer."""

    def __init__(self, data: np.ndarray):
        self._data = np.ascontiguousarray(data)

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def data_ptr(self) -> int:
        """Return the raw pointer address of the underlying data."""
        return self._data.ctypes.data

    def numpy(self) -> np.ndarray:
        """Return the underlying numpy array."""
        return self._data

    def __repr__(self) -> str:
        return f"Buffer(shape={self.shape}, dtype={self.dtype})"
