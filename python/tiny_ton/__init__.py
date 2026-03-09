"""tiny-ton: A Triton-inspired GPU kernel compiler."""

from tiny_ton.jit import jit
from tiny_ton._buffer import Buffer

__version__ = "0.1.0"

_BACKEND_AVAILABLE = False
try:
    import _tiny_ton_core  # noqa: F401
    _BACKEND_AVAILABLE = True
except ImportError:
    pass


def program_id(axis: int = 0):
    """Return the program (block) index along the given axis."""
    raise NotImplementedError("program_id is only valid inside a @tt.jit kernel")


def arange(start: int, end: int):
    """Return a range of consecutive values [start, end)."""
    raise NotImplementedError("arange is only valid inside a @tt.jit kernel")


def load(ptr, mask=None):
    """Load values from memory at the given pointer."""
    raise NotImplementedError("load is only valid inside a @tt.jit kernel")


def store(ptr, val, mask=None):
    """Store values to memory at the given pointer."""
    raise NotImplementedError("store is only valid inside a @tt.jit kernel")


__all__ = [
    "jit",
    "Buffer",
    "program_id",
    "arange",
    "load",
    "store",
]
