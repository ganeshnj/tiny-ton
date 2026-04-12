"""tiny-ton: A GPU kernel compiler."""

from tiny_ton.jit import jit

__version__ = "0.1.0"


def program_id(axis: int = 0):
    """Return the program (block) index along the given axis."""
    raise NotImplementedError("program_id is only valid inside a @tt.jit kernel")


def arange(start: int, end: int):
    """Return a range of consecutive values [start, end)."""
    raise NotImplementedError("arange is only valid inside a @tt.jit kernel")


def load(ptr, mask=None, other=None):
    """Load values from memory at the given pointer."""
    raise NotImplementedError("load is only valid inside a @tt.jit kernel")


def store(ptr, val, mask=None):
    """Store values to memory at the given pointer."""
    raise NotImplementedError("store is only valid inside a @tt.jit kernel")


def relu(x):
    """Element-wise ReLU: max(x, 0)."""
    raise NotImplementedError("relu is only valid inside a @tt.jit kernel")


def reduce_sum(x):
    """Sum reduction across all threads in the block."""
    raise NotImplementedError("reduce_sum is only valid inside a @tt.jit kernel")


def reduce_max(x):
    """Max reduction across all threads in the block."""
    raise NotImplementedError("reduce_max is only valid inside a @tt.jit kernel")


def sync():
    """Block-wide barrier — all threads must reach this point before any proceed."""
    raise NotImplementedError("sync is only valid inside a @tt.jit kernel")


def shared_store(idx, val):
    """Write a value into block-local shared memory at the given index."""
    raise NotImplementedError("shared_store is only valid inside a @tt.jit kernel")


def shared_load(idx):
    """Read a value from block-local shared memory at the given index."""
    raise NotImplementedError("shared_load is only valid inside a @tt.jit kernel")


class constexpr:
    """Marks a kernel parameter as a compile-time constant.

    When a parameter is annotated with ``tt.constexpr``, its value is baked
    into the compiled kernel at JIT time rather than being passed as a runtime
    argument.  Different values produce different compiled kernels (separate
    cache entries).

    The primary use is to set the block size via ``tt.arange``::

        @tt.jit
        def kernel(src, dst, N, BLOCK: tt.constexpr):
            tid  = tt.arange(0, BLOCK)
            mask = tid < N
            ...

        kernel[(1,)](src, dst, N, 16)   # 16 threads per block
        kernel[(1,)](src, dst, N, 64)   # 64 threads per block (cached separately)
    """


__all__ = [
    "jit",
    "constexpr",
    "program_id",
    "arange",
    "load",
    "store",
    "relu",
    "reduce_sum",
    "reduce_max",
    "sync",
    "shared_store",
    "shared_load",
]
