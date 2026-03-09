"""End-to-end test: vector addition kernel."""

import pytest
import numpy as np


def test_vector_add_smoke():
    """Verify the @jit decorator and launch syntax don't crash on import."""
    import tiny_ton as tt

    @tt.jit
    def vector_add(a_ptr, b_ptr, c_ptr, N):
        pid = tt.program_id(0)
        offsets = pid * 64 + tt.arange(0, 64)
        mask = offsets < N
        a = tt.load(a_ptr + offsets, mask=mask)
        b = tt.load(b_ptr + offsets, mask=mask)
        tt.store(c_ptr + offsets, a + b, mask=mask)

    assert vector_add is not None
    assert hasattr(vector_add, "_compile")


def test_buffer_wraps_numpy():
    from tiny_ton import Buffer

    arr = np.array([1, 2, 3, 4], dtype=np.int32)
    buf = Buffer(arr)
    assert buf.shape == (4,)
    assert buf.dtype == np.int32
    assert buf.data_ptr() != 0
    np.testing.assert_array_equal(buf.numpy(), arr)


@pytest.mark.skip(reason="compilation not yet implemented")
def test_vector_add_execution():
    """Full end-to-end test — enable once compilation pipeline works."""
    import tiny_ton as tt

    @tt.jit
    def vector_add(a_ptr, b_ptr, c_ptr, N):
        pid = tt.program_id(0)
        offsets = pid * 64 + tt.arange(0, 64)
        mask = offsets < N
        a = tt.load(a_ptr + offsets, mask=mask)
        b = tt.load(b_ptr + offsets, mask=mask)
        tt.store(c_ptr + offsets, a + b, mask=mask)

    a = np.array([1, 2, 3, 4], dtype=np.int32)
    b = np.array([10, 20, 30, 40], dtype=np.int32)
    c = np.zeros(4, dtype=np.int32)

    vector_add[(1,)](a, b, c, len(a))
    np.testing.assert_array_equal(c, [11, 22, 33, 44])
