"""Vector addition — the north-star example.

This is the target user experience. It will work end-to-end once
the compilation pipeline and runtime are implemented.
"""

import numpy as np
import tiny_ton as tt


@tt.jit
def vector_add(a_ptr, b_ptr, c_ptr, N):
    pid = tt.program_id(0)
    offsets = pid * 64 + tt.arange(0, 64)
    mask = offsets < N
    a = tt.load(a_ptr + offsets, mask=mask)
    b = tt.load(b_ptr + offsets, mask=mask)
    tt.store(c_ptr + offsets, a + b, mask=mask)


def main():
    N = 256
    a = np.arange(N, dtype=np.int32)
    b = np.arange(N, dtype=np.int32)
    c = np.zeros(N, dtype=np.int32)

    grid = (N // 64,)
    vector_add[grid](a, b, c, N)

    print("c =", c[:8], "...")
    expected = a + b
    assert np.array_equal(c, expected), "mismatch!"
    print("PASS")


if __name__ == "__main__":
    main()
