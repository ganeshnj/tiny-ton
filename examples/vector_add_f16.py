"""Float16 vector addition -- identical kernel code, just half-precision arrays.

Run: PYTHONPATH="build/bindings:python" python3 examples/vector_add_f16.py
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
    a = np.random.randn(N).astype(np.float16)
    b = np.random.randn(N).astype(np.float16)
    c = np.zeros(N, dtype=np.float16)

    grid = (N // 64,)
    vector_add[grid](a, b, c, N)

    print("c[:8]    =", c[:8])
    expected = (a + b).astype(np.float16)
    print("expected =", expected[:8])

    if np.allclose(c, expected, atol=1e-2):
        print("PASS")
    else:
        print("FAIL!")
        diff = np.max(np.abs(c.astype(np.float32) - expected.astype(np.float32)))
        print(f"  max diff = {diff}")


if __name__ == "__main__":
    main()
