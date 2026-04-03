"""Gather (embedding lookup) -- verified against NumPy indexing.

Run: PYTHONPATH="build/bindings:python" python3 examples/gather_test.py
"""

import numpy as np
import tiny_ton as tt


@tt.jit
def gather_kernel(table_ptr, index, dst_ptr, n_embd):
    tid = tt.arange(0, 64)
    mask = tid < n_embd
    val = tt.load(table_ptr + index * n_embd + tid, mask=mask)
    tt.store(dst_ptr + tid, val, mask=mask)


def main():
    np.random.seed(42)

    rows, cols = 8, 16
    table = np.random.randn(rows, cols).astype(np.float32)
    table_flat = table.flatten()

    all_ok = True

    for idx in [0, 3, 7]:
        expected = table[idx]
        dst = np.zeros(cols, dtype=np.float32)
        gather_kernel[(1,)](table_flat.copy(), idx, dst, cols)

        ok = np.allclose(dst, expected, atol=0.0)
        print(f"gather row[{idx}]: {'PASS' if ok else 'FAIL'}")
        if not ok:
            for i in range(min(8, cols)):
                print(f"  [{i}] got={dst[i]:.6f}  expected={expected[i]:.6f}")
            all_ok = False

    assert all_ok, "gather test failed"
    print("All gather tests passed.")


if __name__ == "__main__":
    main()
