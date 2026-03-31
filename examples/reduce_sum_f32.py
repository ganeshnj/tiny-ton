"""Float32 reduce_sum -- per-block sum reduction, verified against NumPy.

Run: PYTHONPATH="build/bindings:python" python3 examples/reduce_sum_f32.py
"""

import numpy as np
import tiny_ton as tt


@tt.jit
def kern_reduce_sum(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    total = tt.reduce_sum(x)
    tt.store(dst + pid, total)


@tt.jit
def kern_reduce_max(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    mx = tt.reduce_max(x)
    tt.store(dst + pid, mx)


def main():
    np.random.seed(42)
    BLOCK = 64
    N = 256
    grid = (N // BLOCK,)
    x = np.random.randn(N).astype(np.float32)

    # --- reduce_sum ---
    out_sum = np.zeros(N // BLOCK, dtype=np.float32)
    kern_reduce_sum[grid](x.copy(), out_sum, N)

    print("reduce_sum:")
    all_ok = True
    for i in range(N // BLOCK):
        expected = np.sum(x[i * BLOCK:(i + 1) * BLOCK])
        ok = np.allclose(out_sum[i], expected, atol=1e-4)
        print(f"  block {i}: got {out_sum[i]:.6f}, expected {expected:.6f}"
              f" -- {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok

    # --- reduce_max ---
    out_max = np.zeros(N // BLOCK, dtype=np.float32)
    kern_reduce_max[grid](x.copy(), out_max, N)

    print("reduce_max:")
    for i in range(N // BLOCK):
        expected = np.max(x[i * BLOCK:(i + 1) * BLOCK])
        ok = np.allclose(out_max[i], expected, atol=1e-4)
        print(f"  block {i}: got {out_max[i]:.6f}, expected {expected:.6f}"
              f" -- {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok

    print(f"\n{'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
