# attention (Scaled Dot-Product Attention) — Design & Implementation

## What it does

Single-head scaled dot-product attention computes a context-weighted sum of value vectors, where the weights come from softmax-normalized dot products between a query and stored keys.

In microgpt:

```python
attn_logits = [sum(q[j] * k_t[j] for j in range(dim)) / dim**0.5
               for t in range(seq_len)]
attn_weights = softmax(attn_logits)
out = [sum(attn_weights[t] * v_t[j] for t in range(seq_len))
       for j in range(dim)]
```

With Q/K/V projections and output projection, attention is the core building block of every transformer layer.

## 12-kernel decomposition

```
Step  1:   q = Wq @ x                →  linear_kernel    (1 launch)
Step  2:   k = Wk @ x                →  linear_kernel    (1 launch)
Step  3:   v = Wv @ x                →  linear_kernel    (1 launch)
Step  4:   scores = K @ q             →  matvec_kernel    (1 launch)
Step  5:   scores /= sqrt(n_embd)     →  kern_div_scalar  (1 launch)
Steps 6-10: weights = softmax(scores) →  5 launches
Step 11:   attn_out = V^T @ weights   →  matvec_kernel    (1 launch)
Step 12:   output = Wo @ attn_out     →  linear_kernel    (1 launch)
```

All 12 steps use existing kernels — no new kernels needed.

## Key design decisions

**Single head only.** Multi-head adds slicing/concatenation complexity without exercising new kernels. For Stage 1, single-head (head_dim = n_embd) is sufficient.

**V^T transpose on host.** The weighted sum `out[j] = sum_t(weights[t] * V[t,j])` is the matrix-vector product `V^T @ weights`. We transpose V on the host to `(n_embd, seq_len)` row-major before passing to `matvec_kernel`.

**seq_len <= 64.** Softmax requires the full scores vector in one block. microgpt processes tokens one at a time, so seq_len grows with context. For the test, seq_len = 4.

**sqrt(n_embd) as a 1-element f32 array.** Reuses the scalar broadcast `kern_div_scalar` pattern from softmax.

## Python orchestrator

```python
def attention(x, Wq, Wk, Wv, Wo, K_cache, V_cache, n_embd):
    q = np.zeros(n_embd, dtype=np.float32)
    k = np.zeros(n_embd, dtype=np.float32)
    v = np.zeros(n_embd, dtype=np.float32)

    linear_kernel[(n_embd,)](Wq, x, q, n_embd)
    linear_kernel[(n_embd,)](Wk, x, k, n_embd)
    linear_kernel[(n_embd,)](Wv, x, v, n_embd)

    K_cache.append(k.copy())
    V_cache.append(v.copy())
    K = np.vstack(K_cache)
    V = np.vstack(V_cache)
    seq_len = len(K_cache)

    scores = np.zeros(seq_len, dtype=np.float32)
    matvec_kernel[(seq_len,)](K.flatten(), q, scores, n_embd)

    sqrt_d = np.array([np.sqrt(float(n_embd))], dtype=np.float32)
    scores_scaled = np.zeros(seq_len, dtype=np.float32)
    kern_div_scalar[(1,)](scores, sqrt_d, scores_scaled, seq_len)

    weights = np.zeros(seq_len, dtype=np.float32)
    softmax(scores_scaled, weights, seq_len)

    V_T = np.ascontiguousarray(V.T)
    attn_out = np.zeros(n_embd, dtype=np.float32)
    matvec_kernel[(n_embd,)](V_T.flatten(), weights, attn_out, seq_len)

    output = np.zeros(n_embd, dtype=np.float32)
    linear_kernel[(n_embd,)](Wo, attn_out, output, n_embd)
    return output
```

## How it maps to microgpt

microgpt's transformer block runs attention + residual + MLP:

```python
q = linear(x, wq)
k = linear(x, wk)
v = linear(x, wv)
keys[li].append(k)
values[li].append(v)
# per-head attention scores, softmax, weighted sum
x = linear(x_attn, wo)
x = [a + b for a, b in zip(x, x_residual)]  # residual
```

With tiny-ton, each of these calls maps to the kernel launches above.

## How it flows through the stack

```
Python orchestrator: 12 x kernel[grid](...)
    │
    ▼
JIT (for each kernel): existing builtins
    │  linear (3x Q/K/V + 1x Wo)  → tinyton.load + tinyton.mul + tinyton.reduce_sum
    │  matvec (scores, weighted)   → same as linear
    │  div_scalar                  → tinyton.load + tinyton.div
    │  softmax (5x)                → reduce_max, sub, exp, reduce_sum, div
    │
    ▼
GPU lowering (existing patterns)
    │  all ops already lowered
    │
    ▼
PTX: 12 separate kernel launches via cuLaunchKernel
```

## Files changed

| File | What |
|------|------|
| `docs/09-attention.md` | This design doc |
| `examples/attention_test.py` | Standalone test: 12-launch attention vs NumPy |

No C++ files changed. No new builtins in `jit.py`. All kernels are user-written.

## Testing strategy

`examples/attention_test.py` creates random weight matrices (Wq, Wk, Wv, Wo) of shape (16, 16), processes 4 input vectors to build a KV cache, then runs the full 12-launch attention for the last position. Compares against a NumPy reference that performs the same Q/K/V projections, scaled dot-product attention, and output projection. Tolerance: `atol=1e-3` (12 launches of float accumulation).
