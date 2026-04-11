# linear (Matrix-Vector Multiply) — Design & Implementation

## What it does

`linear(x, W)` computes `y = W @ x` — a matrix-vector multiply where each output element is the dot product of one row of W with the input vector x.

In microgpt:

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

This is the most common operation in the model — every attention projection (`wq`, `wk`, `wv`, `wo`), every MLP layer (`fc1`, `fc2`), and the final LM head use it.

## Kernel

`linear` reuses `matvec_kernel` from [dot-matvec.md](dot-matvec.md) — no new kernel needed:

```python
@tt.jit
def linear_kernel(W_ptr, x_ptr, y_ptr, in_features):
    pid = tt.program_id(0)
    tid = tt.arange(0, 64)
    mask = tid < in_features
    w = tt.load(W_ptr + pid * in_features + tid, mask=mask)
    x = tt.load(x_ptr + tid, mask=mask)
    dot = tt.reduce_sum(w * x)
    tt.store(y_ptr + pid, dot)
```

Each block computes one output element. Launched with `grid = (out_features,)`.

## Python helper

```python
def linear(W_flat, x, y, out_features, in_features):
    linear_kernel[(out_features,)](W_flat, x, y, in_features)
```

## How it maps to microgpt

microgpt uses linear layers throughout:

```python
q = linear(x, state_dict[f'layers.{l}.attn.wq'])   # query projection
k = linear(x, state_dict[f'layers.{l}.attn.wk'])   # key projection
v = linear(x, state_dict[f'layers.{l}.attn.wv'])   # value projection
h = linear(x, state_dict[f'layers.{l}.mlp.fc1'])   # MLP up-projection
y = linear(h, state_dict[f'layers.{l}.mlp.fc2'])   # MLP down-projection
logits = linear(x, state_dict['lm_head'])           # final output
```

With tiny-ton, each call is a single `linear_kernel` launch.

## Chaining linear layers

A key test is verifying that outputs of one kernel can feed as inputs to the next — the pattern used in every transformer block:

```python
# x -> W1 -> relu -> W2 -> y  (simplified MLP)
linear(W1_flat, x, h, hidden_dim, in_features)
kern_relu[grid](h, h_relu, hidden_dim)
linear(W2_flat, h_relu, y, out_features, hidden_dim)
```

## Constraint: in_features <= 64

Same as dot/matvec — a single block loads the full input row. microgpt uses `n_embd = 16`.

## Files changed

| File | What |
|------|------|
| `docs/07-linear.md` | This design doc |
| `examples/linear_test.py` | Test at microgpt scale + chained layers |

No C++ files changed. No new builtins. Kernel is identical to `matvec_kernel`.

## Testing strategy

`examples/linear_test.py` tests:

1. **Single linear**: `W (8, 16)`, `x (16,)` — like an attention projection, compare against `W @ x`
2. **Chained linears**: `x -> W1 -> relu -> W2 -> y` — verifies multi-kernel pipeline works end-to-end
3. Tolerance: `atol=1e-4` (float accumulation in reduce_sum)
