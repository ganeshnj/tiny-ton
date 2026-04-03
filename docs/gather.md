# gather (Embedding Lookup) — Design & Implementation

## What it does

`gather` loads a single row from a 2D table stored in flat memory. In microgpt this is the embedding lookup: `tok_emb = state_dict['wte'][token_id]` — given a vocabulary table of shape `(vocab_size, n_embd)` and an integer index, extract the corresponding embedding vector.

```python
@tt.jit
def gather_kernel(table_ptr, index, dst_ptr, n_embd):
    tid = tt.arange(0, 64)
    mask = tid < n_embd
    val = tt.load(table_ptr + index * n_embd + tid, mask=mask)
    tt.store(dst_ptr + tid, val, mask=mask)
```

Each thread loads one element of the selected row. The block size (64) must be >= `n_embd`. For microgpt's default `n_embd=16`, this is well within a single block.

## Why no new MLIR op

Gather is purely address arithmetic on top of existing `tt.load`:

```
row_start = index * n_embd
element_addr = table_ptr + row_start + tid
value = tt.load(element_addr, mask=tid < n_embd)
```

The multiplication, addition, load, and store are all existing ops. No new entry in `TinyTonOps.td`, no builder method, no GPU lowering, no simulator opcode.

## How it maps to microgpt

microgpt uses two embedding tables:

```python
tok_emb = state_dict['wte'][token_id]   # token embedding
pos_emb = state_dict['wpe'][pos_id]     # position embedding
x = [t + p for t, p in zip(tok_emb, pos_emb)]
```

With tiny-ton, each becomes a `gather_kernel` launch, then a `vector_add` launch:

```python
gather_kernel[(1,)](wte_flat, token_id, tok_buf, n_embd)
gather_kernel[(1,)](wpe_flat, pos_id,   pos_buf, n_embd)
vector_add[(1,)](tok_buf, pos_buf, x_buf, n_embd)
```

The tables are flattened to 1D f32 arrays before upload.

## How it flows through the stack

```
Python: table_ptr + index * n_embd + tid
    │
    ▼
JIT AST visitor
    │  emit_mul(index, n_embd) → tinyton.mul
    │  emit_add(table_ptr, ...) → tinyton.add (→ GEP in GPU lowering)
    │  emit_add(..., tid)       → tinyton.add (→ GEP)
    │  emit_load(addr, mask)    → tinyton.load
    │
    ▼
GPU lowering (TinyTonToGPU.cpp, existing patterns)
    │  tinyton.add with pointer → llvm.gep
    │  tinyton.load → llvm.load
    │
    ▼
PTX: ld.global.f32 with computed address
```

## Files changed

| File | What |
|------|------|
| `docs/gather.md` | This design doc |
| `examples/gather_test.py` | Standalone test vs `table[index]` in NumPy |

No C++ files changed. No new builtins in `jit.py` — gather is a user-written kernel.

## Testing strategy

`examples/gather_test.py` creates a random f32 table `(rows, cols)`, picks several indices, runs the gather kernel for each, and compares the output row against `table[index]` in NumPy. Tolerance: exact match (gather involves no arithmetic approximation).
