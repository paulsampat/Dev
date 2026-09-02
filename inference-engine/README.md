# inference-engine

A from-scratch ML model inference engine: load a trained model, optimize its
computation graph, and run forward passes efficiently.

It implements the full pipeline an engine like ONNX Runtime or llama.cpp is
built from — model format, graph IR, operator kernels, graph optimization,
memory planning, execution, and request batching — at a size you can read in
one sitting. Numpy provides the BLAS; everything above it is here.

```python
from engine import InferenceSession

with InferenceSession("models/mlp.iem") as session:
    outputs = session.run({"x": batch})
```

## Quickstart

```bash
pip install -r requirements.txt

python -m tools.export mlp models/mlp.iem      # write a demo model
python -m pytest tests/ -q                     # 133 tests
python -m bench.benchmark --model mlp          # measure it
```

## How it is put together

Each stage maps to one module, in the order data flows through them.

| Stage | Module | What it does |
|---|---|---|
| 1. Scope | — | float32/int8 CPU inference, static graphs, dynamic batch dimension |
| 2. Model format | `engine/format.py`, `engine/loader.py` | `.iem` files: JSON header + weight blob, mmap'd zero-copy |
| 3. Tensors & kernels | `engine/tensor.py`, `engine/ops.py` | Operator registry: compute + shape inference + dtype inference |
| 4. Execution | `engine/executor.py` | Topological plan, liveness analysis, the run loop |
| 5. Memory | `engine/allocator.py` | Size-bucketed arena that recycles intermediate buffers |
| 6. Backends | `engine/backends/` | The kernel interface a hardware target implements |
| 7. Graph optimization | `engine/optimizer.py` | Constant folding, fusion, DCE, int8 quantization |
| 8. Batching | `engine/scheduler.py` | Dynamic batching for concurrent serving |
| 9. API | `engine/api.py` | `InferenceSession` |
| 10. Validation | `tests/`, `bench/` | Reference comparison and benchmarking |

### 2. Model format

A `.iem` file is `magic | header_len | JSON header | weight blob`. The header
holds the graph structure and, per weight, a `(dtype, shape, offset, nbytes)`
entry into the blob. Loading mmaps the file and makes each weight a numpy view
into the mapping, so weights cost no heap and startup does not scale with model
size. This is the same reason safetensors and GGUF are laid out this way.

### 3. Operators

Every operator registers three functions: how to compute it, how to infer its
output shape, and how to infer its output dtype. Shape inference is not
optional bookkeeping — the memory planner needs the output shape *before* the
kernel runs in order to hand it a buffer.

Implemented: `MatMul`, `Gemm`, `QGemmInt8`, `Add`, `Mul`, `Relu`, `Gelu`,
`Sigmoid`, `Tanh`, `Scale`, `Softmax`, `LayerNorm`, `Reshape`, `Transpose`,
`Gather`, `Cast` — enough for MLPs and transformer blocks.

### 4-5. Planning and memory

`ExecutionPlan.compile()` runs once per model: topological sort, kernel
resolution, and a liveness analysis that records each value's last use. At run
time, a value's buffer returns to the arena the instant it dies, so peak memory
tracks the widest point of the graph rather than the sum of every intermediate.

The subtle part is aliasing. `Reshape` and `Transpose` return *views* of their
input, so releasing that input's buffer would let a later node overwrite memory
a live view still points at. Before returning a buffer to the pool, the run
loop checks whether any live value is a view into it and transfers ownership
instead of freeing. `tests/test_executor.py::test_a_view_is_not_corrupted_when_its_source_buffer_dies`
covers this — with the naive "always release" version, that graph's output is
wrong by 5.09 rather than 0.

### 7. Graph optimization

Passes run at load time, each returning a new graph plus a count of what it
changed:

- **`constant_fold`** — evaluates subgraphs that depend only on weights.
- **`fuse_matmul_bias`** — `MatMul → Add(bias)` becomes one `Gemm`.
- **`fuse_activation`** — folds a trailing `Relu`/`Gelu`/… into that `Gemm`.
- **`eliminate_dead_nodes`** — drops nodes and weights nothing reads.
- **`quantize_int8`** (opt-in) — int8 weights with per-output-column scales.

Fusion only fires when the intermediate value has exactly one consumer and is
not a graph output; otherwise it would delete a result somebody still needs.

On the demo MLP: 8 nodes → 3 fused `Gemm`s. On the transformer block: 26 → 23.

## Measured results

Ryzen-class CPU, numpy 2.4, single thread. `max err` is against the reference
implementations in `tools/models.py`, which compute the same maths directly.

**MLP** (784→256→256→10), batch 64:

| configuration | p50 (ms) | rows/s | max err |
|---|---|---|---|
| unoptimized, no arena | 0.203 | 314,728 | 0 |
| fused, no arena | 0.197 | 325,354 | 0 |
| fused + arena | 0.269 | 237,828 | 0 |
| fused + arena + int8 | 19.512 | 3,280 | 3.6e-02 |

**Transformer block** (d=128, 4 heads, seq 32), batch 8:

| configuration | p50 (ms) | rows/s | max err |
|---|---|---|---|
| unoptimized, no arena | 1.830 | 4,370 | 1.3e-06 |
| fused + arena | 2.199 | 3,639 | 1.3e-06 |

**Dynamic batching**, 200 concurrent single-row MLP requests:
12,271 → 20,509 req/s (**1.67x**), mean batch size 28.6.

Two of these results are worth stating plainly rather than burying:

**The arena costs latency here, and buys memory.** It is ~20-30% slower, because
in Python the allocator's own bookkeeping (bisect, dict updates, constructing
views) costs more than the `malloc` it avoids. What it buys is peak memory: on
the transformer block it holds **0.52 MB against 18.35 MB of total requests**, a
35x reduction, at 98% pool hit rate. That trade — memory ceiling for a little
latency — is usually the right one when the constraint is how many models fit
on a box, and it inverts in a compiled engine where the bookkeeping is nearly
free. It defaults to on; pass `use_arena=False` to turn it off.

**int8 quantization shrinks weights 4x and makes inference slower.** Weights go
from 1.08 MB to 0.27 MB with 1.8% relative error and unchanged argmax
predictions. But numpy has no int8 GEMM, so `QGemmInt8` upcasts to int32 and
does an int32 matmul that never reaches BLAS. Quantization pays off only with
kernels built for it (VNNI, dp4a, or an ARM dot-product path); without those it
is a memory optimization that costs throughput. Use it when weight size is the
binding constraint, not for speed.

## Testing

133 tests, ~0.5s. The load-bearing ones compare engine output against the
independent numpy references in `tools/models.py`; a mismatch points at the
engine rather than at a shared bug. Others pin down the parts most likely to
break silently: buffer aliasing, arena-on vs arena-off agreement, repeated runs
over recycled buffers, batch-size independence (row *i* of a batch equals that
row run alone), and optimized-vs-unoptimized equivalence.

```bash
python -m pytest tests/ -q
```

## Not implemented

Deliberate scope cuts, each a real extension point:

- **KV caching** — required for autoregressive LLM decoding; needs mutable
  state across calls, which this stateless plan does not model.
- **Convolutions** — `Conv2d` via im2col + GEMM would slot into `engine/ops.py`
  without touching anything else.
- **A GPU backend** — `engine/backends/base.py` is the interface to implement;
  `to_device`/`to_host` are already threaded through the executor.
- **Real int8 kernels** — see above.
- **Operator autotuning and multi-threaded execution** — numpy's BLAS threads
  the matmuls; the graph itself runs single-threaded and in order.
