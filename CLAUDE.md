# deepseek-rs — context for Claude

Rust/CUDA port of the ds4 C reference implementation of DeepSeek V4 Flash 284B.

## Workspace layout

```
deepseek-rs/
  Cargo.toml           [workspace, members = engine + chat]
  engine/              deepseek-engine lib crate — all model logic
    src/
      engine.rs        GPU context, kernel loading, weight upload cache
      gguf.rs          Memory-mapped GGUF v3 parser
      model.rs         Fixed architecture constants + TensorMeta/ModelWeights
      session.rs       GPU scratch buffers + decode/prefill/decode_next
      tokenizer.rs     BPE tokenizer (loaded from GGUF vocabulary)
      kernels/         One file per kernel group; all are cuda-oxide #[kernel] fns
        attention.rs   prefill_raw, decode_mixed, indexed_mixed, …
        hc.rs          Hierarchical-compression kernels (sinkhorn, compressor_*)
        indexer.rs     indexer_scores, indexer_topk_*, topk_mask
        matmul.rs      matmul_f16, matmul_q8_0_*, matmul_q4k_*, …
        moe.rs         iq2_xxs_dequant, moe_route_and_reduce, …
        (+ embed, kv_cache, norm, quantize, rope, utils)
    tests/             Integration tests (skip gracefully if no model/GPU)
  chat/                deepseek-chat bin crate
    src/main.rs        rustyline multi-turn chat loop
```

The reference C engine lives at `/srv/data/work/ds4/`.
The model file is `/srv/data/work/ds4/ds4flash.gguf` (~81 GB, memory-mapped).

## Build

```sh
# Must use cargo-oxide, not cargo, to compile the NVPTX kernels.
cargo oxide build                         # debug
cargo oxide build --release               # release

# Per-crate (also works):
cargo oxide build -p deepseek-engine
cargo oxide build -p deepseek-chat
```

The build compiles every `kernels/*.rs` file to NVVM IR (`deepseek_rs.ll`) via the
`cuda-device` proc-macro crate, then `ltoir::load_kernel_module` JIT-links it to a
cubin at first run (cubin is cached next to the .ll so subsequent starts are fast).

## Run

```sh
# Multi-turn interactive chat (rustyline):
./target/release/deepseek-chat [model_path] [max_tokens] [ctx_size]
# defaults: ds4flash.gguf  2048  8192

# Integration tests (require env var pointing at the model file):
DEEPSEEK_RS_MODEL_PATH=/path/to/ds4flash.gguf cargo test -p deepseek-engine
```

## Key architecture facts

**Model** — DeepSeek V4 Flash 284B (43 layers, N_EMBD=4096, N_HEAD=64, N_HEAD_KV=1).

**Hierarchical compression (HC)** — The token embedding is not a plain vector but an
HC tensor: shape `[N_HC=4, N_EMBD]` = 16 384 floats.  Every operation on the residual
stream works in this HC space.  `embed_token_hc` WRITES (not accumulates) `cur_hc`,
so each token's HC state is independent — this is what makes batch prefill correct.

**Layer classification** (`compress_ratio` in session.rs):
- Layers 0–1 (il < 2): `ratio=0` — pure sliding-window attention (N_SWA=128 tokens).
- Even layers ≥ 2: `ratio=4` — compressed KV + indexer top-K attention.
- Odd layers ≥ 3: `ratio=128` — compressed KV + static full attention.

**Attention** — MLA: Q is low-rank (`W_qa` → Q_RANK=1024 → `W_qb` → Q_DIM=32768),
KV is a single compressed vector (N_HEAD_DIM=512) shared across all heads.  RoPE is
applied to the tail 64 dims of Q and KV; the leading 448 dims are NoPE.

**Quantization in use**:
- Embedding: F16
- Attention weights: Q8_0 (most), F16 (ffn_gate_inp, compressor weights, hc_attn_fn, …)
- FFN experts: IQ2_XXS (256 experts × gate/up/down)
- Router: F16 — `ffn_gate_inp` is **F16, not Q8_0**.  Using the Q8_0 matmul path here
  was the main router correctness bug; fixed by calling `matmul_f16` directly on the raw
  `ffn_norm` f32 vector.

## Session API

```rust
let mut sess = Session::new(&engine, ctx_size)?;

// Low-level: explicit position (used by tests).
sess.decode(&engine, token, pos)?;

// High-level: position tracked internally by n_filled.
sess.decode_next(&engine, token)?;   // sequential; always correct
sess.prefill(&engine, &tokens)?;     // batch; fast for the first turn
sess.pos()                           // current fill level
```

`decode_next` and `prefill` are both correct for all turns including multi-turn.
`prefill` processes in chunks of `raw_cap=128`; only the very first chunk of a fresh
session uses fast batch `prefill_raw` for attention — all other chunks use per-token
`decode_mixed` which attends to the full KV history.  The chat binary uses
`decode_next` for per-turn tokens purely for simplicity; `prefill` would also work.

## Compressor and indexer

**Two compressor paths per layer** — `run_attn_compressor_step(engine, il, pos)` builds
`attn_comp_cache` (512-dim rows used in attention).  `run_index_compressor_step(engine,
il, pos)` builds `index_comp_cache` (128-dim rows used only for indexer scoring).  Both
read `self.attn_norm`.

**Call order matters** — Index compressor must be called BEFORE attn compressor.  Only
the attn compressor increments `layer_n_comp[il]`; both use the pre-increment value as
their `comp_row` (they always emit in lockstep at the same positions).

**Indexer dispatch for ratio=4** — When `layer_n_comp[il] > N_INDEXER_TOP_K` (512):
`indexer_score_one_direct` scores each `index_comp_cache` row against indexer Q, then
`indexer_topk_bitonic` selects top-512 indices, then `indexed_mixed` runs attention
with those rows from `attn_comp_cache`.  When `n_comp ≤ 512`, all rows are visible and
`decode_mixed` is used directly (same path as ratio=128).

**`indexer_topk_bitonic` launch** — `n_sort` must be a power of 2 ≥ n_comp;
`shared_mem_bytes = n_sort * 8`.  The kernel writes `*mut u32`; cast
`indexer_topk_buf.cu_deviceptr() as *mut u32` since `indexed_mixed` reads `&[i32]` and
all indices are small enough that the bit patterns are identical.

**Compressor during prefill** — Both compressor methods are called inside the per-token
sequential loop in `prefill` (after `b_attn_norm[t]` → `self.attn_norm` memcpy, before
`decode_attn_out_and_ffn`), so the compressed KV caches are populated even for batch
prefill tokens.

## Important gotchas

**NVPTX alloca bug** — Inside a kernel, writing `&slice[i] as *const T` silently
generates a 1-byte alloca and produces a garbage pointer.  Always use:
```rust
unsafe { slice.as_ptr().add(i) }
```
This bit us in the IQ2_XXS dequant kernel and is easy to reintroduce.

**ffn_gate_inp is F16** — The router weight matrix (`blk.N.ffn_gate_inp`) is stored as
F16 (type_id=1) in the GGUF, not Q8_0 (type_id=8).  Feeding it through the Q8_0 matmul
path produces completely wrong expert selection.  The fix is to call `matmul_f16` and
pass the unquantized `ffn_norm` f32 buffer directly as input.

**GPT-2 byte-to-unicode** — The tokenizer vocabulary encodes raw bytes as Unicode
surrogates.  Mapping: printable ASCII (33–126) → itself; bytes 0–32 → U+0100–U+0120;
bytes 127–160 → U+0121–U+0142; byte 173 → U+0143.  Space (0x20=32) → Ġ (U+0120),
newline (0x0A=10) → Ċ (U+010A).  See `gpt2_token_to_bytes` in tokenizer.rs.

**BPE pre-tokenizer regex** — Uses the cl100k pattern.  The Rust `regex` crate does not
support `\p{L}` Unicode properties or lookaheads.  The `\s+(?!\S)` lookahead piece is
dropped (negligible impact).  For verifying token IDs against Python, use a venv with
the `regex` package (`pip install regex`), not the stdlib `re` module.

**GGUF type_id=1 is F16** — Confirmed empirically and via GGUF spec.  An early version
of the parser had this as Q8_0, causing the router bug above.  There is now a dedicated
test (`type_id_1_is_f16_not_q8_0` in tests/gguf_parse.rs) guarding this.

**Golden test OOM** — Loading the 81 GB model in multiple parallel test functions
exhausts VRAM.  All model-level assertions live in a single `bos_golden` test function
that loads once.  Do not split it into multiple `#[test]` fns.

## Known limitations

**`prefill_raw` batch attention is only used for the very first chunk** — The
`prefill` method processes long prompts in chunks of `raw_cap = N_SWA = 128` tokens.
Only the first chunk of a fresh session (`abs_start == 0`) uses batch `prefill_raw`
(fast, but sees only the current batch).  All other chunks — including multi-turn
continuations and chunk 2+ of long prompts — use per-token `decode_mixed`, which is
correct but sequential for the attention step.  QKV projections remain batched across
all tokens within each chunk.

## Chat template tokens

```
BOS           = 0
EOS           = 1
<｜User｜>     = 128803   (TOK_USER in chat/src/main.rs)
<｜Assistant｜> = 128804  (TOK_ASST)
<think>       = 128821   (TOK_THINK)
</think>      = 128822   (TOK_ETHINK)
```

The model generates a `<think>…</think>` block before its actual response.
The chat binary renders thinking output dimmed (ANSI `\x1b[2m…\x1b[0m`).
