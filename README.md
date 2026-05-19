# deepseek-rs

Rust/CUDA inference engine for **DeepSeek V4 Flash**, ported from [ds4](https://github.com/antirez/ds4).

Built on [cuda-oxide](https://github.com/NVlabs/cuda-oxide) — CUDA kernels are written in Rust and compiled to PTX via NVPTX, with no C++, no CUDA SDK headers, and no cuDNN.

Why? Because all great software should be re-written in Rust. Is this safe rust? No! Is the cuda compiler less than a month old and still a little wonky? Yes, but that's okay, it's written in rust! Did I write this all myself? No! Me and by buddy Claude did this "together.

## Requirements

- **GPU:** A big honking nvidia GPU
- **Model:** `ds4flash.gguf` (~81 GB) — the DeepSeek V4 Flash GGUF (get this from Ds4)
- **Rust:** nightly-2026-04-03 (see `rust-toolchain.toml`)
- **cuda-oxide:** - install nightly

## Build

```sh
# Compiles Rust CUDA kernels to NVVM IR and links them
cargo oxide build

# Release build
cargo oxide build --release
```

`cargo oxide` is the cuda-oxide build driver. It compiles every file in
`engine/src/kernels/` to NVVM IR (`deepseek_rs.ll`), then `ltoir::load_kernel_module`
JIT-links it to a cubin on first run. The cubin is cached next to the `.ll` file so
subsequent starts skip JIT compilation.

Do **not** use plain `cargo build` — it will compile the Rust code but skip the
NVPTX kernel compilation, and the binary will fail at runtime when it can't find
the kernel artifact.

## Run

```sh
# Interactive multi-turn chat
./target/release/deepseek-chat [model_path] [max_tokens] [ctx_size]

# Defaults:
#   model_path  ds4flash.gguf
#   max_tokens  2048
#   ctx_size    8192
```

The chat loop:

1. Loads the model and tokenizer from the GGUF file
2. Seeds the KV cache with BOS
3. Reads user input via rustyline (with history)
4. Encodes the turn as `<｜User｜>…<｜Assistant｜>` and feeds it via `decode_next`
5. Generates until EOS, rendering `<think>…</think>` blocks dimmed

## Test

```sh
# Unit + integration tests (no model required)
cargo test -p deepseek-engine

# Full golden tests (requires model)
DEEPSEEK_RS_MODEL_PATH=/path/to/ds4flash.gguf \
  cargo test --test golden_bos --release
```

The golden test suite (`engine/tests/golden_bos.rs`) loads the model once and runs:

1. Sequential `decode_next` for BOS — checks top-5 against ds4 reference
2. `prefill([BOS])` — checks top-5 matches the decode path
3. 200-token `prefill` (two chunks) — checks pinned top-1 to guard the compressor
   and chunked-attention paths
4. `prefill` then `decode_next` — verifies position tracking

## Known limitations

- `prefill_raw` batch attention is only used for the first 128-token chunk of a fresh
  session. Subsequent chunks use per-token `decode_mixed` — QKV projections remain
  batched, but attention is sequential.
- Only greedy decoding (no temperature / top-p / top-k sampling).
- Only the IQ2_XXS + Q2_K expert quantization variant is implemented; other
  quantization formats for expert weights are not supported.
