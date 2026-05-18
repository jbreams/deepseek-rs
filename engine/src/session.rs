/// Session: allocates GPU scratch buffers and runs single-token decode.
///
/// Full implementation: sliding-window attention for ratio=0 layers,
/// compressed-KV attention for ratio=128 layers, and indexed (top-K) attention
/// for ratio=4 layers.  Compressor state is populated during both `decode` and
/// `prefill`.
use anyhow::{Context, Result};
use cuda_core::{DeviceBuffer, LaunchConfig};

use crate::engine::Engine;
use crate::kernels::quantize::{IQ2_SIGNS, IQ2_XXS_GRID};
use crate::model::{
    N_EMBD, N_EXPERT, N_EXPERT_USED, N_FF_EXP, N_HC, N_HC_SINKHORN_ITER, N_HEAD, N_HEAD_DIM,
    N_HEAD_KV, N_INDEXER_HEAD, N_INDEXER_HEAD_DIM, N_INDEXER_TOP_K, N_LAYER, N_LORA_O, N_LORA_Q,
    N_OUT_GROUP, N_ROT, N_SWA, N_VALUE_DIM, N_VOCAB, QK_K, TensorMeta,
};

// ─── Model hyperparameters ───────────────────────────────────────────────────
const RMS_EPS: f32 = 1e-6;
const HC_EPS: f32 = 1e-6;
const SWIGLU_CLAMP: f32 = 10.0;
const ROPE_FREQ_BASE: f32 = 10000.0;
const ROPE_SCALE_FACTOR: f32 = 16.0;
const ROPE_YARN_BETA_FAST: f32 = 32.0;
const ROPE_YARN_BETA_SLOW: f32 = 1.0;
const COMPRESS_ROPE_FREQ_BASE: f32 = 160000.0;
const ROPE_ORIG_CTX: u32 = 65536;

// ─── Scalar dimension constants ──────────────────────────────────────────────
const HC_DIM: usize = N_HC * N_EMBD; // 16384
const MIX_HC: usize = 2 * N_HC + N_HC * N_HC; // 24
const Q_RANK: usize = N_LORA_Q; // 1024
const Q_DIM: usize = N_HEAD * N_HEAD_DIM; // 32768
const LOW_DIM: usize = N_OUT_GROUP * N_LORA_O; // 8192
const GROUP_DIM: usize = N_HEAD_DIM * (N_HEAD / N_OUT_GROUP); // 4096
const INDEXER_Q_DIM: usize = N_INDEXER_HEAD * N_INDEXER_HEAD_DIM; // 8192

// ─── Layer properties ────────────────────────────────────────────────────────
fn compress_ratio(il: usize) -> u32 {
    if il < 2 {
        0
    } else if il % 2 == 0 {
        4
    } else {
        128
    }
}
fn rope_freq_base(il: usize) -> f32 {
    if compress_ratio(il) != 0 {
        COMPRESS_ROPE_FREQ_BASE
    } else {
        ROPE_FREQ_BASE
    }
}
fn rope_freq_scale(il: usize) -> f32 {
    if compress_ratio(il) == 0 {
        1.0
    } else {
        1.0 / ROPE_SCALE_FACTOR
    }
}
fn rope_ext_factor(il: usize) -> f32 {
    if compress_ratio(il) != 0 { 1.0 } else { 0.0 }
}
fn rope_attn_factor(il: usize, freq_scale: f32) -> f32 {
    if compress_ratio(il) != 0 {
        1.0 / (1.0 + 0.1 * (1.0 / freq_scale).log2() * std::f32::consts::LN_2)
    } else {
        1.0
    }
}

// ─── Session struct ───────────────────────────────────────────────────────────

pub struct Session {
    // ── Single-token scratch buffers ──────────────────────────────────────
    cur_hc: DeviceBuffer<f32>,          // [HC_DIM=16384]
    flat_hc: DeviceBuffer<f32>,         // [HC_DIM]
    hc_mix: DeviceBuffer<f32>,          // [MIX_HC=24]
    hc_split: DeviceBuffer<f32>,        // [MIX_HC]
    attn_cur: DeviceBuffer<f32>,        // [N_EMBD]
    attn_norm: DeviceBuffer<f32>,       // [N_EMBD]
    qr: DeviceBuffer<f32>,              // [Q_RANK=1024]
    qr_norm: DeviceBuffer<f32>,         // [Q_RANK]
    q: DeviceBuffer<f32>,               // [Q_DIM=32768]
    kv_raw: DeviceBuffer<f32>,          // [N_HEAD_DIM=512]
    kv: DeviceBuffer<f32>,              // [N_HEAD_DIM]
    comp_kv_cur: DeviceBuffer<f32>,     // [2*N_HEAD_DIM=1024]
    comp_sc_cur: DeviceBuffer<f32>,     // [2*N_HEAD_DIM]
    heads: DeviceBuffer<f32>,           // [N_HEAD * N_VALUE_DIM = 32768]
    attn_out: DeviceBuffer<f32>,        // [N_EMBD]
    after_attn_hc: DeviceBuffer<f32>,   // [HC_DIM]
    ffn_cur: DeviceBuffer<f32>,         // [N_EMBD]
    ffn_norm: DeviceBuffer<f32>,        // [N_EMBD]
    shared_gate: DeviceBuffer<f32>,     // [N_FF_EXP=2048]
    shared_up: DeviceBuffer<f32>,       // [N_FF_EXP]
    shared_mid: DeviceBuffer<f32>,      // [N_FF_EXP]
    shared_out: DeviceBuffer<f32>,      // [N_EMBD]
    router_logits: DeviceBuffer<f32>,   // [N_EXPERT=256]
    router_probs: DeviceBuffer<f32>,    // [N_EXPERT]
    router_selected: DeviceBuffer<i32>, // [N_EXPERT_USED=6]
    router_weights: DeviceBuffer<f32>,  // [N_EXPERT_USED]
    routed_gate: DeviceBuffer<f32>,     // [N_EXPERT_USED * N_FF_EXP = 12288]
    routed_up: DeviceBuffer<f32>,       // [N_EXPERT_USED * N_FF_EXP]
    routed_mid: DeviceBuffer<f32>,      // [N_EXPERT_USED * N_FF_EXP]
    routed_out: DeviceBuffer<f32>,      // [N_EXPERT_USED * N_EMBD = 24576]
    after_ffn_hc: DeviceBuffer<f32>,    // [HC_DIM]
    output_weights: DeviceBuffer<f32>,  // [N_HC=4]
    output_embd: DeviceBuffer<f32>,     // [N_EMBD]
    output_norm: DeviceBuffer<f32>,     // [N_EMBD]
    logits_buf: DeviceBuffer<f32>,      // [N_VOCAB=129280]
    // ── Q8_0 quantisation scratch (max in_dim = Q_DIM = 32768) ────────────
    xq_i8: DeviceBuffer<i8>,   // [max_in_dim = 32768]
    xscale: DeviceBuffer<f32>, // [max_in_dim / 32 = 1024]
    // ── Attention output low-rank scratch ─────────────────────────────────
    attn_low: DeviceBuffer<f32>, // [LOW_DIM=8192]
    // ── Compressor emit scratch (one emitted row, pre- and post-norm) ────
    comp_emit_pre: DeviceBuffer<f32>, // [N_HEAD_DIM] — pool reduction output
    comp_emit_post: DeviceBuffer<f32>, // [N_HEAD_DIM] — after norm+rope+fp8
    // ── Indexer compressor emit scratch ───────────────────────────────────────
    index_comp_emit_pre: DeviceBuffer<f32>, // [N_INDEXER_HEAD_DIM=128]
    index_comp_emit_post: DeviceBuffer<f32>, // [N_INDEXER_HEAD_DIM]
    // ── Indexer scratch buffers ───────────────────────────────────────────────
    indexer_q: DeviceBuffer<f32>,        // [INDEXER_Q_DIM=8192]
    indexer_weights: DeviceBuffer<f32>,  // [N_INDEXER_HEAD=64]
    indexer_scores: DeviceBuffer<f32>,   // [comp_cap]
    indexer_topk_buf: DeviceBuffer<i32>, // [N_INDEXER_TOP_K=512]
    // ── MoE Q8_K scratch ─────────────────────────────────────────────────
    moe_xq: DeviceBuffer<u8>, // Q8_K of ffn_norm: xq_blocks * Q8K_BLOCK_BYTES
    moe_midq: DeviceBuffer<u8>, // Q8_K of routed_mid: N_EXPERT_USED * midq_blocks * Q8K_BLOCK_BYTES
    // ── IQ2_XXS lookup tables (uploaded once) ─────────────────────────────
    iq2_grid_gpu: DeviceBuffer<u64>,
    iq2_signs_gpu: DeviceBuffer<u8>,
    // ── Per-layer KV caches ───────────────────────────────────────────────
    raw_cache: Vec<DeviceBuffer<f32>>,
    attn_comp_cache: Vec<DeviceBuffer<f32>>,
    attn_state_kv: Vec<DeviceBuffer<f32>>,
    attn_state_score: Vec<DeviceBuffer<f32>>,
    // ── Per-layer indexer compressor state (ratio=4 only) ────────────────────
    index_comp_cache: Vec<DeviceBuffer<f32>>,
    index_state_kv: Vec<DeviceBuffer<f32>>,
    index_state_score: Vec<DeviceBuffer<f32>>,
    // ── Per-layer counters ────────────────────────────────────────────────
    layer_n_comp: Vec<u32>,
    // ── Config ────────────────────────────────────────────────────────────
    ctx_size: usize,
    raw_cap: usize,
    comp_cap: usize,
    // ── Position tracking ─────────────────────────────────────────────────
    /// Number of tokens whose KV state has been committed to the cache.
    /// Updated by both `prefill` and `decode_next`.
    n_filled: usize,
}

// ─── Helper: safe reinterpret of DeviceBuffer<u8> as &DeviceBuffer<T> ────────
// We never actually move the bytes — the buffer stays as-is.  The pointer
// arithmetic inside the kernel is responsible for correct alignment.
fn buf_as_f32(b: &DeviceBuffer<u8>) -> &DeviceBuffer<f32> {
    // SAFETY: DeviceBuffer<T> is a transparent newtype over a raw CUDA pointer;
    // the size/alignment of the pointed-to type is only meaningful on the device,
    // and the kernel is written to receive raw pointers anyway.
    unsafe { &*(b as *const DeviceBuffer<u8> as *const DeviceBuffer<f32>) }
}
fn buf_as_u16(b: &DeviceBuffer<u8>) -> &DeviceBuffer<u16> {
    unsafe { &*(b as *const DeviceBuffer<u8> as *const DeviceBuffer<u16>) }
}
fn buf_as_i32(b: &DeviceBuffer<u8>) -> &DeviceBuffer<i32> {
    unsafe { &*(b as *const DeviceBuffer<u8> as *const DeviceBuffer<i32>) }
}

// ─── Helper: fetch a pre-uploaded GPU tensor ─────────────────────────────────
fn upload<'e>(
    engine: &'e Engine,
    meta: &Option<TensorMeta>,
    name: &str,
) -> Result<&'e DeviceBuffer<u8>> {
    let m = meta
        .as_ref()
        .with_context(|| format!("missing weight: {name}"))?;
    engine
        .gpu
        .get(&m.name)
        .with_context(|| format!("GPU buffer not found for '{name}' (GGUF name '{}')", m.name))
}

// ─── Session::new ─────────────────────────────────────────────────────────────
impl Session {
    pub fn new(engine: &Engine, ctx_size: usize) -> Result<Self> {
        let s = &engine.stream;
        macro_rules! zbuf {
            ($ty:ty, $n:expr) => {
                DeviceBuffer::<$ty>::zeroed(s, $n).context(concat!(
                    "zeroed ",
                    stringify!($ty),
                    "[",
                    stringify!($n),
                    "]"
                ))?
            };
        }

        let raw_cap = N_SWA;
        let comp_cap = ctx_size / 4 + 2;

        // Scalar scratch
        let cur_hc = zbuf!(f32, HC_DIM);
        let flat_hc = zbuf!(f32, HC_DIM);
        let hc_mix = zbuf!(f32, MIX_HC);
        let hc_split = zbuf!(f32, MIX_HC);
        let attn_cur = zbuf!(f32, N_EMBD);
        let attn_norm = zbuf!(f32, N_EMBD);
        let qr = zbuf!(f32, Q_RANK);
        let qr_norm = zbuf!(f32, Q_RANK);
        let q = zbuf!(f32, Q_DIM);
        let kv_raw = zbuf!(f32, N_HEAD_DIM);
        let kv = zbuf!(f32, N_HEAD_DIM);
        let comp_kv_cur = zbuf!(f32, 2 * N_HEAD_DIM);
        let comp_sc_cur = zbuf!(f32, 2 * N_HEAD_DIM);
        let heads = zbuf!(f32, N_HEAD * N_VALUE_DIM);
        let attn_out = zbuf!(f32, N_EMBD);
        let after_attn_hc = zbuf!(f32, HC_DIM);
        let ffn_cur = zbuf!(f32, N_EMBD);
        let ffn_norm = zbuf!(f32, N_EMBD);
        let shared_gate = zbuf!(f32, N_FF_EXP);
        let shared_up = zbuf!(f32, N_FF_EXP);
        let shared_mid = zbuf!(f32, N_FF_EXP);
        let shared_out = zbuf!(f32, N_EMBD);
        let router_logits = zbuf!(f32, N_EXPERT);
        let router_probs = zbuf!(f32, N_EXPERT);
        let router_selected = zbuf!(i32, N_EXPERT_USED);
        let router_weights = zbuf!(f32, N_EXPERT_USED);
        let routed_gate = zbuf!(f32, N_EXPERT_USED * N_FF_EXP);
        let routed_up = zbuf!(f32, N_EXPERT_USED * N_FF_EXP);
        let routed_mid = zbuf!(f32, N_EXPERT_USED * N_FF_EXP);
        let routed_out = zbuf!(f32, N_EXPERT_USED * N_EMBD);
        let after_ffn_hc = zbuf!(f32, HC_DIM);
        let output_weights = zbuf!(f32, N_HC);
        let output_embd = zbuf!(f32, N_EMBD);
        let output_norm = zbuf!(f32, N_EMBD);
        let logits_buf = zbuf!(f32, N_VOCAB);

        // Q8_0 quantisation scratch for max in_dim = Q_DIM = 32768
        let max_in_dim = Q_DIM;
        let xq_i8 = zbuf!(i8, max_in_dim);
        let xscale = zbuf!(f32, max_in_dim / 32);
        let attn_low = zbuf!(f32, LOW_DIM);

        // Compressor emit scratch
        let comp_emit_pre = zbuf!(f32, N_HEAD_DIM);
        let comp_emit_post = zbuf!(f32, N_HEAD_DIM);

        // Indexer compressor emit scratch
        let index_comp_emit_pre = zbuf!(f32, N_INDEXER_HEAD_DIM);
        let index_comp_emit_post = zbuf!(f32, N_INDEXER_HEAD_DIM);
        let indexer_q = zbuf!(f32, INDEXER_Q_DIM);
        let indexer_weights = zbuf!(f32, N_INDEXER_HEAD);
        let indexer_scores = zbuf!(f32, comp_cap);
        let indexer_topk_buf = zbuf!(i32, N_INDEXER_TOP_K);

        // MoE Q8_K scratch buffers
        const Q8K_BLOCK_BYTES: usize = 292; // f32 d + i8[256] qs + i16[16] bsums
        let xq_blocks = N_EMBD / QK_K; // 16 — Q8_K blocks per ffn_norm row
        let midq_blocks = N_FF_EXP / QK_K; // 8  — Q8_K blocks per expert mid row
        let moe_xq =
            DeviceBuffer::<u8>::zeroed(s, xq_blocks * Q8K_BLOCK_BYTES).context("moe_xq alloc")?;
        let moe_midq = DeviceBuffer::<u8>::zeroed(s, N_EXPERT_USED * midq_blocks * Q8K_BLOCK_BYTES)
            .context("moe_midq alloc")?;

        // IQ2_XXS tables
        let iq2_grid_gpu = DeviceBuffer::from_host(s, &IQ2_XXS_GRID).context("iq2_grid upload")?;
        let iq2_signs_gpu = DeviceBuffer::from_host(s, &IQ2_SIGNS).context("iq2_signs upload")?;

        // Per-layer caches
        let mut raw_cache = Vec::with_capacity(N_LAYER);
        let mut attn_comp_cache = Vec::with_capacity(N_LAYER);
        let mut attn_state_kv = Vec::with_capacity(N_LAYER);
        let mut attn_state_score = Vec::with_capacity(N_LAYER);
        let mut index_comp_cache = Vec::with_capacity(N_LAYER);
        let mut index_state_kv = Vec::with_capacity(N_LAYER);
        let mut index_state_score = Vec::with_capacity(N_LAYER);

        for il in 0..N_LAYER {
            // raw cache: circular ring of raw_cap KV rows
            raw_cache.push(zbuf!(f32, raw_cap * N_HEAD_DIM));

            let ratio = compress_ratio(il);
            if ratio != 0 {
                attn_comp_cache.push(zbuf!(f32, comp_cap * N_HEAD_DIM));
                // State ring: coff * N_HEAD_DIM * coff * ratio
                let coff: usize = if ratio == 4 { 2 } else { 1 };
                let state_rows = coff * ratio as usize; // 8 for r=4, 128 for r=128
                let state_width = coff * N_HEAD_DIM;
                attn_state_kv.push(zbuf!(f32, state_rows * state_width));
                attn_state_score.push(zbuf!(f32, state_rows * state_width));
            } else {
                // Placeholder 1-elem buffers for SWA-only layers
                attn_comp_cache.push(zbuf!(f32, 1));
                attn_state_kv.push(zbuf!(f32, 1));
                attn_state_score.push(zbuf!(f32, 1));
            }

            // Indexer compressor state (ratio=4 only)
            if ratio == 4 {
                index_comp_cache.push(zbuf!(f32, comp_cap * N_INDEXER_HEAD_DIM));
                let state_rows = 2 * ratio as usize; // coff=2 for ratio=4 → 8 rows
                let state_width = 2 * N_INDEXER_HEAD_DIM; // 256
                index_state_kv.push(zbuf!(f32, state_rows * state_width));
                index_state_score.push(zbuf!(f32, state_rows * state_width));
            } else {
                index_comp_cache.push(zbuf!(f32, 1));
                index_state_kv.push(zbuf!(f32, 1));
                index_state_score.push(zbuf!(f32, 1));
            }
        }

        let layer_n_comp = vec![0u32; N_LAYER];

        Ok(Self {
            cur_hc,
            flat_hc,
            hc_mix,
            hc_split,
            attn_cur,
            attn_norm,
            qr,
            qr_norm,
            q,
            kv_raw,
            kv,
            comp_kv_cur,
            comp_sc_cur,
            heads,
            attn_out,
            after_attn_hc,
            ffn_cur,
            ffn_norm,
            shared_gate,
            shared_up,
            shared_mid,
            shared_out,
            router_logits,
            router_probs,
            router_selected,
            router_weights,
            routed_gate,
            routed_up,
            routed_mid,
            routed_out,
            after_ffn_hc,
            output_weights,
            output_embd,
            output_norm,
            logits_buf,
            xq_i8,
            xscale,
            attn_low,
            comp_emit_pre,
            comp_emit_post,
            index_comp_emit_pre,
            index_comp_emit_post,
            indexer_q,
            indexer_weights,
            indexer_scores,
            indexer_topk_buf,
            moe_xq,
            moe_midq,
            iq2_grid_gpu,
            iq2_signs_gpu,
            raw_cache,
            attn_comp_cache,
            attn_state_kv,
            attn_state_score,
            index_comp_cache,
            index_state_kv,
            index_state_score,
            layer_n_comp,
            ctx_size,
            raw_cap,
            comp_cap,
            n_filled: 0,
        })
    }

    /// Current KV-cache fill level (number of tokens whose state is committed).
    pub fn pos(&self) -> usize {
        self.n_filled
    }

    /// Decode one token at the current position and advance the position counter.
    pub fn decode_next(&mut self, engine: &Engine, token: i32) -> Result<Vec<f32>> {
        let pos = self.n_filled as u32;
        let logits = self.decode(engine, token, pos)?;
        self.n_filled += 1;
        Ok(logits)
    }

    // ─── Main decode entry point ───────────────────────────────────────────
    pub fn decode(&mut self, engine: &Engine, token: i32, pos: u32) -> Result<Vec<f32>> {
        let stream = &engine.stream;

        // ── 1. Embed token → cur_hc ──────────────────────────────────────
        {
            let w_embd_buf = upload(engine, &engine.weights.token_embd, "token_embd")?;
            let n_elem = HC_DIM; // n_embd * n_hc
            engine.kernels.embed.embed_token_hc(
                stream,
                Engine::cfg1d(n_elem, 256),
                buf_as_u16(&w_embd_buf),
                &mut self.cur_hc,
                token as u32,
                N_EMBD as u32,
                N_HC as u32,
            )?;
        }

        // ── 2. Layer loop ─────────────────────────────────────────────────
        for il in 0..N_LAYER {
            self.decode_layer(engine, il, pos, token)?;
        }

        // ── 3. Output projection ──────────────────────────────────────────
        self.decode_output(engine, pos)?;

        // ── 4. Return logits ──────────────────────────────────────────────
        let logits = self
            .logits_buf
            .to_host_vec(stream)
            .context("copying logits to host")?;
        Ok(logits)
    }

    // ─── Batch prefill ────────────────────────────────────────────────────────
    /// Process a sequence of prompt tokens, populating the KV cache and returning
    /// the logits for the token that would follow.
    ///
    /// The prompt is processed in chunks of `raw_cap = N_SWA = 128` tokens.  Within
    /// each chunk, QKV projections and normalisations run in batch (the main speedup
    /// vs. sequential `decode_next`).  Attention runs in batch via `prefill_raw` only
    /// for the first chunk of a fresh session (`pos == 0`); all other chunks use
    /// per-token `decode_mixed`, which correctly attends to the full circular KV cache
    /// and all accumulated compressed rows.
    ///
    /// Compressor state is populated for every token.  This method also works
    /// correctly for multi-turn: calling `prefill` after prior tokens have been
    /// committed uses per-token attention throughout.
    pub fn prefill(&mut self, engine: &Engine, tokens: &[i32]) -> Result<Vec<f32>> {
        let n = tokens.len();
        anyhow::ensure!(n > 0, "prefill: empty token list");
        anyhow::ensure!(
            self.n_filled + n <= self.ctx_size,
            "prefill: {} + {n} tokens exceeds ctx_size={}",
            self.n_filled,
            self.ctx_size
        );
        let start_pos = self.n_filled;
        let chunk_size = self.raw_cap; // = N_SWA = 128

        let stream = &engine.stream;

        // Allocate batch scratch buffers sized for one chunk (≤ chunk_size tokens).
        // Using chunk_size rather than n avoids large allocations for long prompts.
        let nc_max = n.min(chunk_size);
        let mut b_hc = DeviceBuffer::<f32>::zeroed(stream, nc_max * HC_DIM).context("b_hc")?;
        let mut b_hc_next =
            DeviceBuffer::<f32>::zeroed(stream, nc_max * HC_DIM).context("b_hc_next")?;
        let mut b_flat_hc =
            DeviceBuffer::<f32>::zeroed(stream, nc_max * HC_DIM).context("b_flat_hc")?;
        let mut b_hc_mix =
            DeviceBuffer::<f32>::zeroed(stream, nc_max * MIX_HC).context("b_hc_mix")?;
        let b_hc_split =
            DeviceBuffer::<f32>::zeroed(stream, nc_max * MIX_HC).context("b_hc_split")?;
        let mut b_attn_cur =
            DeviceBuffer::<f32>::zeroed(stream, nc_max * N_EMBD).context("b_attn_cur")?;
        let mut b_attn_norm =
            DeviceBuffer::<f32>::zeroed(stream, nc_max * N_EMBD).context("b_attn_norm")?;
        let max_xq = nc_max * Q_DIM;
        let max_xs = nc_max * (Q_DIM / 32);
        let mut b_xq_i8 = DeviceBuffer::<i8>::zeroed(stream, max_xq).context("b_xq_i8")?;
        let mut b_xscale = DeviceBuffer::<f32>::zeroed(stream, max_xs).context("b_xscale")?;
        let mut b_qr = DeviceBuffer::<f32>::zeroed(stream, nc_max * Q_RANK).context("b_qr")?;
        let mut b_qr_norm =
            DeviceBuffer::<f32>::zeroed(stream, nc_max * Q_RANK).context("b_qr_norm")?;
        let mut b_q = DeviceBuffer::<f32>::zeroed(stream, nc_max * Q_DIM).context("b_q")?;
        let mut b_kv_raw =
            DeviceBuffer::<f32>::zeroed(stream, nc_max * N_HEAD_DIM).context("b_kv_raw")?;
        let mut b_kv = DeviceBuffer::<f32>::zeroed(stream, nc_max * N_HEAD_DIM).context("b_kv")?;
        // b_heads only used when batch_attn == true (first fresh chunk)
        let mut b_heads = DeviceBuffer::<f32>::zeroed(stream, nc_max * Q_DIM).context("b_heads")?;

        let mut processed = 0usize;

        while processed < n {
            let nc = chunk_size.min(n - processed);
            let abs_start = (start_pos + processed) as u32;
            // Use batch prefill_raw attention only for the very first chunk of a
            // fresh session.  Any other chunk (including multi-turn continuations)
            // uses per-token decode_mixed, which correctly handles the circular KV
            // cache and all accumulated comp rows.
            let batch_attn = abs_start == 0;
            let chunk_tokens = &tokens[processed..processed + nc];

            // Upload this chunk's token IDs to GPU.
            let tok_buf: DeviceBuffer<i32> =
                DeviceBuffer::from_host(stream, chunk_tokens).context("prefill: upload tokens")?;

            // ── 1. Embed chunk tokens → b_hc ──────────────────────────────────
            {
                let w_embd = upload(engine, &engine.weights.token_embd, "token_embd")?;
                engine.kernels.embed.embed_tokens_hc(
                    stream,
                    Engine::cfg1d(nc * HC_DIM, 256),
                    &tok_buf,
                    buf_as_u16(&w_embd),
                    &mut b_hc,
                    N_VOCAB as u32,
                    nc as u32,
                    N_EMBD as u32,
                    N_HC as u32,
                )?;
            }

            // ── 2. Layer loop ──────────────────────────────────────────────────
            for il in 0..N_LAYER {
                let lw = &engine.weights.layers[il];
                let ratio = compress_ratio(il);
                let blocks_embd = (N_EMBD / 32) as u64;
                let blocks_qrank = (Q_RANK / 32) as u64;

                // ── a. Batch HC attn pre-norm ──────────────────────────────────
                engine.kernels.norm.rms_norm_plain(
                    stream,
                    LaunchConfig {
                        grid_dim: (nc as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &b_hc,
                    &mut b_flat_hc,
                    HC_DIM as u32,
                    nc as u32,
                    RMS_EPS,
                )?;
                {
                    let w_fn = upload(engine, &lw.hc_attn_fn, &format!("blk.{il}.hc_attn_fn"))?;
                    engine.kernels.matmul.matmul_f16(
                        stream,
                        LaunchConfig {
                            grid_dim: (MIX_HC as u32, nc as u32, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        buf_as_u16(&w_fn),
                        &b_flat_hc,
                        &mut b_hc_mix,
                        HC_DIM as u64,
                        MIX_HC as u64,
                        nc as u64,
                    )?;
                }
                {
                    let w_scale = upload(
                        engine,
                        &lw.hc_attn_scale,
                        &format!("blk.{il}.hc_attn_scale"),
                    )?;
                    let w_base =
                        upload(engine, &lw.hc_attn_base, &format!("blk.{il}.hc_attn_base"))?;
                    let w_norm = upload(engine, &lw.attn_norm, &format!("blk.{il}.attn_norm"))?;
                    engine.kernels.hc.hc_split_weighted_sum_norm_fused(
                        stream,
                        LaunchConfig {
                            grid_dim: (nc as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &b_hc_mix,
                        buf_as_f32(&w_scale),
                        buf_as_f32(&w_base),
                        &b_hc,
                        buf_as_f32(&w_norm),
                        b_hc_split.cu_deviceptr() as *mut f32,
                        &mut b_attn_cur,
                        &mut b_attn_norm,
                        N_EMBD as u32,
                        N_HC as u32,
                        nc as u32,
                        N_HC_SINKHORN_ITER as u32,
                        HC_EPS,
                        RMS_EPS,
                    )?;
                }

                // ── b. Batch QKV projections ───────────────────────────────────
                engine.kernels.quantize.quantize_q8_0(
                    stream,
                    LaunchConfig {
                        grid_dim: (blocks_embd as u32, nc as u32, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &b_attn_norm,
                    &mut b_xq_i8,
                    &mut b_xscale,
                    N_EMBD as u64,
                    blocks_embd,
                )?;
                {
                    let w_qa = upload(engine, &lw.attn_q_a, &format!("blk.{il}.attn_q_a"))?;
                    engine.kernels.matmul.matmul_q8_0_preq_batch_warp8(
                        stream,
                        LaunchConfig {
                            grid_dim: ((Q_RANK as u32 + 7) / 8, nc as u32, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &w_qa,
                        &b_xq_i8,
                        &b_xscale,
                        &mut b_qr,
                        N_EMBD as u64,
                        Q_RANK as u64,
                        nc as u64,
                        blocks_embd,
                    )?;
                }
                {
                    let w_kv = upload(engine, &lw.attn_kv, &format!("blk.{il}.attn_kv_a_mla"))?;
                    engine.kernels.matmul.matmul_q8_0_preq_batch_warp8(
                        stream,
                        LaunchConfig {
                            grid_dim: ((N_HEAD_DIM as u32 + 7) / 8, nc as u32, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &w_kv,
                        &b_xq_i8,
                        &b_xscale,
                        &mut b_kv_raw,
                        N_EMBD as u64,
                        N_HEAD_DIM as u64,
                        nc as u64,
                        blocks_embd,
                    )?;
                }
                {
                    let w_qa_norm = upload(
                        engine,
                        &lw.attn_q_a_norm,
                        &format!("blk.{il}.attn_q_a_norm"),
                    )?;
                    let w_kv_norm = upload(
                        engine,
                        &lw.attn_kv_a_norm,
                        &format!("blk.{il}.attn_kv_a_norm"),
                    )?;
                    engine.kernels.norm.dsv4_qkv_rms_norm_rows(
                        stream,
                        LaunchConfig {
                            grid_dim: (nc as u32, 2, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &b_qr,
                        buf_as_f32(&w_qa_norm),
                        &mut b_qr_norm,
                        Q_RANK as u32,
                        &b_kv_raw,
                        buf_as_f32(&w_kv_norm),
                        &mut b_kv,
                        N_HEAD_DIM as u32,
                        nc as u32,
                        RMS_EPS,
                    )?;
                }
                engine.kernels.quantize.quantize_q8_0(
                    stream,
                    LaunchConfig {
                        grid_dim: (blocks_qrank as u32, nc as u32, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &b_qr_norm,
                    &mut b_xq_i8,
                    &mut b_xscale,
                    Q_RANK as u64,
                    blocks_qrank,
                )?;
                {
                    let w_qb = upload(engine, &lw.attn_q_b, &format!("blk.{il}.attn_q_b"))?;
                    engine.kernels.matmul.matmul_q8_0_preq_batch_warp8(
                        stream,
                        LaunchConfig {
                            grid_dim: ((Q_DIM as u32 + 7) / 8, nc as u32, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &w_qb,
                        &b_xq_i8,
                        &b_xscale,
                        &mut b_q,
                        Q_RANK as u64,
                        Q_DIM as u64,
                        nc as u64,
                        blocks_qrank,
                    )?;
                }
                engine.kernels.norm.head_rms_norm(
                    stream,
                    LaunchConfig {
                        grid_dim: (nc as u32 * N_HEAD as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut b_q,
                    nc as u32,
                    N_HEAD as u32,
                    N_HEAD_DIM as u32,
                    RMS_EPS,
                )?;
                let fs = rope_freq_scale(il);
                let fa = rope_attn_factor(il, fs);
                engine.kernels.rope.rope_tail(
                    stream,
                    Engine::cfg1d(nc * N_HEAD * (N_ROT / 2), 256),
                    &mut b_q,
                    nc as u32,
                    N_HEAD as u32,
                    N_HEAD_DIM as u32,
                    N_ROT as u32,
                    abs_start,
                    ROPE_ORIG_CTX,
                    0,
                    rope_freq_base(il),
                    fs,
                    rope_ext_factor(il),
                    fa,
                    ROPE_YARN_BETA_FAST,
                    ROPE_YARN_BETA_SLOW,
                )?;
                engine.kernels.rope.rope_tail(
                    stream,
                    Engine::cfg1d(nc * N_HEAD_KV * (N_ROT / 2), 256),
                    &mut b_kv,
                    nc as u32,
                    N_HEAD_KV as u32,
                    N_HEAD_DIM as u32,
                    N_ROT as u32,
                    abs_start,
                    ROPE_ORIG_CTX,
                    0,
                    rope_freq_base(il),
                    fs,
                    rope_ext_factor(il),
                    fa,
                    ROPE_YARN_BETA_FAST,
                    ROPE_YARN_BETA_SLOW,
                )?;

                // ── c. Store KV entries at positions abs_start..abs_start+nc-1 ─
                engine.kernels.kv_cache.store_raw_kv_batch(
                    stream,
                    Engine::cfg1d(nc * N_HEAD_DIM, 256),
                    &b_kv,
                    &mut self.raw_cache[il],
                    self.raw_cap as u32,
                    abs_start,
                    nc as u32,
                    N_HEAD_DIM as u32,
                )?;

                // ── d. Batch attention (first fresh chunk only) ─────────────────
                if batch_attn {
                    let sinks_buf =
                        upload(engine, &lw.attn_sinks, &format!("blk.{il}.attn_sinks"))?;
                    engine.kernels.attention.prefill_raw(
                        stream,
                        LaunchConfig {
                            grid_dim: (nc as u32, N_HEAD as u32, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &b_q,
                        &self.raw_cache[il],
                        buf_as_f32(&sinks_buf),
                        &mut b_heads,
                        nc as u32,
                        N_SWA as u32,
                        N_HEAD as u32,
                        N_HEAD_DIM as u32,
                    )?;
                }

                // ── e-i. Per-token: compressors + attention (if not batch) + FFN ─
                let f32_size = std::mem::size_of::<f32>() as u64;
                for t in 0..nc {
                    let pos_t = abs_start + t as u32;
                    unsafe {
                        // b_hc[t] → self.cur_hc
                        cuda_core::memory::memcpy_dtod_async(
                            self.cur_hc.cu_deviceptr(),
                            b_hc.cu_deviceptr() + (t * HC_DIM) as u64 * f32_size,
                            HC_DIM * 4,
                            stream.cu_stream(),
                        )?;
                        // b_hc_split[t] → self.hc_split
                        cuda_core::memory::memcpy_dtod_async(
                            self.hc_split.cu_deviceptr(),
                            b_hc_split.cu_deviceptr() + (t * MIX_HC) as u64 * f32_size,
                            MIX_HC * 4,
                            stream.cu_stream(),
                        )?;
                        // b_attn_norm[t] → self.attn_norm (needed by compressors)
                        cuda_core::memory::memcpy_dtod_async(
                            self.attn_norm.cu_deviceptr(),
                            b_attn_norm.cu_deviceptr() + (t * N_EMBD) as u64 * f32_size,
                            N_EMBD * 4,
                            stream.cu_stream(),
                        )?;
                    }

                    // Compressors — must run before attention to match decode_layer ordering.
                    if ratio == 4 {
                        self.run_index_compressor_step(engine, il, pos_t)?;
                    }
                    self.run_attn_compressor_step(engine, il, pos_t)?;

                    // Attention for this token.
                    if batch_attn {
                        // Use the precomputed batch result.
                        unsafe {
                            cuda_core::memory::memcpy_dtod_async(
                                self.heads.cu_deviceptr(),
                                b_heads.cu_deviceptr() + (t * Q_DIM) as u64 * f32_size,
                                Q_DIM * 4,
                                stream.cu_stream(),
                            )?;
                        }
                    } else {
                        // Per-token decode_mixed: correct for circular raw cache and comp rows.
                        unsafe {
                            cuda_core::memory::memcpy_dtod_async(
                                self.q.cu_deviceptr(),
                                b_q.cu_deviceptr() + (t * Q_DIM) as u64 * f32_size,
                                Q_DIM * 4,
                                stream.cu_stream(),
                            )?;
                        }
                        let n_comp = if ratio != 0 { self.layer_n_comp[il] } else { 0 };
                        let n_raw = if pos_t + 1 < self.raw_cap as u32 {
                            pos_t + 1
                        } else {
                            self.raw_cap as u32
                        };
                        let raw_start = (pos_t + 1).wrapping_sub(n_raw) % self.raw_cap as u32;
                        let sinks_buf =
                            upload(engine, &lw.attn_sinks, &format!("blk.{il}.attn_sinks"))?;
                        engine.kernels.attention.decode_mixed(
                            stream,
                            LaunchConfig {
                                grid_dim: (1, N_HEAD as u32, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            &self.q,
                            &self.raw_cache[il],
                            &self.attn_comp_cache[il],
                            &self.attn_comp_cache[il],
                            buf_as_f32(&sinks_buf),
                            &mut self.heads,
                            0,
                            1,
                            pos_t,
                            n_raw,
                            self.raw_cap as u32,
                            raw_start,
                            n_comp,
                            N_SWA as u32,
                            ratio,
                            N_HEAD as u32,
                            N_HEAD_DIM as u32,
                        )?;
                    }

                    self.decode_attn_out_and_ffn(engine, il, pos_t, ratio, chunk_tokens[t])?;

                    // self.cur_hc → b_hc_next[t]
                    unsafe {
                        cuda_core::memory::memcpy_dtod_async(
                            b_hc_next.cu_deviceptr() + (t * HC_DIM) as u64 * f32_size,
                            self.cur_hc.cu_deviceptr(),
                            HC_DIM * 4,
                            stream.cu_stream(),
                        )?;
                    }
                }

                std::mem::swap(&mut b_hc, &mut b_hc_next);
            }

            processed += nc;
        }

        // After all chunks: self.cur_hc holds the last token's HC state (output of
        // the final layer), ready for decode_output.
        let last_pos = (start_pos + n - 1) as u32;
        self.decode_output(engine, last_pos)?;
        self.n_filled += n;

        let logits = self
            .logits_buf
            .to_host_vec(stream)
            .context("prefill: copying logits")?;
        Ok(logits)
    }

    /// Run one step of the attention compressor for layer `il` at position `pos`.
    /// Reads `self.attn_norm`; may write to `attn_comp_cache[il]`; increments
    /// `layer_n_comp[il]` when a row is emitted.
    fn run_attn_compressor_step(&mut self, engine: &Engine, il: usize, pos: u32) -> Result<()> {
        let ratio = compress_ratio(il);
        if ratio == 0 {
            return Ok(());
        }
        let lw = &engine.weights.layers[il];
        if lw.attn_compressor_kv.is_none() {
            return Ok(());
        }

        let stream = &engine.stream;
        let coff: usize = if ratio == 4 { 2 } else { 1 };
        let comp_width = (coff * N_HEAD_DIM) as u32;

        // Project attn_norm → comp_kv_cur, comp_sc_cur
        {
            let w_kv = upload(
                engine,
                &lw.attn_compressor_kv,
                &format!("blk.{il}.attn_compressor_kv"),
            )?;
            let w_gate = upload(
                engine,
                &lw.attn_compressor_gate,
                &format!("blk.{il}.attn_compressor_gate"),
            )?;
            engine.kernels.matmul.matmul_f16(
                stream,
                LaunchConfig {
                    grid_dim: (comp_width, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                buf_as_u16(&w_kv),
                &self.attn_norm,
                &mut self.comp_kv_cur,
                N_EMBD as u64,
                comp_width as u64,
                1,
            )?;
            engine.kernels.matmul.matmul_f16(
                stream,
                LaunchConfig {
                    grid_dim: (comp_width, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                buf_as_u16(&w_gate),
                &self.attn_norm,
                &mut self.comp_sc_cur,
                N_EMBD as u64,
                comp_width as u64,
                1,
            )?;
        }
        // Store into state ring with APE positional bias
        {
            let w_ape = upload(
                engine,
                &lw.attn_compressor_ape,
                &format!("blk.{il}.attn_compressor_ape"),
            )?;
            engine.kernels.hc.compressor_store(
                stream,
                Engine::cfg1d(comp_width as usize, 256),
                &self.comp_kv_cur,
                &self.comp_sc_cur,
                &mut self.attn_state_kv[il],
                &mut self.attn_state_score[il],
                w_ape.cu_deviceptr() as *const u8,
                0u64,
                1u32,
                N_HEAD_DIM as u32,
                ratio,
                pos,
                1u32,
            )?;
        }
        // Emit a compressed KV row every `ratio` tokens
        let emit = (pos + 1) % ratio == 0;
        if emit && self.layer_n_comp[il] < self.comp_cap as u32 {
            let comp_row = self.layer_n_comp[il];

            engine.kernels.hc.compressor_update_pool(
                stream,
                Engine::cfg1d(N_HEAD_DIM, 256),
                &self.attn_state_kv[il],
                &self.attn_state_score[il],
                &mut self.comp_emit_pre,
                N_HEAD_DIM as u32,
                ratio,
            )?;
            {
                let w_norm = upload(
                    engine,
                    &lw.attn_compressor_norm,
                    &format!("blk.{il}.attn_compressor_norm"),
                )?;
                engine.kernels.norm.rms_norm_weight(
                    stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &self.comp_emit_pre,
                    buf_as_f32(&w_norm),
                    &mut self.comp_emit_post,
                    N_HEAD_DIM as u32,
                    1,
                    RMS_EPS,
                )?;
            }
            let comp_pos = pos + 1 - ratio;
            let fs = rope_freq_scale(il);
            let fa = rope_attn_factor(il, fs);
            engine.kernels.rope.rope_tail(
                stream,
                Engine::cfg1d(N_ROT / 2, 256),
                &mut self.comp_emit_post,
                1,
                1u32,
                N_HEAD_DIM as u32,
                N_ROT as u32,
                comp_pos,
                ROPE_ORIG_CTX,
                0,
                rope_freq_base(il),
                fs,
                rope_ext_factor(il),
                fa,
                ROPE_YARN_BETA_FAST,
                ROPE_YARN_BETA_SLOW,
            )?;
            engine.kernels.quantize.fp8_kv_quantize(
                stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (64, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut self.comp_emit_post,
                1u32,
                N_HEAD_DIM as u32,
                N_ROT as u32,
            )?;
            engine.kernels.utils.copy_f32_at_offset(
                stream,
                Engine::cfg1d(N_HEAD_DIM, 256),
                &self.comp_emit_post,
                &mut self.attn_comp_cache[il],
                (comp_row as usize * N_HEAD_DIM) as u32,
            )?;
            if ratio == 4 {
                engine.kernels.hc.compressor_shift_ratio4(
                    stream,
                    Engine::cfg1d(4 * comp_width as usize, 256),
                    &mut self.attn_state_kv[il],
                    &mut self.attn_state_score[il],
                    comp_width,
                )?;
            }
            self.layer_n_comp[il] += 1;
        }
        Ok(())
    }

    /// Run one step of the indexer compressor for layer `il` at position `pos`.
    /// Only fires for ratio=4 layers. Uses the same `layer_n_comp[il]` counter
    /// as the attn compressor (they emit in lockstep); does NOT increment it.
    fn run_index_compressor_step(&mut self, engine: &Engine, il: usize, pos: u32) -> Result<()> {
        let ratio = compress_ratio(il);
        if ratio != 4 {
            return Ok(());
        }
        let lw = &engine.weights.layers[il];
        if lw.indexer_compressor_kv.is_none() {
            return Ok(());
        }

        let stream = &engine.stream;
        // coff=2 for ratio=4; comp_width = 2 * N_INDEXER_HEAD_DIM = 256
        let comp_width = (2 * N_INDEXER_HEAD_DIM) as u32;

        // Project attn_norm → comp_kv_cur, comp_sc_cur (reuse attn scratch; smaller size is fine)
        {
            let w_kv = upload(
                engine,
                &lw.indexer_compressor_kv,
                &format!("blk.{il}.indexer_compressor_kv"),
            )?;
            let w_gate = upload(
                engine,
                &lw.indexer_compressor_gate,
                &format!("blk.{il}.indexer_compressor_gate"),
            )?;
            engine.kernels.matmul.matmul_f16(
                stream,
                LaunchConfig {
                    grid_dim: (comp_width, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                buf_as_u16(&w_kv),
                &self.attn_norm,
                &mut self.comp_kv_cur,
                N_EMBD as u64,
                comp_width as u64,
                1,
            )?;
            engine.kernels.matmul.matmul_f16(
                stream,
                LaunchConfig {
                    grid_dim: (comp_width, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                buf_as_u16(&w_gate),
                &self.attn_norm,
                &mut self.comp_sc_cur,
                N_EMBD as u64,
                comp_width as u64,
                1,
            )?;
        }
        {
            let w_ape = upload(
                engine,
                &lw.indexer_compressor_ape,
                &format!("blk.{il}.indexer_compressor_ape"),
            )?;
            engine.kernels.hc.compressor_store(
                stream,
                Engine::cfg1d(comp_width as usize, 256),
                &self.comp_kv_cur,
                &self.comp_sc_cur,
                &mut self.index_state_kv[il],
                &mut self.index_state_score[il],
                w_ape.cu_deviceptr() as *const u8,
                0u64,
                1u32,
                N_INDEXER_HEAD_DIM as u32,
                ratio,
                pos,
                1u32,
            )?;
        }
        let emit = (pos + 1) % ratio == 0;
        if emit && self.layer_n_comp[il] < self.comp_cap as u32 {
            let comp_row = self.layer_n_comp[il];

            engine.kernels.hc.compressor_update_pool(
                stream,
                Engine::cfg1d(N_INDEXER_HEAD_DIM, 256),
                &self.index_state_kv[il],
                &self.index_state_score[il],
                &mut self.index_comp_emit_pre,
                N_INDEXER_HEAD_DIM as u32,
                ratio,
            )?;
            {
                let w_norm = upload(
                    engine,
                    &lw.indexer_compressor_norm,
                    &format!("blk.{il}.indexer_compressor_norm"),
                )?;
                engine.kernels.norm.rms_norm_weight(
                    stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &self.index_comp_emit_pre,
                    buf_as_f32(&w_norm),
                    &mut self.index_comp_emit_post,
                    N_INDEXER_HEAD_DIM as u32,
                    1,
                    RMS_EPS,
                )?;
            }
            let comp_pos = pos + 1 - ratio;
            let fs = rope_freq_scale(il);
            let fa = rope_attn_factor(il, fs);
            engine.kernels.rope.rope_tail(
                stream,
                Engine::cfg1d(N_ROT / 2, 256),
                &mut self.index_comp_emit_post,
                1,
                1u32,
                N_INDEXER_HEAD_DIM as u32,
                N_ROT as u32,
                comp_pos,
                ROPE_ORIG_CTX,
                0,
                rope_freq_base(il),
                fs,
                rope_ext_factor(il),
                fa,
                ROPE_YARN_BETA_FAST,
                ROPE_YARN_BETA_SLOW,
            )?;
            engine.kernels.quantize.fp8_kv_quantize(
                stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (64, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut self.index_comp_emit_post,
                1u32,
                N_INDEXER_HEAD_DIM as u32,
                N_ROT as u32,
            )?;
            engine.kernels.utils.copy_f32_at_offset(
                stream,
                Engine::cfg1d(N_INDEXER_HEAD_DIM, 256),
                &self.index_comp_emit_post,
                &mut self.index_comp_cache[il],
                (comp_row as usize * N_INDEXER_HEAD_DIM) as u32,
            )?;
            // ratio=4 always needs a state ring shift
            engine.kernels.hc.compressor_shift_ratio4(
                stream,
                Engine::cfg1d(4 * comp_width as usize, 256),
                &mut self.index_state_kv[il],
                &mut self.index_state_score[il],
                comp_width,
            )?;
            // NOTE: do NOT increment layer_n_comp here; run_attn_compressor_step does it
        }
        Ok(())
    }

    // ─── Per-layer decode ─────────────────────────────────────────────────────
    fn decode_layer(&mut self, engine: &Engine, il: usize, pos: u32, token: i32) -> Result<()> {
        let stream = &engine.stream;
        let lw = &engine.weights.layers[il];
        let ratio = compress_ratio(il);

        // ── a. HC attn pre-norm ──────────────────────────────────────────
        // rms_norm_plain: flat_hc ← cur_hc
        engine.kernels.norm.rms_norm_plain(
            stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            &self.cur_hc,
            &mut self.flat_hc,
            HC_DIM as u32,
            1,
            RMS_EPS,
        )?;

        // matmul_f16: hc_mix ← flat_hc × W_hc_attn_fn  [hc_dim × mix_hc]
        {
            let w_fn = upload(engine, &lw.hc_attn_fn, &format!("blk.{il}.hc_attn_fn"))?;
            engine.kernels.matmul.matmul_f16(
                stream,
                LaunchConfig {
                    grid_dim: (MIX_HC as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                buf_as_u16(&w_fn),
                &self.flat_hc,
                &mut self.hc_mix,
                HC_DIM as u64,
                MIX_HC as u64,
                1,
            )?;
        }

        // hc_split_weighted_sum_norm_fused:
        //   attn_cur, attn_norm ← f(hc_mix, cur_hc, W_hc_attn_scale, W_hc_attn_base, W_attn_norm)
        {
            let w_scale = upload(
                engine,
                &lw.hc_attn_scale,
                &format!("blk.{il}.hc_attn_scale"),
            )?;
            let w_base = upload(engine, &lw.hc_attn_base, &format!("blk.{il}.hc_attn_base"))?;
            let w_norm = upload(engine, &lw.attn_norm, &format!("blk.{il}.attn_norm"))?;
            engine.kernels.hc.hc_split_weighted_sum_norm_fused(
                stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &self.hc_mix,
                buf_as_f32(&w_scale),
                buf_as_f32(&w_base),
                &self.cur_hc,
                buf_as_f32(&w_norm),
                self.hc_split.cu_deviceptr() as *mut f32,
                &mut self.attn_cur,
                &mut self.attn_norm,
                N_EMBD as u32,
                N_HC as u32,
                1,
                N_HC_SINKHORN_ITER as u32,
                HC_EPS,
                RMS_EPS,
            )?;
        }

        // ── b. Q/KV projections ──────────────────────────────────────────
        // matmul_q8_0_preq_warp8: qr ← attn_norm × W_attn_q_a  [N_EMBD → Q_RANK]
        {
            let w_qa = upload(engine, &lw.attn_q_a, &format!("blk.{il}.attn_q_a"))?;
            let blocks = (N_EMBD / 32) as u64;
            engine.kernels.quantize.quantize_q8_0(
                stream,
                LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &self.attn_norm,
                &mut self.xq_i8,
                &mut self.xscale,
                N_EMBD as u64,
                blocks,
            )?;
            engine.kernels.matmul.matmul_q8_0_preq_warp8(
                stream,
                Engine::cfg_warp8(Q_RANK, 1),
                &w_qa,
                &self.xq_i8,
                &self.xscale,
                &mut self.qr,
                N_EMBD as u64,
                Q_RANK as u64,
                blocks,
            )?;
        }

        // matmul_q8_0_preq_warp8: kv_raw ← attn_norm × W_attn_kv  [N_EMBD → N_HEAD_DIM]
        {
            let w_kv = upload(engine, &lw.attn_kv, &format!("blk.{il}.attn_kv_a_mla"))?;
            let blocks = (N_EMBD / 32) as u64;
            // xq_i8 / xscale already quantized above (same input attn_norm, same blocks)
            engine.kernels.matmul.matmul_q8_0_preq_warp8(
                stream,
                Engine::cfg_warp8(N_HEAD_DIM, 1),
                &w_kv,
                &self.xq_i8,
                &self.xscale,
                &mut self.kv_raw,
                N_EMBD as u64,
                N_HEAD_DIM as u64,
                blocks,
            )?;
        }

        // dsv4_qkv_rms_norm_rows: qr_norm, kv ← rms_norm(qr, W_q_a_norm), rms_norm(kv_raw, W_kv_a_norm)
        {
            let w_qa_norm = upload(
                engine,
                &lw.attn_q_a_norm,
                &format!("blk.{il}.attn_q_a_norm"),
            )?;
            let w_kv_norm = upload(
                engine,
                &lw.attn_kv_a_norm,
                &format!("blk.{il}.attn_kv_a_norm"),
            )?;
            engine.kernels.norm.dsv4_qkv_rms_norm_rows(
                stream,
                LaunchConfig {
                    grid_dim: (1, 2, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &self.qr,
                buf_as_f32(&w_qa_norm),
                &mut self.qr_norm,
                Q_RANK as u32,
                &self.kv_raw,
                buf_as_f32(&w_kv_norm),
                &mut self.kv,
                N_HEAD_DIM as u32,
                1,
                RMS_EPS,
            )?;
        }

        // matmul_q8_0_preq_warp8: q ← qr_norm × W_attn_q_b  [Q_RANK → Q_DIM]
        {
            let w_qb = upload(engine, &lw.attn_q_b, &format!("blk.{il}.attn_q_b"))?;
            let blocks = (Q_RANK / 32) as u64;
            engine.kernels.quantize.quantize_q8_0(
                stream,
                LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &self.qr_norm,
                &mut self.xq_i8,
                &mut self.xscale,
                Q_RANK as u64,
                blocks,
            )?;
            engine.kernels.matmul.matmul_q8_0_preq_warp8(
                stream,
                Engine::cfg_warp8(Q_DIM, 1),
                &w_qb,
                &self.xq_i8,
                &self.xscale,
                &mut self.q,
                Q_RANK as u64,
                Q_DIM as u64,
                blocks,
            )?;
        }

        // head_rms_norm: q in-place  [n_tok=1, n_head=N_HEAD, head_dim=N_HEAD_DIM]
        // Grid: one block per head (blockIdx_x = head), NOT cfg1d which would collapse to 1 block.
        engine.kernels.norm.head_rms_norm(
            stream,
            LaunchConfig {
                grid_dim: (N_HEAD as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            &mut self.q,
            1,
            N_HEAD as u32,
            N_HEAD_DIM as u32,
            RMS_EPS,
        )?;

        // rope_tail: q  [n_tok=1, n_head=N_HEAD, head_dim=N_HEAD_DIM, n_rot=N_ROT]
        {
            let fs = rope_freq_scale(il);
            let fa = rope_attn_factor(il, fs);
            engine.kernels.rope.rope_tail(
                stream,
                Engine::cfg1d(1 * N_HEAD * (N_ROT / 2), 256),
                &mut self.q,
                1,
                N_HEAD as u32,
                N_HEAD_DIM as u32,
                N_ROT as u32,
                pos,
                ROPE_ORIG_CTX,
                0, // not inverse
                rope_freq_base(il),
                fs,
                rope_ext_factor(il),
                fa,
                ROPE_YARN_BETA_FAST,
                ROPE_YARN_BETA_SLOW,
            )?;
        }

        // rope_tail: kv  [n_tok=1, n_head=N_HEAD_KV=1]
        {
            let fs = rope_freq_scale(il);
            let fa = rope_attn_factor(il, fs);
            engine.kernels.rope.rope_tail(
                stream,
                Engine::cfg1d(1 * N_HEAD_KV * (N_ROT / 2), 256),
                &mut self.kv,
                1,
                N_HEAD_KV as u32,
                N_HEAD_DIM as u32,
                N_ROT as u32,
                pos,
                ROPE_ORIG_CTX,
                0,
                rope_freq_base(il),
                fs,
                rope_ext_factor(il),
                fa,
                ROPE_YARN_BETA_FAST,
                ROPE_YARN_BETA_SLOW,
            )?;
        }

        // store_raw_kv_batch: raw_cache[il] ← kv at position pos
        engine.kernels.kv_cache.store_raw_kv_batch(
            stream,
            Engine::cfg1d(N_HEAD_DIM, 256),
            &self.kv,
            &mut self.raw_cache[il],
            self.raw_cap as u32,
            pos,
            1,
            N_HEAD_DIM as u32,
        )?;

        // ── b2. Indexer Q and weights (ratio=4 layers only) ──────────────────
        if ratio == 4 && lw.indexer_attn_q_b.is_some() {
            // indexer_q ← qr_norm × W_indexer_q_b  [Q_RANK → INDEXER_Q_DIM]
            // Note: xq_i8/xscale still hold quantized qr_norm from step b
            let w_iqb = upload(
                engine,
                &lw.indexer_attn_q_b,
                &format!("blk.{il}.indexer.attn_q_b"),
            )?;
            engine.kernels.matmul.matmul_f16(
                stream,
                LaunchConfig {
                    grid_dim: (INDEXER_Q_DIM as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                buf_as_u16(&w_iqb),
                &self.qr_norm,
                &mut self.indexer_q,
                Q_RANK as u64,
                INDEXER_Q_DIM as u64,
                1,
            )?;
            let fs = rope_freq_scale(il);
            let fa = rope_attn_factor(il, fs);
            engine.kernels.rope.rope_tail(
                stream,
                Engine::cfg1d(N_INDEXER_HEAD * (N_ROT / 2), 256),
                &mut self.indexer_q,
                1,
                N_INDEXER_HEAD as u32,
                N_INDEXER_HEAD_DIM as u32,
                N_ROT as u32,
                pos,
                ROPE_ORIG_CTX,
                0,
                rope_freq_base(il),
                fs,
                rope_ext_factor(il),
                fa,
                ROPE_YARN_BETA_FAST,
                ROPE_YARN_BETA_SLOW,
            )?;
            // indexer_weights ← attn_cur × W_indexer_proj  [N_EMBD → N_INDEXER_HEAD]
            let w_iproj = upload(engine, &lw.indexer_proj, &format!("blk.{il}.indexer.proj"))?;
            engine.kernels.matmul.matmul_f16(
                stream,
                LaunchConfig {
                    grid_dim: (N_INDEXER_HEAD as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                buf_as_u16(&w_iproj),
                &self.attn_cur,
                &mut self.indexer_weights,
                N_EMBD as u64,
                N_INDEXER_HEAD as u64,
                1,
            )?;
        }

        // ── c. Compressor update ─────────────────────────────────────────
        // index compressor must run BEFORE attn compressor so both use the same
        // layer_n_comp value before the attn step increments it.
        if ratio == 4 {
            self.run_index_compressor_step(engine, il, pos)?;
        }
        self.run_attn_compressor_step(engine, il, pos)?;

        // ── d. Attention ─────────────────────────────────────────────────
        {
            let sinks_buf = upload(engine, &lw.attn_sinks, &format!("blk.{il}.attn_sinks"))?;
            let n_raw = if pos + 1 < self.raw_cap as u32 {
                pos + 1
            } else {
                self.raw_cap as u32
            };
            let raw_start = (pos + 1).wrapping_sub(n_raw) % self.raw_cap as u32;
            let n_comp = if ratio != 0 { self.layer_n_comp[il] } else { 0 };

            if ratio == 4 && n_comp as usize > N_INDEXER_TOP_K && lw.indexer_attn_q_b.is_some() {
                // Score every index-comp row, select top-K, then indexed attention.
                let scale = 1.0_f32 / ((N_INDEXER_HEAD_DIM * N_INDEXER_HEAD) as f32).sqrt();
                engine.kernels.indexer.indexer_score_one_direct(
                    stream,
                    LaunchConfig {
                        grid_dim: (n_comp, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &self.indexer_q,
                    &self.indexer_weights,
                    &self.index_comp_cache[il],
                    &mut self.indexer_scores,
                    n_comp,
                    pos,
                    4,
                    scale,
                    1,
                )?;
                let n_sort = n_comp.next_power_of_two();
                engine.kernels.indexer.indexer_topk_bitonic(
                    stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: (n_sort * 8) as u32,
                    },
                    &self.indexer_scores,
                    self.indexer_topk_buf.cu_deviceptr() as *mut u32,
                    n_comp,
                    1,
                    N_INDEXER_TOP_K as u32,
                    n_sort,
                )?;
                engine.kernels.attention.indexed_mixed(
                    stream,
                    LaunchConfig {
                        grid_dim: (1, N_HEAD as u32, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &self.q,
                    &self.raw_cache[il],
                    &self.attn_comp_cache[il],
                    &self.indexer_topk_buf,
                    buf_as_f32(&sinks_buf),
                    &mut self.heads,
                    1,
                    pos,
                    n_raw,
                    self.raw_cap as u32,
                    raw_start,
                    n_comp,
                    N_INDEXER_TOP_K as u32,
                    N_SWA as u32,
                    4,
                    N_HEAD as u32,
                    N_HEAD_DIM as u32,
                )?;
            } else {
                // ratio=0: n_comp=0 (SWA only)
                // ratio=4, n_comp≤top_k: all comp rows visible (no indexer needed)
                // ratio=128: all comp rows (static attention)
                engine.kernels.attention.decode_mixed(
                    stream,
                    LaunchConfig {
                        grid_dim: (1, N_HEAD as u32, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &self.q,
                    &self.raw_cache[il],
                    &self.attn_comp_cache[il],
                    &self.attn_comp_cache[il],
                    buf_as_f32(&sinks_buf),
                    &mut self.heads,
                    0,
                    1,
                    pos,
                    n_raw,
                    self.raw_cap as u32,
                    raw_start,
                    n_comp,
                    N_SWA as u32,
                    ratio,
                    N_HEAD as u32,
                    N_HEAD_DIM as u32,
                )?;
            }
        }

        // ── e-i. Attention output + FFN (extracted so prefill can call it per-token)
        self.decode_attn_out_and_ffn(engine, il, pos, ratio, token)
    }

    // ─── Attention output projection + HC FFN + FFN ───────────────────────────
    // Called by decode_layer (single-token) and by prefill (per-token after batch attn).
    // Requires self.heads, self.cur_hc, self.hc_split to be set for this token.
    // Produces updated self.cur_hc (= next layer's input for this token).
    fn decode_attn_out_and_ffn(
        &mut self,
        engine: &Engine,
        il: usize,
        pos: u32,
        _ratio: u32,
        token: i32,
    ) -> Result<()> {
        let stream = &engine.stream;
        let lw = &engine.weights.layers[il];

        // ── e. Attention output projection ───────────────────────────────
        //
        // 1. Inverse RoPE on heads (undo the value-space rotation)
        // 2. Quantize heads as N_OUT_GROUP groups of GROUP_DIM → xq_i8/xscale
        // 3. Grouped down-proj: attn_low[LOW_DIM] = heads × W_attn_output_a
        // 4. Quantize attn_low → xq_i8/xscale (reuse scratch)
        // 5. HC-expanded up-proj: attn_out + after_attn_hc = attn_low × W_attn_output_b
        //
        // Mirrors ds4_gpu_attention_output_low_q8_tensor +
        //         ds4_gpu_matmul_q8_0_hc_expand_tensor in ds4.c.
        {
            // Step 1: inverse RoPE on heads[N_HEAD × N_VALUE_DIM]
            // Same freq params as the forward RoPE on q; n_ctx_orig = ROPE_ORIG_CTX for
            // compressed (ratio != 0) layers, matching ds4.c line 9607-9617.
            let n_ctx_orig_inv = if compress_ratio(il) != 0 {
                ROPE_ORIG_CTX
            } else {
                0
            };
            let fs = rope_freq_scale(il);
            let fa = rope_attn_factor(il, fs);
            engine.kernels.rope.rope_tail(
                stream,
                Engine::cfg1d(1 * N_HEAD * (N_ROT / 2), 256),
                &mut self.heads,
                1,
                N_HEAD as u32,
                N_HEAD_DIM as u32,
                N_ROT as u32,
                pos,
                n_ctx_orig_inv,
                1, // inverse = true
                rope_freq_base(il),
                fs,
                rope_ext_factor(il),
                fa,
                ROPE_YARN_BETA_FAST,
                ROPE_YARN_BETA_SLOW,
            )?;

            // Step 2: quantize heads as N_OUT_GROUP rows of GROUP_DIM
            // Grid: (blocks_a, N_OUT_GROUP, 1) — 2D so each group is an independent row.
            // Our quantize_q8_0 kernel uses blockIdx_y as the "token" index, which maps
            // directly to the group index here.
            let blocks_a = (GROUP_DIM / 32) as u64; // 128
            engine.kernels.quantize.quantize_q8_0(
                stream,
                LaunchConfig {
                    grid_dim: (blocks_a as u32, N_OUT_GROUP as u32, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &self.heads,
                &mut self.xq_i8,
                &mut self.xscale,
                GROUP_DIM as u64,
                blocks_a,
            )?;

            // Step 3: grouped Q8_0 down-projection → attn_low[LOW_DIM]
            // W_attn_output_a: [LOW_DIM × GROUP_DIM] in Q8_0.
            // Output row r uses input from group (r / N_LORA_O); weight rows are sequential.
            let w_oa = upload(
                engine,
                &lw.attn_output_a,
                &format!("blk.{il}.attn_output_a"),
            )?;
            engine.kernels.matmul.matmul_q8_0_grouped_preq_warp8(
                stream,
                Engine::cfg_warp8(LOW_DIM, 1),
                &w_oa,
                &self.xq_i8,
                &self.xscale,
                &mut self.attn_low,
                GROUP_DIM as u64,
                N_LORA_O as u64,
                N_OUT_GROUP as u32,
                blocks_a,
            )?;

            // Step 4: quantize attn_low[LOW_DIM] as a single row (1D grid)
            let blocks_b = (LOW_DIM / 32) as u64; // 256
            engine.kernels.quantize.quantize_q8_0(
                stream,
                LaunchConfig {
                    grid_dim: (blocks_b as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &self.attn_low,
                &mut self.xq_i8,
                &mut self.xscale,
                LOW_DIM as u64,
                blocks_b,
            )?;

            // Step 5: HC-expanded up-proj → attn_out[N_EMBD] + after_attn_hc[HC_DIM]
            // W_attn_output_b: [N_EMBD × LOW_DIM] in Q8_0.
            // Simultaneously blends with cur_hc via hc_split to produce after_attn_hc.
            let w_ob = upload(
                engine,
                &lw.attn_output_b,
                &format!("blk.{il}.attn_output_b"),
            )?;
            engine.kernels.matmul.matmul_q8_0_hc_expand_preq_warp8(
                stream,
                Engine::cfg_warp8(N_EMBD, 1),
                &w_ob,
                &self.xq_i8,
                &self.xscale,
                &self.cur_hc,
                &self.hc_split,
                &mut self.attn_out,
                &self.heads, // block_add — unused when has_add=0; any buffer is fine
                &mut self.after_attn_hc,
                LOW_DIM as u64,
                N_EMBD as u64,
                N_EMBD as u32,
                N_HC as u32,
                blocks_b,
                0, // has_add = 0
            )?;
        }

        // ── f. HC FFN pre-norm ───────────────────────────────────────────
        // rms_norm_plain: flat_hc ← after_attn_hc
        engine.kernels.norm.rms_norm_plain(
            stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            &self.after_attn_hc,
            &mut self.flat_hc,
            HC_DIM as u32,
            1,
            RMS_EPS,
        )?;

        // matmul_f16: hc_mix ← flat_hc × W_hc_ffn_fn
        {
            let w_fn = upload(engine, &lw.hc_ffn_fn, &format!("blk.{il}.hc_ffn_fn"))?;
            engine.kernels.matmul.matmul_f16(
                stream,
                LaunchConfig {
                    grid_dim: (MIX_HC as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                buf_as_u16(&w_fn),
                &self.flat_hc,
                &mut self.hc_mix,
                HC_DIM as u64,
                MIX_HC as u64,
                1,
            )?;
        }

        // hc_split_weighted_sum_norm_fused: ffn_cur, ffn_norm ← f(hc_mix, after_attn_hc, ...)
        {
            let w_scale = upload(engine, &lw.hc_ffn_scale, &format!("blk.{il}.hc_ffn_scale"))?;
            let w_base = upload(engine, &lw.hc_ffn_base, &format!("blk.{il}.hc_ffn_base"))?;
            let w_norm = upload(engine, &lw.ffn_norm, &format!("blk.{il}.ffn_norm"))?;
            engine.kernels.hc.hc_split_weighted_sum_norm_fused(
                stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &self.hc_mix,
                buf_as_f32(&w_scale),
                buf_as_f32(&w_base),
                &self.after_attn_hc,
                buf_as_f32(&w_norm),
                self.hc_split.cu_deviceptr() as *mut f32,
                &mut self.ffn_cur,
                &mut self.ffn_norm,
                N_EMBD as u32,
                N_HC as u32,
                1,
                N_HC_SINKHORN_ITER as u32,
                HC_EPS,
                RMS_EPS,
            )?;
        }

        // ── g. Shared expert ─────────────────────────────────────────────
        if lw.ffn_gate_shexp.is_some() {
            let blocks = (N_EMBD / 32) as u64;
            engine.kernels.quantize.quantize_q8_0(
                stream,
                LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &self.ffn_norm,
                &mut self.xq_i8,
                &mut self.xscale,
                N_EMBD as u64,
                blocks,
            )?;
            // gate_shexp: [N_EMBD → N_FF_EXP]
            {
                let w = upload(
                    engine,
                    &lw.ffn_gate_shexp,
                    &format!("blk.{il}.ffn_gate_shexp"),
                )?;
                engine.kernels.matmul.matmul_q8_0_preq_warp8(
                    stream,
                    Engine::cfg_warp8(N_FF_EXP, 1),
                    &w,
                    &self.xq_i8,
                    &self.xscale,
                    &mut self.shared_gate,
                    N_EMBD as u64,
                    N_FF_EXP as u64,
                    blocks,
                )?;
            }
            // up_shexp: [N_EMBD → N_FF_EXP]
            {
                let w = upload(engine, &lw.ffn_up_shexp, &format!("blk.{il}.ffn_up_shexp"))?;
                engine.kernels.matmul.matmul_q8_0_preq_warp8(
                    stream,
                    Engine::cfg_warp8(N_FF_EXP, 1),
                    &w,
                    &self.xq_i8,
                    &self.xscale,
                    &mut self.shared_up,
                    N_EMBD as u64,
                    N_FF_EXP as u64,
                    blocks,
                )?;
            }
            // SwiGLU: shared_mid[i] = silu(shared_gate[i]) * shared_up[i]
            engine.kernels.utils.swiglu(
                stream,
                Engine::cfg1d(N_FF_EXP, 256),
                &self.shared_gate,
                &self.shared_up,
                &mut self.shared_mid,
                SWIGLU_CLAMP,
                1.0,
            )?;
            // down_shexp: shared_out ← shared_mid × W_down_shexp  [N_FF_EXP → N_EMBD]
            {
                let w = upload(
                    engine,
                    &lw.ffn_down_shexp,
                    &format!("blk.{il}.ffn_down_shexp"),
                )?;
                let blocks = (N_FF_EXP / 32) as u64;
                engine.kernels.quantize.quantize_q8_0(
                    stream,
                    LaunchConfig {
                        grid_dim: (blocks as u32, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &self.shared_mid,
                    &mut self.xq_i8,
                    &mut self.xscale,
                    N_FF_EXP as u64,
                    blocks,
                )?;
                engine.kernels.matmul.matmul_q8_0_preq_warp8(
                    stream,
                    Engine::cfg_warp8(N_EMBD, 1),
                    &w,
                    &self.xq_i8,
                    &self.xscale,
                    &mut self.shared_out,
                    N_FF_EXP as u64,
                    N_EMBD as u64,
                    blocks,
                )?;
            }
        }

        // ── h. MoE router + experts ──────────────────────────────────────
        if lw.ffn_gate_inp.is_some() {
            // router_logits ← ffn_norm × W_ffn_gate_inp  [N_EMBD → N_EXPERT]
            // ffn_gate_inp is F16; use matmul_f16 with raw f32 ffn_norm input.
            {
                let w = upload(engine, &lw.ffn_gate_inp, &format!("blk.{il}.ffn_gate_inp"))?;
                engine.kernels.matmul.matmul_f16(
                    stream,
                    LaunchConfig {
                        grid_dim: (N_EXPERT as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    buf_as_u16(&w),
                    &self.ffn_norm,
                    &mut self.router_logits,
                    N_EMBD as u64,
                    N_EXPERT as u64,
                    1,
                )?;
            }
            // router_select_parallel: probs, selected, weights ← f(router_logits, bias, ...)
            {
                let has_bias = if lw.ffn_exp_probs_b.is_some() {
                    1u32
                } else {
                    0u32
                };
                let bias_ref: &DeviceBuffer<f32> = if lw.ffn_exp_probs_b.is_some() {
                    buf_as_f32(upload(
                        engine,
                        &lw.ffn_exp_probs_b,
                        &format!("blk.{il}.exp_probs_b"),
                    )?)
                } else {
                    &self.router_logits // dummy — kernel ignores when has_bias=0
                };
                // empty_tokens first so it can serve as dummy for hash_ref when hash_mode=0
                let empty_tokens: DeviceBuffer<i32> =
                    DeviceBuffer::zeroed(stream, 0).context("empty_tokens alloc")?;
                let hash_ref: &DeviceBuffer<i32> = if lw.ffn_gate_tid2eid.is_some() {
                    buf_as_i32(upload(
                        engine,
                        &lw.ffn_gate_tid2eid,
                        &format!("blk.{il}.ffn_gate_tid2eid"),
                    )?)
                } else {
                    &empty_tokens // dummy — kernel ignores when hash_mode=0
                };
                engine.kernels.moe.router_select_parallel(
                    stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &self.router_logits,
                    bias_ref,
                    hash_ref,
                    &empty_tokens,
                    &mut self.router_probs,
                    &mut self.router_selected,
                    &mut self.router_weights,
                    token as i32,
                    0, // hash_rows (unused when hash_mode=0)
                    1, // n_tokens
                    has_bias,
                    0, // hash_mode=0 (standard top-k)
                )?;
            }
            // Q8_K quantize ffn_norm → moe_xq (16 blocks × 292 bytes, 1 row)
            const MOE_XQ_BLOCKS: u32 = (N_EMBD / QK_K) as u32; // 16
            const MOE_MID_BLOCKS: u32 = (N_FF_EXP / QK_K) as u32; // 8
            engine.kernels.quantize.q8k_quantize(
                stream,
                LaunchConfig {
                    grid_dim: (MOE_XQ_BLOCKS, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &self.ffn_norm,
                self.moe_xq.cu_deviceptr() as *mut u8,
                N_EMBD as u32,
                1,
            )?;

            // Gate+Up+Mid: all 6 expert slots in one kernel launch
            // W layout: [N_EXPERT × N_FF_EXP × xq_blocks × IQ2_XXS_BLOCK_BYTES]
            const IQ2_BLOCK: u64 = 66;
            const Q2K_BLOCK: u64 = 84;
            let gate_row_bytes = MOE_XQ_BLOCKS as u64 * IQ2_BLOCK; // 1056
            let gate_expert_bytes = N_FF_EXP as u64 * gate_row_bytes; // 2162688
            {
                let w_gate = upload(
                    engine,
                    &lw.ffn_gate_exps,
                    &format!("blk.{il}.ffn_gate_exps"),
                )?;
                let w_up = upload(engine, &lw.ffn_up_exps, &format!("blk.{il}.ffn_up_exps"))?;
                engine.kernels.moe.moe_gate_up_mid_qwarp32(
                    stream,
                    LaunchConfig {
                        grid_dim: ((N_FF_EXP as u32 + 127) / 128, N_EXPERT_USED as u32, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &w_gate,
                    &w_up,
                    &self.moe_xq,
                    &self.router_selected,
                    &self.router_weights,
                    &self.iq2_grid_gpu,
                    &self.iq2_signs_gpu,
                    &mut self.routed_gate,
                    &mut self.routed_up,
                    &mut self.routed_mid,
                    gate_expert_bytes,
                    gate_row_bytes,
                    MOE_XQ_BLOCKS,
                    N_FF_EXP as u32,
                    N_EXPERT_USED as u32,
                    SWIGLU_CLAMP,
                )?;
            }

            // Q8_K quantize routed_mid → moe_midq (8 blocks × 292 bytes, 6 rows)
            engine.kernels.quantize.q8k_quantize(
                stream,
                LaunchConfig {
                    grid_dim: (MOE_MID_BLOCKS, N_EXPERT_USED as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &self.routed_mid,
                self.moe_midq.cu_deviceptr() as *mut u8,
                N_FF_EXP as u32,
                N_EXPERT_USED as u32,
            )?;

            // Down projection + sum across 6 experts → routed_out[0..N_EMBD]
            // W layout: [N_EXPERT × N_EMBD × mid_blocks × Q2K_BLOCK_BYTES]
            let down_row_bytes = MOE_MID_BLOCKS as u64 * Q2K_BLOCK; // 672
            let down_expert_bytes = N_EMBD as u64 * down_row_bytes; // 2752512
            {
                let w_down = upload(
                    engine,
                    &lw.ffn_down_exps,
                    &format!("blk.{il}.ffn_down_exps"),
                )?;
                engine.kernels.moe.moe_down_sum6_qwarp32(
                    stream,
                    LaunchConfig {
                        grid_dim: ((N_EMBD as u32 + 31) / 32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &w_down,
                    &self.moe_midq,
                    &self.router_selected,
                    &mut self.routed_out,
                    down_expert_bytes,
                    down_row_bytes,
                    MOE_MID_BLOCKS,
                    N_EMBD as u32,
                )?;
            }
        }

        // ── i. FFN residual merge → after_ffn_hc ─────────────────────────
        // hc_expand: after_ffn_hc = hc_expand(shared_out [+ routed_sum], after_attn_hc, hc_split)
        // When MoE ran, shared_out + routed_out[0..N_EMBD] are added (has_add=1).
        // When no MoE (dense layers), has_add=0 and routed_out is ignored.
        let has_moe = lw.ffn_gate_inp.is_some();
        engine.kernels.hc.hc_expand(
            stream,
            Engine::cfg1d(HC_DIM, 256),
            &self.shared_out,    // block_out = shared expert output
            &self.routed_out,    // block_add = MoE expert sum (unused when has_add=0)
            &self.after_attn_hc, // residual_hc
            &self.hc_split,      // full split [pre(N_HC), post(N_HC), comb(N_HC²)]
            &mut self.after_ffn_hc,
            N_EMBD as u32,
            N_HC as u32,
            1,
            MIX_HC as u32, // mix_hc row stride
            has_moe as u32,
        )?;

        // Advance cur_hc ← after_ffn_hc for the next layer by swapping pointers.
        // Both buffers are HC_DIM f32 elements; swapping the DeviceBuffer (GPU pointer)
        // is zero-cost and correct since after_ffn_hc is not read again in this layer.
        std::mem::swap(&mut self.cur_hc, &mut self.after_ffn_hc);

        Ok(())
    }

    // ─── Output projection ────────────────────────────────────────────────────
    fn decode_output(&mut self, engine: &Engine, _pos: u32) -> Result<()> {
        let stream = &engine.stream;
        let w = &engine.weights;

        // output_hc_weights: output_weights ← sigmoid(output_hc_fn(flat_hc))
        {
            // Compute flat_hc = rms_norm_plain(cur_hc)
            engine.kernels.norm.rms_norm_plain(
                stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &self.cur_hc,
                &mut self.flat_hc,
                HC_DIM as u32,
                1,
                RMS_EPS,
            )?;
            // hc_pre for output: matmul_f16 with output_hc_fn [hc_dim → N_HC]
            // Actually the output path computes: pre[N_HC] = output_hc_fn × flat_hc
            // then output_hc_weights(pre, scale, base) → output_weights[N_HC]
            // output_hc_fn: [hc_dim × N_HC] f16 weight
            let w_fn = upload(engine, &w.output_hc_fn, "output_hc_fn")?;
            let w_scale = upload(engine, &w.output_hc_scale, "output_hc_scale")?;
            let w_base = upload(engine, &w.output_hc_base, "output_hc_base")?;
            // pre = matmul_f16(flat_hc, output_hc_fn) → [N_HC]
            // Reuse hc_mix as pre buffer (it's [MIX_HC=24], N_HC=4 fits)
            engine.kernels.matmul.matmul_f16(
                stream,
                LaunchConfig {
                    grid_dim: (N_HC as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                buf_as_u16(&w_fn),
                &self.flat_hc,
                &mut self.hc_mix, // output (only first N_HC elements used)
                HC_DIM as u64,
                N_HC as u64,
                1,
            )?;
            // output_hc_weights: output_weights ← sigmoid gating
            engine.kernels.hc.output_hc_weights(
                stream,
                Engine::cfg1d(N_HC, 256),
                &self.hc_mix,         // pre activations [N_HC]
                buf_as_f32(&w_scale), // scale [1]
                buf_as_f32(&w_base),  // base [N_HC]
                &mut self.output_weights,
                N_HC as u32,
                1,
                HC_EPS,
            )?;
        }

        // hc_weighted_sum: output_embd ← Σ_h cur_hc[h] * output_weights[h]
        engine.kernels.hc.hc_weighted_sum(
            stream,
            Engine::cfg1d(N_EMBD, 256),
            &self.cur_hc,
            &self.output_weights,
            &mut self.output_embd,
            N_EMBD as u32,
            N_HC as u32,
            1,
            N_HC as u32, // weight_stride_f32
        )?;

        // rms_norm_weight: output_norm ← rms_norm(output_embd) * W_output_norm
        {
            let w_on = upload(engine, &w.output_norm, "output_norm")?;
            engine.kernels.norm.rms_norm_weight(
                stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &self.output_embd,
                buf_as_f32(&w_on),
                &mut self.output_norm,
                N_EMBD as u32,
                1,
                RMS_EPS,
            )?;
        }

        // matmul_q8_0_preq_warp8: logits ← output_norm × W_output  [N_EMBD → N_VOCAB]
        {
            let w_out = upload(engine, &w.output, "output.weight")?;
            let blocks = (N_EMBD / 32) as u64;
            engine.kernels.quantize.quantize_q8_0(
                stream,
                LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &self.output_norm,
                &mut self.xq_i8,
                &mut self.xscale,
                N_EMBD as u64,
                blocks,
            )?;
            engine.kernels.matmul.matmul_q8_0_preq_warp8(
                stream,
                Engine::cfg_warp8(N_VOCAB, 1),
                &w_out,
                &self.xq_i8,
                &self.xscale,
                &mut self.logits_buf,
                N_EMBD as u64,
                N_VOCAB as u64,
                blocks,
            )?;
        }

        Ok(())
    }
}
