use anyhow::{Context, Result};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_host::ltoir;
/// GPU inference engine: loads the model and all compiled CUDA kernels.
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::gguf::GgufFile;
use crate::kernels::{
    attention, embed, hc, indexer, kv_cache, matmul, moe, norm, quantize, rope, utils,
};
use crate::model::ModelWeights;

/// All kernel modules loaded from the compiled PTX.
pub struct Kernels {
    pub utils: utils::utils::LoadedModule,
    pub norm: norm::norm::LoadedModule,
    pub embed: embed::embed::LoadedModule,
    pub rope: rope::rope::LoadedModule,
    pub kv_cache: kv_cache::kv_cache::LoadedModule,
    pub quantize: quantize::quantize::LoadedModule,
    pub matmul: matmul::matmul::LoadedModule,
    pub hc: hc::hc::LoadedModule,
    pub moe: moe::moe::LoadedModule,
    pub indexer: indexer::indexer::LoadedModule,
    pub attention: attention::attention::LoadedModule,
}

impl Kernels {
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        // Compile the NVVM IR (deepseek_rs.ll) → cubin via libNVVM+nvJitLink,
        // then load it.  The cubin is cached next to the .ll so subsequent
        // runs are fast.  Looks in $CARGO_MANIFEST_DIR or cwd for the .ll file.
        let m: Arc<cuda_core::CudaModule> = ltoir::load_kernel_module(ctx, "deepseek_rs")
            .context("loading CUDA module from deepseek_rs.ll / .cubin")?;

        Ok(Self {
            utils: utils::utils::from_module(Arc::clone(&m)).context("utils kernels")?,
            norm: norm::norm::from_module(Arc::clone(&m)).context("norm kernels")?,
            embed: embed::embed::from_module(Arc::clone(&m)).context("embed kernels")?,
            rope: rope::rope::from_module(Arc::clone(&m)).context("rope kernels")?,
            kv_cache: kv_cache::kv_cache::from_module(Arc::clone(&m))
                .context("kv_cache kernels")?,
            quantize: quantize::quantize::from_module(Arc::clone(&m))
                .context("quantize kernels")?,
            matmul: matmul::matmul::from_module(Arc::clone(&m)).context("matmul kernels")?,
            hc: hc::hc::from_module(Arc::clone(&m)).context("hc kernels")?,
            moe: moe::moe::from_module(Arc::clone(&m)).context("moe kernels")?,
            indexer: indexer::indexer::from_module(Arc::clone(&m)).context("indexer kernels")?,
            attention: attention::attention::from_module(Arc::clone(&m))
                .context("attention kernels")?,
        })
    }
}

/// Loaded model: GGUF file + parsed weight tensor metadata + all weights pre-uploaded to GPU.
pub struct Engine {
    pub gguf: GgufFile,
    pub weights: ModelWeights,
    /// All tensor data pre-uploaded: name → GPU buffer (raw quantised bytes).
    pub gpu: HashMap<String, DeviceBuffer<u8>>,
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub kernels: Kernels,
}

impl Engine {
    pub fn open(model_path: &Path) -> Result<Self> {
        let gguf = GgufFile::open(model_path)?;
        let weights = bind_weights(&gguf)?;

        let ctx = CudaContext::new(0).context("creating CUDA context")?;
        let stream = ctx.new_stream().context("creating CUDA stream")?;
        let kernels = Kernels::load(&ctx)?;

        // Upload every tensor to GPU once and keep it resident.
        // With 96 GB VRAM and an 81 GB model this fits comfortably.
        let n = gguf.tensors.len();
        let mut gpu = HashMap::with_capacity(n);
        for (i, t) in gguf.tensors.iter().enumerate() {
            if i % 100 == 0 {
                println!("  uploading tensors {}/{} ...", i, n);
            }
            let data = gguf.tensor_data(t);
            let buf = DeviceBuffer::from_host(&stream, data)
                .with_context(|| format!("uploading '{}'", t.name))?;
            gpu.insert(t.name.clone(), buf);
        }
        println!("  all {} tensors on GPU", n);

        Ok(Self {
            gguf,
            weights,
            gpu,
            ctx,
            stream,
            kernels,
        })
    }

    /// Return a LaunchConfig for a 1D grid covering `n` elements with `block` threads/block.
    pub fn cfg1d(n: usize, block: u32) -> LaunchConfig {
        let grid = ((n as u32) + block - 1) / block;
        LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Return a LaunchConfig for warp-per-row Q8_0 kernels.
    ///
    /// These kernels assign `row = blockIdx.x * 8 + warp_id` with 256-thread
    /// (8-warp) blocks, so the grid must be `ceil(out_dim / 8)` — NOT
    /// `ceil(out_dim / 8 / block_size)` as `cfg1d(out_dim/8, 256)` would give.
    pub fn cfg_warp8(out_dim: usize, n_tok: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((out_dim as u32 + 7) / 8, n_tok as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Return a LaunchConfig for a 2D grid.
    pub fn cfg2d(x: u32, y: u32, bx: u32, by: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((x + bx - 1) / bx, (y + by - 1) / by, 1),
            block_dim: (bx, by, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// Match GGUF tensor names to our model weight struct fields.
fn bind_weights(gguf: &GgufFile) -> Result<ModelWeights> {
    let mut w = ModelWeights::new();

    // Helper: try to find a tensor by name (returns None if absent — some
    // tensors are optional depending on layer type).
    let get = |name: &str| gguf.find(name).cloned();

    w.token_embd = get("token_embd.weight");
    w.output_norm = get("output_norm.weight");
    w.output = get("output.weight");
    w.output_hc_fn = get("output_hc_fn.weight");
    w.output_hc_scale = get("output_hc_scale.weight");
    w.output_hc_base = get("output_hc_base.weight");

    for il in 0..crate::model::N_LAYER {
        let l = &mut w.layers[il];
        let p = |s: &str| format!("blk.{il}.{s}");

        l.attn_norm = get(&p("attn_norm.weight"));
        l.attn_q_a = get(&p("attn_q_a.weight"));
        l.attn_q_a_norm = get(&p("attn_q_a_norm.weight"));
        l.attn_q_b = get(&p("attn_q_b.weight"));
        l.attn_kv = get(&p("attn_kv.weight"));
        l.attn_kv_a_norm = get(&p("attn_kv_a_norm.weight"));
        l.attn_sinks = get(&p("attn_sinks.weight"));
        l.attn_output_a = get(&p("attn_output_a.weight"));
        l.attn_output_b = get(&p("attn_output_b.weight"));
        l.attn_compressor_ape = get(&p("attn_compressor_ape.weight"));
        l.attn_compressor_kv = get(&p("attn_compressor_kv.weight"));
        l.attn_compressor_gate = get(&p("attn_compressor_gate.weight"));
        l.attn_compressor_norm = get(&p("attn_compressor_norm.weight"));
        l.hc_attn_fn = get(&p("hc_attn_fn.weight"));
        l.hc_attn_scale = get(&p("hc_attn_scale.weight"));
        l.hc_attn_base = get(&p("hc_attn_base.weight"));
        l.indexer_attn_q_b = get(&p("indexer.attn_q_b.weight"));
        l.indexer_proj = get(&p("indexer.proj.weight"));
        l.indexer_compressor_ape = get(&p("indexer_compressor_ape.weight"));
        l.indexer_compressor_kv = get(&p("indexer_compressor_kv.weight"));
        l.indexer_compressor_gate = get(&p("indexer_compressor_gate.weight"));
        l.indexer_compressor_norm = get(&p("indexer_compressor_norm.weight"));
        l.ffn_norm = get(&p("ffn_norm.weight"));
        l.ffn_gate_tid2eid = get(&p("ffn_gate_tid2eid.weight"));
        l.ffn_gate_inp = get(&p("ffn_gate_inp.weight"));
        l.ffn_exp_probs_b = None; // not present in this GGUF variant
        l.ffn_gate_exps = get(&p("ffn_gate_exps.weight"));
        l.ffn_up_exps = get(&p("ffn_up_exps.weight"));
        l.ffn_down_exps = get(&p("ffn_down_exps.weight"));
        l.ffn_gate_shexp = get(&p("ffn_gate_shexp.weight"));
        l.ffn_up_shexp = get(&p("ffn_up_shexp.weight"));
        l.ffn_down_shexp = get(&p("ffn_down_shexp.weight"));
        l.hc_ffn_fn = get(&p("hc_ffn_fn.weight"));
        l.hc_ffn_scale = get(&p("hc_ffn_scale.weight"));
        l.hc_ffn_base = get(&p("hc_ffn_base.weight"));
    }

    Ok(w)
}
