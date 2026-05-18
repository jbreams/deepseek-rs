/// Fixed model architecture constants for DeepSeek V4 Flash 284B.
/// These are validated against the GGUF metadata on load.
pub const N_LAYER: usize = 43;
pub const N_EMBD: usize = 4096;
pub const N_VOCAB: usize = 129280;
pub const N_HEAD: usize = 64;
pub const N_HEAD_KV: usize = 1;
pub const N_HEAD_DIM: usize = 512;
pub const N_VALUE_DIM: usize = 512;
pub const N_ROT: usize = 64;
pub const N_NOPE: usize = N_HEAD_DIM - N_ROT; // 448
pub const N_OUT_GROUP: usize = 8; // GQA output heads per KV head
pub const N_LORA_Q: usize = 1024;
pub const N_LORA_O: usize = 1024;
pub const N_EXPERT: usize = 256;
pub const N_EXPERT_USED: usize = 6;
pub const N_EXPERT_SHARED: usize = 1;
pub const N_FF_EXP: usize = 2048;
pub const N_HASH_LAYER: usize = 3; // first 3 layers use sliding-window attn
pub const N_SWA: usize = 128; // sliding window size for those layers
pub const N_INDEXER_HEAD: usize = 64;
pub const N_INDEXER_HEAD_DIM: usize = 128;
pub const N_INDEXER_TOP_K: usize = 512;
pub const N_HC: usize = 4; // hierarchical-compression heads
pub const N_HC_SINKHORN_ITER: usize = 20;
pub const QK_K: usize = 256; // quantization super-block size

/// GGUF tensor type IDs (matching ds4's DS4_TENSOR_* enum).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    F32 = 0,
    F16 = 1,
    Q8_0 = 8,
    Q2K = 10,
    Q4K = 12,
    Iq2Xxs = 16,
    I32 = 26,
}

impl TensorType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            8 => Some(Self::Q8_0),
            10 => Some(Self::Q2K),
            12 => Some(Self::Q4K),
            16 => Some(Self::Iq2Xxs),
            26 => Some(Self::I32),
            _ => None,
        }
    }

    /// Bytes per element (or per packed unit for block quantizations).
    /// For block formats, returns bytes per QK_K-element block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q8_0 => 34,   // u16 scale + i8×32 per 32-elem block
            Self::Q2K => 84,    // scales[16] + qs[64] + u16 d + u16 dmin per 256-elem block
            Self::Q4K => 144,   // scales[12] + qs[128] + u16 d + u16 dmin per 256-elem block
            Self::Iq2Xxs => 66, // u16 d + u16 qs[32] per 256-elem block
            Self::I32 => 4,
        }
    }

    /// Elements per block (1 for scalar types, QK_K for block quants).
    pub fn elems_per_block(self) -> usize {
        match self {
            Self::Q8_0 => 32,
            Self::Q2K | Self::Q4K | Self::Iq2Xxs => QK_K,
            _ => 1,
        }
    }

    /// Total bytes for `n_elems` elements of this type.
    pub fn byte_size(self, n_elems: usize) -> usize {
        let blocks = (n_elems + self.elems_per_block() - 1) / self.elems_per_block();
        blocks * self.block_bytes()
    }
}

/// Metadata for one tensor loaded from the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub ty: TensorType,
    pub dims: Vec<u64>, // fastest-varying last (row-major)
    pub offset: u64,    // byte offset from the GGUF data section start
}

impl TensorMeta {
    /// Total number of elements (product of dims).
    pub fn n_elems(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Byte size of this tensor's data.
    pub fn byte_size(&self) -> usize {
        self.ty.byte_size(self.n_elems() as usize)
    }
}

/// All weight tensors needed for one transformer layer.
#[derive(Default, Debug)]
pub struct LayerWeights {
    // Hierarchical compression — attention
    pub hc_attn_fn: Option<TensorMeta>,
    pub hc_attn_scale: Option<TensorMeta>,
    pub hc_attn_base: Option<TensorMeta>,
    // Attention
    pub attn_norm: Option<TensorMeta>,
    pub attn_q_a: Option<TensorMeta>,
    pub attn_q_a_norm: Option<TensorMeta>,
    pub attn_q_b: Option<TensorMeta>,
    pub attn_kv: Option<TensorMeta>,
    pub attn_kv_a_norm: Option<TensorMeta>,
    pub attn_sinks: Option<TensorMeta>,
    pub attn_output_a: Option<TensorMeta>,
    pub attn_output_b: Option<TensorMeta>,
    // Attention compressor
    pub attn_compressor_ape: Option<TensorMeta>,
    pub attn_compressor_kv: Option<TensorMeta>,
    pub attn_compressor_gate: Option<TensorMeta>,
    pub attn_compressor_norm: Option<TensorMeta>,
    // Indexer
    pub indexer_attn_q_b: Option<TensorMeta>,
    pub indexer_proj: Option<TensorMeta>,
    pub indexer_compressor_ape: Option<TensorMeta>,
    pub indexer_compressor_kv: Option<TensorMeta>,
    pub indexer_compressor_gate: Option<TensorMeta>,
    pub indexer_compressor_norm: Option<TensorMeta>,
    // Hierarchical compression — FFN
    pub hc_ffn_fn: Option<TensorMeta>,
    pub hc_ffn_scale: Option<TensorMeta>,
    pub hc_ffn_base: Option<TensorMeta>,
    // FFN / MoE
    pub ffn_norm: Option<TensorMeta>,
    pub ffn_gate_tid2eid: Option<TensorMeta>,
    pub ffn_gate_inp: Option<TensorMeta>,
    pub ffn_exp_probs_b: Option<TensorMeta>,
    pub ffn_gate_exps: Option<TensorMeta>,
    pub ffn_up_exps: Option<TensorMeta>,
    pub ffn_down_exps: Option<TensorMeta>,
    // Shared expert
    pub ffn_gate_shexp: Option<TensorMeta>,
    pub ffn_up_shexp: Option<TensorMeta>,
    pub ffn_down_shexp: Option<TensorMeta>,
}

/// All weight tensors for the full model.
#[derive(Default, Debug)]
pub struct ModelWeights {
    pub token_embd: Option<TensorMeta>,
    pub output_hc_base: Option<TensorMeta>,
    pub output_hc_fn: Option<TensorMeta>,
    pub output_hc_scale: Option<TensorMeta>,
    pub output_norm: Option<TensorMeta>,
    pub output: Option<TensorMeta>,
    pub layers: Vec<LayerWeights>,
}

impl ModelWeights {
    pub fn new() -> Self {
        let mut w = Self::default();
        w.layers = (0..N_LAYER).map(|_| LayerWeights::default()).collect();
        w
    }
}
