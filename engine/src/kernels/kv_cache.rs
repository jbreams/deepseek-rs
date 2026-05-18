use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kv_cache {
    use super::*;

    /// Store a batch of KV tokens into the circular raw-window buffer,
    /// quantising through FP16 and immediately dequantising back to FP32
    /// (i.e. rounding each value to FP16 precision).
    ///
    /// The raw buffer is a circular queue of size `raw_cap` rows.
    /// Token `t` is stored at row `(pos0 + t) % raw_cap`.
    ///
    /// Grid: covers `n_tokens * head_dim` threads (1D).
    /// Ported from store_raw_kv_batch_kernel.
    #[kernel]
    pub fn store_raw_kv_batch(
        kv: &[f32],
        mut raw: DisjointSlice<f32>,
        raw_cap: u32,
        pos0: u32,
        n_tokens: u32,
        head_dim: u32,
    ) {
        let gid = thread::index_1d().get() as u64;
        let n = n_tokens as u64 * head_dim as u64;
        if gid >= n {
            return;
        }
        let d = (gid % head_dim as u64) as usize;
        let t = (gid / head_dim as u64) as u32;
        let row = ((pos0 + t) % raw_cap) as usize;
        let dst = row * head_dim as usize + d;
        // Round through f16 precision (matching the C code's __half2float(__float2half(x)) pattern)
        let v = super::super::utils::f16_bits_to_f32(super::super::utils::f32_to_f16_bits(
            kv[gid as usize],
        ));
        unsafe {
            *raw.get_unchecked_mut(dst) = v;
        }
    }
}
