use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

// F16 weights are passed as `&[u16]` (raw bit patterns) to avoid pulling in
// x86 intrinsics from the `half` crate.  See utils.rs for the conversion helpers.

#[cuda_module]
pub mod embed {
    use super::*;

    /// Embed a single token into an HC (hierarchical-compression) output buffer.
    ///
    /// The embedding table `w` is stored as raw f16 bit patterns (`u16`);
    /// output `out` is FP32.  The token's row is broadcast across all `n_hc`
    /// HC heads: `out[hc * n_embd + e] = f32(w[token * n_embd + e])`.
    ///
    /// Grid: covers `n_embd * n_hc` threads.
    /// Ported from embed_token_hc_kernel.
    #[kernel]
    pub fn embed_token_hc(
        w: &[u16],
        mut out: DisjointSlice<f32>,
        token: u32,
        n_embd: u32,
        n_hc: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let n = n_embd as usize * n_hc as usize;
        if i >= n {
            return;
        }
        if let Some(out_elem) = out.get_mut(idx) {
            let e = i % n_embd as usize;
            *out_elem =
                super::super::utils::f16_bits_to_f32(w[token as usize * n_embd as usize + e]);
        }
    }

    /// Embed a batch of tokens into an HC output buffer.
    ///
    /// Output layout: `out[t * n_hc * n_embd + hc * n_embd + d]`
    /// where the same embedding row is replicated across all `n_hc` heads.
    ///
    /// `tokens` values are clamped to `[0, n_vocab)`.
    ///
    /// Grid: covers `n_tokens * n_hc * n_embd` threads.
    /// Ported from embed_tokens_hc_kernel.
    #[kernel]
    pub fn embed_tokens_hc(
        tokens: &[i32],
        w: &[u16],
        mut out: DisjointSlice<f32>,
        n_vocab: u32,
        n_tokens: u32,
        n_embd: u32,
        n_hc: u32,
    ) {
        let idx = thread::index_1d();
        let gid = idx.get();
        let n = n_tokens as usize * n_hc as usize * n_embd as usize;
        if gid >= n {
            return;
        }

        let d = gid % n_embd as usize;
        let tmp = gid / n_embd as usize;
        let t = tmp / n_hc as usize;

        let tok_i = tokens[t];
        let tok = if tok_i < 0 {
            0usize
        } else {
            let t = tok_i as usize;
            if t >= n_vocab as usize { 0 } else { t }
        };

        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = super::super::utils::f16_bits_to_f32(w[tok * n_embd as usize + d]);
        }
    }
}
