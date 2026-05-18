use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

// ============================================================================
// Device helpers
// ============================================================================

/// Compute 24-element HC split for one row: pre-weights (sigmoid), post-weights
/// (2·sigmoid), and a 4×4 comb matrix (row-softmax + Sinkhorn normalisation).
///
/// `scale[0..3]` = pre/post/comb scaling factors.
/// `base[0..24]`  = bias terms (4 pre + 4 post + 16 comb).
/// `mix[0..24]`   = raw pre-activations (input, per-row).
/// `out[0..24]`   = result (written by this function).
///
/// Ported from hc4_split_one.
#[cuda_device::device]
pub fn hc4_split_one(
    out: *mut f32,
    mix: *const f32,
    scale: *const f32,
    base: *const f32,
    sinkhorn_iters: u32,
    epsv: f32,
) {
    unsafe {
        let pre_s = *scale.add(0);
        let post_s = *scale.add(1);
        let comb_s = *scale.add(2);

        // Pre-weights: sigmoid
        for i in 0..4usize {
            let z = *mix.add(i) * pre_s + *base.add(i);
            *out.add(i) = 1.0 / (1.0 + (-z).exp()) + epsv;
        }
        // Post-weights: 2·sigmoid
        for i in 0..4usize {
            let z = *mix.add(4 + i) * post_s + *base.add(4 + i);
            *out.add(4 + i) = 2.0 / (1.0 + (-z).exp());
        }

        // Comb matrix (4×4): row-softmax then Sinkhorn
        let mut c = [0.0f32; 16];
        for r in 0..4usize {
            let mut m = f32::NEG_INFINITY;
            for col in 0..4usize {
                let v = *mix.add(8 + r * 4 + col) * comb_s + *base.add(8 + r * 4 + col);
                c[r * 4 + col] = v;
                if v > m {
                    m = v;
                }
            }
            let mut s = 0.0f32;
            for col in 0..4usize {
                let v = (c[r * 4 + col] - m).exp();
                c[r * 4 + col] = v;
                s += v;
            }
            for col in 0..4usize {
                c[r * 4 + col] = c[r * 4 + col] / s + epsv;
            }
        }
        // Initial column normalisation
        for col in 0..4usize {
            let mut s = epsv;
            for r in 0..4usize {
                s += c[r * 4 + col];
            }
            for r in 0..4usize {
                c[r * 4 + col] /= s;
            }
        }
        // Sinkhorn iterations
        let mut iter = 1u32;
        while iter < sinkhorn_iters {
            for r in 0..4usize {
                let mut s = epsv;
                for col in 0..4usize {
                    s += c[r * 4 + col];
                }
                for col in 0..4usize {
                    c[r * 4 + col] /= s;
                }
            }
            for col in 0..4usize {
                let mut s = epsv;
                for r in 0..4usize {
                    s += c[r * 4 + col];
                }
                for r in 0..4usize {
                    c[r * 4 + col] /= s;
                }
            }
            iter += 1;
        }
        for i in 0..16usize {
            *out.add(8 + i) = c[i];
        }
    }
}

/// Read one scalar from the model weight buffer (f16 or f32 depending on type).
/// Inlined conversion avoids cross-crate device-function resolution issues.
/// Ported from model_scalar_dev.
#[cuda_device::device]
pub fn model_scalar_hc(base: *const u8, offset: u64, type_: u32, idx: u64) -> f32 {
    unsafe {
        let p = base.add(offset as usize);
        if type_ == 1 {
            let bits = *(p.add(idx as usize * 2) as *const u16);
            // Inline f16_bits_to_f32
            let h = bits as u32;
            let sign = (h & 0x8000) << 16;
            let exp = (h >> 10) & 0x1F;
            let mant = h & 0x3FF;
            let b32 = if exp == 0 {
                sign
            } else if exp == 31 {
                sign | 0x7F80_0000 | (mant << 13)
            } else {
                sign | ((exp + 112) << 23) | (mant << 13)
            };
            f32::from_bits(b32)
        } else {
            *(p.add(idx as usize * 4) as *const f32)
        }
    }
}

// ============================================================================
// Kernels
// ============================================================================

#[cuda_module]
pub mod hc {
    use super::*;

    /// Apply Sinkhorn-normalised HC split to every row of `mix`.
    ///
    /// Each thread processes one row (24 floats).
    /// Grid: covers `n_rows` threads (1D).
    /// Ported from hc_split_sinkhorn_kernel.
    #[kernel]
    pub fn hc_split_sinkhorn(
        mix: &[f32],
        scale: &[f32],
        base: &[f32],
        mut out: DisjointSlice<f32>,
        n_rows: u32,
        sinkhorn_iters: u32,
        epsv: f32,
    ) {
        let row = thread::index_1d().get() as u32;
        if row >= n_rows {
            return;
        }
        let off = row as usize * 24;
        unsafe {
            super::hc4_split_one(
                out.get_unchecked_mut(off),
                mix.as_ptr().add(off),
                scale.as_ptr(),
                base.as_ptr(),
                sinkhorn_iters,
                epsv,
            );
        }
    }

    /// Weighted sum of HC residuals across heads:
    /// `out[t, d] = Σ_h x[t, h, d] * w[t, h]`
    ///
    /// Grid: covers `n_embd * n_tokens` threads (1D).
    /// Ported from hc_weighted_sum_kernel.
    #[kernel]
    pub fn hc_weighted_sum(
        x: &[f32],
        w: &[f32],
        mut out: DisjointSlice<f32>,
        n_embd: u32,
        n_hc: u32,
        n_tokens: u32,
        weight_stride_f32: u32,
    ) {
        let idx = thread::index_1d();
        let gid = idx.get() as u64;
        let n = n_embd as u64 * n_tokens as u64;
        if gid >= n {
            return;
        }
        let d = (gid % n_embd as u64) as usize;
        let t = (gid / n_embd as u64) as usize;
        let mut acc = 0.0f32;
        for h in 0..n_hc as usize {
            acc += x[t * n_hc as usize * n_embd as usize + h * n_embd as usize + d]
                * w[t * weight_stride_f32 as usize + h];
        }
        if let Some(o) = out.get_mut(idx) {
            *o = acc;
        }
    }

    /// Expand block output through the HC combination matrix.
    ///
    /// `split` is the full 24-element HC split array [pre(n_hc), post(n_hc), comb(n_hc²)].
    /// The kernel reads post-weights at `split[t*mix_hc + n_hc + dst_hc]` and
    /// comb at `split[t*mix_hc + 2*n_hc + dst_hc + src_hc*n_hc]`, matching
    /// ds4's hc_expand_split_tensor which passes `split + n_hc` and `split + 2*n_hc`.
    ///
    /// Grid: covers `n_tokens * n_hc * n_embd` threads (1D).
    /// Ported from hc_expand_kernel + hc_expand_split_tensor offset convention.
    #[kernel]
    pub fn hc_expand(
        block_out: &[f32],
        block_add: &[f32],
        residual_hc: &[f32],
        split: &[f32],
        mut out_hc: DisjointSlice<f32>,
        n_embd: u32,
        n_hc: u32,
        n_tokens: u32,
        mix_hc: u32,
        has_add: u32,
    ) {
        let idx = thread::index_1d();
        let gid = idx.get() as u64;
        let n_elem = n_tokens as u64 * n_hc as u64 * n_embd as u64;
        if gid >= n_elem {
            return;
        }
        let d = (gid % n_embd as u64) as usize;
        let tmp = gid / n_embd as u64;
        let dst_hc = (tmp % n_hc as u64) as usize;
        let t = (tmp / n_hc as u64) as usize;

        let mut block_v = block_out[t * n_embd as usize + d];
        if has_add != 0 {
            block_v += block_add[t * n_embd as usize + d];
        }
        // Post-weights live at split[t*mix_hc + n_hc .. t*mix_hc + 2*n_hc]
        let post_base = t * mix_hc as usize + n_hc as usize;
        let mut acc = block_v * split[post_base + dst_hc];
        // Comb matrix lives at split[t*mix_hc + 2*n_hc .. t*mix_hc + 2*n_hc + n_hc²]
        let comb_base = t * mix_hc as usize + 2 * n_hc as usize;
        for src_hc in 0..n_hc as usize {
            acc += split[comb_base + dst_hc + src_hc * n_hc as usize]
                * residual_hc[t * n_hc as usize * n_embd as usize + src_hc * n_embd as usize + d];
        }
        if let Some(o) = out_hc.get_mut(idx) {
            *o = acc;
        }
    }

    /// Fused HC split + weighted sum (n_hc == 4).
    ///
    /// Thread 0 computes the 24-element Sinkhorn split into `split[t*24..]`,
    /// then all threads compute the weighted residual sum.
    ///
    /// Grid: (n_rows, 1, 1), Block: (n_embd_threads, 1, 1).
    /// Ported from hc_split_weighted_sum_fused_kernel.
    #[kernel]
    pub fn hc_split_weighted_sum_fused(
        mix: &[f32],
        scale: &[f32],
        base: &[f32],
        residual_hc: &[f32],
        split: *mut f32,
        mut out: DisjointSlice<f32>,
        n_embd: u32,
        n_hc: u32,
        n_rows: u32,
        sinkhorn_iters: u32,
        epsv: f32,
    ) {
        let t = thread::blockIdx_x() as usize;
        let d = thread::threadIdx_x() as usize;
        if t as u32 >= n_rows || n_hc != 4 {
            return;
        }
        let sp = unsafe { split.add(t * 24) };
        if d == 0 {
            unsafe {
                super::hc4_split_one(
                    sp,
                    mix.as_ptr().add(t * 24),
                    scale.as_ptr(),
                    base.as_ptr(),
                    sinkhorn_iters,
                    epsv,
                );
            }
        }
        thread::sync_threads();
        let bx = thread::blockDim_x() as usize;
        let mut col = d;
        while col < n_embd as usize {
            let mut acc = 0.0f32;
            for h in 0..4usize {
                let sp_h = unsafe { *sp.add(h) };
                acc += residual_hc[t * 4 * n_embd as usize + h * n_embd as usize + col] * sp_h;
            }
            unsafe {
                *out.get_unchecked_mut(t * n_embd as usize + col) = acc;
            }
            col += bx;
        }
    }

    /// Fused HC split + weighted sum + RMS normalisation (n_hc == 4).
    ///
    /// Grid: (n_rows, 1, 1), Block: (256, 1, 1).
    /// Ported from hc_split_weighted_sum_norm_fused_kernel.
    #[kernel]
    pub fn hc_split_weighted_sum_norm_fused(
        mix: &[f32],
        scale: &[f32],
        base: &[f32],
        residual_hc: &[f32],
        norm_w: &[f32],
        split: *mut f32,
        mut out: DisjointSlice<f32>,
        mut norm_out: DisjointSlice<f32>,
        n_embd: u32,
        n_hc: u32,
        n_rows: u32,
        sinkhorn_iters: u32,
        epsv: f32,
        norm_eps: f32,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        let d = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        if t as u32 >= n_rows || n_hc != 4 {
            return;
        }

        let sp = unsafe { split.add(t * 24) };
        if d == 0 {
            unsafe {
                super::hc4_split_one(
                    sp,
                    mix.as_ptr().add(t * 24),
                    scale.as_ptr(),
                    base.as_ptr(),
                    sinkhorn_iters,
                    epsv,
                );
            }
        }
        thread::sync_threads();

        let mut sum_sq = 0.0f32;
        let mut col = d;
        while col < n_embd as usize {
            let mut acc = 0.0f32;
            for h in 0..4usize {
                let sp_h = unsafe { *sp.add(h) };
                acc += residual_hc[t * 4 * n_embd as usize + h * n_embd as usize + col] * sp_h;
            }
            unsafe {
                *out.get_unchecked_mut(t * n_embd as usize + col) = acc;
            }
            sum_sq += acc * acc;
            col += bx;
        }

        unsafe {
            PARTIAL[d] = sum_sq;
        }
        thread::sync_threads();
        let mut stride = bx >> 1;
        while stride > 0 {
            if d < stride {
                unsafe {
                    let hi = PARTIAL[d + stride];
                    PARTIAL[d] += hi;
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        let norm_scale = unsafe { (PARTIAL[0] / n_embd as f32 + norm_eps).sqrt().recip() };

        let mut col = d;
        while col < n_embd as usize {
            let off = t * n_embd as usize + col;
            unsafe {
                let v = *out.get_unchecked_mut(off);
                *norm_out.get_unchecked_mut(off) = v * norm_scale * norm_w[col];
            }
            col += bx;
        }
    }

    /// Sigmoid gate for HC output weights: `out[i] = σ(pre[i]*scale + base[h]) + epsv`.
    ///
    /// Grid: covers `n_tokens * n_hc` threads (1D).
    /// Ported from output_hc_weights_kernel.
    #[kernel]
    pub fn output_hc_weights(
        pre: &[f32],
        scale: &[f32],
        base: &[f32],
        mut out: DisjointSlice<f32>,
        n_hc: u32,
        n_tokens: u32,
        epsv: f32,
    ) {
        let idx = thread::index_1d();
        let gid = idx.get() as u32;
        if gid >= n_tokens * n_hc {
            return;
        }
        let h = (gid % n_hc) as usize;
        let z = pre[gid as usize] * scale[0] + base[h];
        if let Some(o) = out.get_mut(idx) {
            *o = 1.0 / (1.0 + (-z).exp()) + epsv;
        }
    }

    /// Softmax-pool K/V candidates from `kv`/`sc` into compressed slot `c`.
    ///
    /// Grid: (n_comp * blockDim.x / blockDim.x, n_comp, 1), Block: (head_dim_threads, 1, 1).
    /// Ported from compressor_prefill_pool_kernel.
    #[kernel]
    pub fn compressor_prefill_pool(
        kv: &[f32],
        sc: &[f32],
        state_kv: &[f32],
        state_score: &[f32],
        model_map: *const u8,
        ape_offset: u64,
        ape_type: u32,
        mut comp: DisjointSlice<f32>,
        head_dim: u32,
        ratio: u32,
        pos0: u32,
        n_comp: u32,
        replay: u32,
    ) {
        let d = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let c = thread::blockIdx_y() as u32;
        if d as u32 >= head_dim || c >= n_comp {
            return;
        }
        let coff = if ratio == 4 { 2u32 } else { 1u32 };
        let width = (coff * head_dim) as usize;

        let mut vals = [0.0f32; 16];
        let mut scores = [0.0f32; 16];
        let mut max_s = f32::NEG_INFINITY;
        let mut n_cand = 0usize;

        if ratio == 4 {
            if replay != 0 && c == 0 {
                for r in 0..4usize {
                    vals[n_cand] = state_kv[r * width + d];
                    scores[n_cand] = state_score[r * width + d];
                    if scores[n_cand] > max_s {
                        max_s = scores[n_cand];
                    }
                    n_cand += 1;
                }
            } else if c > 0 {
                let base_t = ((c - 1) * ratio) as usize;
                for r in 0..4usize {
                    let t = base_t + r;
                    let ape = super::model_scalar_hc(
                        model_map,
                        ape_offset,
                        ape_type,
                        ((pos0 as usize + t) % ratio as usize * width + d) as u64,
                    );
                    vals[n_cand] = kv[t * width + d];
                    scores[n_cand] = sc[t * width + d] + ape;
                    if scores[n_cand] > max_s {
                        max_s = scores[n_cand];
                    }
                    n_cand += 1;
                }
            }
            let base_t = (c * ratio) as usize;
            for r in 0..4usize {
                let t = base_t + r;
                let ape = super::model_scalar_hc(
                    model_map,
                    ape_offset,
                    ape_type,
                    ((pos0 as usize + t) % ratio as usize * width + head_dim as usize + d) as u64,
                );
                vals[n_cand] = kv[t * width + head_dim as usize + d];
                scores[n_cand] = sc[t * width + head_dim as usize + d] + ape;
                if scores[n_cand] > max_s {
                    max_s = scores[n_cand];
                }
                n_cand += 1;
            }
        } else {
            let base_t = (c * ratio) as usize;
            for r in 0..ratio as usize {
                let t = base_t + r;
                let ape = super::model_scalar_hc(
                    model_map,
                    ape_offset,
                    ape_type,
                    ((pos0 as usize + t) % ratio as usize * width + d) as u64,
                );
                vals[n_cand] = kv[t * width + d];
                scores[n_cand] = sc[t * width + d] + ape;
                if scores[n_cand] > max_s {
                    max_s = scores[n_cand];
                }
                n_cand += 1;
            }
        }

        let mut den = 0.0f32;
        let mut acc = 0.0f32;
        for i in 0..n_cand {
            let w = (scores[i] - max_s).exp();
            den += w;
            acc += vals[i] * w;
        }
        unsafe {
            *comp.get_unchecked_mut(c as usize * head_dim as usize + d) =
                if den != 0.0 { acc / den } else { 0.0 };
        }
    }

    /// Softmax-pool the current compressor state window into `row[d]`.
    ///
    /// Grid: covers `head_dim` threads (1D).
    /// Ported from compressor_update_pool_kernel.
    #[kernel]
    pub fn compressor_update_pool(
        state_kv: &[f32],
        state_score: &[f32],
        mut row: DisjointSlice<f32>,
        head_dim: u32,
        ratio: u32,
    ) {
        let idx = thread::index_1d();
        let d = idx.get();
        if d as u32 >= head_dim {
            return;
        }
        let coff = if ratio == 4 { 2u32 } else { 1u32 };
        let width = (coff * head_dim) as usize;

        let mut vals = [0.0f32; 16];
        let mut scores = [0.0f32; 16];
        let mut max_s = f32::NEG_INFINITY;
        let mut n_cand = 0usize;

        if ratio == 4 {
            for r in 0..4usize {
                vals[n_cand] = state_kv[r * width + d];
                scores[n_cand] = state_score[r * width + d];
                if scores[n_cand] > max_s {
                    max_s = scores[n_cand];
                }
                n_cand += 1;
            }
            for r in 0..4usize {
                vals[n_cand] = state_kv[(ratio as usize + r) * width + head_dim as usize + d];
                scores[n_cand] = state_score[(ratio as usize + r) * width + head_dim as usize + d];
                if scores[n_cand] > max_s {
                    max_s = scores[n_cand];
                }
                n_cand += 1;
            }
        } else {
            for r in 0..ratio as usize {
                vals[n_cand] = state_kv[r * width + d];
                scores[n_cand] = state_score[r * width + d];
                if scores[n_cand] > max_s {
                    max_s = scores[n_cand];
                }
                n_cand += 1;
            }
        }

        let mut den = 0.0f32;
        let mut acc = 0.0f32;
        for i in 0..n_cand {
            let w = (scores[i] - max_s).exp();
            den += w;
            acc += vals[i] * w;
        }
        if let Some(o) = row.get_mut(idx) {
            *o = if den != 0.0 { acc / den } else { 0.0 };
        }
    }

    /// Shift the top half of the ratio-4 compressor state into the bottom half.
    ///
    /// Grid: covers `4 * width` threads (1D).
    /// Ported from compressor_shift_ratio4_kernel.
    #[kernel]
    pub fn compressor_shift_ratio4(
        mut state_kv: DisjointSlice<f32>,
        mut state_score: DisjointSlice<f32>,
        width: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u64;
        let half = 4u64 * width as u64;
        if i >= half {
            return;
        }
        let v = unsafe { *state_kv.get_unchecked_mut((half + i) as usize) };
        let s = unsafe { *state_score.get_unchecked_mut((half + i) as usize) };
        unsafe {
            *state_kv.get_unchecked_mut(i as usize) = v;
            *state_score.get_unchecked_mut(i as usize) = s;
            *state_kv.get_unchecked_mut((half + i) as usize) = v;
            *state_score.get_unchecked_mut((half + i) as usize) = s;
        }
    }

    /// Store compressed KV state from `kv`/`sc` into `state_kv`/`state_score`.
    /// Adds APE bias from the model map to the score.
    ///
    /// Grid: covers n_tokens * width threads (1D). Block: any.
    /// Ported from compressor_store_kernel.
    #[kernel]
    pub fn compressor_store(
        kv: &[f32],
        sc: &[f32],
        mut state_kv: DisjointSlice<f32>,
        mut state_score: DisjointSlice<f32>,
        model_map: *const u8,
        ape_offset: u64,
        ape_type: u32,
        head_dim: u32,
        ratio: u32,
        pos0: u32,
        n_tokens: u32,
    ) {
        let idx = thread::index_1d();
        let gid = idx.get() as u64;
        let coff: u32 = if ratio == 4 { 2 } else { 1 };
        let width = coff * head_dim;
        let n = n_tokens as u64 * width as u64;
        if gid >= n {
            return;
        }
        let t = (gid / width as u64) as u32;
        let j = gid - t as u64 * width as u64;
        let pos_mod = (pos0 + t) % ratio;
        let dst_row: u32 = if ratio == 4 { ratio + pos_mod } else { pos_mod };
        let dst = dst_row as u64 * width as u64 + j;
        let src = t as u64 * width as u64 + j;
        let ape = super::model_scalar_hc(
            model_map,
            ape_offset,
            ape_type,
            pos_mod as u64 * width as u64 + j,
        );
        unsafe {
            *state_kv.get_unchecked_mut(dst as usize) = kv[src as usize];
            *state_score.get_unchecked_mut(dst as usize) = sc[src as usize] + ape;
        }
    }

    /// Copy rows [src0, src0+rows) from kv/sc into state rows [dst0, dst0+rows),
    /// adding APE bias at the destination phase.
    ///
    /// Grid: covers rows * width threads (1D). Block: any.
    /// Ported from compressor_set_rows_kernel.
    #[kernel]
    pub fn compressor_set_rows(
        kv: &[f32],
        sc: &[f32],
        mut state_kv: DisjointSlice<f32>,
        mut state_score: DisjointSlice<f32>,
        model_map: *const u8,
        ape_offset: u64,
        ape_type: u32,
        width: u32,
        ratio: u32,
        pos0: u32,
        src0: u32,
        dst0: u32,
        rows: u32,
    ) {
        let idx = thread::index_1d();
        let gid = idx.get() as u64;
        let n = rows as u64 * width as u64;
        if gid >= n {
            return;
        }
        let r = (gid / width as u64) as u32;
        let j = gid - r as u64 * width as u64;
        let src = src0 + r;
        let dst = dst0 + r;
        let phase = (pos0 + src) % ratio;
        let ape = super::model_scalar_hc(
            model_map,
            ape_offset,
            ape_type,
            phase as u64 * width as u64 + j,
        );
        let src_off = src as u64 * width as u64 + j;
        let dst_off = dst as u64 * width as u64 + j;
        unsafe {
            *state_kv.get_unchecked_mut(dst_off as usize) = kv[src_off as usize];
            *state_score.get_unchecked_mut(dst_off as usize) = sc[src_off as usize] + ape;
        }
    }
}
