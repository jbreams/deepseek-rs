use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

// ============================================================================
// Device helpers
// ============================================================================

/// Warp-wide reduction of `v` (32 lanes → lane 0 holds the sum).
/// Ported from warp_sum_f32.
#[cuda_device::device]
pub fn warp_sum_f32(mut v: f32) -> f32 {
    v += warp::shuffle_down_f32(v, 16);
    v += warp::shuffle_down_f32(v, 8);
    v += warp::shuffle_down_f32(v, 4);
    v += warp::shuffle_down_f32(v, 2);
    v += warp::shuffle_down_f32(v, 1);
    v
}

/// Scalar dot product of `n` int8 values accessed via raw pointers.
///
/// TODO(cuda-oxide): Two workarounds in play here:
///   1. Takes `*const i8` instead of `&[i8]` because `core::slice::from_raw_parts`
///      calls ptr::metadata which NVPTX cannot lower. Fix when from_raw_parts works.
///   2. Declared as safe `fn` (not `unsafe fn`) because marking a `#[cuda_device::device]`
///      function unsafe causes the macro to emit an unsafe host wrapper that fails to
///      compile. Fix when the macro handles unsafe device functions correctly.
/// Callers are responsible for pointer validity.
/// Ported from the plain path of dot_i8_block.
#[cuda_device::device]
pub fn dot_i8(a: *const i8, b: *const i8, n: usize) -> i32 {
    let mut dot = 0i32;
    let mut i = 0usize;
    while i < n {
        unsafe {
            dot += *a.add(i) as i32 * *b.add(i) as i32;
        }
        i += 1;
    }
    dot
}

// ============================================================================
// Kernels
// ============================================================================

#[cuda_module]
pub mod matmul {
    use super::*;

    /// FP16-weight × FP32-input matrix multiply: `out[tok, row] = dot(w[row], x[tok])`.
    ///
    /// Weights `w` are stored as raw f16 bit patterns (u16).
    ///
    /// Grid: (out_dim, n_tok, 1), Block: (256, 1, 1).
    /// Ported from matmul_f16_kernel.
    #[kernel]
    pub fn matmul_f16(
        w: &[u16],
        x: &[f32],
        mut out: DisjointSlice<f32>,
        in_dim: u64,
        out_dim: u64,
        n_tok: u64,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as u64;
        let tok = thread::blockIdx_y() as u64;
        if row >= out_dim || tok >= n_tok {
            return;
        }
        let tx = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        let wr_off = (row * in_dim) as usize;
        let xr_off = (tok * in_dim) as usize;

        let mut sum = 0.0f32;
        let mut i = tx;
        while i < in_dim as usize {
            sum += super::super::utils::f16_bits_to_f32(w[wr_off + i]) * x[xr_off + i];
            i += bx;
        }

        unsafe {
            PARTIAL[tx] = sum;
        }
        thread::sync_threads();
        let mut stride = bx >> 1;
        while stride > 0 {
            if tx < stride {
                unsafe {
                    let hi = PARTIAL[tx + stride];
                    PARTIAL[tx] += hi;
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }

        if tx == 0 {
            unsafe {
                *out.get_unchecked_mut((tok * out_dim + row) as usize) = PARTIAL[0];
            }
        }
    }

    /// Single-threaded (serial) FP16-weight × FP32-input dot product.
    /// Used as a fallback for very small matrices or when block-level
    /// parallelism is unnecessary.
    ///
    /// Grid: (out_dim, n_tok, 1), Block: (1, 1, 1).
    /// Ported from matmul_f16_serial_kernel.
    #[kernel]
    pub fn matmul_f16_serial(
        w: &[u16],
        x: &[f32],
        mut out: DisjointSlice<f32>,
        in_dim: u64,
        out_dim: u64,
        n_tok: u64,
    ) {
        let row = thread::blockIdx_x() as u64;
        let tok = thread::blockIdx_y() as u64;
        if row >= out_dim || tok >= n_tok || thread::threadIdx_x() != 0 {
            return;
        }
        let wr_off = (row * in_dim) as usize;
        let xr_off = (tok * in_dim) as usize;
        let mut sum = 0.0f32;
        let mut i = 0usize;
        while i < in_dim as usize {
            sum += super::super::utils::f16_bits_to_f32(w[wr_off + i]) * x[xr_off + i];
            i += 1;
        }
        unsafe {
            *out.get_unchecked_mut((tok * out_dim + row) as usize) = sum;
        }
    }

    /// Ordered-chunk FP16-weight × FP32-input matrix multiply.
    ///
    /// 32 threads each own a contiguous chunk `[k0, k1)` of the input
    /// dimension; they accumulate in registers, write to shared memory,
    /// and thread 0 performs the final serial sum.
    ///
    /// Grid: (out_dim, n_tok, 1), Block: (32, 1, 1).
    /// Ported from matmul_f16_ordered_chunks_kernel.
    #[kernel]
    pub fn matmul_f16_ordered_chunks(
        w: &[u16],
        x: &[f32],
        mut out: DisjointSlice<f32>,
        in_dim: u64,
        out_dim: u64,
        n_tok: u64,
    ) {
        static mut PARTIAL: SharedArray<f32, 32> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as u64;
        let tok = thread::blockIdx_y() as u64;
        if row >= out_dim || tok >= n_tok {
            return;
        }
        let tid = thread::threadIdx_x() as u64;
        let wr_off = (row * in_dim) as usize;
        let xr_off = (tok * in_dim) as usize;
        let chunk = (in_dim + 31) / 32;
        let k0 = (tid * chunk) as usize;
        let k1 = {
            let end = k0 + chunk as usize;
            if end > in_dim as usize {
                in_dim as usize
            } else {
                end
            }
        };

        let mut sum = 0.0f32;
        let mut i = k0;
        while i < k1 {
            sum += super::super::utils::f16_bits_to_f32(w[wr_off + i]) * x[xr_off + i];
            i += 1;
        }
        unsafe {
            PARTIAL[tid as usize] = sum;
        }
        thread::sync_threads();

        if tid == 0 {
            let mut total = 0.0f32;
            let mut j = 0usize;
            while j < 32 {
                total += unsafe { PARTIAL[j] };
                j += 1;
            }
            unsafe {
                *out.get_unchecked_mut((tok * out_dim + row) as usize) = total;
            }
        }
    }

    /// Dual ordered-chunk FP16-weight matrix multiply (single token).
    ///
    /// Computes two matrix-vector products (`w0 × x` and `w1 × x`) in one
    /// kernel to amortise memory bandwidth.  Only the per-row outputs for
    /// the active dimension(s) are written.
    ///
    /// Grid: (max(out0_dim, out1_dim), 1, 1), Block: (32, 1, 1).
    /// Ported from matmul_f16_pair_ordered_chunks_kernel.
    #[kernel]
    pub fn matmul_f16_pair_ordered_chunks(
        w0: &[u16],
        w1: &[u16],
        x: &[f32],
        mut out0: DisjointSlice<f32>,
        mut out1: DisjointSlice<f32>,
        in_dim: u64,
        out0_dim: u64,
        out1_dim: u64,
    ) {
        static mut PARTIAL0: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut PARTIAL1: SharedArray<f32, 32> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as u64;
        if row >= out0_dim && row >= out1_dim {
            return;
        }
        let tid = thread::threadIdx_x() as u64;
        let chunk = (in_dim + 31) / 32;
        let k0 = (tid * chunk) as usize;
        let k1 = {
            let end = k0 + chunk as usize;
            if end > in_dim as usize {
                in_dim as usize
            } else {
                end
            }
        };

        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let wr0_off = if row < out0_dim {
            (row * in_dim) as usize
        } else {
            0
        };
        let wr1_off = if row < out1_dim {
            (row * in_dim) as usize
        } else {
            0
        };

        let mut i = k0;
        while i < k1 {
            let xv = x[i];
            if row < out0_dim {
                sum0 += super::super::utils::f16_bits_to_f32(w0[wr0_off + i]) * xv;
            }
            if row < out1_dim {
                sum1 += super::super::utils::f16_bits_to_f32(w1[wr1_off + i]) * xv;
            }
            i += 1;
        }
        unsafe {
            PARTIAL0[tid as usize] = sum0;
            PARTIAL1[tid as usize] = sum1;
        }
        thread::sync_threads();

        if tid == 0 {
            let mut total0 = 0.0f32;
            let mut total1 = 0.0f32;
            let mut j = 0usize;
            while j < 32 {
                unsafe {
                    total0 += PARTIAL0[j];
                    total1 += PARTIAL1[j];
                }
                j += 1;
            }
            if row < out0_dim {
                unsafe {
                    *out0.get_unchecked_mut(row as usize) = total0;
                }
            }
            if row < out1_dim {
                unsafe {
                    *out1.get_unchecked_mut(row as usize) = total1;
                }
            }
        }
    }

    /// FP32-weight × FP32-input matrix multiply.
    ///
    /// Grid: (out_dim, n_tok, 1), Block: (256, 1, 1).
    /// Ported from matmul_f32_kernel.
    #[kernel]
    pub fn matmul_f32(
        w: &[f32],
        x: &[f32],
        mut out: DisjointSlice<f32>,
        in_dim: u64,
        out_dim: u64,
        n_tok: u64,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as u64;
        let tok = thread::blockIdx_y() as u64;
        if row >= out_dim || tok >= n_tok {
            return;
        }
        let tx = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        let wr_off = (row * in_dim) as usize;
        let xr_off = (tok * in_dim) as usize;

        let mut sum = 0.0f32;
        let mut i = tx;
        while i < in_dim as usize {
            sum += w[wr_off + i] * x[xr_off + i];
            i += bx;
        }

        unsafe {
            PARTIAL[tx] = sum;
        }
        thread::sync_threads();
        let mut stride = bx >> 1;
        while stride > 0 {
            if tx < stride {
                unsafe {
                    let hi = PARTIAL[tx + stride];
                    PARTIAL[tx] += hi;
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }

        if tx == 0 {
            unsafe {
                *out.get_unchecked_mut((tok * out_dim + row) as usize) = PARTIAL[0];
            }
        }
    }

    /// Warp-per-row Q8_0-weight × pre-quantised-input matrix-vector multiply.
    ///
    /// Q8_0 block layout (34 bytes): `[u16 f16-scale | i8×32 quantised]`.
    /// The block is already quantised to Q8_0 (`xq`/`xscale`), one row per warp.
    ///
    /// Grid: (out_dim / 8, 1, 1), Block: (256, 1, 1) [8 warps × 32 lanes].
    /// Ported from matmul_q8_0_preq_warp8_kernel.
    #[kernel]
    pub fn matmul_q8_0_preq_warp8(
        w: &[u8],
        xq: &[i8],
        xscale: &[f32],
        mut out: DisjointSlice<f32>,
        in_dim: u64,
        out_dim: u64,
        blocks: u64,
    ) {
        let row = (thread::blockIdx_x() as u64) * 8 + (thread::threadIdx_x() as u64 >> 5);
        let lane = (thread::threadIdx_x() & 31) as usize;
        if row >= out_dim {
            return;
        }
        let wr_off = (row * blocks * 34) as usize;
        let mut acc = 0.0f32;
        let mut b = lane;
        while (b as u64) < blocks {
            let i0 = (b as u64 * 32) as usize;
            let bn = if in_dim as usize - i0 < 32 {
                in_dim as usize - i0
            } else {
                32
            };
            let blk_off = wr_off + b * 34;
            let scale = super::super::utils::f16_bits_to_f32(
                (w[blk_off] as u16) | ((w[blk_off + 1] as u16) << 8),
            );
            // Use w.as_ptr().add() so the compiler emits a GEP instruction
            // rather than materialising a 1-byte alloca.  The old form
            // `&w[blk_off+2] as *const u8 as *const i8` causes the compiler to
            // load a single byte from global memory, spill it to a local alloca,
            // then hand that 1-byte stack address to dot_i8, which then reads 32
            // bytes of stack garbage for elements 1..31.
            let dot = super::dot_i8(
                unsafe { w.as_ptr().add(blk_off + 2) as *const i8 },
                unsafe { xq.as_ptr().add(b * 32) },
                bn,
            );
            acc += scale * xscale[b] * dot as f32;
            b += 32;
        }
        acc = super::warp_sum_f32(acc);
        if lane == 0 {
            unsafe {
                *out.get_unchecked_mut(row as usize) = acc;
            }
        }
    }

    /// Batched warp-per-row Q8_0 matrix multiply (multiple tokens).
    ///
    /// Grid: (out_dim / 8, n_tok, 1), Block: (256, 1, 1).
    /// Ported from matmul_q8_0_preq_batch_warp8_kernel.
    #[kernel]
    pub fn matmul_q8_0_preq_batch_warp8(
        w: &[u8],
        xq: &[i8],
        xscale: &[f32],
        mut out: DisjointSlice<f32>,
        in_dim: u64,
        out_dim: u64,
        n_tok: u64,
        blocks: u64,
    ) {
        let row = (thread::blockIdx_x() as u64) * 8 + (thread::threadIdx_x() as u64 >> 5);
        let tok = thread::blockIdx_y() as u64;
        let lane = (thread::threadIdx_x() & 31) as usize;
        if row >= out_dim || tok >= n_tok {
            return;
        }
        let wr_off = (row * blocks * 34) as usize;
        let xq_off = (tok * blocks * 32) as usize;
        let xs_off = (tok * blocks) as usize;
        let mut acc = 0.0f32;
        let mut b = lane;
        while (b as u64) < blocks {
            let i0 = (b as u64 * 32) as usize;
            let bn = if in_dim as usize - i0 < 32 {
                in_dim as usize - i0
            } else {
                32
            };
            let blk_off = wr_off + b * 34;
            let scale = super::super::utils::f16_bits_to_f32(
                (w[blk_off] as u16) | ((w[blk_off + 1] as u16) << 8),
            );
            let dot = super::dot_i8(
                unsafe { w.as_ptr().add(blk_off + 2) as *const i8 },
                unsafe { xq.as_ptr().add(xq_off + b * 32) },
                bn,
            );
            acc += scale * xscale[xs_off + b] * dot as f32;
            b += 32;
        }
        acc = super::warp_sum_f32(acc);
        if lane == 0 {
            unsafe {
                *out.get_unchecked_mut((tok * out_dim + row) as usize) = acc;
            }
        }
    }

    /// Grouped warp-per-row Q8_0 down-projection for the attention output LoRA.
    ///
    /// Each output row `r` belongs to group `g = r / rank`.  The weight layout
    /// is standard sequential (`w[r * blocks * 34 ..]`), but the quantised input
    /// row is `xq[g * blocks * 32 ..]` (one quantised slice per group, not per
    /// output row).  This mirrors `grouped_q8_0_a_preq_warp8_kernel` in ds4.
    ///
    /// Grid: (low_dim / 8, 1, 1), Block: (256, 1, 1).
    /// Ported from grouped_q8_0_a_preq_warp8_kernel (n_tokens=1 path).
    #[kernel]
    pub fn matmul_q8_0_grouped_preq_warp8(
        w: &[u8],
        xq: &[i8],      // layout: [n_groups * blocks * 32]
        xscale: &[f32], // layout: [n_groups * blocks]
        mut out: DisjointSlice<f32>,
        group_dim: u64, // elements per group in the input
        rank: u64,      // output rows per group (N_LORA_O)
        n_groups: u32,  // number of head groups (N_OUT_GROUP)
        blocks: u64,    // blocks per group = (group_dim + 31) / 32
    ) {
        let row = (thread::blockIdx_x() as u64) * 8 + (thread::threadIdx_x() as u64 >> 5);
        let lane = (thread::threadIdx_x() & 31) as usize;
        let low_dim = n_groups as u64 * rank;
        if row >= low_dim {
            return;
        }

        let group = row / rank;
        let wr_off = (row * blocks * 34) as usize;
        let xq_off = (group * blocks * 32) as usize;
        let xs_off = (group * blocks) as usize;

        let mut acc = 0.0f32;
        let mut b = lane;
        while (b as u64) < blocks {
            let i0 = (b as u64 * 32) as usize;
            let bn = if group_dim as usize - i0 < 32 {
                group_dim as usize - i0
            } else {
                32
            };
            let blk_off = wr_off + b * 34;
            let scale = super::super::utils::f16_bits_to_f32(
                (w[blk_off] as u16) | ((w[blk_off + 1] as u16) << 8),
            );
            let dot = super::dot_i8(
                unsafe { w.as_ptr().add(blk_off + 2) as *const i8 },
                unsafe { xq.as_ptr().add(xq_off + b * 32) },
                bn,
            );
            acc += scale * xscale[xs_off + b] * dot as f32;
            b += 32;
        }
        acc = super::warp_sum_f32(acc);
        if lane == 0 {
            unsafe {
                *out.get_unchecked_mut(row as usize) = acc;
            }
        }
    }

    /// Warp-per-row Q8_0 matrix multiply with simultaneous HC expansion.
    ///
    /// After computing the warp-reduced dot product, lane 0 writes the result
    /// back and also computes the HC-expanded output by blending with the
    /// `split` tensor and the residual HC state.
    ///
    /// Grid: (out_dim / 8, 1, 1), Block: (256, 1, 1).
    /// Ported from matmul_q8_0_hc_expand_preq_warp8_kernel.
    #[kernel]
    pub fn matmul_q8_0_hc_expand_preq_warp8(
        w: &[u8],
        xq: &[i8],
        xscale: &[f32],
        residual_hc: &[f32],
        split: &[f32],
        mut block_out: DisjointSlice<f32>,
        block_add: &[f32],
        mut out_hc: DisjointSlice<f32>,
        in_dim: u64,
        out_dim: u64,
        n_embd: u32,
        n_hc: u32,
        blocks: u64,
        has_add: u32,
    ) {
        let row = (thread::blockIdx_x() as u64) * 8 + (thread::threadIdx_x() as u64 >> 5);
        let lane = (thread::threadIdx_x() & 31) as usize;
        if row >= out_dim {
            return;
        }
        let wr_off = (row * blocks * 34) as usize;
        let mut acc = 0.0f32;
        let mut b = lane;
        while (b as u64) < blocks {
            let i0 = (b as u64 * 32) as usize;
            let bn = if in_dim as usize - i0 < 32 {
                in_dim as usize - i0
            } else {
                32
            };
            let blk_off = wr_off + b * 34;
            let scale = super::super::utils::f16_bits_to_f32(
                (w[blk_off] as u16) | ((w[blk_off + 1] as u16) << 8),
            );
            let dot = super::dot_i8(
                unsafe { w.as_ptr().add(blk_off + 2) as *const i8 },
                unsafe { xq.as_ptr().add(b * 32) },
                bn,
            );
            acc += scale * xscale[b] * dot as f32;
            b += 32;
        }
        acc = super::warp_sum_f32(acc);

        if lane == 0 {
            let d = row as usize;
            unsafe {
                *block_out.get_unchecked_mut(d) = acc;
            }
            let mut block_v = acc;
            if has_add != 0 {
                block_v += block_add[d];
            }
            let post_off = n_hc as usize; // split[n_hc..2*n_hc]
            let comb_off = 2 * n_hc as usize; // split[2*n_hc..]
            let mut dst_hc = 0u32;
            while dst_hc < n_hc {
                let mut hc_acc = block_v * split[post_off + dst_hc as usize];
                let mut src_hc = 0u32;
                while src_hc < n_hc {
                    let comb_v =
                        split[comb_off + dst_hc as usize + (src_hc as usize * n_hc as usize)];
                    let res_v = residual_hc[src_hc as usize * n_embd as usize + d];
                    hc_acc += comb_v * res_v;
                    src_hc += 1;
                }
                unsafe {
                    *out_hc.get_unchecked_mut(dst_hc as usize * n_embd as usize + d) = hc_acc;
                }
                dst_hc += 1;
            }
        }
    }
}
