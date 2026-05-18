use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

// ============================================================================
// Device helpers
// ============================================================================

/// Return the positive float value encoded by FP8 E4M3FN index `i` (0-126).
/// Ported from dsv4_e4m3fn_value_dev.
#[cuda_device::device]
pub fn e4m3fn_value(i: i32) -> f32 {
    let exp = (i >> 3) & 15;
    let mant = i & 7;
    if exp == 0 {
        mant as f32 * 0.001_953_125 // mant / 512
    } else {
        (1.0 + mant as f32 * 0.125) * (2.0f32).powi(exp - 7)
    }
}

/// Round-trip quantise a float to the nearest E4M3FN value via binary search.
/// Operates in the positive domain; sign is handled by the caller.
/// Ported from dsv4_e4m3fn_dequant_dev.
#[cuda_device::device]
pub fn e4m3fn_dequant(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0f32 } else { 1.0 };
    let ax_raw = if x < 0.0 { -x } else { x };
    let ax = if ax_raw < 448.0 { ax_raw } else { 448.0 };
    let mut lo = 0i32;
    let mut hi = 126i32;
    while lo < hi {
        let mid = (lo + hi + 1) >> 1;
        if e4m3fn_value(mid) <= ax {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let mut best = lo;
    if best < 126 {
        let bd = {
            let d = ax - e4m3fn_value(best);
            if d < 0.0 { -d } else { d }
        };
        let nd = {
            let d = ax - e4m3fn_value(best + 1);
            if d < 0.0 { -d } else { d }
        };
        if nd < bd || (nd == bd && ((best + 1) & 1 == 0) && (best & 1 != 0)) {
            best += 1;
        }
    }
    sign * e4m3fn_value(best)
}

// ============================================================================
// Kernels
// ============================================================================

#[cuda_module]
pub mod quantize {
    use super::*;

    /// In-place FP8 (E4M3FN) round-trip quantisation of the non-RoPE
    /// (`n_nope`) dimensions of a KV head buffer.
    ///
    /// For each 64-element chunk of `n_nope`:
    ///   1. Compute the chunk maximum (shared-memory tree).
    ///   2. Derive a power-of-two scale = 2^⌈log2(max / 448)⌉.
    ///   3. Quantise each element to the nearest E4M3FN value, rescaled.
    ///
    /// Grid: (n_tok, 1, 1), Block: (64, 1, 1).
    /// Ported from fp8_kv_quantize_kernel.
    #[kernel]
    pub fn fp8_kv_quantize(mut x: DisjointSlice<f32>, _n_tok: u32, head_dim: u32, n_rot: u32) {
        static mut SCRATCH: SharedArray<f32, 64> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let n_nope = (head_dim - n_rot) as usize;
        let row_off = row * head_dim as usize;

        let mut off = 0usize;
        while off < n_nope {
            let v = if off + tid < n_nope {
                unsafe { *x.get_unchecked_mut(row_off + off + tid) }
            } else {
                0.0
            };
            let abs_v = if v < 0.0 { -v } else { v };
            unsafe {
                SCRATCH[tid] = if off + tid < n_nope { abs_v } else { 0.0 };
            }
            thread::sync_threads();

            // Max reduction in shared memory
            let mut stride = 32usize;
            while stride > 0 {
                if tid < stride {
                    unsafe {
                        let a = SCRATCH[tid];
                        let b = SCRATCH[tid + stride];
                        SCRATCH[tid] = if a > b { a } else { b };
                    }
                }
                thread::sync_threads();
                stride >>= 1;
            }

            let max_val = unsafe {
                let m = SCRATCH[0];
                if m < 1.0e-4 { 1.0e-4 } else { m }
            };
            let scale = (max_val / 448.0).log2().ceil().exp2();

            if off + tid < n_nope {
                let clamped = {
                    let q = v / scale;
                    if q > 448.0 {
                        448.0
                    } else if q < -448.0 {
                        -448.0
                    } else {
                        q
                    }
                };
                let q = super::e4m3fn_dequant(clamped) * scale;
                unsafe {
                    *x.get_unchecked_mut(row_off + off + tid) = q;
                }
            }
            thread::sync_threads();
            off += 64;
        }
    }

    /// Quantise pre-computed activations to Q8_0 format (separate quantised
    /// values and per-block scales).
    ///
    /// Q8_0 block size = 32 elements.
    /// Outputs:
    ///   - `xq[tok * blocks * 32 + b * 32 + tid]` : `i8` quantised value
    ///   - `xscale[tok * blocks + b]`              : `f32` scale factor
    ///
    /// Grid: (blocks, n_tok, 1), Block: (32, 1, 1).
    /// Ported from quantize_q8_0_f32_kernel.
    #[kernel]
    pub fn quantize_q8_0(
        x: &[f32],
        mut xq: DisjointSlice<i8>,
        mut xscale: DisjointSlice<f32>,
        in_dim: u64,
        blocks: u64,
    ) {
        static mut VALS: SharedArray<f32, 32> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as u64;
        let tok = thread::blockIdx_y() as u64;
        let tid = thread::threadIdx_x() as usize;
        if b >= blocks {
            return;
        }
        let i0 = b * 32;
        let bn = if in_dim - i0 < 32 {
            (in_dim - i0) as usize
        } else {
            32
        };
        let xr_off = (tok * in_dim + i0) as usize;

        // Each thread finds |x[i]|; threads past bn contribute 0
        let v = if tid < bn { x[xr_off + tid] } else { 0.0 };
        let a = if v < 0.0 { -v } else { v };
        unsafe {
            VALS[tid] = a;
        }
        thread::sync_threads();

        // Max reduction (16-way tree for 32 threads)
        let mut stride = 16usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    let hi = VALS[tid + stride];
                    if hi > VALS[tid] {
                        VALS[tid] = hi;
                    }
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }

        let d = unsafe { VALS[0] } / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        if tid == 0 {
            let scale_idx = (tok * blocks + b) as usize;
            unsafe {
                *xscale.get_unchecked_mut(scale_idx) = d;
            }
        }

        // Write quantised value (or 0 for padding threads)
        let xq_idx = ((tok * blocks + b) * 32) as usize + tid;
        let qv = if tid < bn {
            let raw = v * id;
            let rounded = if raw >= 0.0 {
                (raw + 0.5) as i32
            } else {
                (raw - 0.5) as i32
            };
            if rounded > 127 {
                127i8
            } else if rounded < -128 {
                -128i8
            } else {
                rounded as i8
            }
        } else {
            0i8
        };
        unsafe {
            *xq.get_unchecked_mut(xq_idx) = qv;
        }
    }

    /// Dequantise Q8_0 weights to raw FP16 bit patterns (`u16`).
    ///
    /// Q8_0 block layout: `[u16 scale (f16 bits)] [i8×32 quantised values]`
    ///                   = 34 bytes per block.
    ///
    /// `out[gid] = f16(scale * q)` (stored as raw u16).
    ///
    /// Grid: covers `in_dim * out_dim` threads (1D).
    /// Ported from dequant_q8_0_to_f16_kernel.
    #[kernel]
    pub fn dequant_q8_0_to_f16(
        w: &[u8],
        mut out: DisjointSlice<u16>,
        in_dim: u64,
        out_dim: u64,
        blocks: u64,
    ) {
        let idx = thread::index_1d();
        let gid = idx.get() as u64;
        let n = in_dim * out_dim;
        if gid >= n {
            return;
        }
        let row = gid / in_dim;
        let i = gid - row * in_dim;
        let b = i / 32;
        let j = (i - b * 32) as usize;
        let blk_off = ((row * blocks + b) * 34) as usize;
        let scale_f32 = super::super::utils::f16_bits_to_f32(
            (w[blk_off] as u16) | ((w[blk_off + 1] as u16) << 8),
        );
        let q = w[blk_off + 2 + j] as i8 as f32;
        let val_f32 = scale_f32 * q;
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = super::super::utils::f32_to_f16_bits(val_f32);
        }
    }

    /// Dequantise Q8_0 weights to FP32.
    ///
    /// `out[gid] = f32(scale * q)` where `scale` is stored as f16 in the block.
    ///
    /// Grid: covers `in_dim * out_dim` threads (1D).
    /// Ported from dequant_q8_0_to_f32_kernel.
    #[kernel]
    pub fn dequant_q8_0_to_f32(
        w: &[u8],
        mut out: DisjointSlice<f32>,
        in_dim: u64,
        out_dim: u64,
        blocks: u64,
    ) {
        let idx = thread::index_1d();
        let gid = idx.get() as u64;
        let n = in_dim * out_dim;
        if gid >= n {
            return;
        }
        let row = gid / in_dim;
        let i = gid - row * in_dim;
        let b = i / 32;
        let j = (i - b * 32) as usize;
        let blk_off = ((row * blocks + b) * 34) as usize;
        let scale = super::super::utils::f16_bits_to_f32(
            (w[blk_off] as u16) | ((w[blk_off + 1] as u16) << 8),
        );
        let q = w[blk_off + 2 + j] as i8 as f32;
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = scale * q;
        }
    }

    /// Quantise activations to Q8_K format (256-element blocks).
    ///
    /// Q8_K block layout (292 bytes, written into raw `u8` output):
    ///   offset 0  : f32 scale `d`
    ///   offset 4  : i8[256] quantised values `qs`
    ///   offset 260: i16[16] row partial sums `bsums` (16 elements × 16 per sum)
    ///
    /// Grid: (in_dim/256, n_rows, 1), Block: (256, 1, 1).
    /// Ported from q8_K_quantize_kernel.
    #[kernel]
    pub fn q8k_quantize(x: &[f32], out: *mut u8, in_dim: u32, n_rows: u32) {
        static mut ABS_PART: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut VAL_PART: SharedArray<f32, 256> = SharedArray::UNINIT;

        const QK_K: u32 = 256;
        const BLOCK_BYTES: usize = 4 + 256 + 32; // f32 d + i8[256] qs + i16[16] bsums

        let b = thread::blockIdx_x();
        let row = thread::blockIdx_y();
        let n_blocks = in_dim / QK_K;
        if row >= n_rows || b >= n_blocks {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let xr_off = (row as u64 * in_dim as u64 + b as u64 * QK_K as u64) as usize;

        let v = if (tid as u32) < QK_K {
            x[xr_off + tid]
        } else {
            0.0
        };
        let abs_v = if v < 0.0 { -v } else { v };
        unsafe {
            ABS_PART[tid] = if (tid as u32) < QK_K { abs_v } else { 0.0 };
            VAL_PART[tid] = v;
        }
        thread::sync_threads();

        // Max-absolute-value reduction (pick element with larger abs, break ties with value)
        let mut stride = thread::blockDim_x() as usize >> 1;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    if ABS_PART[tid + stride] > ABS_PART[tid] {
                        ABS_PART[tid] = ABS_PART[tid + stride];
                        VAL_PART[tid] = VAL_PART[tid + stride];
                    }
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }

        let amax = unsafe { ABS_PART[0] };
        let yb = unsafe { out.add((row as usize * n_blocks as usize + b as usize) * BLOCK_BYTES) };

        if amax == 0.0 {
            // Zero block
            if tid == 0 {
                // Write d = 0.0f32
                unsafe {
                    *(yb as *mut f32) = 0.0;
                }
            }
            if (tid as u32) < QK_K {
                unsafe {
                    *(yb.add(4).add(tid) as *mut i8) = 0;
                }
            }
            if (tid as u32) < QK_K / 16 {
                unsafe {
                    *(yb.add(4 + 256).add(tid * 2) as *mut i16) = 0;
                }
            }
            return;
        }

        // iscale = -127 / maxval (so that maxval maps to -127, preserving sign)
        let maxval = unsafe { VAL_PART[0] };
        let iscale = -127.0 / maxval;

        if tid == 0 {
            let d = 1.0 / iscale;
            unsafe {
                *(yb as *mut f32) = d;
            }
        }
        thread::sync_threads();

        if (tid as u32) < QK_K {
            let raw = iscale * x[xr_off + tid];
            let qv = if raw >= 0.0 {
                (raw + 0.5) as i32
            } else {
                (raw - 0.5) as i32
            };
            let qv = if qv > 127 {
                127i8
            } else if qv < -128 {
                -128i8
            } else {
                qv as i8
            };
            unsafe {
                *(yb.add(4).add(tid) as *mut i8) = qv;
            }
        }
        thread::sync_threads();

        if (tid as u32) < QK_K / 16 {
            let mut sum = 0i32;
            for k in 0..16usize {
                sum += unsafe { *(yb.add(4).add(tid * 16 + k) as *const i8) } as i32;
            }
            unsafe {
                *(yb.add(4 + 256).add(tid * 2) as *mut i16) = sum as i16;
            }
        }
    }
}

// ============================================================================
// IQ2_XXS / Q2_K lookup tables and dot-product device helpers
// ============================================================================
// Note: cuda-oxide only supports zero-initialized device statics, so the IQ2_XXS
// grid and sign tables are NOT module-level statics. Instead they are embedded as
// host-side consts (visible to host code) and the device functions take them as
// pointer parameters. The MoE kernels pass these tables as `&[u8]` / `&[u64]` slices.

/// IQ2_XXS quantisation grid (256 entries × 8 bytes).
/// Ported from cuda_iq2xxs_grid in ds4_iq2_tables_cuda.inc.
/// Host-side constant — passed to device as a kernel slice argument.
pub const IQ2_XXS_GRID: [u64; 256] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b2b08,
    0x08080808082b2b2b,
    0x0808080819080819,
    0x0808080819081908,
    0x0808080819190808,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b082b2b,
    0x080808082b2b082b,
    0x0808081908080819,
    0x0808081908081908,
    0x0808081908190808,
    0x0808081908191919,
    0x0808081919080808,
    0x080808192b081908,
    0x080808192b192b08,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b082b082b,
    0x0808082b2b08082b,
    0x0808190808080819,
    0x0808190808081908,
    0x0808190808190808,
    0x08081908082b0819,
    0x08081908082b1908,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819082b08,
    0x08081908192b0808,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b190808,
    0x080819082b2b1908,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908082b08,
    0x08081919082b0808,
    0x080819191908192b,
    0x08081919192b2b19,
    0x080819192b080808,
    0x080819192b190819,
    0x0808192b08082b19,
    0x0808192b08190808,
    0x0808192b19080808,
    0x0808192b2b081908,
    0x0808192b2b2b1908,
    0x08082b0808080808,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808191908,
    0x08082b08082b2b08,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b0819190808,
    0x08082b081919082b,
    0x08082b082b082b08,
    0x08082b1908081908,
    0x08082b1919080808,
    0x08082b2b0808082b,
    0x08082b2b08191908,
    0x0819080808080819,
    0x0819080808081908,
    0x0819080808190808,
    0x08190808082b0819,
    0x0819080819080808,
    0x08190808192b0808,
    0x081908082b081908,
    0x081908082b190808,
    0x081908082b191919,
    0x0819081908080808,
    0x0819081908082b08,
    0x08190819082b0808,
    0x0819081919190808,
    0x0819081919192b2b,
    0x081908192b080808,
    0x0819082b082b1908,
    0x0819082b19081919,
    0x0819190808080808,
    0x0819190808082b08,
    0x08191908082b0808,
    0x08191908082b1919,
    0x0819190819082b19,
    0x081919082b080808,
    0x0819191908192b08,
    0x08191919192b082b,
    0x0819192b08080808,
    0x0819192b0819192b,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b0808190808,
    0x08192b0819080808,
    0x08192b082b080819,
    0x08192b1908080808,
    0x08192b1908081919,
    0x08192b192b2b0808,
    0x08192b2b19190819,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808082b2b,
    0x082b080819081908,
    0x082b0808192b0819,
    0x082b08082b080808,
    0x082b08082b08082b,
    0x082b0819082b2b19,
    0x082b081919082b08,
    0x082b082b08080808,
    0x082b082b0808082b,
    0x082b190808080819,
    0x082b190808081908,
    0x082b190808190808,
    0x082b190819080808,
    0x082b19081919192b,
    0x082b191908080808,
    0x082b191919080819,
    0x082b1919192b1908,
    0x082b192b2b190808,
    0x082b2b0808082b08,
    0x082b2b08082b0808,
    0x082b2b082b191908,
    0x082b2b2b19081908,
    0x1908080808080819,
    0x1908080808081908,
    0x1908080808190808,
    0x1908080808192b08,
    0x19080808082b0819,
    0x19080808082b1908,
    0x1908080819080808,
    0x1908080819082b08,
    0x190808081919192b,
    0x19080808192b0808,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x1908081908080808,
    0x19080819082b0808,
    0x19080819192b0819,
    0x190808192b080808,
    0x190808192b081919,
    0x1908082b08080819,
    0x1908082b08190808,
    0x1908082b19082b08,
    0x1908082b1919192b,
    0x1908082b192b2b08,
    0x1908190808080808,
    0x1908190808082b08,
    0x19081908082b0808,
    0x190819082b080808,
    0x190819082b192b19,
    0x190819190819082b,
    0x19081919082b1908,
    0x1908192b08080808,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808190808,
    0x19082b0819080808,
    0x19082b0819081919,
    0x19082b1908080808,
    0x19082b1919192b08,
    0x19082b19192b0819,
    0x19082b192b08082b,
    0x19082b2b19081919,
    0x19082b2b2b190808,
    0x1919080808080808,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808192b19,
    0x19190808082b0808,
    0x191908082b080808,
    0x191908082b082b08,
    0x1919081908081908,
    0x191908191908082b,
    0x191908192b2b1908,
    0x1919082b2b190819,
    0x191919082b190808,
    0x191919082b19082b,
    0x1919191908082b2b,
    0x1919192b08080819,
    0x1919192b19191908,
    0x19192b0808080808,
    0x19192b0808190819,
    0x19192b0808192b19,
    0x19192b08192b1908,
    0x19192b1919080808,
    0x19192b2b08082b08,
    0x192b080808081908,
    0x192b080808190808,
    0x192b080819080808,
    0x192b0808192b2b08,
    0x192b081908080808,
    0x192b081919191919,
    0x192b082b08192b08,
    0x192b082b192b0808,
    0x192b190808080808,
    0x192b190808081919,
    0x192b191908190808,
    0x192b19190819082b,
    0x192b19192b081908,
    0x192b2b081908082b,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808082b2b,
    0x2b08080819080819,
    0x2b0808082b08082b,
    0x2b08081908081908,
    0x2b08081908192b08,
    0x2b08081919080808,
    0x2b08082b08190819,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b08190808190808,
    0x2b08190808191919,
    0x2b08190819080808,
    0x2b081908192b0808,
    0x2b08191908080808,
    0x2b0819191908192b,
    0x2b0819192b191908,
    0x2b08192b08082b19,
    0x2b08192b19080808,
    0x2b08192b192b0808,
    0x2b082b080808082b,
    0x2b082b1908081908,
    0x2b082b2b08190819,
    0x2b19080808081908,
    0x2b19080808190808,
    0x2b190808082b1908,
    0x2b19080819080808,
    0x2b1908082b2b0819,
    0x2b1908190819192b,
    0x2b1908192b080808,
    0x2b19082b19081919,
    0x2b19190808080808,
    0x2b191908082b082b,
    0x2b19190819081908,
    0x2b19191919190819,
    0x2b192b082b080819,
    0x2b192b19082b0808,
    0x2b2b08080808082b,
    0x2b2b080819190808,
    0x2b2b08082b081919,
    0x2b2b081908082b19,
    0x2b2b082b08080808,
    0x2b2b190808192b08,
    0x2b2b2b0819190808,
    0x2b2b2b1908081908,
];

/// IQ2_XXS sign table (128 entries).
/// Ported from cuda_ksigns_iq2xs in ds4_cuda.cu.
/// Host-side constant — passed to device as a kernel slice argument.
pub const IQ2_SIGNS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149,
    150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170,
    43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212,
    85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232,
    105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123,
    252, 125, 126, 255,
];

/// Parity-correct an 8-bit sign mask: flip bit 7 if popcount is odd.
#[cuda_device::device]
pub fn unpack_iq2_signs(sign: u8) -> u8 {
    let p = sign.count_ones() & 1;
    sign ^ ((p as u8) << 7)
}

/// Dot product of 8 int8 values decoded from one IQ2_XXS grid entry with 8 Q8_K int8 values.
/// `sign_byte` is the raw value from the IQ2_SIGNS table (parity-corrected inside).
/// `q8` points to 8 consecutive i8 values.
/// TODO(cuda-oxide): implements __dp4a+__vsub4+__vcmpne4 in scalar because those
/// CUDA SIMD intrinsics are not yet available in cuda-oxide device code.
#[cuda_device::device]
pub fn dot_iq2_dp8(grid: u64, sign_byte: u8, q8: *const i8) -> i32 {
    let sb = unpack_iq2_signs(sign_byte);
    let mut acc = 0i32;
    let mut i = 0usize;
    while i < 8 {
        let g = ((grid >> (i * 8)) & 0xFF) as u8;
        let s = (sb >> i) & 1 != 0;
        let w: i8 = if s { -(g as i8) } else { g as i8 };
        acc += w as i32 * unsafe { *q8.add(i) } as i32;
        i += 1;
    }
    acc
}

/// Read a little-endian u16 from a `*const u8` pointer (avoids read_unaligned).
#[cuda_device::device]
pub fn read_u16_le(p: *const u8) -> u16 {
    let lo = unsafe { *p } as u16;
    let hi = unsafe { *p.add(1) } as u16;
    lo | (hi << 8)
}

/// Read a little-endian f32 from a `*const u8` pointer (avoids read_unaligned).
#[cuda_device::device]
pub fn read_f32_le(p: *const u8) -> f32 {
    let b0 = unsafe { *p } as u32;
    let b1 = unsafe { *p.add(1) } as u32;
    let b2 = unsafe { *p.add(2) } as u32;
    let b3 = unsafe { *p.add(3) } as u32;
    f32::from_bits(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
}

/// Read a little-endian i16 from a `*const u8` pointer (avoids read_unaligned).
#[cuda_device::device]
pub fn read_i16_le(p: *const u8) -> i16 {
    let lo = unsafe { *p } as u16;
    let hi = unsafe { *p.add(1) } as u16;
    (lo | (hi << 8)) as i16
}

/// Dot product of one IQ2_XXS block (66 bytes: u16 d | u16 qs[32])
/// with one Q8_K block (292 bytes: f32 d | i8 qs[256] | i16 bsums[16]).
/// Both arguments are raw byte pointers to the block data.
/// `grid` and `signs` are the IQ2_XXS_GRID and IQ2_SIGNS tables (passed by caller).
/// CUDA_QK_K = 256, so each IQ2_XXS block covers 256 2-bit elements.
/// TODO(cuda-oxide): scalar fallback — see dot_iq2_dp8 for intrinsics note.
#[cuda_device::device]
pub fn dot_iq2_xxs_q8k(
    iq2_blk: *const u8,
    q8k_blk: *const u8,
    grid: *const u64,
    signs: *const u8,
) -> f32 {
    // IQ2_XXS block: offset 0 = u16 d (f16), offset 2 = u16 qs[32]
    let xd = super::utils::f16_bits_to_f32(read_u16_le(iq2_blk));
    let yd = read_f32_le(q8k_blk);
    // q8 data starts at offset 4 in Q8_K block
    let q8_base: *const i8 = unsafe { q8k_blk.add(4) as *const i8 };
    // qs data starts at offset 2 in IQ2_XXS block (u16 pairs, byte-addressed)
    let q2_base: *const u8 = unsafe { iq2_blk.add(2) };

    let mut bsum = 0i32;
    let mut ib32 = 0usize;
    while ib32 < 8 {
        // CUDA_QK_K/32 = 8
        // Each group is 4 u16 values = 8 bytes; ib32*4 u16s = ib32*8 bytes
        let q2 = unsafe { q2_base.add(ib32 * 8) };
        let w0 = read_u16_le(q2) as u32;
        let w1 = read_u16_le(unsafe { q2.add(2) }) as u32;
        let w2 = read_u16_le(unsafe { q2.add(4) }) as u32;
        let w3 = read_u16_le(unsafe { q2.add(6) }) as u32;
        let aux0 = w0 | (w1 << 16);
        let aux1 = w2 | (w3 << 16);
        let ls = (2 * (aux1 >> 28) + 1) as i32;
        let a0 = (aux0 & 0xFF) as usize;
        let a1 = ((aux0 >> 8) & 0xFF) as usize;
        let a2 = ((aux0 >> 16) & 0xFF) as usize;
        let a3 = ((aux0 >> 24) & 0xFF) as usize;
        let s0 = unsafe { *signs.add(((aux1 >> 0) & 127) as usize) };
        let s1 = unsafe { *signs.add(((aux1 >> 7) & 127) as usize) };
        let s2 = unsafe { *signs.add(((aux1 >> 14) & 127) as usize) };
        let s3 = unsafe { *signs.add(((aux1 >> 21) & 127) as usize) };
        let g0 = unsafe { *grid.add(a0) };
        let g1 = unsafe { *grid.add(a1) };
        let g2 = unsafe { *grid.add(a2) };
        let g3 = unsafe { *grid.add(a3) };
        let q8 = unsafe { q8_base.add(ib32 * 32) };
        let mut sumi = 0i32;
        sumi += dot_iq2_dp8(g0, s0, q8);
        sumi += dot_iq2_dp8(g1, s1, unsafe { q8.add(8) });
        sumi += dot_iq2_dp8(g2, s2, unsafe { q8.add(16) });
        sumi += dot_iq2_dp8(g3, s3, unsafe { q8.add(24) });
        bsum += sumi * ls;
        ib32 += 1;
    }
    0.125 * xd * yd * bsum as f32
}

/// Dot product of 16 Q2_K elements with 16 Q8_K int8 values.
/// q2 points to raw Q2_K quantized bytes (2 bits per element, packed).
/// shift selects which 2-bit field (0, 2, 4, or 6).
#[cuda_device::device]
pub fn dot_q2_16(q2: *const u8, q8: *const i8, shift: i32) -> i32 {
    // TODO(cuda-oxide): scalar fallback for __dp4a (SIMD 4-way int8 dot product)
    let mut sum = 0i32;
    let mut i = 0usize;
    while i < 16 {
        // Read 4 bytes individually and combine (avoids *const i32 cast + read_unaligned)
        let b0 = unsafe { *q2.add(i) } as u32;
        let b1 = unsafe { *q2.add(i + 1) } as u32;
        let b2 = unsafe { *q2.add(i + 2) } as u32;
        let b3 = unsafe { *q2.add(i + 3) } as u32;
        let raw = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)) as i32;
        let v_packed = (raw >> shift) & 0x03030303i32;
        let mut j = 0;
        while j < 4 {
            let q2v = ((v_packed >> (j * 8)) & 0x03) as i32;
            let q8v = unsafe { *q8.add(i + j) } as i32;
            sum += q2v * q8v;
            j += 1;
        }
        i += 4;
    }
    sum
}

/// Dot product of one Q2_K block (84 bytes) with one Q8_K block (292 bytes).
/// Q2_K layout: u8 scales[16] | u8 qs[64] | u16 d | u16 dmin
/// Q8_K layout: f32 d | i8 qs[256] | i16 bsums[16]
#[cuda_device::device]
pub fn dot_q2k_q8k(q2k_blk: *const u8, q8k_blk: *const u8) -> f32 {
    let scales: *const u8 = q2k_blk; // offset 0
    let q2_ptr: *const u8 = unsafe { q2k_blk.add(16) }; // offset 16
    let xd_bits = read_u16_le(unsafe { q2k_blk.add(80) });
    let xdmin_bits = read_u16_le(unsafe { q2k_blk.add(82) });
    let yd = read_f32_le(q8k_blk);
    let q8_base: *const i8 = unsafe { q8k_blk.add(4) as *const i8 };
    // bsums at offset 260 in Q8_K block, i16[16] — read via byte helpers
    let bsums_base: *const u8 = unsafe { q8k_blk.add(260) };

    let dall = yd * super::utils::f16_bits_to_f32(xd_bits);
    let dmin = yd * super::utils::f16_bits_to_f32(xdmin_bits);

    // summs = Σ bsums[j] * (scales[j] >> 4)
    let mut summs = 0i32;
    let mut j = 0usize;
    while j < 16 {
        let bsum_j = read_i16_le(unsafe { bsums_base.add(j * 2) }) as i32;
        summs += bsum_j * (unsafe { (*scales.add(j) >> 4) as i32 });
        j += 1;
    }

    let mut isum = 0i32;
    let mut is = 0usize;
    let mut k = 0usize;
    let mut q2_off = 0usize;
    let mut q8_off = 0usize;
    while k < 2 {
        // CUDA_QK_K/128 = 2
        let mut shift = 0i32;
        let mut j = 0usize;
        while j < 4 {
            let d = unsafe { (*scales.add(is) & 0x0f) as i32 };
            is += 1;
            isum += d * dot_q2_16(
                unsafe { q2_ptr.add(q2_off) },
                unsafe { q8_base.add(q8_off) },
                shift,
            );
            let d2 = unsafe { (*scales.add(is) & 0x0f) as i32 };
            is += 1;
            isum += d2
                * dot_q2_16(
                    unsafe { q2_ptr.add(q2_off + 16) },
                    unsafe { q8_base.add(q8_off + 16) },
                    shift,
                );
            shift += 2;
            q8_off += 32;
            j += 1;
        }
        q2_off += 32;
        k += 1;
    }
    dall * isum as f32 - dmin * summs as f32
}
