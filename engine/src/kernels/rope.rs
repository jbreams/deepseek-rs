use core::f32::consts::{LN_2, PI};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

// TODO(cuda-oxide): f32::min/max call intrinsics::minimum_number_nsz_f32 which
// NVPTX cannot lower — using explicit if-comparisons throughout this file.
// TODO(cuda-oxide): f32::powf emits llvm.pow which NVPTX cannot lower — using
// log2/exp2 via cuda_powf below until powf is supported natively.

/// Clamp `x` to `[0, 1]` then form the ramp weight for YaRN frequency blending.
/// Ported from rope_yarn_ramp_dev / rope_yarn_ramp_cpu_equiv_dev.
#[cuda_device::device]
pub fn rope_yarn_ramp(low: f32, high: f32, i0: i32) -> f32 {
    let denom = {
        let d = high - low;
        if d < 0.001 { 0.001 } else { d }
    };
    let y = ((i0 / 2) as f32 - low) / denom;
    // TODO(cuda-oxide): replace with y.clamp(0.0, 1.0) once f32::min/max are lowerable.
    let clamped = if y < 0.0 {
        0.0
    } else if y > 1.0 {
        1.0
    } else {
        y
    };
    1.0 - clamped
}

/// `a^b` via `exp2(b * log2(a))`.
/// TODO(cuda-oxide): replace with `a.powf(b)` once llvm.pow.f32 is lowerable by NVPTX.
#[cuda_device::device]
pub fn cuda_powf(a: f32, b: f32) -> f32 {
    (b * a.log2()).exp2()
}

#[cuda_module]
pub mod rope {
    use super::*;

    /// Tail-only RoPE: applies YaRN-scaled rotary embeddings to the last
    /// `n_rot` dimensions of each head, leaving the first `n_nope` dimensions
    /// untouched.
    ///
    /// Each thread handles one (x0, x1) rotation pair.
    ///
    /// Grid: covers `n_tok * n_head * (n_rot / 2)` threads (1D).
    /// Ported from rope_tail_kernel.
    #[kernel]
    pub fn rope_tail(
        mut x: DisjointSlice<f32>,
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        n_rot: u32,
        pos0: u32,
        n_ctx_orig: u32,
        inverse: i32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) {
        let gid = thread::index_1d().get() as u32;
        let half_rot = n_rot / 2;
        let pairs = n_tok * n_head * half_rot;
        if gid >= pairs {
            return;
        }
        let pair = gid % half_rot;
        let tmp = gid / half_rot;
        let h = tmp % n_head;
        let t = tmp / n_head;
        let n_nope = head_dim - n_rot;
        let i = pair * 2;

        // YaRN correction range
        let (corr0, corr1) = if ext_factor != 0.0 {
            let denom = 2.0 * freq_base.log2() * LN_2; // = 2 * ln(freq_base)
            let c0 = (n_rot as f32 * ((n_ctx_orig as f32 / (beta_fast * 2.0 * PI)).log2() * LN_2)
                / denom)
                .floor();
            let c1 = (n_rot as f32 * ((n_ctx_orig as f32 / (beta_slow * 2.0 * PI)).log2() * LN_2)
                / denom)
                .ceil();
            let c0 = if c0 < 0.0 { 0.0 } else { c0 };
            let c1 = if c1 > (n_rot - 1) as f32 {
                (n_rot - 1) as f32
            } else {
                c1
            };
            (c0, c1)
        } else {
            (0.0f32, 0.0f32)
        };

        let theta_extrap =
            (pos0 + t) as f32 * super::cuda_powf(freq_base, -(i as f32) / n_rot as f32);
        let theta_interp = freq_scale * theta_extrap;
        let (theta, mscale) = if ext_factor != 0.0 {
            let ramp_mix = super::rope_yarn_ramp(corr0, corr1, i as i32) * ext_factor;
            let th = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
            let ms = attn_factor * (1.0 + 0.1 * (1.0 / freq_scale).log2() * LN_2);
            (th, ms)
        } else {
            (theta_interp, attn_factor)
        };

        let c = theta.cos() * mscale;
        let mut s = theta.sin() * mscale;
        if inverse != 0 {
            s = -s;
        }

        let base =
            ((t as u64 * n_head as u64 + h as u64) * head_dim as u64 + n_nope as u64) as usize;
        let i = i as usize;
        unsafe {
            let x0 = *x.get_unchecked_mut(base + i);
            let x1 = *x.get_unchecked_mut(base + i + 1);
            *x.get_unchecked_mut(base + i) = x0 * c - x1 * s;
            *x.get_unchecked_mut(base + i + 1) = x0 * s + x1 * c;
        }
    }

    /// Fused per-head RMS-normalisation + tail RoPE.
    ///
    /// Each block processes one head of one token:
    ///   1. RMS-normalise all `head_dim` elements.
    ///   2. Apply the scale to the first `n_nope` elements.
    ///   3. Apply YaRN-scaled RoPE rotation to the last `n_rot` elements.
    ///
    /// Grid: (n_tok * n_head, 1, 1), Block: (256, 1, 1).
    /// Ported from head_rms_norm_rope_tail_kernel.
    #[kernel]
    pub fn head_rms_norm_rope_tail(
        mut x: DisjointSlice<f32>,
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        n_rot: u32,
        pos0: u32,
        n_ctx_orig: u32,
        inverse: i32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        eps: f32,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= n_tok * n_head {
            return;
        }
        let t = row / n_head;
        let tx = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        let row_off = row as usize * head_dim as usize;

        // RMS norm: accumulate squared sum
        let mut sum = 0.0f32;
        let mut i = tx;
        while i < head_dim as usize {
            unsafe {
                let v = *x.get_unchecked_mut(row_off + i);
                sum += v * v;
            }
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
        let scale = unsafe { (PARTIAL[0] / head_dim as f32 + eps).sqrt().recip() };

        let n_nope = (head_dim - n_rot) as usize;

        // Apply scale to the no-RoPE prefix
        let mut i = tx;
        while i < n_nope {
            unsafe {
                let v = *x.get_unchecked_mut(row_off + i);
                *x.get_unchecked_mut(row_off + i) = v * scale;
            }
            i += bx;
        }

        // YaRN correction range
        let (corr0, corr1) = if ext_factor != 0.0 {
            let denom = 2.0 * freq_base.log2() * LN_2;
            let c0 = (n_rot as f32 * ((n_ctx_orig as f32 / (beta_fast * 2.0 * PI)).log2() * LN_2)
                / denom)
                .floor();
            let c1 = (n_rot as f32 * ((n_ctx_orig as f32 / (beta_slow * 2.0 * PI)).log2() * LN_2)
                / denom)
                .ceil();
            let c0 = if c0 < 0.0 { 0.0 } else { c0 };
            let c1 = if c1 > (n_rot - 1) as f32 {
                (n_rot - 1) as f32
            } else {
                c1
            };
            (c0, c1)
        } else {
            (0.0f32, 0.0f32)
        };

        // Apply scale + RoPE to the rotary tail (one pair per thread, strided)
        let half_rot = (n_rot / 2) as usize;
        let mut pair = tx;
        while pair < half_rot {
            let i = pair * 2;
            let theta_extrap =
                (pos0 + t) as f32 * super::cuda_powf(freq_base, -(i as f32) / n_rot as f32);
            let theta_interp = freq_scale * theta_extrap;
            let (theta, mscale) = if ext_factor != 0.0 {
                let ramp_mix = super::rope_yarn_ramp(corr0, corr1, i as i32) * ext_factor;
                let th = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
                let ms = attn_factor * (1.0 + 0.1 * (1.0 / freq_scale).log2() * LN_2);
                (th, ms)
            } else {
                (theta_interp, attn_factor)
            };

            let c = theta.cos() * mscale;
            let mut s = theta.sin() * mscale;
            if inverse != 0 {
                s = -s;
            }

            unsafe {
                let x0 = *x.get_unchecked_mut(row_off + n_nope + i) * scale;
                let x1 = *x.get_unchecked_mut(row_off + n_nope + i + 1) * scale;
                *x.get_unchecked_mut(row_off + n_nope + i) = x0 * c - x1 * s;
                *x.get_unchecked_mut(row_off + n_nope + i + 1) = x0 * s + x1 * c;
            }
            pair += bx;
        }
    }
}
