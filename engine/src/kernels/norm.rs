use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

#[cuda_module]
pub mod norm {
    use super::*;

    /// RMS-normalise every row of `x` and write to `out`.
    ///
    /// `out[row, i] = x[row, i] * rsqrt(mean(x[row]²) + eps)`
    ///
    /// Grid: (rows, 1, 1), Block: (256, 1, 1).
    /// Ported from rms_norm_plain_kernel.
    #[kernel]
    pub fn rms_norm_plain(x: &[f32], mut out: DisjointSlice<f32>, n: u32, rows: u32, eps: f32) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let tx = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        let row_off = row as usize * n as usize;

        // Compute squared-sum contribution of this thread across its stride.
        let mut sum = 0.0f32;
        let mut i = tx;
        while i < n as usize {
            let v = x[row_off + i];
            sum += v * v;
            i += bx;
        }

        // Block-tree reduction → PARTIAL[0] holds the total sum².
        // TODO(cuda-oxide): this loop is inlined in every kernel rather than extracted
        // into a shared helper because device functions that take `*mut SharedArray`
        // have calling-convention issues with static mut references in the current
        // cuda-oxide backend. Refactor into a helper once that is fixed.
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

        let scale = unsafe { (PARTIAL[0] / n as f32 + eps).sqrt().recip() };
        let mut i = tx;
        while i < n as usize {
            unsafe {
                *out.get_unchecked_mut(row_off + i) = x[row_off + i] * scale;
            }
            i += bx;
        }
    }

    /// RMS-normalise with a learned per-element weight vector.
    ///
    /// `out[row, i] = x[row, i] * w[i] * rsqrt(mean(x[row]²) + eps)`
    ///
    /// Grid: (rows, 1, 1), Block: (256, 1, 1).
    /// Ported from rms_norm_weight_kernel.
    #[kernel]
    pub fn rms_norm_weight(
        x: &[f32],
        w: &[f32],
        mut out: DisjointSlice<f32>,
        n: u32,
        rows: u32,
        eps: f32,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let tx = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        let row_off = row as usize * n as usize;

        let mut sum = 0.0f32;
        let mut i = tx;
        while i < n as usize {
            let v = x[row_off + i];
            sum += v * v;
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

        let scale = unsafe { (PARTIAL[0] / n as f32 + eps).sqrt().recip() };
        let mut i = tx;
        while i < n as usize {
            unsafe {
                *out.get_unchecked_mut(row_off + i) = x[row_off + i] * scale * w[i];
            }
            i += bx;
        }
    }

    /// Fused Q/KV RMS-norm for DeepSeek V4 attention projection.
    ///
    /// blockIdx.y == 0 → normalise the Q rows; blockIdx.y == 1 → KV rows.
    /// Allows a single kernel launch to normalise both projections in parallel.
    ///
    /// Grid: (rows, 2, 1), Block: (256, 1, 1).
    /// Ported from dsv4_qkv_rms_norm_rows_kernel.
    #[kernel]
    pub fn dsv4_qkv_rms_norm_rows(
        q: &[f32],
        q_w: &[f32],
        mut q_out: DisjointSlice<f32>,
        q_n: u32,
        kv: &[f32],
        kv_w: &[f32],
        mut kv_out: DisjointSlice<f32>,
        kv_n: u32,
        rows: u32,
        eps: f32,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let which = thread::blockIdx_y(); // 0 = Q, 1 = KV
        if row >= rows || which > 1 {
            return;
        }
        let n = if which == 0 {
            q_n as usize
        } else {
            kv_n as usize
        };
        let row_off = row as usize * n;
        let tx = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;

        let mut sum = 0.0f32;
        let mut i = tx;
        while i < n {
            let v = if which == 0 {
                q[row_off + i]
            } else {
                kv[row_off + i]
            };
            sum += v * v;
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

        let scale = unsafe { (PARTIAL[0] / n as f32 + eps).sqrt().recip() };
        let mut i = tx;
        while i < n {
            if which == 0 {
                unsafe {
                    *q_out.get_unchecked_mut(row_off + i) = q[row_off + i] * scale * q_w[i];
                }
            } else {
                unsafe {
                    *kv_out.get_unchecked_mut(row_off + i) = kv[row_off + i] * scale * kv_w[i];
                }
            }
            i += bx;
        }
    }

    /// Per-attention-head in-place RMS normalisation.
    ///
    /// Treats `x` as a flat `(n_tok * n_head, head_dim)` matrix.
    /// Each block normalises one head of one token.
    ///
    /// Grid: (n_tok * n_head, 1, 1), Block: (≤ head_dim, 1, 1).
    /// Ported from head_rms_norm_kernel.
    #[kernel]
    pub fn head_rms_norm(
        mut x: DisjointSlice<f32>,
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        eps: f32,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= n_tok * n_head {
            return;
        }
        let tx = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        let row_off = row as usize * head_dim as usize;

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
        let mut i = tx;
        while i < head_dim as usize {
            unsafe {
                let v = *x.get_unchecked_mut(row_off + i);
                *x.get_unchecked_mut(row_off + i) = v * scale;
            }
            i += bx;
        }
    }
}
