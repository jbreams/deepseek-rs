use core::sync::atomic::{AtomicU32, Ordering};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

// Unique names to avoid PTX symbol clashes with matmul.rs helpers.
// TODO(cuda-oxide): all #[cuda_device::device] functions share a flat PTX symbol
// namespace across the entire crate, so names must be globally unique. If cuda-oxide
// gains proper per-module namespacing, the `attn_` prefix on helpers can be dropped.

#[cuda_device::device]
pub fn attn_warp_sum(mut v: f32) -> f32 {
    v += warp::shuffle_down_f32(v, 16);
    v += warp::shuffle_down_f32(v, 8);
    v += warp::shuffle_down_f32(v, 4);
    v += warp::shuffle_down_f32(v, 2);
    v += warp::shuffle_down_f32(v, 1);
    v
}

#[cuda_device::device]
pub fn attn_warp_max(mut v: f32) -> f32 {
    let o = warp::shuffle_down_f32(v, 16);
    if o > v {
        v = o;
    }
    let o = warp::shuffle_down_f32(v, 8);
    if o > v {
        v = o;
    }
    let o = warp::shuffle_down_f32(v, 4);
    if o > v {
        v = o;
    }
    let o = warp::shuffle_down_f32(v, 2);
    if o > v {
        v = o;
    }
    let o = warp::shuffle_down_f32(v, 1);
    if o > v {
        v = o;
    }
    v
}

/// Dot product of two 4-float vectors.
#[cuda_device::device]
pub fn attn_dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// Shared memory fmax-reduction (blockDim.x elements, already stored in partial).
/// Caller must have written partial[tx] before calling and sync_threads() afterward.
/// Leaves result in partial[0].
macro_rules! block_reduce_max {
    ($partial:ident, $tx:expr, $bx:expr) => {{
        let mut stride = $bx >> 1;
        while stride > 0 {
            if $tx < stride {
                unsafe {
                    let hi = $partial[$tx + stride];
                    if hi > $partial[$tx] {
                        $partial[$tx] = hi;
                    }
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
    }};
}

macro_rules! block_reduce_sum {
    ($partial:ident, $tx:expr, $bx:expr) => {{
        let mut stride = $bx >> 1;
        while stride > 0 {
            if $tx < stride {
                unsafe {
                    let hi = $partial[$tx + stride];
                    $partial[$tx] += hi;
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
    }};
}

#[cuda_module]
pub mod attention {
    use super::*;

    // ─── Prefill: raw-window-only attention ─────────────────────────────────
    /// Scaled-dot-product attention over a raw KV window (prefill).
    ///
    /// Grid: (n_tokens, n_head, 1), Block: (128, 1, 1).
    /// Ported from attention_prefill_raw_kernel.
    #[kernel]
    pub fn prefill_raw(
        q: &[f32],
        raw_kv: &[f32],
        sinks: &[f32],
        mut heads: DisjointSlice<f32>,
        n_tokens: u32,
        window: u32,
        n_head: u32,
        head_dim: u32,
    ) {
        static mut SCORES: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut PARTIAL: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut MAX_S: SharedArray<f32, 1> = SharedArray::UNINIT;
        static mut DENOM: SharedArray<f32, 1> = SharedArray::UNINIT;

        let t = thread::blockIdx_x();
        let h = thread::blockIdx_y();
        if t >= n_tokens || h >= n_head {
            return;
        }
        let tx = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        let raw_count = if t + 1 < window { t + 1 } else { window } as usize;
        let raw_start = (t + 1 - raw_count as u32) as usize;
        let qh_off = (t as usize * n_head as usize + h as usize) * head_dim as usize;
        let scale = (head_dim as f32).sqrt().recip();

        let mut local_max = sinks[h as usize];
        let mut r = tx;
        while r < raw_count {
            let kv_off = (raw_start + r) * head_dim as usize;
            let mut dot = 0.0f32;
            for d in 0..head_dim as usize {
                dot += q[qh_off + d] * raw_kv[kv_off + d];
            }
            let s = dot * scale;
            unsafe {
                SCORES[r] = s;
            }
            if s > local_max {
                local_max = s;
            }
            r += bx;
        }
        unsafe {
            PARTIAL[tx] = local_max;
        }
        thread::sync_threads();
        block_reduce_max!(PARTIAL, tx, bx);
        if tx == 0 {
            unsafe {
                MAX_S[0] = PARTIAL[0];
            }
        }
        thread::sync_threads();

        if tx == 0 {
            let max_s = unsafe { MAX_S[0] };
            let mut den = (sinks[h as usize] - max_s).exp();
            for r in 0..raw_count {
                unsafe {
                    SCORES[r] = (SCORES[r] - max_s).exp();
                    den += SCORES[r];
                }
            }
            unsafe {
                DENOM[0] = den;
            }
        }
        thread::sync_threads();
        let denom = unsafe { DENOM[0] };

        let oh_off = (t as usize * n_head as usize + h as usize) * head_dim as usize;
        let mut d = tx;
        while d < head_dim as usize {
            let mut acc = 0.0f32;
            for r in 0..raw_count {
                acc += raw_kv[(raw_start + r) * head_dim as usize + d] * unsafe { SCORES[r] };
            }
            unsafe {
                *heads.get_unchecked_mut(oh_off + d) = acc / denom;
            }
            d += bx;
        }
    }

    // ─── Prefill: mixed raw + compressed attention ───────────────────────────
    /// Grid: (n_tokens, n_head, 1), Block: (256, 1, 1).
    /// Ported from attention_prefill_mixed_kernel.
    #[kernel]
    pub fn prefill_mixed(
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        comp_mask: &[f32],
        sinks: &[f32],
        mut heads: DisjointSlice<f32>,
        use_comp_mask: u32,
        n_tokens: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) {
        static mut SCORES: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut MAX_S: SharedArray<f32, 1> = SharedArray::UNINIT;
        static mut DENOM: SharedArray<f32, 1> = SharedArray::UNINIT;

        let t = thread::blockIdx_x();
        let h = thread::blockIdx_y();
        if t >= n_tokens || h >= n_head {
            return;
        }
        let tx = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        let raw_start = if window != 0 && t + 1 > window {
            (t + 1 - window) as usize
        } else {
            0
        };
        let raw_count = (t + 1) as usize - raw_start;
        let mut visible_comp = ((t + 1) / ratio) as usize;
        if visible_comp as u32 > n_comp {
            visible_comp = n_comp as usize;
        }
        let n_score = raw_count + visible_comp;
        let qh_off = (t as usize * n_head as usize + h as usize) * head_dim as usize;
        let scale = (head_dim as f32).sqrt().recip();

        let mut local_max = sinks[h as usize];
        let mut r = tx;
        while r < raw_count {
            let kv_off = (raw_start + r) * head_dim as usize;
            let mut dot = 0.0f32;
            for d in 0..head_dim as usize {
                dot += q[qh_off + d] * raw_kv[kv_off + d];
            }
            let s = dot * scale;
            unsafe {
                SCORES[r] = s;
            }
            if s > local_max {
                local_max = s;
            }
            r += bx;
        }
        let mut c = tx;
        while c < visible_comp {
            let add = if use_comp_mask != 0 {
                comp_mask[t as usize * n_comp as usize + c]
            } else {
                0.0
            };
            let s = if add > -1.0e20 {
                let kv_off = c * head_dim as usize;
                let mut dot = 0.0f32;
                for d in 0..head_dim as usize {
                    dot += q[qh_off + d] * comp_kv[kv_off + d];
                }
                dot * scale + add
            } else {
                f32::NEG_INFINITY
            };
            unsafe {
                SCORES[raw_count + c] = s;
            }
            if s > local_max {
                local_max = s;
            }
            c += bx;
        }

        unsafe {
            PARTIAL[tx] = local_max;
        }
        thread::sync_threads();
        block_reduce_max!(PARTIAL, tx, bx);
        if tx == 0 {
            unsafe {
                MAX_S[0] = PARTIAL[0];
            }
        }
        thread::sync_threads();
        let max_s = unsafe { MAX_S[0] };

        let mut den_local = 0.0f32;
        let mut i = tx;
        while i < n_score {
            let p = unsafe { (SCORES[i] - max_s).exp() };
            unsafe {
                SCORES[i] = p;
            }
            den_local += p;
            i += bx;
        }
        unsafe {
            PARTIAL[tx] = den_local;
        }
        thread::sync_threads();
        block_reduce_sum!(PARTIAL, tx, bx);
        if tx == 0 {
            unsafe {
                DENOM[0] = PARTIAL[0] + (sinks[h as usize] - max_s).exp();
            }
        }
        thread::sync_threads();
        let denom = unsafe { DENOM[0] };

        let oh_off = (t as usize * n_head as usize + h as usize) * head_dim as usize;
        let mut d = tx;
        while d < head_dim as usize {
            let mut acc = 0.0f32;
            for r in 0..raw_count {
                acc += raw_kv[(raw_start + r) * head_dim as usize + d] * unsafe { SCORES[r] };
            }
            for c in 0..visible_comp {
                acc += comp_kv[c * head_dim as usize + d] * unsafe { SCORES[raw_count + c] };
            }
            unsafe {
                *heads.get_unchecked_mut(oh_off + d) = acc / denom;
            }
            d += bx;
        }
    }

    // ─── Prefill: in-place softmax on pre-computed score matrix ─────────────
    /// Grid: (n_tokens, n_head, 1), Block: (256, 1, 1).
    /// Ported from attention_prefill_raw_softmax_kernel.
    #[kernel]
    pub fn prefill_raw_softmax(
        sinks: &[f32],
        mut scores: DisjointSlice<f32>,
        n_tokens: u32,
        window: u32,
        n_keys: u32,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut MAX_S: SharedArray<f32, 1> = SharedArray::UNINIT;
        static mut DENOM: SharedArray<f32, 1> = SharedArray::UNINIT;

        let t = thread::blockIdx_x();
        let h = thread::blockIdx_y();
        if t >= n_tokens {
            return;
        }
        let tx = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        let row_off = (h as usize * n_tokens as usize + t as usize) * n_keys as usize;

        let mut local_max = sinks[h as usize];
        let mut k = tx;
        while k < n_keys as usize {
            let valid = k <= t as usize && (window == 0 || t as usize - k < window as usize);
            let s = if valid {
                unsafe { *scores.get_unchecked_mut(row_off + k) }
            } else {
                f32::NEG_INFINITY
            };
            unsafe {
                *scores.get_unchecked_mut(row_off + k) = s;
            }
            if s > local_max {
                local_max = s;
            }
            k += bx;
        }
        unsafe {
            PARTIAL[tx] = local_max;
        }
        thread::sync_threads();
        block_reduce_max!(PARTIAL, tx, bx);
        if tx == 0 {
            unsafe {
                MAX_S[0] = PARTIAL[0];
            }
        }
        thread::sync_threads();
        let max_s = unsafe { MAX_S[0] };

        let mut den_local = 0.0f32;
        let mut k = tx;
        while k < n_keys as usize {
            let sv = unsafe { *scores.get_unchecked_mut(row_off + k) };
            let p = if sv.is_finite() {
                (sv - max_s).exp()
            } else {
                0.0
            };
            unsafe {
                *scores.get_unchecked_mut(row_off + k) = p;
            }
            den_local += p;
            k += bx;
        }
        unsafe {
            PARTIAL[tx] = den_local;
        }
        thread::sync_threads();
        block_reduce_sum!(PARTIAL, tx, bx);
        if tx == 0 {
            unsafe {
                DENOM[0] = PARTIAL[0] + (sinks[h as usize] - max_s).exp();
            }
        }
        thread::sync_threads();
        let denom = unsafe { DENOM[0] };

        let mut k = tx;
        while k < n_keys as usize {
            unsafe {
                *scores.get_unchecked_mut(row_off + k) /= denom;
            }
            k += bx;
        }
    }

    // ─── Prefill: in-place mixed softmax ─────────────────────────────────────
    /// Grid: (n_tokens, n_head, 1), Block: (256, 1, 1).
    /// Ported from attention_prefill_mixed_softmax_kernel.
    #[kernel]
    pub fn prefill_mixed_softmax(
        comp_mask: &[f32],
        sinks: &[f32],
        mut scores: DisjointSlice<f32>,
        use_comp_mask: u32,
        n_tokens: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_keys: u32,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut MAX_S: SharedArray<f32, 1> = SharedArray::UNINIT;
        static mut DENOM: SharedArray<f32, 1> = SharedArray::UNINIT;

        let t = thread::blockIdx_x();
        let h = thread::blockIdx_y();
        if t >= n_tokens || ratio == 0 {
            return;
        }
        let tx = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        let row_off = (h as usize * n_tokens as usize + t as usize) * n_keys as usize;
        let visible_comp = ((t + 1) / ratio) as usize;

        let mut local_max = sinks[h as usize];
        let mut k = tx;
        while k < n_keys as usize {
            let s = if k < n_tokens as usize {
                if k <= t as usize && (window == 0 || t as usize - k < window as usize) {
                    unsafe { *scores.get_unchecked_mut(row_off + k) }
                } else {
                    f32::NEG_INFINITY
                }
            } else {
                let c = k - n_tokens as usize;
                if c < n_comp as usize && c < visible_comp {
                    let add = if use_comp_mask != 0 {
                        comp_mask[t as usize * n_comp as usize + c]
                    } else {
                        0.0
                    };
                    // TODO(cuda-oxide): `{ unsafe { expr } + add }` fails because the
                    // parser closes the if-arm at `}`, leaving `+ add` orphaned. Moving
                    // `+ add` inside the unsafe block is the workaround.
                    if add > -1.0e20 {
                        unsafe { *scores.get_unchecked_mut(row_off + k) + add }
                    } else {
                        f32::NEG_INFINITY
                    }
                } else {
                    f32::NEG_INFINITY
                }
            };
            unsafe {
                *scores.get_unchecked_mut(row_off + k) = s;
            }
            if s > local_max {
                local_max = s;
            }
            k += bx;
        }
        unsafe {
            PARTIAL[tx] = local_max;
        }
        thread::sync_threads();
        block_reduce_max!(PARTIAL, tx, bx);
        if tx == 0 {
            unsafe {
                MAX_S[0] = PARTIAL[0];
            }
        }
        thread::sync_threads();
        let max_s = unsafe { MAX_S[0] };

        let mut den_local = 0.0f32;
        let mut k = tx;
        while k < n_keys as usize {
            let sv = unsafe { *scores.get_unchecked_mut(row_off + k) };
            let p = if sv.is_finite() {
                (sv - max_s).exp()
            } else {
                0.0
            };
            unsafe {
                *scores.get_unchecked_mut(row_off + k) = p;
            }
            den_local += p;
            k += bx;
        }
        unsafe {
            PARTIAL[tx] = den_local;
        }
        thread::sync_threads();
        block_reduce_sum!(PARTIAL, tx, bx);
        if tx == 0 {
            unsafe {
                DENOM[0] = PARTIAL[0] + (sinks[h as usize] - max_s).exp();
            }
        }
        thread::sync_threads();
        let denom = unsafe { DENOM[0] };

        let mut k = tx;
        while k < n_keys as usize {
            unsafe {
                *scores.get_unchecked_mut(row_off + k) /= denom;
            }
            k += bx;
        }
    }

    // ─── KV packing / unpacking helpers ────────────────────────────────────
    /// Concatenate raw_kv and comp_kv into a single contiguous buffer.
    /// Ported from attention_prefill_pack_mixed_kv_kernel.
    #[kernel]
    pub fn prefill_pack_mixed_kv(
        raw_kv: &[f32],
        comp_kv: &[f32],
        mut dst: DisjointSlice<f32>,
        n_tokens: u32,
        n_comp: u32,
        head_dim: u32,
    ) {
        let idx = thread::index_1d();
        let gid = idx.get() as u64;
        let n = (n_tokens + n_comp) as u64 * head_dim as u64;
        if gid >= n {
            return;
        }
        let d = (gid % head_dim as u64) as usize;
        let r = (gid / head_dim as u64) as usize;
        let v = if r < n_tokens as usize {
            raw_kv[r * head_dim as usize + d]
        } else {
            comp_kv[(r - n_tokens as usize) * head_dim as usize + d]
        };
        if let Some(o) = dst.get_mut(idx) {
            *o = v;
        }
    }

    /// Reshape heads from [n_head, n_tokens, head_dim] to [n_tokens, n_head, head_dim].
    /// Ported from attention_prefill_unpack_heads_kernel.
    #[kernel]
    pub fn prefill_unpack_heads(
        tmp: &[f32],
        mut heads: DisjointSlice<f32>,
        n_tokens: u32,
        n_head: u32,
        head_dim: u32,
    ) {
        let idx = thread::index_1d();
        let gid = idx.get() as u64;
        let n = n_tokens as u64 * n_head as u64 * head_dim as u64;
        if gid >= n {
            return;
        }
        let d = (gid % head_dim as u64) as usize;
        let q = gid / head_dim as u64;
        let h = (q % n_head as u64) as usize;
        let t = (q / n_head as u64) as usize;
        let v = tmp[(h * n_tokens as usize + t) * head_dim as usize + d];
        if let Some(o) = heads.get_mut(idx) {
            *o = v;
        }
    }

    /// Convert grouped heads to f16 (raw u16 bits) with layout transposition.
    /// Ported from attention_pack_group_heads_f16_kernel.
    #[kernel]
    pub fn pack_group_heads_f16(
        heads: &[f32],
        mut dst: DisjointSlice<u16>,
        n_tokens: u32,
        n_groups: u32,
        group_dim: u32,
    ) {
        let idx = thread::index_1d();
        let gid = idx.get() as u64;
        let n = n_groups as u64 * n_tokens as u64 * group_dim as u64;
        if gid >= n {
            return;
        }
        let d = (gid % group_dim as u64) as usize;
        let q = gid / group_dim as u64;
        let t = (q % n_tokens as u64) as usize;
        let g = (q / n_tokens as u64) as usize;
        let v = heads[(t * n_groups as usize + g) * group_dim as usize + d];
        if let Some(o) = dst.get_mut(idx) {
            *o = super::super::utils::f32_to_f16_bits(v);
        }
    }

    /// Unpack low-rank output from [n_groups, n_tokens, rank] to [n_tokens, n_groups*rank].
    /// Ported from attention_unpack_group_low_kernel.
    #[kernel]
    pub fn unpack_group_low(
        tmp: &[f32],
        mut low: DisjointSlice<f32>,
        n_tokens: u32,
        n_groups: u32,
        rank: u32,
    ) {
        let idx = thread::index_1d();
        let gid = idx.get() as u64;
        let n = n_groups as u64 * n_tokens as u64 * rank as u64;
        if gid >= n {
            return;
        }
        let r = (gid % rank as u64) as usize;
        let q = gid / rank as u64;
        let t = (q % n_tokens as u64) as usize;
        let g = (q / n_tokens as u64) as usize;
        let low_dim = n_groups as usize * rank as usize;
        let dst = t * low_dim + g * rank as usize + r;
        if let Some(o) = low.get_mut(idx) {
            *o = tmp[gid as usize];
        } else {
            unsafe {
                *low.get_unchecked_mut(dst) = tmp[gid as usize];
            }
        }
    }

    // ─── Decode: mixed raw + compressed attention ────────────────────────────
    /// Grid: (n_tokens, n_head, 1), Block: (256, 1, 1).
    /// Ported from attention_decode_mixed_kernel.
    #[kernel]
    pub fn decode_mixed(
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        comp_mask: &[f32],
        sinks: &[f32],
        mut heads: DisjointSlice<f32>,
        use_comp_mask: u32,
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) {
        static mut SCORES: SharedArray<f32, 8192> = SharedArray::UNINIT;
        static mut RAW_ROWS: SharedArray<u32, 256> = SharedArray::UNINIT;
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut MAX_S: SharedArray<f32, 1> = SharedArray::UNINIT;
        static mut DENOM: SharedArray<f32, 1> = SharedArray::UNINIT;
        static mut RAW_COUNT: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut RAW_FIRST_IDX: SharedArray<u32, 1> = SharedArray::UNINIT;

        let t = thread::blockIdx_x();
        let h = thread::blockIdx_y();
        if t >= n_tokens || h >= n_head {
            return;
        }
        let tx = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        let single_all = n_tokens == 1 && ratio == 0;
        let qpos = pos0 + t;
        let first_raw_pos = pos0 + n_tokens - n_raw;
        let mut visible_comp = if single_all {
            n_comp
        } else if n_comp != 0 {
            (qpos + 1) / ratio
        } else {
            0
        };
        if visible_comp > n_comp {
            visible_comp = n_comp;
        }

        if tx == 0 {
            // TODO(cuda-oxide): this raw-range computation was originally a
            // #[cuda_device::device] fn returning (u32, u32), but the backend treats
            // ALL MirTupleType returns as void and then panics (ICE: "Operation with
            // use(s) being erased") when the result is used. Inline with scalar vars
            // until tuple-returning device functions are supported.
            let mut fi = 0u32;
            let mut rc = 0u32;
            if n_raw != 0 {
                if single_all {
                    fi = 0;
                    rc = if n_raw > 256 { 256 } else { n_raw };
                } else {
                    let raw_last = first_raw_pos + n_raw - 1;
                    let mut lo = first_raw_pos;
                    if window != 0 && qpos + 1 > window {
                        let wlo = qpos + 1 - window;
                        if wlo > lo {
                            lo = wlo;
                        }
                    }
                    let hi = if qpos < raw_last { qpos } else { raw_last };
                    if hi >= lo {
                        fi = lo - first_raw_pos;
                        rc = hi - lo + 1;
                        if rc > 256 {
                            rc = 256;
                        }
                    }
                }
            }
            unsafe {
                RAW_FIRST_IDX[0] = fi;
                RAW_COUNT[0] = rc;
            }
        }
        thread::sync_threads();
        let raw_first_idx = unsafe { RAW_FIRST_IDX[0] };
        let raw_count = unsafe { RAW_COUNT[0] } as usize;
        let mut r = tx;
        while r < raw_count {
            unsafe {
                RAW_ROWS[r] = (raw_start + raw_first_idx + r as u32) % raw_cap;
            }
            r += bx;
        }
        thread::sync_threads();

        let n_score = raw_count + visible_comp as usize;
        let qh_off = (t as usize * n_head as usize + h as usize) * head_dim as usize;
        let scale = (head_dim as f32).sqrt().recip();
        let mut local_max = sinks[h as usize];

        // Quarter-warp dot products in mixed path, plain loops otherwise
        if visible_comp == 0 || n_tokens == 1 {
            let mut r = tx;
            while r < raw_count {
                let kv_off = unsafe { RAW_ROWS[r] } as usize * head_dim as usize;
                let mut dot = 0.0f32;
                for d in 0..head_dim as usize {
                    dot += q[qh_off + d] * raw_kv[kv_off + d];
                }
                let s = dot * scale;
                unsafe {
                    SCORES[r] = s;
                }
                if s > local_max {
                    local_max = s;
                }
                r += bx;
            }
            let mut c = tx;
            while c < visible_comp as usize {
                let add = if use_comp_mask != 0 {
                    comp_mask[t as usize * n_comp as usize + c]
                } else {
                    0.0
                };
                let s = if add > -1.0e20 {
                    let kv_off = c * head_dim as usize;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim as usize {
                        dot += q[qh_off + d] * comp_kv[kv_off + d];
                    }
                    dot * scale + add
                } else {
                    f32::NEG_INFINITY
                };
                unsafe {
                    SCORES[raw_count + c] = s;
                }
                if s > local_max {
                    local_max = s;
                }
                c += bx;
            }
        } else {
            let qlane = tx & 7;
            let qgroup = tx >> 3;
            let mut row0 = 0usize;
            while row0 < n_score {
                let row = row0 + qgroup;
                if row < n_score {
                    let (add, kv_base, valid) = if row < raw_count {
                        let kv_base = unsafe { RAW_ROWS[row] } as usize * head_dim as usize;
                        (0.0f32, kv_base, true)
                    } else {
                        let c = row - raw_count;
                        let add = if use_comp_mask != 0 {
                            comp_mask[t as usize * n_comp as usize + c]
                        } else {
                            0.0
                        };
                        if add > -1.0e20 {
                            (add, c * head_dim as usize, true)
                        } else {
                            (add, 0, false)
                        }
                    };
                    let s = if valid {
                        let data = if row < raw_count { raw_kv } else { comp_kv };
                        let mut dot = 0.0f32;
                        let mut d = qlane;
                        while d < head_dim as usize {
                            dot += q[qh_off + d] * data[kv_base + d];
                            d += 8;
                        }
                        dot += warp::shuffle_xor_f32(dot, 4);
                        dot += warp::shuffle_xor_f32(dot, 2);
                        dot += warp::shuffle_xor_f32(dot, 1);
                        dot * scale + add
                    } else {
                        f32::NEG_INFINITY
                    };
                    if qlane == 0 {
                        unsafe {
                            SCORES[row] = s;
                        }
                    }
                }
                row0 += 32;
            }
            thread::sync_threads();
            let mut i = tx;
            while i < n_score {
                let s = unsafe { SCORES[i] };
                if s > local_max {
                    local_max = s;
                }
                i += bx;
            }
        }

        unsafe {
            PARTIAL[tx] = local_max;
        }
        thread::sync_threads();
        block_reduce_max!(PARTIAL, tx, bx);
        if tx == 0 {
            unsafe {
                MAX_S[0] = PARTIAL[0];
            }
        }
        thread::sync_threads();
        let max_s = unsafe { MAX_S[0] };

        let mut den_local = 0.0f32;
        let mut i = tx;
        while i < n_score {
            let p = unsafe { (SCORES[i] - max_s).exp() };
            unsafe {
                SCORES[i] = p;
            }
            den_local += p;
            i += bx;
        }
        unsafe {
            PARTIAL[tx] = den_local;
        }
        thread::sync_threads();
        block_reduce_sum!(PARTIAL, tx, bx);
        if tx == 0 {
            unsafe {
                DENOM[0] = PARTIAL[0] + (sinks[h as usize] - max_s).exp();
            }
        }
        thread::sync_threads();
        let denom = unsafe { DENOM[0] };

        let oh_off = (t as usize * n_head as usize + h as usize) * head_dim as usize;
        if head_dim == 512 && bx == 256 {
            let d0 = tx;
            let d1 = tx + 256;
            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            for r in 0..raw_count {
                let s = unsafe { SCORES[r] };
                let kv_off = unsafe { RAW_ROWS[r] } as usize * head_dim as usize;
                acc0 += raw_kv[kv_off + d0] * s;
                acc1 += raw_kv[kv_off + d1] * s;
            }
            for c in 0..visible_comp as usize {
                let s = unsafe { SCORES[raw_count + c] };
                let kv_off = c * head_dim as usize;
                acc0 += comp_kv[kv_off + d0] * s;
                acc1 += comp_kv[kv_off + d1] * s;
            }
            unsafe {
                *heads.get_unchecked_mut(oh_off + d0) = acc0 / denom;
                *heads.get_unchecked_mut(oh_off + d1) = acc1 / denom;
            }
        } else {
            let mut d = tx;
            while d < head_dim as usize {
                let mut acc = 0.0f32;
                for r in 0..raw_count {
                    acc += raw_kv[unsafe { RAW_ROWS[r] } as usize * head_dim as usize + d]
                        * unsafe { SCORES[r] };
                }
                for c in 0..visible_comp as usize {
                    acc += comp_kv[c * head_dim as usize + d] * unsafe { SCORES[raw_count + c] };
                }
                unsafe {
                    *heads.get_unchecked_mut(oh_off + d) = acc / denom;
                }
                d += bx;
            }
        }
    }

    // ─── Decode: indexed (top-K) mixed attention ──────────────────────────────
    /// Grid: (n_tokens, n_head, 1), Block: (256, 1, 1).
    /// Ported from attention_indexed_mixed_kernel.
    #[kernel]
    pub fn indexed_mixed(
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        topk: &[i32],
        sinks: &[f32],
        mut heads: DisjointSlice<f32>,
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        n_comp: u32,
        top_k: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) {
        static mut SCORES: SharedArray<f32, 768> = SharedArray::UNINIT;
        static mut RAW_ROWS: SharedArray<u32, 256> = SharedArray::UNINIT;
        static mut COMP_ROWS: SharedArray<u32, 512> = SharedArray::UNINIT;
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut MAX_S: SharedArray<f32, 1> = SharedArray::UNINIT;
        static mut DENOM: SharedArray<f32, 1> = SharedArray::UNINIT;
        static mut RAW_COUNT: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut RAW_FIRST_IDX: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut COMP_COUNT: SharedArray<u32, 1> = SharedArray::UNINIT;

        let t = thread::blockIdx_x();
        let h = thread::blockIdx_y();
        if t >= n_tokens || h >= n_head {
            return;
        }
        let tx = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        let qpos = pos0 + t;
        let first_raw_pos = pos0 + n_tokens - n_raw;
        let mut visible_comp = n_comp;
        if ratio != 0 {
            visible_comp = (qpos + 1) / ratio;
            if visible_comp > n_comp {
                visible_comp = n_comp;
            }
        }

        if tx == 0 {
            // TODO(cuda-oxide): inlined raw-range computation — see decode_mixed for explanation.
            let mut fi = 0u32;
            let mut rc = 0u32;
            if n_raw != 0 {
                let raw_last = first_raw_pos + n_raw - 1;
                let mut lo = first_raw_pos;
                if window != 0 && qpos + 1 > window {
                    let wlo = qpos + 1 - window;
                    if wlo > lo {
                        lo = wlo;
                    }
                }
                let hi = if qpos < raw_last { qpos } else { raw_last };
                if hi >= lo {
                    fi = lo - first_raw_pos;
                    rc = hi - lo + 1;
                    if rc > 256 {
                        rc = 256;
                    }
                }
            }
            unsafe {
                RAW_FIRST_IDX[0] = fi;
                RAW_COUNT[0] = rc;
                COMP_COUNT[0] = 0;
            }
        }
        thread::sync_threads();
        let raw_first_idx = unsafe { RAW_FIRST_IDX[0] };
        let raw_count = unsafe { RAW_COUNT[0] } as usize;
        let mut r = tx;
        while r < raw_count {
            unsafe {
                RAW_ROWS[r] = (raw_start + raw_first_idx + r as u32) % raw_cap;
            }
            r += bx;
        }
        // Parallel top-K → comp_rows via shared atomic
        let mut i = tx;
        while i < top_k as usize {
            let c = topk[t as usize * top_k as usize + i];
            if c >= 0 && (c as u32) < visible_comp {
                let slot = unsafe {
                    let ptr: *mut u32 = &mut COMP_COUNT[0];
                    let atomic = AtomicU32::from_ptr(ptr);
                    (*atomic).fetch_add(1, Ordering::Relaxed)
                };
                if slot < 512 {
                    unsafe {
                        COMP_ROWS[slot as usize] = c as u32;
                    }
                }
            }
            i += bx;
        }
        thread::sync_threads();
        if tx == 0 {
            unsafe {
                if COMP_COUNT[0] > 512 {
                    COMP_COUNT[0] = 512;
                }
            }
        }
        thread::sync_threads();
        let comp_count = unsafe { COMP_COUNT[0] } as usize;
        let n_score = raw_count + comp_count;
        let qh_off = (t as usize * n_head as usize + h as usize) * head_dim as usize;
        let scale = (head_dim as f32).sqrt().recip();
        let mut local_max = sinks[h as usize];

        if comp_count == 0 {
            let mut r = tx;
            while r < raw_count {
                let kv_off = unsafe { RAW_ROWS[r] } as usize * head_dim as usize;
                let mut dot = 0.0f32;
                for d in 0..head_dim as usize {
                    dot += q[qh_off + d] * raw_kv[kv_off + d];
                }
                let s = dot * scale;
                unsafe {
                    SCORES[r] = s;
                }
                if s > local_max {
                    local_max = s;
                }
                r += bx;
            }
        } else {
            let qlane = tx & 7;
            let qgroup = tx >> 3;
            let mut row0 = 0usize;
            while row0 < n_score {
                let row = row0 + qgroup;
                if row < n_score {
                    let (data, kv_off) = if row < raw_count {
                        (
                            raw_kv,
                            unsafe { RAW_ROWS[row] } as usize * head_dim as usize,
                        )
                    } else {
                        (
                            comp_kv,
                            unsafe { COMP_ROWS[row - raw_count] } as usize * head_dim as usize,
                        )
                    };
                    let mut dot = 0.0f32;
                    let mut d = qlane;
                    while d < head_dim as usize {
                        dot += q[qh_off + d] * data[kv_off + d];
                        d += 8;
                    }
                    dot += warp::shuffle_xor_f32(dot, 4);
                    dot += warp::shuffle_xor_f32(dot, 2);
                    dot += warp::shuffle_xor_f32(dot, 1);
                    if qlane == 0 {
                        unsafe {
                            SCORES[row] = dot * scale;
                        }
                    }
                }
                row0 += 32;
            }
            thread::sync_threads();
            let mut i = tx;
            while i < n_score {
                let s = unsafe { SCORES[i] };
                if s > local_max {
                    local_max = s;
                }
                i += bx;
            }
        }

        unsafe {
            PARTIAL[tx] = local_max;
        }
        thread::sync_threads();
        block_reduce_max!(PARTIAL, tx, bx);
        if tx == 0 {
            unsafe {
                MAX_S[0] = PARTIAL[0];
            }
        }
        thread::sync_threads();
        let max_s = unsafe { MAX_S[0] };

        let mut den_local = 0.0f32;
        let mut i = tx;
        while i < n_score {
            let p = unsafe { (SCORES[i] - max_s).exp() };
            unsafe {
                SCORES[i] = p;
            }
            den_local += p;
            i += bx;
        }
        unsafe {
            PARTIAL[tx] = den_local;
        }
        thread::sync_threads();
        block_reduce_sum!(PARTIAL, tx, bx);
        if tx == 0 {
            unsafe {
                DENOM[0] = PARTIAL[0] + (sinks[h as usize] - max_s).exp();
            }
        }
        thread::sync_threads();
        let denom = unsafe { DENOM[0] };

        let oh_off = (t as usize * n_head as usize + h as usize) * head_dim as usize;
        if head_dim == 512 && bx == 256 {
            let d0 = tx;
            let d1 = tx + 256;
            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            for r in 0..raw_count {
                let s = unsafe { SCORES[r] };
                let kv_off = unsafe { RAW_ROWS[r] } as usize * 512;
                acc0 += raw_kv[kv_off + d0] * s;
                acc1 += raw_kv[kv_off + d1] * s;
            }
            for c in 0..comp_count {
                let s = unsafe { SCORES[raw_count + c] };
                let kv_off = unsafe { COMP_ROWS[c] } as usize * 512;
                acc0 += comp_kv[kv_off + d0] * s;
                acc1 += comp_kv[kv_off + d1] * s;
            }
            unsafe {
                *heads.get_unchecked_mut(oh_off + d0) = acc0 / denom;
                *heads.get_unchecked_mut(oh_off + d1) = acc1 / denom;
            }
        } else {
            let mut d = tx;
            while d < head_dim as usize {
                let mut acc = 0.0f32;
                for r in 0..raw_count {
                    acc += raw_kv[unsafe { RAW_ROWS[r] } as usize * head_dim as usize + d]
                        * unsafe { SCORES[r] };
                }
                for c in 0..comp_count {
                    acc += comp_kv[unsafe { COMP_ROWS[c] } as usize * head_dim as usize + d]
                        * unsafe { SCORES[raw_count + c] };
                }
                unsafe {
                    *heads.get_unchecked_mut(oh_off + d) = acc / denom;
                }
                d += bx;
            }
        }
    }

    // ─── Online-softmax kernels (head_dim = 512, 8 heads per block) ─────────
    // Shared helper: compute online softmax update for one KV row.
    // sum_s, o[0..3][0..3] are updated in-place using the rescaling trick.

    /// 8-head-per-block indexed attention with online softmax, 4 KV rows per stage.
    ///
    /// Requires head_dim == 512. Grid: (n_tokens, n_head/8, 1), Block: (256, 1, 1).
    /// Ported from attention_indexed_mixed_heads8_online_kernel<4, 8>.
    #[kernel]
    pub fn indexed_mixed_heads8_online(
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        topk: &[i32],
        sinks: &[f32],
        mut heads: DisjointSlice<f32>,
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        n_comp: u32,
        top_k: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) {
        // kv_shared: 4 stages × 128 float4-groups × 4 floats = 2048 floats
        static mut KV_SHARED: SharedArray<f32, 2048> = SharedArray::UNINIT;
        static mut RAW_ROWS: SharedArray<u32, 256> = SharedArray::UNINIT;
        static mut RAW_COUNT: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut RAW_FIRST_IDX: SharedArray<u32, 1> = SharedArray::UNINIT;

        const ROWS_PER_STAGE: usize = 4;
        let t = thread::blockIdx_x();
        let head_group = thread::blockIdx_y();
        if t >= n_tokens || head_dim != 512 {
            return;
        }
        let lane = (thread::threadIdx_x() & 31) as usize;
        let warp_id = (thread::threadIdx_x() >> 5) as usize;
        let head = head_group * 8 + warp_id as u32;
        let valid_head = head < n_head;
        let tx = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;

        let qpos = pos0 + t;
        let first_raw_pos = pos0 + n_tokens - n_raw;
        let mut visible_comp = n_comp;
        if ratio != 0 {
            visible_comp = (qpos + 1) / ratio;
            if visible_comp > n_comp {
                visible_comp = n_comp;
            }
        }
        let mut comp_count = if top_k < visible_comp {
            top_k
        } else {
            visible_comp
        } as usize;
        if comp_count > 512 {
            comp_count = 512;
        }

        if tx == 0 {
            // TODO(cuda-oxide): inlined raw-range computation — see decode_mixed for explanation.
            let mut fi = 0u32;
            let mut rc = 0u32;
            if n_raw != 0 {
                let raw_last = first_raw_pos + n_raw - 1;
                let mut lo = first_raw_pos;
                if window != 0 && qpos + 1 > window {
                    let wlo = qpos + 1 - window;
                    if wlo > lo {
                        lo = wlo;
                    }
                }
                let hi = if qpos < raw_last { qpos } else { raw_last };
                if hi >= lo {
                    fi = lo - first_raw_pos;
                    rc = hi - lo + 1;
                    if rc > 256 {
                        rc = 256;
                    }
                }
            }
            unsafe {
                RAW_FIRST_IDX[0] = fi;
                RAW_COUNT[0] = rc;
            }
        }
        thread::sync_threads();
        let raw_first_idx = unsafe { RAW_FIRST_IDX[0] };
        let raw_count = unsafe { RAW_COUNT[0] } as usize;
        let mut r = tx;
        while r < raw_count {
            unsafe {
                RAW_ROWS[r] = (raw_start + raw_first_idx + r as u32) % raw_cap;
            }
            r += bx;
        }
        thread::sync_threads();

        let n_score = raw_count + comp_count;
        let scale = (head_dim as f32).sqrt().recip();

        // Load Q for this head into registers (4 groups of 4 floats per lane)
        let mut q0 = [0.0f32; 4];
        let mut q1 = [0.0f32; 4];
        let mut q2 = [0.0f32; 4];
        let mut q3 = [0.0f32; 4];
        if valid_head {
            let qb = (t as usize * n_head as usize + head as usize) * 512 + lane * 4;
            for j in 0..4 {
                q0[j] = q[qb + j];
                q1[j] = q[qb + 128 + j];
                q2[j] = q[qb + 256 + j];
                q3[j] = q[qb + 384 + j];
            }
        }

        let mut max_s = f32::NEG_INFINITY;
        let mut sum_s = 0.0f32;
        let mut o0 = [0.0f32; 4];
        let mut o1 = [0.0f32; 4];
        let mut o2 = [0.0f32; 4];
        let mut o3 = [0.0f32; 4];

        let mut row0 = 0usize;
        while row0 < n_score {
            let nr = {
                let r = n_score - row0;
                if r < ROWS_PER_STAGE {
                    r
                } else {
                    ROWS_PER_STAGE
                }
            };
            // Load KV tile into KV_SHARED
            let mut off = tx;
            while off < nr * 128 {
                let rr = off >> 7;
                let c4 = off & 127;
                let sr = row0 + rr;
                let comp_idx = if sr >= raw_count {
                    topk[t as usize * top_k as usize + (sr - raw_count)] as usize
                } else {
                    0
                };
                let src_base = if sr < raw_count {
                    // TODO(cuda-oxide): parens around the unsafe block are required here
                    // because `unsafe { X } as T * N` inside an if-arm is parsed as
                    // `unsafe { X }` (closing the arm) then `as T * N` (syntax error).
                    (unsafe { RAW_ROWS[sr] as usize }) * 512 + c4 * 4
                } else {
                    comp_idx * 512 + c4 * 4
                };
                let data = if sr < raw_count { raw_kv } else { comp_kv };
                unsafe {
                    KV_SHARED[off * 4] = data[src_base];
                    KV_SHARED[off * 4 + 1] = data[src_base + 1];
                    KV_SHARED[off * 4 + 2] = data[src_base + 2];
                    KV_SHARED[off * 4 + 3] = data[src_base + 3];
                }
                off += bx;
            }
            thread::sync_threads();
            if valid_head {
                for rr in 0..nr {
                    let kb = rr * 128 + lane;
                    let k0 = unsafe {
                        [
                            KV_SHARED[kb * 4],
                            KV_SHARED[kb * 4 + 1],
                            KV_SHARED[kb * 4 + 2],
                            KV_SHARED[kb * 4 + 3],
                        ]
                    };
                    let k1 = unsafe {
                        let o = (kb + 32) * 4;
                        [
                            KV_SHARED[o],
                            KV_SHARED[o + 1],
                            KV_SHARED[o + 2],
                            KV_SHARED[o + 3],
                        ]
                    };
                    let k2 = unsafe {
                        let o = (kb + 64) * 4;
                        [
                            KV_SHARED[o],
                            KV_SHARED[o + 1],
                            KV_SHARED[o + 2],
                            KV_SHARED[o + 3],
                        ]
                    };
                    let k3 = unsafe {
                        let o = (kb + 96) * 4;
                        [
                            KV_SHARED[o],
                            KV_SHARED[o + 1],
                            KV_SHARED[o + 2],
                            KV_SHARED[o + 3],
                        ]
                    };
                    let mut score = super::attn_dot4(q0, k0)
                        + super::attn_dot4(q1, k1)
                        + super::attn_dot4(q2, k2)
                        + super::attn_dot4(q3, k3);
                    score = super::attn_warp_sum(score) * scale;
                    score = warp::shuffle_f32(score, 0);
                    let new_m = if score > max_s { score } else { max_s };
                    let old_sc = (max_s - new_m).exp();
                    let row_sc = (score - new_m).exp();
                    sum_s = sum_s * old_sc + row_sc;
                    for j in 0..4 {
                        o0[j] = o0[j] * old_sc + k0[j] * row_sc;
                        o1[j] = o1[j] * old_sc + k1[j] * row_sc;
                        o2[j] = o2[j] * old_sc + k2[j] * row_sc;
                        o3[j] = o3[j] * old_sc + k3[j] * row_sc;
                    }
                    max_s = new_m;
                }
            }
            thread::sync_threads();
            row0 += ROWS_PER_STAGE;
        }

        if valid_head {
            let sink = sinks[head as usize];
            let new_m = if sink > max_s { sink } else { max_s };
            let old_sc = (max_s - new_m).exp();
            let sink_sc = (sink - new_m).exp();
            sum_s = sum_s * old_sc + sink_sc;
            let inv_s = if sum_s == 0.0 { 0.0 } else { 1.0 / sum_s };
            for j in 0..4 {
                o0[j] = o0[j] * old_sc * inv_s;
                o1[j] = o1[j] * old_sc * inv_s;
                o2[j] = o2[j] * old_sc * inv_s;
                o3[j] = o3[j] * old_sc * inv_s;
            }
            let ob = (t as usize * n_head as usize + head as usize) * 512 + lane * 4;
            unsafe {
                for j in 0..4 {
                    *heads.get_unchecked_mut(ob + j) = o0[j];
                }
                for j in 0..4 {
                    *heads.get_unchecked_mut(ob + 128 + j) = o1[j];
                }
                for j in 0..4 {
                    *heads.get_unchecked_mut(ob + 256 + j) = o2[j];
                }
                for j in 0..4 {
                    *heads.get_unchecked_mut(ob + 384 + j) = o3[j];
                }
            }
        }
    }

    /// Static (all compressed rows) online-softmax attention, 8 heads/block.
    ///
    /// head_dim must be 512. Grid: (n_tokens, n_head/8, 1), Block: (256, 1, 1).
    /// Ported from attention_static_mixed_heads8_online_kernel.
    #[kernel]
    pub fn static_mixed_heads8_online(
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        sinks: &[f32],
        mut heads: DisjointSlice<f32>,
        n_tokens: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) {
        static mut KV_SHARED: SharedArray<f32, 2048> = SharedArray::UNINIT;

        let t = thread::blockIdx_x();
        let head_group = thread::blockIdx_y();
        if t >= n_tokens || head_dim != 512 {
            return;
        }
        let lane = (thread::threadIdx_x() & 31) as usize;
        let warp_id = (thread::threadIdx_x() >> 5) as usize;
        let head = head_group * 8 + warp_id as u32;
        let valid_head = head < n_head;
        let tx = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;

        let raw_count = if window != 0 && t + 1 > window {
            window as usize
        } else {
            t as usize + 1
        };
        let raw_start_pos = t as usize + 1 - raw_count;
        let comp_count = if n_comp != 0 && ratio != 0 {
            let c = ((t + 1) / ratio) as usize;
            if c > n_comp as usize {
                n_comp as usize
            } else {
                c
            }
        } else {
            0
        };
        let n_score = raw_count + comp_count;
        let scale = (head_dim as f32).sqrt().recip();

        let mut q0 = [0.0f32; 4];
        let mut q1 = [0.0f32; 4];
        let mut q2 = [0.0f32; 4];
        let mut q3 = [0.0f32; 4];
        if valid_head {
            let qb = (t as usize * n_head as usize + head as usize) * 512 + lane * 4;
            for j in 0..4 {
                q0[j] = q[qb + j];
                q1[j] = q[qb + 128 + j];
                q2[j] = q[qb + 256 + j];
                q3[j] = q[qb + 384 + j];
            }
        }
        let mut max_s = f32::NEG_INFINITY;
        let mut sum_s = 0.0f32;
        let mut o0 = [0.0f32; 4];
        let mut o1 = [0.0f32; 4];
        let mut o2 = [0.0f32; 4];
        let mut o3 = [0.0f32; 4];

        let mut row0 = 0usize;
        while row0 < n_score {
            let nr = {
                let r = n_score - row0;
                if r < 4 { r } else { 4 }
            };
            let mut off = tx;
            while off < nr * 128 {
                let rr = off >> 7;
                let c4 = off & 127;
                let sr = row0 + rr;
                let (data, src_base) = if sr < raw_count {
                    (raw_kv, (raw_start_pos + sr) * 512 + c4 * 4)
                } else {
                    (comp_kv, (sr - raw_count) * 512 + c4 * 4)
                };
                unsafe {
                    KV_SHARED[off * 4] = data[src_base];
                    KV_SHARED[off * 4 + 1] = data[src_base + 1];
                    KV_SHARED[off * 4 + 2] = data[src_base + 2];
                    KV_SHARED[off * 4 + 3] = data[src_base + 3];
                }
                off += bx;
            }
            thread::sync_threads();
            if valid_head {
                for rr in 0..nr {
                    let kb = rr * 128 + lane;
                    let k0 = unsafe {
                        [
                            KV_SHARED[kb * 4],
                            KV_SHARED[kb * 4 + 1],
                            KV_SHARED[kb * 4 + 2],
                            KV_SHARED[kb * 4 + 3],
                        ]
                    };
                    let k1 = unsafe {
                        let o = (kb + 32) * 4;
                        [
                            KV_SHARED[o],
                            KV_SHARED[o + 1],
                            KV_SHARED[o + 2],
                            KV_SHARED[o + 3],
                        ]
                    };
                    let k2 = unsafe {
                        let o = (kb + 64) * 4;
                        [
                            KV_SHARED[o],
                            KV_SHARED[o + 1],
                            KV_SHARED[o + 2],
                            KV_SHARED[o + 3],
                        ]
                    };
                    let k3 = unsafe {
                        let o = (kb + 96) * 4;
                        [
                            KV_SHARED[o],
                            KV_SHARED[o + 1],
                            KV_SHARED[o + 2],
                            KV_SHARED[o + 3],
                        ]
                    };
                    let mut score = super::attn_dot4(q0, k0)
                        + super::attn_dot4(q1, k1)
                        + super::attn_dot4(q2, k2)
                        + super::attn_dot4(q3, k3);
                    score = super::attn_warp_sum(score) * scale;
                    score = warp::shuffle_f32(score, 0);
                    let new_m = if score > max_s { score } else { max_s };
                    let old_sc = (max_s - new_m).exp();
                    let row_sc = (score - new_m).exp();
                    sum_s = sum_s * old_sc + row_sc;
                    for j in 0..4 {
                        o0[j] = o0[j] * old_sc + k0[j] * row_sc;
                        o1[j] = o1[j] * old_sc + k1[j] * row_sc;
                        o2[j] = o2[j] * old_sc + k2[j] * row_sc;
                        o3[j] = o3[j] * old_sc + k3[j] * row_sc;
                    }
                    max_s = new_m;
                }
            }
            thread::sync_threads();
            row0 += 4;
        }
        if valid_head {
            let sink = sinks[head as usize];
            let new_m = if sink > max_s { sink } else { max_s };
            let old_sc = (max_s - new_m).exp();
            let sink_sc = (sink - new_m).exp();
            sum_s = sum_s * old_sc + sink_sc;
            let inv_s = if sum_s == 0.0 { 0.0 } else { 1.0 / sum_s };
            let ob = (t as usize * n_head as usize + head as usize) * 512 + lane * 4;
            unsafe {
                for j in 0..4 {
                    *heads.get_unchecked_mut(ob + j) = o0[j] * old_sc * inv_s;
                }
                for j in 0..4 {
                    *heads.get_unchecked_mut(ob + 128 + j) = o1[j] * old_sc * inv_s;
                }
                for j in 0..4 {
                    *heads.get_unchecked_mut(ob + 256 + j) = o2[j] * old_sc * inv_s;
                }
                for j in 0..4 {
                    *heads.get_unchecked_mut(ob + 384 + j) = o3[j] * old_sc * inv_s;
                }
            }
        }
    }

    /// Decode online-softmax attention, 8 heads/block, circular raw buffer.
    ///
    /// head_dim must be 512. Grid: (n_tokens, n_head/8, 1), Block: (256, 1, 1).
    /// Ported from attention_decode_mixed_heads8_online_kernel.
    #[kernel]
    pub fn decode_mixed_heads8_online(
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        sinks: &[f32],
        mut heads: DisjointSlice<f32>,
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) {
        static mut KV_SHARED: SharedArray<f32, 2048> = SharedArray::UNINIT;
        static mut RAW_ROWS: SharedArray<u32, 256> = SharedArray::UNINIT;
        static mut RAW_COUNT_S: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut RAW_FIRST_IDX: SharedArray<u32, 1> = SharedArray::UNINIT;

        let t = thread::blockIdx_x();
        let head_group = thread::blockIdx_y();
        if t >= n_tokens || head_dim != 512 {
            return;
        }
        let lane = (thread::threadIdx_x() & 31) as usize;
        let warp_id = (thread::threadIdx_x() >> 5) as usize;
        let head = head_group * 8 + warp_id as u32;
        let valid_head = head < n_head;
        let tx = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;

        let qpos = pos0 + t;
        let first_raw_pos = pos0 + n_tokens - n_raw;
        let comp_count = if n_comp != 0 {
            if n_tokens == 1 && ratio == 0 {
                n_comp as usize
            } else if ratio != 0 {
                let c = ((qpos + 1) / ratio) as usize;
                if c > n_comp as usize {
                    n_comp as usize
                } else {
                    c
                }
            } else {
                0
            }
        } else {
            0
        };

        if tx == 0 {
            // TODO(cuda-oxide): inlined raw-range computation — see decode_mixed for explanation.
            let mut fi = 0u32;
            let mut rc = 0u32;
            if n_raw != 0 {
                let raw_last = first_raw_pos + n_raw - 1;
                let mut lo = first_raw_pos;
                if window != 0 && qpos + 1 > window {
                    let wlo = qpos + 1 - window;
                    if wlo > lo {
                        lo = wlo;
                    }
                }
                let hi = if qpos < raw_last { qpos } else { raw_last };
                if hi >= lo {
                    fi = lo - first_raw_pos;
                    rc = hi - lo + 1;
                    if rc > 256 {
                        rc = 256;
                    }
                }
            }
            unsafe {
                RAW_FIRST_IDX[0] = fi;
                RAW_COUNT_S[0] = rc;
            }
        }
        thread::sync_threads();
        let raw_first_idx = unsafe { RAW_FIRST_IDX[0] };
        let raw_count = unsafe { RAW_COUNT_S[0] } as usize;
        let mut r = tx;
        while r < raw_count {
            unsafe {
                RAW_ROWS[r] = (raw_start + raw_first_idx + r as u32) % raw_cap;
            }
            r += bx;
        }
        thread::sync_threads();

        let n_score = raw_count + comp_count;
        let scale = (head_dim as f32).sqrt().recip();
        let mut q0 = [0.0f32; 4];
        let mut q1 = [0.0f32; 4];
        let mut q2 = [0.0f32; 4];
        let mut q3 = [0.0f32; 4];
        if valid_head {
            let qb = (t as usize * n_head as usize + head as usize) * 512 + lane * 4;
            for j in 0..4 {
                q0[j] = q[qb + j];
                q1[j] = q[qb + 128 + j];
                q2[j] = q[qb + 256 + j];
                q3[j] = q[qb + 384 + j];
            }
        }
        let mut max_s = f32::NEG_INFINITY;
        let mut sum_s = 0.0f32;
        let mut o0 = [0.0f32; 4];
        let mut o1 = [0.0f32; 4];
        let mut o2 = [0.0f32; 4];
        let mut o3 = [0.0f32; 4];

        let mut row0 = 0usize;
        while row0 < n_score {
            let nr = {
                let r = n_score - row0;
                if r < 4 { r } else { 4 }
            };
            let mut off = tx;
            while off < nr * 128 {
                let rr = off >> 7;
                let c4 = off & 127;
                let sr = row0 + rr;
                let (data, src_base) = if sr < raw_count {
                    (raw_kv, unsafe { RAW_ROWS[sr] } as usize * 512 + c4 * 4)
                } else {
                    (comp_kv, (sr - raw_count) * 512 + c4 * 4)
                };
                unsafe {
                    KV_SHARED[off * 4] = data[src_base];
                    KV_SHARED[off * 4 + 1] = data[src_base + 1];
                    KV_SHARED[off * 4 + 2] = data[src_base + 2];
                    KV_SHARED[off * 4 + 3] = data[src_base + 3];
                }
                off += bx;
            }
            thread::sync_threads();
            if valid_head {
                for rr in 0..nr {
                    let kb = rr * 128 + lane;
                    let k0 = unsafe {
                        [
                            KV_SHARED[kb * 4],
                            KV_SHARED[kb * 4 + 1],
                            KV_SHARED[kb * 4 + 2],
                            KV_SHARED[kb * 4 + 3],
                        ]
                    };
                    let k1 = unsafe {
                        let o = (kb + 32) * 4;
                        [
                            KV_SHARED[o],
                            KV_SHARED[o + 1],
                            KV_SHARED[o + 2],
                            KV_SHARED[o + 3],
                        ]
                    };
                    let k2 = unsafe {
                        let o = (kb + 64) * 4;
                        [
                            KV_SHARED[o],
                            KV_SHARED[o + 1],
                            KV_SHARED[o + 2],
                            KV_SHARED[o + 3],
                        ]
                    };
                    let k3 = unsafe {
                        let o = (kb + 96) * 4;
                        [
                            KV_SHARED[o],
                            KV_SHARED[o + 1],
                            KV_SHARED[o + 2],
                            KV_SHARED[o + 3],
                        ]
                    };
                    let mut score = super::attn_dot4(q0, k0)
                        + super::attn_dot4(q1, k1)
                        + super::attn_dot4(q2, k2)
                        + super::attn_dot4(q3, k3);
                    score = super::attn_warp_sum(score) * scale;
                    score = warp::shuffle_f32(score, 0);
                    let new_m = if score > max_s { score } else { max_s };
                    let old_sc = (max_s - new_m).exp();
                    let row_sc = (score - new_m).exp();
                    sum_s = sum_s * old_sc + row_sc;
                    for j in 0..4 {
                        o0[j] = o0[j] * old_sc + k0[j] * row_sc;
                        o1[j] = o1[j] * old_sc + k1[j] * row_sc;
                        o2[j] = o2[j] * old_sc + k2[j] * row_sc;
                        o3[j] = o3[j] * old_sc + k3[j] * row_sc;
                    }
                    max_s = new_m;
                }
            }
            thread::sync_threads();
            row0 += 4;
        }
        if valid_head {
            let sink = sinks[head as usize];
            let new_m = if sink > max_s { sink } else { max_s };
            let old_sc = (max_s - new_m).exp();
            let sink_sc = (sink - new_m).exp();
            sum_s = sum_s * old_sc + sink_sc;
            let inv_s = if sum_s == 0.0 { 0.0 } else { 1.0 / sum_s };
            let ob = (t as usize * n_head as usize + head as usize) * 512 + lane * 4;
            unsafe {
                for j in 0..4 {
                    *heads.get_unchecked_mut(ob + j) = o0[j] * old_sc * inv_s;
                }
                for j in 0..4 {
                    *heads.get_unchecked_mut(ob + 128 + j) = o1[j] * old_sc * inv_s;
                }
                for j in 0..4 {
                    *heads.get_unchecked_mut(ob + 256 + j) = o2[j] * old_sc * inv_s;
                }
                for j in 0..4 {
                    *heads.get_unchecked_mut(ob + 384 + j) = o3[j] * old_sc * inv_s;
                }
            }
        }
    }
}
