use cuda_device::{
    DisjointSlice, DynamicSharedArray, SharedArray, cuda_module, kernel, thread, warp,
};

// ============================================================================
// Device helpers
// ============================================================================

/// True if (av, ai) scores better than (bv, bi): higher value wins, lower index breaks ties.
/// Ported from topk_score_better.
#[cuda_device::device]
pub fn topk_score_better(av: f32, ai: u32, bv: f32, bi: u32) -> bool {
    av > bv || (av == bv && ai < bi)
}

// ============================================================================
// Kernels
// ============================================================================

#[cuda_module]
pub mod indexer {
    use super::*;

    // -----------------------------------------------------------------------
    // Score computation
    // -----------------------------------------------------------------------

    /// Multi-head dot-product score for each (compressed-row, token) pair.
    ///
    /// `scores[t, c] = Σ_h max(dot(q[t,h], k[c]), 0) * weights[t, h]`
    ///
    /// Causal masking: if c ≥ (pos0 + t + 1) / ratio, score = -∞.
    ///
    /// Grid: (n_comp, n_tokens, 1), Block: (256, 1, 1).
    /// Ported from indexer_scores_kernel.
    #[kernel]
    pub fn indexer_scores(
        q: &[f32],
        weights: &[f32],
        index_comp: &[f32],
        mut scores: DisjointSlice<f32>,
        n_comp: u32,
        n_tokens: u32,
        pos0: u32,
        n_head: u32,
        head_dim: u32,
        ratio: u32,
        scale: f32,
        causal: u32,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let c = thread::blockIdx_x() as usize;
        let t = thread::blockIdx_y() as usize;
        if c as u32 >= n_comp || t as u32 >= n_tokens {
            return;
        }

        if causal != 0 {
            let visible = ((pos0 + t as u32 + 1) / ratio) as usize;
            if c >= visible {
                if thread::threadIdx_x() == 0 {
                    unsafe {
                        *scores.get_unchecked_mut(t * n_comp as usize + c) = f32::NEG_INFINITY;
                    }
                }
                return;
            }
        }

        let tx = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        let mut total = 0.0f32;

        for h in 0..n_head as usize {
            let qh_off = (t * n_head as usize + h) * head_dim as usize;
            let kh_off = c * head_dim as usize;
            let mut dot = 0.0f32;
            let mut i = tx;
            while i < head_dim as usize {
                dot += q[qh_off + i] * index_comp[kh_off + i];
                i += bx;
            }
            unsafe {
                PARTIAL[tx] = dot;
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
            let head_dot = unsafe { PARTIAL[0] };
            let head_dot = if head_dot < 0.0 { 0.0 } else { head_dot };
            total += head_dot * weights[t * n_head as usize + h];
            thread::sync_threads();
        }

        if tx == 0 {
            unsafe {
                *scores.get_unchecked_mut(t * n_comp as usize + c) = total * scale;
            }
        }
    }

    /// Optimised single-token decode indexer score (head_dim=128, n_head=64 assumed).
    ///
    /// 4 warps × 128 threads; each warp handles 4 heads per iteration.
    ///
    /// Grid: (n_comp, 1, 1), Block: (128, 1, 1).
    /// Ported from indexer_score_one_direct_kernel.
    #[kernel]
    pub fn indexer_score_one_direct(
        q: &[f32],
        weights: &[f32],
        index_comp: &[f32],
        mut scores: DisjointSlice<f32>,
        n_comp: u32,
        pos0: u32,
        ratio: u32,
        scale: f32,
        causal: u32,
    ) {
        static mut KROW: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut PARTIAL4: SharedArray<f32, 4> = SharedArray::UNINIT;

        let c = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        if c as u32 >= n_comp || tid >= 128 {
            return;
        }

        if causal != 0 {
            let visible = ((pos0 + 1) / ratio) as usize;
            if c >= visible {
                if tid == 0 {
                    unsafe {
                        *scores.get_unchecked_mut(c) = f32::NEG_INFINITY;
                    }
                }
                return;
            }
        }

        unsafe {
            KROW[tid] = index_comp[c * 128 + tid];
        }
        thread::sync_threads();

        let lane = tid & 31;
        let warp_id = tid >> 5;
        let mut total = 0.0f32;

        let mut h0 = 0usize;
        while h0 < 64 {
            let h = h0 + warp_id;
            let q_off = h * 128 + lane * 4;
            let k_off = lane * 4;
            let dot = {
                let q0 = q[q_off];
                let q1 = q[q_off + 1];
                let q2 = q[q_off + 2];
                let q3 = q[q_off + 3];
                let k0 = unsafe { KROW[k_off] };
                let k1 = unsafe { KROW[k_off + 1] };
                let k2 = unsafe { KROW[k_off + 2] };
                let k3 = unsafe { KROW[k_off + 3] };
                q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3
            };
            let dot = {
                let mut v = dot;
                v += warp::shuffle_down_f32(v, 16);
                v += warp::shuffle_down_f32(v, 8);
                v += warp::shuffle_down_f32(v, 4);
                v += warp::shuffle_down_f32(v, 2);
                v += warp::shuffle_down_f32(v, 1);
                v
            };
            if lane == 0 {
                let head_dot = if dot < 0.0 { 0.0 } else { dot };
                unsafe {
                    PARTIAL4[warp_id] = head_dot * weights[h] * scale;
                }
            }
            thread::sync_threads();
            if tid == 0 {
                total += unsafe { PARTIAL4[0] + PARTIAL4[1] + PARTIAL4[2] + PARTIAL4[3] };
            }
            thread::sync_threads();
            h0 += 4;
        }

        if tid == 0 {
            unsafe {
                *scores.get_unchecked_mut(c) = total;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Top-K selection
    // -----------------------------------------------------------------------

    /// Reference single-thread insertion-sort top-K (one thread per token).
    ///
    /// Grid: (n_tokens, 1, 1), Block: (1, 1, 1).
    /// Ported from indexer_topk_kernel.
    #[kernel]
    pub fn indexer_topk(
        scores: &[f32],
        selected: *mut u32,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
    ) {
        let t = thread::blockIdx_x() as usize;
        if t as u32 >= n_tokens || thread::threadIdx_x() != 0 {
            return;
        }
        let row_base = t * n_comp as usize;
        let k = top_k as usize;
        for i in 0..k {
            unsafe {
                *selected.add(t * k + i) = 0u32;
            }
        }
        for c in 0..n_comp as usize {
            let v = scores[row_base + c];
            for i in 0..k {
                let cur = unsafe { *selected.add(t * k + i) };
                if c < k
                    || super::topk_score_better(v, c as u32, scores[row_base + cur as usize], cur)
                {
                    let mut j = k - 1;
                    while j > i {
                        let prev = unsafe { *selected.add(t * k + j - 1) };
                        unsafe {
                            *selected.add(t * k + j) = prev;
                        }
                        j -= 1;
                    }
                    unsafe {
                        *selected.add(t * k + i) = c as u32;
                    }
                    break;
                }
            }
        }
    }

    /// Bitonic sort top-K using dynamic shared memory.
    ///
    /// Sorts `n_sort` elements (a power of 2) in descending score order, then
    /// writes the first `top_k` indices to `selected`.
    ///
    /// Launch with `shared_mem_bytes = n_sort * 8` (4 bytes val + 4 bytes idx).
    ///
    /// Grid: (n_tokens, 1, 1).
    /// Ported from indexer_topk_pow2_kernel (all SORT_N variants unified).
    #[kernel]
    pub fn indexer_topk_bitonic(
        scores: &[f32],
        selected: *mut u32,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        n_sort: u32,
    ) {
        let t = thread::blockIdx_x() as usize;
        if t as u32 >= n_tokens {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        let ns = n_sort as usize;

        let vals: *mut f32 = DynamicSharedArray::<f32>::get();
        let idxs: *mut u32 = DynamicSharedArray::<u32>::offset(ns * 4);

        let row_off = t * n_comp as usize;
        let mut i = tid;
        while i < ns {
            unsafe {
                if (i as u32) < n_comp {
                    *vals.add(i) = scores[row_off + i];
                    *idxs.add(i) = i as u32;
                } else {
                    *vals.add(i) = f32::NEG_INFINITY;
                    *idxs.add(i) = u32::MAX;
                }
            }
            i += bx;
        }
        thread::sync_threads();

        let mut k = 2usize;
        while k <= ns {
            let mut j = k >> 1;
            while j > 0 {
                let mut i = tid;
                while i < ns {
                    let other = i ^ j;
                    if other > i && other < ns {
                        unsafe {
                            let av = *vals.add(i);
                            let bv = *vals.add(other);
                            let ai = *idxs.add(i);
                            let bi = *idxs.add(other);
                            let desc = (i & k) == 0;
                            let swap = if desc {
                                super::topk_score_better(bv, bi, av, ai)
                            } else {
                                super::topk_score_better(av, ai, bv, bi)
                            };
                            if swap {
                                *vals.add(i) = bv;
                                *idxs.add(i) = bi;
                                *vals.add(other) = av;
                                *idxs.add(other) = ai;
                            }
                        }
                    }
                    i += bx;
                }
                thread::sync_threads();
                j >>= 1;
            }
            k <<= 1;
        }

        let mut i = tid;
        while i < top_k as usize {
            unsafe {
                *selected.add(t * top_k as usize + i) = *idxs.add(i);
            }
            i += bx;
        }
    }

    /// Bitonic sort top-K with u16 indices (n_comp ≤ 65535).
    ///
    /// Launch with `shared_mem_bytes = n_sort * 6` (4 bytes f32 val + 2 bytes u16 idx).
    ///
    /// Grid: (n_tokens, 1, 1).
    /// Ported from indexer_topk_pow2_u16_kernel.
    #[kernel]
    pub fn indexer_topk_bitonic_u16(
        scores: &[f32],
        selected: *mut u32,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        n_sort: u32,
    ) {
        let t = thread::blockIdx_x() as usize;
        if t as u32 >= n_tokens {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        let ns = n_sort as usize;

        let vals: *mut f32 = DynamicSharedArray::<f32>::get();
        let idxs: *mut u16 = DynamicSharedArray::<u16>::offset(ns * 4);

        let row_off = t * n_comp as usize;
        let mut i = tid;
        while i < ns {
            unsafe {
                if (i as u32) < n_comp {
                    *vals.add(i) = scores[row_off + i];
                    *idxs.add(i) = i as u16;
                } else {
                    *vals.add(i) = f32::NEG_INFINITY;
                    *idxs.add(i) = u16::MAX;
                }
            }
            i += bx;
        }
        thread::sync_threads();

        let mut k = 2usize;
        while k <= ns {
            let mut j = k >> 1;
            while j > 0 {
                let mut i = tid;
                while i < ns {
                    let other = i ^ j;
                    if other > i && other < ns {
                        unsafe {
                            let av = *vals.add(i);
                            let bv = *vals.add(other);
                            let ai = *idxs.add(i) as u32;
                            let bi = *idxs.add(other) as u32;
                            let desc = (i & k) == 0;
                            let swap = if desc {
                                super::topk_score_better(bv, bi, av, ai)
                            } else {
                                super::topk_score_better(av, ai, bv, bi)
                            };
                            if swap {
                                *vals.add(i) = bv;
                                *idxs.add(i) = bi as u16;
                                *vals.add(other) = av;
                                *idxs.add(other) = ai as u16;
                            }
                        }
                    }
                    i += bx;
                }
                thread::sync_threads();
                j >>= 1;
            }
            k <<= 1;
        }

        let mut i = tid;
        while i < top_k as usize {
            unsafe {
                *selected.add(t * top_k as usize + i) = *idxs.add(i) as u32;
            }
            i += bx;
        }
    }

    /// Chunk-parallel bitonic sort: sort a contiguous `n_sort`-element chunk of
    /// the score array, emitting the top-K candidates from that chunk.
    ///
    /// Grid: (n_tokens, n_chunks, 1). `shared_mem_bytes = n_sort * 8`.
    /// Ported from indexer_topk_chunk_pow2_kernel.
    #[kernel]
    pub fn indexer_topk_chunk_bitonic(
        scores: &[f32],
        candidates: *mut u32,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        candidate_stride: u32,
        n_sort: u32,
    ) {
        let t = thread::blockIdx_x() as usize;
        let chunk = thread::blockIdx_y() as usize;
        if t as u32 >= n_tokens {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        let ns = n_sort as usize;
        let chunk_start = chunk * ns;
        if chunk_start >= n_comp as usize {
            return;
        }
        let rem = n_comp as usize - chunk_start;
        let chunk_n = if rem < ns { rem } else { ns };

        let vals: *mut f32 = DynamicSharedArray::<f32>::get();
        let idxs: *mut u32 = DynamicSharedArray::<u32>::offset(ns * 4);

        let row_off = t * n_comp as usize;
        let mut i = tid;
        while i < ns {
            unsafe {
                if i < chunk_n {
                    *vals.add(i) = scores[row_off + chunk_start + i];
                    *idxs.add(i) = (chunk_start + i) as u32;
                } else {
                    *vals.add(i) = f32::NEG_INFINITY;
                    *idxs.add(i) = u32::MAX;
                }
            }
            i += bx;
        }
        thread::sync_threads();

        let mut k = 2usize;
        while k <= ns {
            let mut j = k >> 1;
            while j > 0 {
                let mut i = tid;
                while i < ns {
                    let other = i ^ j;
                    if other > i && other < ns {
                        unsafe {
                            let av = *vals.add(i);
                            let bv = *vals.add(other);
                            let ai = *idxs.add(i);
                            let bi = *idxs.add(other);
                            let desc = (i & k) == 0;
                            let swap = if desc {
                                super::topk_score_better(bv, bi, av, ai)
                            } else {
                                super::topk_score_better(av, ai, bv, bi)
                            };
                            if swap {
                                *vals.add(i) = bv;
                                *idxs.add(i) = bi;
                                *vals.add(other) = av;
                                *idxs.add(other) = ai;
                            }
                        }
                    }
                    i += bx;
                }
                thread::sync_threads();
                j >>= 1;
            }
            k <<= 1;
        }

        let out = unsafe { candidates.add(t * candidate_stride as usize + chunk * top_k as usize) };
        let mut i = tid;
        while i < top_k as usize {
            unsafe {
                *out.add(i) = *idxs.add(i);
            }
            i += bx;
        }
    }

    /// Merge candidate lists from multiple chunks into a single sorted top-K.
    ///
    /// Grid: (n_tokens, 1, 1). `shared_mem_bytes = n_sort * 8`.
    /// Ported from indexer_topk_merge_pow2_kernel.
    #[kernel]
    pub fn indexer_topk_merge_bitonic(
        scores: &[f32],
        candidates: &[u32],
        selected: *mut u32,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        candidate_count: u32,
        candidate_stride: u32,
        n_sort: u32,
    ) {
        let t = thread::blockIdx_x() as usize;
        if t as u32 >= n_tokens {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        let ns = n_sort as usize;

        let vals: *mut f32 = DynamicSharedArray::<f32>::get();
        let idxs: *mut u32 = DynamicSharedArray::<u32>::offset(ns * 4);

        let cand_base = t * candidate_stride as usize;
        let row_base = t * n_comp as usize;
        let mut i = tid;
        while i < ns {
            unsafe {
                let (idx, v) = if (i as u32) < candidate_count {
                    let idx = candidates[cand_base + i];
                    let v = if idx < n_comp {
                        scores[row_base + idx as usize]
                    } else {
                        f32::NEG_INFINITY
                    };
                    (idx, v)
                } else {
                    (u32::MAX, f32::NEG_INFINITY)
                };
                *vals.add(i) = v;
                *idxs.add(i) = idx;
            }
            i += bx;
        }
        thread::sync_threads();

        let mut k = 2usize;
        while k <= ns {
            let mut j = k >> 1;
            while j > 0 {
                let mut i = tid;
                while i < ns {
                    let other = i ^ j;
                    if other > i && other < ns {
                        unsafe {
                            let av = *vals.add(i);
                            let bv = *vals.add(other);
                            let ai = *idxs.add(i);
                            let bi = *idxs.add(other);
                            let desc = (i & k) == 0;
                            let swap = if desc {
                                super::topk_score_better(bv, bi, av, ai)
                            } else {
                                super::topk_score_better(av, ai, bv, bi)
                            };
                            if swap {
                                *vals.add(i) = bv;
                                *idxs.add(i) = bi;
                                *vals.add(other) = av;
                                *idxs.add(other) = ai;
                            }
                        }
                    }
                    i += bx;
                }
                thread::sync_threads();
                j >>= 1;
            }
            k <<= 1;
        }

        let mut i = tid;
        while i < top_k as usize {
            unsafe {
                *selected.add(t * top_k as usize + i) = *idxs.add(i);
            }
            i += bx;
        }
    }

    /// Tree-merge phase: merge `merge_group` candidate sets from a previous chunk-sort
    /// into one sorted set of `top_k` elements, writing to `out`.
    ///
    /// Grid: (n_tokens, n_groups, 1). `shared_mem_bytes = n_sort * 8`.
    /// Ported from indexer_topk_tree_merge_pow2_kernel.
    #[kernel]
    pub fn indexer_topk_tree_merge_bitonic(
        scores: &[f32],
        candidates: &[u32],
        out: *mut u32,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        n_sets: u32,
        merge_group: u32,
        candidate_stride: u32,
        out_stride: u32,
        n_sort: u32,
    ) {
        let t = thread::blockIdx_x() as usize;
        let group = thread::blockIdx_y() as usize;
        if t as u32 >= n_tokens {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let bx = thread::blockDim_x() as usize;
        let ns = n_sort as usize;

        let set0 = group * merge_group as usize;
        if set0 as u32 >= n_sets {
            return;
        }
        let rem_sets = n_sets as usize - set0;
        let set_count = if rem_sets < merge_group as usize {
            rem_sets
        } else {
            merge_group as usize
        };
        let candidate_count = set_count * top_k as usize;

        let vals: *mut f32 = DynamicSharedArray::<f32>::get();
        let idxs: *mut u32 = DynamicSharedArray::<u32>::offset(ns * 4);

        let cand_base = t * candidate_stride as usize + set0 * top_k as usize;
        let row_base = t * n_comp as usize;
        let mut i = tid;
        while i < ns {
            unsafe {
                let (idx, v) = if i < candidate_count {
                    let idx = candidates[cand_base + i];
                    let v = if idx < n_comp {
                        scores[row_base + idx as usize]
                    } else {
                        f32::NEG_INFINITY
                    };
                    (idx, v)
                } else {
                    (u32::MAX, f32::NEG_INFINITY)
                };
                *vals.add(i) = v;
                *idxs.add(i) = idx;
            }
            i += bx;
        }
        thread::sync_threads();

        let mut k = 2usize;
        while k <= ns {
            let mut j = k >> 1;
            while j > 0 {
                let mut i = tid;
                while i < ns {
                    let other = i ^ j;
                    if other > i && other < ns {
                        unsafe {
                            let av = *vals.add(i);
                            let bv = *vals.add(other);
                            let ai = *idxs.add(i);
                            let bi = *idxs.add(other);
                            let desc = (i & k) == 0;
                            let swap = if desc {
                                super::topk_score_better(bv, bi, av, ai)
                            } else {
                                super::topk_score_better(av, ai, bv, bi)
                            };
                            if swap {
                                *vals.add(i) = bv;
                                *idxs.add(i) = bi;
                                *vals.add(other) = av;
                                *idxs.add(other) = ai;
                            }
                        }
                    }
                    i += bx;
                }
                thread::sync_threads();
                j >>= 1;
            }
            k <<= 1;
        }

        let dst = unsafe { out.add(t * out_stride as usize + group * top_k as usize) };
        let mut i = tid;
        while i < top_k as usize {
            unsafe {
                *dst.add(i) = *idxs.add(i);
            }
            i += bx;
        }
    }

    /// Ascending bitonic sort of 512 i32 indices (for ordered attention processing).
    ///
    /// Grid: (n_tokens, 1, 1), Block: (512, 1, 1).
    /// Ported from indexed_topk_sort_512_asc_kernel.
    #[kernel]
    pub fn indexed_topk_sort_512_asc(src: &[i32], dst: *mut i32, n_tokens: u32) {
        static mut ROWS: SharedArray<i32, 512> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        if t as u32 >= n_tokens || tid >= 512 {
            return;
        }

        unsafe {
            ROWS[tid] = src[t * 512 + tid];
        }
        thread::sync_threads();

        let mut k = 2usize;
        while k <= 512 {
            let mut j = k >> 1;
            while j > 0 {
                let other = tid ^ j;
                if other > tid && other < 512 {
                    unsafe {
                        let a = ROWS[tid];
                        let b = ROWS[other];
                        let up = (tid & k) == 0;
                        if (up && a > b) || (!up && a < b) {
                            ROWS[tid] = b;
                            ROWS[other] = a;
                        }
                    }
                }
                thread::sync_threads();
                j >>= 1;
            }
            k <<= 1;
        }

        unsafe {
            *dst.add(t * 512 + tid) = ROWS[tid];
        }
    }

    /// Build a float attention mask from a top-K index list.
    ///
    /// `mask[t, c] = 0.0` if c is in topk[t], else `-∞`.
    ///
    /// Grid: covers `n_tokens * n_comp` threads (1D).
    /// Ported from topk_mask_kernel.
    #[kernel]
    pub fn topk_mask(
        topk: &[u32],
        mut mask: DisjointSlice<f32>,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
    ) {
        let idx = thread::index_1d();
        let gid = idx.get() as u64;
        let n = n_tokens as u64 * n_comp as u64;
        if gid >= n {
            return;
        }
        let t = (gid / n_comp as u64) as usize;
        let c = (gid % n_comp as u64) as u32;
        let mut v = f32::NEG_INFINITY;
        for k in 0..top_k as usize {
            if topk[t * top_k as usize + k] == c {
                v = 0.0;
                break;
            }
        }
        if let Some(o) = mask.get_mut(idx) {
            *o = v;
        }
    }
}
