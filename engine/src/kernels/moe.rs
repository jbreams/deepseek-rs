use core::sync::atomic::{AtomicU32, Ordering};
use cuda_device::thread::{blockDim_y, threadIdx_y};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

// ============================================================================
// Device helpers
// ============================================================================

/// Numerically stable softplus: `ln(1 + exp(x))`.
/// Ported from softplus_dev.
#[cuda_device::device]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    }
    // TODO(cuda-oxide): f32::ln_1p calls std::sys::cmath::log1pf which is forbidden
    // in device code (only core + cuda_device are allowed). Replace `(1.0 + x.exp()).ln()`
    // with `x.exp().ln_1p()` once libm/cmath is available in device context.
    else {
        (1.0f32 + x.exp()).ln()
    }
}

/// True if (av, ai) scores better than (bv, bi): higher value wins; ties go to lower index.
/// Ported from router_score_better / topk_score_better.
#[cuda_device::device]
pub fn router_score_better(av: f32, ai: u32, bv: f32, bi: u32) -> bool {
    av > bv || (av == bv && ai < bi)
}

/// Sum `v` across an 8-lane sub-warp using XOR shuffles (butterfly reduction).
/// All 8 lanes end up with the same total. Lane 0 writes the result.
/// Ported from quarter_warp_sum_f32.
#[cuda_device::device]
pub fn quarter_warp_sum(mut v: f32) -> f32 {
    v += warp::shuffle_xor_f32(v, 4);
    v += warp::shuffle_xor_f32(v, 2);
    v += warp::shuffle_xor_f32(v, 1);
    v
}

// ============================================================================
// Kernels
// ============================================================================

#[cuda_module]
pub mod moe {
    use super::*;

    // -----------------------------------------------------------------------
    // Router: top-6 expert selection
    // -----------------------------------------------------------------------

    /// Single-threaded expert selection (one thread per token).
    ///
    /// Computes `sqrt(softplus(logit))` probabilities, selects top-6 by
    /// insertion sort, then normalises weights to sum ≈ 1.5.
    ///
    /// Grid: (n_tokens, 1, 1), Block: (1, 1, 1).
    /// Ported from router_select_kernel.
    #[kernel]
    pub fn router_select(
        logits: &[f32],
        bias: &[f32],
        hash: &[i32],
        tokens: &[i32],
        mut probs: DisjointSlice<f32>,
        mut selected: DisjointSlice<i32>,
        mut weights: DisjointSlice<f32>,
        token_scalar: i32,
        hash_rows: u32,
        n_tokens: u32,
        has_bias: u32,
        hash_mode: u32,
    ) {
        let t = thread::blockIdx_x() as usize;
        if t as u32 >= n_tokens || thread::threadIdx_x() != 0 {
            return;
        }

        let log_base = t * 256;
        for i in 0..256usize {
            let p = super::softplus(logits[log_base + i]).sqrt();
            unsafe {
                *probs.get_unchecked_mut(log_base + i) = p;
            }
        }

        let mut sel = [i32::MIN; 6];
        if hash_mode != 0 {
            let tok = {
                let raw = if !tokens.is_empty() {
                    tokens[t]
                } else {
                    token_scalar
                };
                if raw < 0 || raw as u32 >= hash_rows {
                    0
                } else {
                    raw as usize
                }
            };
            for i in 0..6usize {
                unsafe {
                    *selected.get_unchecked_mut(t * 6 + i) = hash[tok * 6 + i];
                }
            }
        } else {
            let prob_base = t * 256;
            for i in 0..6usize {
                sel[i] = -1i32;
            }
            for e in 0i32..256 {
                let p_e = unsafe { *probs.get_unchecked_mut(prob_base + e as usize) };
                let s_e = p_e + if has_bias != 0 { bias[e as usize] } else { 0.0 };
                for j in 0..6usize {
                    let cur = sel[j];
                    let s_cur = if cur < 0 {
                        f32::NEG_INFINITY
                    } else {
                        let p_c = unsafe { *probs.get_unchecked_mut(prob_base + cur as usize) };
                        p_c + if has_bias != 0 {
                            bias[cur as usize]
                        } else {
                            0.0
                        }
                    };
                    if s_e > s_cur {
                        let mut k = 5usize;
                        while k > j {
                            sel[k] = sel[k - 1];
                            k -= 1;
                        }
                        sel[j] = e;
                        break;
                    }
                }
            }
            for i in 0..6usize {
                unsafe {
                    *selected.get_unchecked_mut(t * 6 + i) = sel[i];
                }
            }
        }

        let mut sum = 0.0f32;
        for i in 0..6usize {
            let e = unsafe { *selected.get_unchecked_mut(t * 6 + i) };
            let v = if e >= 0 && e < 256 {
                unsafe { *probs.get_unchecked_mut(t * 256 + e as usize) }
            } else {
                0.0
            };
            unsafe {
                *weights.get_unchecked_mut(t * 6 + i) = v;
            }
            sum += v;
        }
        if sum < 6.103_515_625e-5 {
            sum = 6.103_515_625e-5;
        }
        for i in 0..6usize {
            unsafe {
                *weights.get_unchecked_mut(t * 6 + i) /= sum / 1.5;
            }
        }
    }

    /// Parallel expert selection: 256 threads compute probabilities simultaneously,
    /// then thread 0 does insertion sort and normalisation.
    ///
    /// Grid: (n_tokens, 1, 1), Block: (256, 1, 1).
    /// Ported from router_select_parallel_kernel.
    #[kernel]
    pub fn router_select_parallel(
        logits: &[f32],
        bias: &[f32],
        hash: &[i32],
        tokens: &[i32],
        mut probs: DisjointSlice<f32>,
        mut selected: DisjointSlice<i32>,
        mut weights: DisjointSlice<f32>,
        token_scalar: i32,
        hash_rows: u32,
        n_tokens: u32,
        has_bias: u32,
        hash_mode: u32,
    ) {
        static mut SPROB: SharedArray<f32, 256> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        let i = thread::threadIdx_x() as usize;
        if t as u32 >= n_tokens || i >= 256 {
            return;
        }

        let p = super::softplus(logits[t * 256 + i]).sqrt();
        unsafe {
            SPROB[i] = p;
        }
        unsafe {
            *probs.get_unchecked_mut(t * 256 + i) = p;
        }
        thread::sync_threads();
        if i != 0 {
            return;
        }

        if hash_mode != 0 {
            let tok = {
                let raw = if !tokens.is_empty() {
                    tokens[t]
                } else {
                    token_scalar
                };
                if raw < 0 || raw as u32 >= hash_rows {
                    0
                } else {
                    raw as usize
                }
            };
            for j in 0..6usize {
                unsafe {
                    *selected.get_unchecked_mut(t * 6 + j) = hash[tok * 6 + j];
                }
            }
        } else {
            let get_score = |e: usize| -> f32 {
                let p = unsafe { SPROB[e] };
                p + if has_bias != 0 { bias[e] } else { 0.0 }
            };
            for j in 0..6usize {
                unsafe {
                    *selected.get_unchecked_mut(t * 6 + j) = -1;
                }
            }
            for e in 0..256usize {
                let score = get_score(e);
                for j in 0..6usize {
                    let cur = unsafe { *selected.get_unchecked_mut(t * 6 + j) };
                    let cur_score = if cur < 0 {
                        f32::NEG_INFINITY
                    } else {
                        get_score(cur as usize)
                    };
                    if score > cur_score {
                        let mut k = 5usize;
                        while k > j {
                            let prev = unsafe { *selected.get_unchecked_mut(t * 6 + k - 1) };
                            unsafe {
                                *selected.get_unchecked_mut(t * 6 + k) = prev;
                            }
                            k -= 1;
                        }
                        unsafe {
                            *selected.get_unchecked_mut(t * 6 + j) = e as i32;
                        }
                        break;
                    }
                }
            }
        }

        let mut sum = 0.0f32;
        for j in 0..6usize {
            let e = unsafe { *selected.get_unchecked_mut(t * 6 + j) };
            let v = if e >= 0 && e < 256 {
                unsafe { SPROB[e as usize] }
            } else {
                0.0
            };
            unsafe {
                *weights.get_unchecked_mut(t * 6 + j) = v;
            }
            sum += v;
        }
        if sum < 6.103_515_625e-5 {
            sum = 6.103_515_625e-5;
        }
        for j in 0..6usize {
            unsafe {
                *weights.get_unchecked_mut(t * 6 + j) /= sum / 1.5;
            }
        }
    }

    // -----------------------------------------------------------------------
    // MoE pair sorting helpers
    // -----------------------------------------------------------------------

    /// Count how many (token, slot) pairs route to each expert.
    /// Uses global atomic increments into `counts[expert_id]`.
    ///
    /// Grid: covers `pair_count` threads (1D).
    /// Ported from moe_count_sorted_pairs_kernel.
    #[kernel]
    pub fn moe_count_sorted_pairs(selected: &[i32], counts: *mut u32, pair_count: u32) {
        let pair = thread::index_1d().get() as u32;
        if pair >= pair_count {
            return;
        }
        let expert_i = {
            let e = selected[pair as usize];
            if e < 0 { 0u32 } else { e as u32 }
        };
        unsafe {
            let atomic = AtomicU32::from_ptr(counts.add(expert_i as usize));
            (*atomic).fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Sequential prefix-sum over 256 expert counts → per-expert offsets/cursors.
    ///
    /// Grid: (1, 1, 1), Block: (1, 1, 1).
    /// Ported from moe_prefix_sorted_pairs_kernel.
    #[kernel]
    pub fn moe_prefix_sorted_pairs(counts: &[u32], offsets: *mut u32, cursors: *mut u32) {
        if thread::threadIdx_x() != 0 {
            return;
        }
        let mut sum = 0u32;
        for e in 0..256usize {
            unsafe {
                *offsets.add(e) = sum;
                *cursors.add(e) = sum;
            }
            sum += counts[e];
        }
        unsafe {
            *offsets.add(256) = sum;
        }
    }

    /// Scatter pairs into expert-sorted order using per-expert atomic cursors.
    ///
    /// Grid: covers `pair_count` threads (1D).
    /// Ported from moe_scatter_sorted_pairs_kernel.
    #[kernel]
    pub fn moe_scatter_sorted_pairs(
        selected: &[i32],
        cursors: *mut u32,
        mut sorted_pairs: DisjointSlice<u32>,
        pair_count: u32,
    ) {
        let pair = thread::index_1d().get() as u32;
        if pair >= pair_count {
            return;
        }
        let expert_i = {
            let e = selected[pair as usize];
            if e < 0 { 0u32 } else { e as u32 }
        };
        let pos = unsafe {
            let atomic = AtomicU32::from_ptr(cursors.add(expert_i as usize));
            (*atomic).fetch_add(1, Ordering::Relaxed)
        };
        unsafe {
            *sorted_pairs.get_unchecked_mut(pos as usize) = pair;
        }
    }

    // -----------------------------------------------------------------------
    // Router: warp-parallel top-6 selection
    // -----------------------------------------------------------------------

    /// Warp-level top-6 expert selection: one warp (32 lanes) per token.
    /// Each lane owns 8 of the 256 experts. Warp shuffle reduces to find the
    /// global top-6 with score breaks by index.
    ///
    /// Grid: (ceil(n_tokens / rows_per_block), 1, 1).
    /// Block: (32, rows_per_block, 1) — threadIdx.x = lane (0..32), threadIdx.y = row in block.
    /// Ported from router_select_warp_topk_kernel.
    #[kernel]
    pub fn router_select_warp_topk(
        logits: &[f32],
        bias: &[f32],
        hash: &[i32],
        tokens: &[i32],
        mut probs: DisjointSlice<f32>,
        mut selected: DisjointSlice<i32>,
        mut weights: DisjointSlice<f32>,
        token_scalar: i32,
        hash_rows: u32,
        n_tokens: u32,
        has_bias: u32,
        hash_mode: u32,
    ) {
        static mut SPROB: SharedArray<f32, 1024> = SharedArray::UNINIT; // 4 rows × 256

        let lane = thread::threadIdx_x() as usize;
        let row_in_block = threadIdx_y() as usize;
        let rows_per_block = blockDim_y() as usize;
        let t = (thread::blockIdx_x() as usize) * rows_per_block + row_in_block;
        if t as u32 >= n_tokens || lane >= 32 {
            return;
        }

        let log_base = t * 256;

        // Each lane computes 8 probabilities (256/32 = 8) and stores to shared mem
        let mut local_prob = [0.0f32; 8];
        let mut local_score = [0.0f32; 8];
        let mut j = 0usize;
        while j < 8 {
            let e = lane + j * 32;
            let p = super::softplus(logits[log_base + e]).sqrt();
            local_prob[j] = p;
            local_score[j] = p + if has_bias != 0 { bias[e] } else { 0.0 };
            unsafe {
                SPROB[row_in_block * 256 + e] = p;
            }
            unsafe {
                *probs.get_unchecked_mut(log_base + e) = p;
            }
            j += 1;
        }
        thread::sync_threads();

        if hash_mode != 0 {
            if lane == 0 {
                let raw = if !tokens.is_empty() {
                    tokens[t]
                } else {
                    token_scalar
                };
                let tok = if raw < 0 || raw as u32 >= hash_rows {
                    0
                } else {
                    raw as usize
                };
                let mut sum = 0.0f32;
                let mut j = 0usize;
                while j < 6 {
                    let e = hash[tok * 6 + j];
                    unsafe {
                        *selected.get_unchecked_mut(t * 6 + j) = e;
                    }
                    let v = if e >= 0 && e < 256 {
                        unsafe { SPROB[row_in_block * 256 + e as usize] }
                    } else {
                        0.0
                    };
                    unsafe {
                        *weights.get_unchecked_mut(t * 6 + j) = v;
                    }
                    sum += v;
                    j += 1;
                }
                if sum < 6.103_515_625e-5 {
                    sum = 6.103_515_625e-5;
                }
                let mut j = 0usize;
                while j < 6 {
                    unsafe {
                        *weights.get_unchecked_mut(t * 6 + j) /= sum / 1.5;
                    }
                    j += 1;
                }
            }
            return;
        }

        // Warp-parallel top-6: each round finds the overall max across 256 experts
        let mut out_idx = [0u32; 6];
        let mut out_prob = [0.0f32; 6];
        let mut k = 0usize;
        while k < 6 {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_prob = 0.0f32;
            let mut best_idx = u32::MAX;
            let mut j = 0usize;
            while j < 8 {
                let e = (lane + j * 32) as u32;
                let s = local_score[j];
                if super::router_score_better(s, e, best_score, best_idx) {
                    best_score = s;
                    best_prob = local_prob[j];
                    best_idx = e;
                }
                j += 1;
            }
            // Warp reduction: find best across all 32 lanes
            let mut mask = 16u32;
            while mask > 0 {
                let other_score = warp::shuffle_xor_f32(best_score, mask);
                let other_prob = warp::shuffle_xor_f32(best_prob, mask);
                let other_idx = warp::shuffle_xor(best_idx, mask);
                if super::router_score_better(other_score, other_idx, best_score, best_idx) {
                    best_score = other_score;
                    best_prob = other_prob;
                    best_idx = other_idx;
                }
                mask >>= 1;
            }
            // Mask out the winner so it's not selected again
            let mut j = 0usize;
            while j < 8 {
                let e = (lane + j * 32) as u32;
                if e == best_idx {
                    local_score[j] = f32::NEG_INFINITY;
                }
                j += 1;
            }
            if lane == 0 {
                out_idx[k] = best_idx;
                out_prob[k] = best_prob;
            }
            k += 1;
        }

        if lane == 0 {
            let mut sum = 0.0f32;
            let mut j = 0usize;
            while j < 6 {
                unsafe {
                    *selected.get_unchecked_mut(t * 6 + j) = out_idx[j] as i32;
                }
                unsafe {
                    *weights.get_unchecked_mut(t * 6 + j) = out_prob[j];
                }
                sum += out_prob[j];
                j += 1;
            }
            if sum < 6.103_515_625e-5 {
                sum = 6.103_515_625e-5;
            }
            let mut j = 0usize;
            while j < 6 {
                unsafe {
                    *weights.get_unchecked_mut(t * 6 + j) /= sum / 1.5;
                }
                j += 1;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Expert tile helpers
    // -----------------------------------------------------------------------

    /// Single-thread prefix-sum to build per-expert tile offsets and total tile count.
    /// Grid: (1,1,1), Block: (1,1,1).
    /// Ported from moe_build_expert_tile_offsets_kernel.
    #[kernel]
    pub fn moe_build_expert_tile_offsets(
        counts: &[u32],
        tile_offsets: *mut u32,
        tile_total: *mut u32,
        block_m: u32,
    ) {
        if thread::threadIdx_x() != 0 {
            return;
        }
        let mut sum = 0u32;
        let mut e = 0usize;
        while e < 256 {
            unsafe {
                *tile_offsets.add(e) = sum;
            }
            sum += (counts[e] + block_m - 1) / block_m;
            e += 1;
        }
        unsafe {
            *tile_offsets.add(256) = sum;
            *tile_total = sum;
        }
    }

    /// Scatter expert indices and tile starts into the tile lists.
    /// Grid: (1,1,1), Block: (256,1,1) — one thread per expert.
    /// Ported from moe_build_expert_tiles_kernel.
    #[kernel]
    pub fn moe_build_expert_tiles(
        counts: &[u32],
        tile_offsets: &[u32],
        tile_experts: *mut u32,
        tile_starts: *mut u32,
        block_m: u32,
    ) {
        let e = thread::threadIdx_x() as usize;
        if e >= 256 {
            return;
        }
        let ntiles = ((counts[e] + block_m - 1) / block_m) as usize;
        let off = tile_offsets[e] as usize;
        let mut t = 0usize;
        while t < ntiles {
            unsafe {
                *tile_experts.add(off + t) = e as u32;
                *tile_starts.add(off + t) = (t as u32) * block_m;
            }
            t += 1;
        }
    }

    // -----------------------------------------------------------------------
    // MoE gate/up/down projections (IQ2_XXS × Q8_K and Q2_K × Q8_K)
    // -----------------------------------------------------------------------
    // Block layouts (passed as raw byte slices):
    //   IQ2_XXS block: 66 bytes  [ u16 d | u16 qs[32] ]
    //   Q2_K block:    84 bytes  [ u8 scales[16] | u8 qs[64] | u16 d | u16 dmin ]
    //   Q8_K block:   292 bytes  [ f32 d | i8 qs[256] | i16 bsums[16] ]
    // TODO(cuda-oxide): swap scalar dot products for SIMD (__dp4a, __vsub4,
    // __vcmpne4) once those PTX intrinsics are available in cuda-oxide device code.
    // TODO: Q4_K and decode-LUT variants.

    const IQ2_XXS_BLOCK_BYTES: u64 = 66;
    const Q2K_BLOCK_BYTES: u64 = 84;
    const Q8K_BLOCK_BYTES: u64 = 292;

    /// Gate+up+mid projection (IQ2_XXS weights × Q8_K input), one row per quarter-warp.
    ///
    /// 4 rows per block: blockIdx.x * 128 + (threadIdx.x >> 3) + rr * 32 for rr in 0..4.
    /// lane = threadIdx.x & 7.
    ///
    /// Grid: (ceil(expert_mid_dim / 128), n_tokens * n_expert, 1).
    /// Block: (256, 1, 1).
    /// Ported from moe_gate_up_mid_qwarp32_kernel.
    #[kernel]
    pub fn moe_gate_up_mid_qwarp32(
        gate_base: &[u8],
        up_base: &[u8],
        xq: &[u8],
        selected: &[i32],
        weights: &[f32],
        iq2_grid: &[u64],
        iq2_signs: &[u8],
        mut gate_out: DisjointSlice<f32>,
        mut up_out: DisjointSlice<f32>,
        mut mid_out: DisjointSlice<f32>,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        clamp: f32,
    ) {
        let lane = (thread::threadIdx_x() & 7) as usize;
        let row_lane = (thread::threadIdx_x() >> 3) as usize;
        let pair = thread::blockIdx_y() as usize;
        let tok = pair / n_expert as usize;
        let slot = pair - tok * n_expert as usize;
        let expert_i = {
            let e = selected[tok * n_expert as usize + slot];
            if e < 0 { 0u64 } else { e as u64 }
        };
        let xq_base_off = tok as u64 * xq_blocks as u64 * Q8K_BLOCK_BYTES;
        let grid_ptr = iq2_grid.as_ptr();
        let signs_ptr = iq2_signs.as_ptr();

        let mut rr = 0usize;
        while rr < 4 {
            let row = (thread::blockIdx_x() as usize) * 128 + row_lane + rr * 32;
            if row < expert_mid_dim as usize {
                let gate_off =
                    (expert_i * gate_expert_bytes + row as u64 * gate_row_bytes) as usize;
                let mut gate_acc = 0.0f32;
                let mut up_acc = 0.0f32;
                let mut b = lane;
                while (b as u32) < xq_blocks {
                    let q8_ptr = unsafe {
                        xq.as_ptr()
                            .add((xq_base_off as usize) + b * Q8K_BLOCK_BYTES as usize)
                    };
                    gate_acc += super::super::quantize::dot_iq2_xxs_q8k(
                        unsafe {
                            gate_base
                                .as_ptr()
                                .add(gate_off + b * IQ2_XXS_BLOCK_BYTES as usize)
                        },
                        q8_ptr,
                        grid_ptr,
                        signs_ptr,
                    );
                    up_acc += super::super::quantize::dot_iq2_xxs_q8k(
                        unsafe {
                            up_base
                                .as_ptr()
                                .add(gate_off + b * IQ2_XXS_BLOCK_BYTES as usize)
                        },
                        q8_ptr,
                        grid_ptr,
                        signs_ptr,
                    );
                    b += 8;
                }
                gate_acc = super::quarter_warp_sum(gate_acc);
                up_acc = super::quarter_warp_sum(up_acc);
                if lane == 0 {
                    if clamp > 1.0e-6 {
                        if gate_acc > clamp {
                            gate_acc = clamp;
                        }
                        if up_acc > clamp {
                            up_acc = clamp;
                        }
                        if up_acc < -clamp {
                            up_acc = -clamp;
                        }
                    }
                    let off = pair * expert_mid_dim as usize + row;
                    let w = weights[tok * n_expert as usize + slot];
                    unsafe {
                        *gate_out.get_unchecked_mut(off) = gate_acc;
                        *up_out.get_unchecked_mut(off) = up_acc;
                        *mid_out.get_unchecked_mut(off) =
                            (gate_acc / (1.0 + (-gate_acc).exp())) * up_acc * w;
                    }
                }
            }
            rr += 1;
        }
    }

    /// Sorted gate+up+mid projection (same as above but pair comes from sorted_pairs lookup).
    /// Grid: (ceil(expert_mid_dim / 128), n_sorted_pairs, 1). Block: (256, 1, 1).
    /// Ported from moe_gate_up_mid_sorted_qwarp32_kernel.
    #[kernel]
    pub fn moe_gate_up_mid_sorted_qwarp32(
        gate_base: &[u8],
        up_base: &[u8],
        xq: &[u8],
        sorted_pairs: &[u32],
        selected: &[i32],
        weights: &[f32],
        iq2_grid: &[u64],
        iq2_signs: &[u8],
        mut gate_out: DisjointSlice<f32>,
        mut up_out: DisjointSlice<f32>,
        mut mid_out: DisjointSlice<f32>,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        clamp: f32,
    ) {
        let lane = (thread::threadIdx_x() & 7) as usize;
        let row_lane = (thread::threadIdx_x() >> 3) as usize;
        let pair = sorted_pairs[thread::blockIdx_y() as usize] as usize;
        let tok = pair / n_expert as usize;
        let slot = pair - tok * n_expert as usize;
        let expert_i = {
            let e = selected[tok * n_expert as usize + slot];
            if e < 0 { 0u64 } else { e as u64 }
        };
        let xq_base_off = tok as u64 * xq_blocks as u64 * Q8K_BLOCK_BYTES;
        let grid_ptr = iq2_grid.as_ptr();
        let signs_ptr = iq2_signs.as_ptr();

        let mut rr = 0usize;
        while rr < 4 {
            let row = (thread::blockIdx_x() as usize) * 128 + row_lane + rr * 32;
            if row < expert_mid_dim as usize {
                let gate_off =
                    (expert_i * gate_expert_bytes + row as u64 * gate_row_bytes) as usize;
                let mut gate_acc = 0.0f32;
                let mut up_acc = 0.0f32;
                let mut b = lane;
                while (b as u32) < xq_blocks {
                    let q8_ptr = unsafe {
                        xq.as_ptr()
                            .add(xq_base_off as usize + b * Q8K_BLOCK_BYTES as usize)
                    };
                    gate_acc += super::super::quantize::dot_iq2_xxs_q8k(
                        unsafe {
                            gate_base
                                .as_ptr()
                                .add(gate_off + b * IQ2_XXS_BLOCK_BYTES as usize)
                        },
                        q8_ptr,
                        grid_ptr,
                        signs_ptr,
                    );
                    up_acc += super::super::quantize::dot_iq2_xxs_q8k(
                        unsafe {
                            up_base
                                .as_ptr()
                                .add(gate_off + b * IQ2_XXS_BLOCK_BYTES as usize)
                        },
                        q8_ptr,
                        grid_ptr,
                        signs_ptr,
                    );
                    b += 8;
                }
                gate_acc = super::quarter_warp_sum(gate_acc);
                up_acc = super::quarter_warp_sum(up_acc);
                if lane == 0 {
                    if clamp > 1.0e-6 {
                        if gate_acc > clamp {
                            gate_acc = clamp;
                        }
                        if up_acc > clamp {
                            up_acc = clamp;
                        }
                        if up_acc < -clamp {
                            up_acc = -clamp;
                        }
                    }
                    let off = pair * expert_mid_dim as usize + row;
                    let w = weights[tok * n_expert as usize + slot];
                    unsafe {
                        *gate_out.get_unchecked_mut(off) = gate_acc;
                        *up_out.get_unchecked_mut(off) = up_acc;
                        *mid_out.get_unchecked_mut(off) =
                            (gate_acc / (1.0 + (-gate_acc).exp())) * up_acc * w;
                    }
                }
            }
            rr += 1;
        }
    }

    /// Down-projection (Q2_K weights × Q8_K mid).
    /// Grid: (ceil(out_dim / 32), n_tokens * n_expert, 1). Block: (256, 1, 1).
    /// Ported from moe_down_qwarp32_kernel.
    #[kernel]
    pub fn moe_down_qwarp32(
        down_base: &[u8],
        midq: &[u8],
        selected: &[i32],
        mut down_out: DisjointSlice<f32>,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
    ) {
        let lane = (thread::threadIdx_x() & 7) as usize;
        let row = (thread::blockIdx_x() as usize) * 32 + (thread::threadIdx_x() >> 3) as usize;
        if row >= out_dim as usize {
            return;
        }
        let pair = thread::blockIdx_y() as usize;
        let tok = pair / n_expert as usize;
        let slot = pair - tok * n_expert as usize;
        let expert_i = {
            let e = selected[tok * n_expert as usize + slot];
            if e < 0 { 0u64 } else { e as u64 }
        };
        let wr_off = (expert_i * down_expert_bytes + row as u64 * down_row_bytes) as usize;
        let xq_base = pair * midq_blocks as usize * Q8K_BLOCK_BYTES as usize;
        let mut acc = 0.0f32;
        let mut b = lane;
        while (b as u32) < midq_blocks {
            acc += super::super::quantize::dot_q2k_q8k(
                unsafe {
                    down_base
                        .as_ptr()
                        .add(wr_off + b * Q2K_BLOCK_BYTES as usize)
                },
                unsafe { midq.as_ptr().add(xq_base + b * Q8K_BLOCK_BYTES as usize) },
            );
            b += 8;
        }
        acc = super::quarter_warp_sum(acc);
        if lane == 0 {
            unsafe {
                *down_out.get_unchecked_mut(pair * out_dim as usize + row) = acc;
            }
        }
    }

    /// Sorted down-projection (pair from sorted_pairs).
    /// Grid: (ceil(out_dim / 32), n_sorted_pairs, 1). Block: (256, 1, 1).
    /// Ported from moe_down_sorted_qwarp32_kernel.
    #[kernel]
    pub fn moe_down_sorted_qwarp32(
        down_base: &[u8],
        midq: &[u8],
        sorted_pairs: &[u32],
        selected: &[i32],
        mut down_out: DisjointSlice<f32>,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
    ) {
        let lane = (thread::threadIdx_x() & 7) as usize;
        let row = (thread::blockIdx_x() as usize) * 32 + (thread::threadIdx_x() >> 3) as usize;
        if row >= out_dim as usize {
            return;
        }
        let pair = sorted_pairs[thread::blockIdx_y() as usize] as usize;
        let tok = pair / n_expert as usize;
        let slot = pair - tok * n_expert as usize;
        let expert_i = {
            let e = selected[tok * n_expert as usize + slot];
            if e < 0 { 0u64 } else { e as u64 }
        };
        let wr_off = (expert_i * down_expert_bytes + row as u64 * down_row_bytes) as usize;
        let xq_base = pair * midq_blocks as usize * Q8K_BLOCK_BYTES as usize;
        let mut acc = 0.0f32;
        let mut b = lane;
        while (b as u32) < midq_blocks {
            acc += super::super::quantize::dot_q2k_q8k(
                unsafe {
                    down_base
                        .as_ptr()
                        .add(wr_off + b * Q2K_BLOCK_BYTES as usize)
                },
                unsafe { midq.as_ptr().add(xq_base + b * Q8K_BLOCK_BYTES as usize) },
            );
            b += 8;
        }
        acc = super::quarter_warp_sum(acc);
        if lane == 0 {
            unsafe {
                *down_out.get_unchecked_mut(pair * out_dim as usize + row) = acc;
            }
        }
    }

    /// Fused down-projection + sum over all 6 expert slots for a single token.
    /// Grid: (ceil(out_dim / 32), 1, 1). Block: (256, 1, 1).
    /// Ported from moe_down_sum6_qwarp32_kernel.
    #[kernel]
    pub fn moe_down_sum6_qwarp32(
        down_base: &[u8],
        midq: &[u8],
        selected: &[i32],
        mut out: DisjointSlice<f32>,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        midq_blocks: u32,
        out_dim: u32,
    ) {
        let lane = (thread::threadIdx_x() & 7) as usize;
        let row = (thread::blockIdx_x() as usize) * 32 + (thread::threadIdx_x() >> 3) as usize;
        if row >= out_dim as usize {
            return;
        }
        let mut total = 0.0f32;
        let mut slot = 0usize;
        while slot < 6 {
            let expert_i = {
                let e = selected[slot];
                if e < 0 { 0u64 } else { e as u64 }
            };
            let wr_off = (expert_i * down_expert_bytes + row as u64 * down_row_bytes) as usize;
            let xq_base = slot * midq_blocks as usize * Q8K_BLOCK_BYTES as usize;
            let mut acc = 0.0f32;
            let mut b = lane;
            while (b as u32) < midq_blocks {
                acc += super::super::quantize::dot_q2k_q8k(
                    unsafe {
                        down_base
                            .as_ptr()
                            .add(wr_off + b * Q2K_BLOCK_BYTES as usize)
                    },
                    unsafe { midq.as_ptr().add(xq_base + b * Q8K_BLOCK_BYTES as usize) },
                );
                b += 8;
            }
            acc = super::quarter_warp_sum(acc);
            if lane == 0 {
                total += acc;
            }
            slot += 1;
        }
        if lane == 0 {
            unsafe {
                *out.get_unchecked_mut(row) = total;
            }
        }
    }

    // -----------------------------------------------------------------------
    // MoE output accumulation
    // -----------------------------------------------------------------------

    /// Sum expert down-projection outputs across all `n_expert` slots for each token.
    ///
    /// `out[tok, row] = Σ_{e=0}^{n_expert} down[(tok * n_expert + e) * out_dim + row]`
    ///
    /// Grid: covers `n_tokens * out_dim` threads (1D).
    /// Ported from moe_sum_kernel.
    #[kernel]
    pub fn moe_sum(
        down: &[f32],
        mut out: DisjointSlice<f32>,
        out_dim: u32,
        n_expert: u32,
        n_tokens: u32,
    ) {
        let idx = thread::index_1d();
        let gid = idx.get() as u64;
        let n = n_tokens as u64 * out_dim as u64;
        if gid >= n {
            return;
        }
        let tok = (gid / out_dim as u64) as usize;
        let row = (gid % out_dim as u64) as usize;
        let mut acc = 0.0f32;
        for e in 0..n_expert as usize {
            acc += down[(tok * n_expert as usize + e) * out_dim as usize + row];
        }
        if let Some(o) = out.get_mut(idx) {
            *o = acc;
        }
    }
}
