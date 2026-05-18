/// CPU-side regression tests for quantization dot-product algorithms and router math.
///
/// These mirror the logic in src/kernels/quantize.rs (device functions) using
/// pure-Rust implementations so they run without a GPU.  If a bug is introduced
/// in the algorithm — e.g. wrong bit extraction, wrong scale factor, wrong
/// sign handling — these catch it before any CUDA compilation happens.
use deepseek_engine::kernels::quantize::{IQ2_SIGNS, IQ2_XXS_GRID};

// ── helpers ──────────────────────────────────────────────────────────────────

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let bits32 = if exp == 0 {
        (sign << 31) | (mant << 13)
    } else if exp == 31 {
        (sign << 31) | 0x7f800000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits32)
}

const F16_ONE: u16 = 0x3C00; // f16 bits for 1.0

/// Q8_0 dot product for one output element (one weight row × pre-quantized input).
/// Mirrors matmul_q8_0_preq_warp8 inner loop logic.
fn q8_0_matvec_row(w: &[u8], xq: &[i8], xscale: &[f32], n_blocks: usize) -> f32 {
    let mut acc = 0.0f32;
    for b in 0..n_blocks {
        let blk = &w[b * 34..];
        let scale_w = f16_bits_to_f32((blk[0] as u16) | ((blk[1] as u16) << 8));
        let dot: i32 = (0..32)
            .map(|i| blk[2 + i] as i8 as i32 * xq[b * 32 + i] as i32)
            .sum();
        acc += scale_w * xscale[b] * dot as f32;
    }
    acc
}

/// IQ2_XXS + Q8_K block dot product. Mirrors dot_iq2_xxs_q8k device function.
fn iq2_xxs_q8k_dot(iq2_blk: &[u8], q8k_blk: &[u8]) -> f32 {
    let xd = f16_bits_to_f32((iq2_blk[0] as u16) | ((iq2_blk[1] as u16) << 8));
    let yd = f32::from_le_bytes([q8k_blk[0], q8k_blk[1], q8k_blk[2], q8k_blk[3]]);
    let q8_base = &q8k_blk[4..]; // i8[256] at offset 4

    let mut bsum = 0i32;
    for ib32 in 0..8usize {
        let q2 = &iq2_blk[2 + ib32 * 8..];
        let w0 = (q2[0] as u32) | ((q2[1] as u32) << 8);
        let w1 = (q2[2] as u32) | ((q2[3] as u32) << 8);
        let w2 = (q2[4] as u32) | ((q2[5] as u32) << 8);
        let w3 = (q2[6] as u32) | ((q2[7] as u32) << 8);
        let aux0 = w0 | (w1 << 16);
        let aux1 = w2 | (w3 << 16);
        let ls = (2 * (aux1 >> 28) + 1) as i32;
        let a = [
            (aux0 & 0xFF) as usize,
            ((aux0 >> 8) & 0xFF) as usize,
            ((aux0 >> 16) & 0xFF) as usize,
            ((aux0 >> 24) & 0xFF) as usize,
        ];
        let si = [
            ((aux1 >> 0) & 127) as usize,
            ((aux1 >> 7) & 127) as usize,
            ((aux1 >> 14) & 127) as usize,
            ((aux1 >> 21) & 127) as usize,
        ];
        let mut sumi = 0i32;
        for (group, (&ai, &sii)) in a.iter().zip(si.iter()).enumerate() {
            let grid = IQ2_XXS_GRID[ai];
            let sign_raw = IQ2_SIGNS[sii];
            let p = sign_raw.count_ones() & 1;
            let sb = sign_raw ^ ((p as u8) << 7);
            let q8_off = ib32 * 32 + group * 8;
            for k in 0..8usize {
                let gv = ((grid >> (k * 8)) & 0xFF) as u8;
                let neg = (sb >> k) & 1 != 0;
                let wk: i8 = if neg { -(gv as i8) } else { gv as i8 };
                sumi += wk as i32 * q8_base[q8_off + k] as i8 as i32;
            }
        }
        bsum += sumi * ls;
    }
    0.125 * xd * yd * bsum as f32
}

/// Q2_K + Q8_K block dot product. Mirrors dot_q2k_q8k device function.
fn q2k_q8k_dot(q2k_blk: &[u8], q8k_blk: &[u8]) -> f32 {
    let scales = &q2k_blk[..16];
    let q2_ptr = &q2k_blk[16..80];
    let xd = f16_bits_to_f32((q2k_blk[80] as u16) | ((q2k_blk[81] as u16) << 8));
    let xdmin = f16_bits_to_f32((q2k_blk[82] as u16) | ((q2k_blk[83] as u16) << 8));
    let yd = f32::from_le_bytes([q8k_blk[0], q8k_blk[1], q8k_blk[2], q8k_blk[3]]);
    let q8_base = &q8k_blk[4..260]; // i8[256]
    // bsums at offset 260: i16[16]
    let bsums: Vec<i32> = (0..16)
        .map(|j| {
            let off = 260 + j * 2;
            i16::from_le_bytes([q8k_blk[off], q8k_blk[off + 1]]) as i32
        })
        .collect();

    let dall = yd * xd;
    let dmin = yd * xdmin;

    let mut summs = 0i32;
    for j in 0..16 {
        summs += bsums[j] * (scales[j] >> 4) as i32;
    }

    let mut isum = 0i32;
    let mut is = 0usize;
    let mut q2_off = 0usize;
    let mut q8_off = 0usize;
    for _k in 0..2 {
        let mut shift = 0i32;
        for _j in 0..4 {
            let d = (scales[is] & 0x0f) as i32;
            is += 1;
            let d2 = (scales[is] & 0x0f) as i32;
            is += 1;
            // dot_q2_16 for 16 elements at q2_off/q8_off and q2_off+16/q8_off+16
            for &(qo, q2o) in &[(q8_off, q2_off), (q8_off + 16, q2_off + 16)] {
                let di = if qo == q8_off { d } else { d2 };
                let mut sum = 0i32;
                let mut i = 0usize;
                while i < 16 {
                    let b0 = q2_ptr[q2o + i] as u32;
                    let b1 = q2_ptr[q2o + i + 1] as u32;
                    let b2 = q2_ptr[q2o + i + 2] as u32;
                    let b3 = q2_ptr[q2o + i + 3] as u32;
                    let raw = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)) as i32;
                    let v_packed = (raw >> shift) & 0x03030303i32;
                    for j2 in 0..4 {
                        let q2v = ((v_packed >> (j2 * 8)) & 0x03) as i32;
                        let q8v = q8_base[qo + i + j2] as i8 as i32;
                        sum += q2v * q8v;
                    }
                    i += 4;
                }
                isum += di * sum;
            }
            shift += 2;
            q8_off += 32;
        }
        q2_off += 32;
    }
    dall * isum as f32 - dmin * summs as f32
}

// ── Q8_0 tests ───────────────────────────────────────────────────────────────

#[test]
fn q8_0_zero_weights_gives_zero() {
    // Any input × zero-weight block = 0.
    let w = {
        let mut b = vec![0u8; 34];
        b[0] = 0x00;
        b[1] = 0x3C; // scale = f16 1.0
        // qs all zero (already zero)
        b
    };
    let xq = vec![1i8; 32];
    let xscale = vec![1.0f32];
    assert_eq!(q8_0_matvec_row(&w, &xq, &xscale, 1), 0.0);
}

#[test]
fn q8_0_zero_input_gives_zero() {
    // Zero input × any weights = 0.
    let w = {
        let mut b = vec![1u8; 34]; // qs all 1
        b[0] = 0x00;
        b[1] = 0x3C; // scale = f16 1.0
        b
    };
    let xq = vec![0i8; 32];
    let xscale = vec![1.0f32];
    assert_eq!(q8_0_matvec_row(&w, &xq, &xscale, 1), 0.0);
}

#[test]
fn q8_0_all_ones_single_block() {
    // w_scale=1, w_qs=[1]*32, x_scale=1, xq=[1]*32 → dot = 32
    let w = {
        let mut b = vec![0u8; 34];
        b[0] = 0x00;
        b[1] = 0x3C; // f16 1.0
        for i in 2..34 {
            b[i] = 1u8;
        } // qs = 1 (as u8, but treated as i8 = 1)
        b
    };
    let xq = vec![1i8; 32];
    let xscale = vec![1.0f32];
    let result = q8_0_matvec_row(&w, &xq, &xscale, 1);
    assert!((result - 32.0).abs() < 1e-4, "expected 32.0, got {result}");
}

#[test]
fn q8_0_two_blocks_accumulate() {
    // Two blocks, each contributing 16.0 → total 32.0
    let mut w = vec![0u8; 68]; // 2 × 34
    // block 0: scale=0.5, qs=[1]*32
    let half_f16: u16 = 0x3800; // f16 0.5
    w[0] = (half_f16 & 0xff) as u8;
    w[1] = (half_f16 >> 8) as u8;
    for i in 2..34 {
        w[i] = 1u8;
    }
    // block 1: scale=0.5, qs=[1]*32
    w[34] = (half_f16 & 0xff) as u8;
    w[35] = (half_f16 >> 8) as u8;
    for i in 36..68 {
        w[i] = 1u8;
    }
    let xq = vec![1i8; 64];
    let xscale = vec![1.0f32, 1.0f32];
    let result = q8_0_matvec_row(&w, &xq, &xscale, 2);
    // 0.5 * 1.0 * 32 + 0.5 * 1.0 * 32 = 32.0
    assert!((result - 32.0).abs() < 1e-4, "expected 32.0, got {result}");
}

#[test]
fn q8_0_negative_weights() {
    // Negative qs values produce negative dot products.
    let w = {
        let mut b = vec![0u8; 34];
        b[0] = 0x00;
        b[1] = 0x3C; // f16 1.0
        for i in 2..34 {
            b[i] = 0xFFu8;
        } // i8 -1
        b
    };
    let xq = vec![1i8; 32];
    let xscale = vec![1.0f32];
    let result = q8_0_matvec_row(&w, &xq, &xscale, 1);
    assert!(
        (result - (-32.0)).abs() < 1e-4,
        "expected -32.0, got {result}"
    );
}

// ── IQ2_XXS tests ────────────────────────────────────────────────────────────

/// Build a minimal IQ2_XXS block (66 bytes).
/// All 8 groups use grid[grid_idx] and sign[sign_idx] with ls factor.
fn make_iq2_blk(d_bits: u16, grid_idx: u8, sign_idx: u8, ls_val: u32) -> Vec<u8> {
    // ls_val: desired ls (1..=31). aux1 >> 28 = (ls_val - 1) / 2
    let aux1_top = ((ls_val - 1) / 2) << 28;
    // sign_idx fits in 7 bits; pack 4 sign indices into aux1 bits [0,7,14,21]
    let aux1 = aux1_top
        | (sign_idx as u32)
        | ((sign_idx as u32) << 7)
        | ((sign_idx as u32) << 14)
        | ((sign_idx as u32) << 21);
    let mut blk = vec![0u8; 66];
    blk[0] = (d_bits & 0xff) as u8;
    blk[1] = (d_bits >> 8) as u8;
    for ib32 in 0..8usize {
        let off = 2 + ib32 * 8;
        // aux0 = all four grid indices set to grid_idx
        let aux0 = (grid_idx as u32)
            | ((grid_idx as u32) << 8)
            | ((grid_idx as u32) << 16)
            | ((grid_idx as u32) << 24);
        let w0 = aux0 & 0xffff;
        let w1 = aux0 >> 16;
        blk[off] = (w0 & 0xff) as u8;
        blk[off + 1] = (w0 >> 8) as u8;
        blk[off + 2] = (w1 & 0xff) as u8;
        blk[off + 3] = (w1 >> 8) as u8;
        let w2 = aux1 & 0xffff;
        let w3 = aux1 >> 16;
        blk[off + 4] = (w2 & 0xff) as u8;
        blk[off + 5] = (w2 >> 8) as u8;
        blk[off + 6] = (w3 & 0xff) as u8;
        blk[off + 7] = (w3 >> 8) as u8;
    }
    blk
}

/// Build a minimal Q8_K block (292 bytes).
fn make_q8k_blk(d: f32, qs: &[i8; 256]) -> Vec<u8> {
    let mut blk = vec![0u8; 292];
    blk[..4].copy_from_slice(&d.to_le_bytes());
    for (i, &v) in qs.iter().enumerate() {
        blk[4 + i] = v as u8;
    }
    // bsums[j] = sum of qs[j*16..(j+1)*16]
    for j in 0..16 {
        let sum: i16 = (0..16).map(|k| qs[j * 16 + k] as i16).sum();
        let off = 260 + j * 2;
        blk[off] = (sum & 0xff) as u8;
        blk[off + 1] = (sum >> 8) as u8;
    }
    blk
}

#[test]
fn iq2_xxs_zero_q8k_scale_gives_zero() {
    let iq2 = make_iq2_blk(F16_ONE, 0, 0, 1);
    let q8k = make_q8k_blk(0.0, &[1i8; 256]);
    let result = iq2_xxs_q8k_dot(&iq2, &q8k);
    assert_eq!(result, 0.0);
}

#[test]
fn iq2_xxs_zero_iq2_scale_gives_zero() {
    let iq2 = make_iq2_blk(0x0000, 0, 0, 1); // f16 0.0
    let q8k = make_q8k_blk(1.0, &[1i8; 256]);
    let result = iq2_xxs_q8k_dot(&iq2, &q8k);
    assert_eq!(result, 0.0);
}

#[test]
fn iq2_xxs_grid0_sign0_ls1_all_ones_q8k() {
    // GRID[0] = 0x0808080808080808 → each byte = 8.
    // SIGNS[0] = 0 → unpack: popcount(0)=0 → sb=0 → all positive.
    // ls=1.
    // dot per 8-group: 4 × sum(8 × q8[i] for i in 0..8) = 4 × 8 × 8 = 256 (q8=1)
    // bsum over 8 groups: 8 × 256 × 1 = 2048
    // result = 0.125 × 1.0 × 1.0 × 2048 = 256.0
    let iq2 = make_iq2_blk(F16_ONE, 0, 0, 1);
    let q8k = make_q8k_blk(1.0, &[1i8; 256]);
    let result = iq2_xxs_q8k_dot(&iq2, &q8k);
    assert!(
        (result - 256.0).abs() < 0.01,
        "expected 256.0, got {result}"
    );
}

#[test]
fn iq2_xxs_ls_scales_result() {
    // Doubling ls from 1 to 3 (aux1>>28 changes from 0 to 1) should 3× the result.
    let iq2_ls1 = make_iq2_blk(F16_ONE, 0, 0, 1);
    let iq2_ls3 = make_iq2_blk(F16_ONE, 0, 0, 3);
    let q8k = make_q8k_blk(1.0, &[1i8; 256]);
    let r1 = iq2_xxs_q8k_dot(&iq2_ls1, &q8k);
    let r3 = iq2_xxs_q8k_dot(&iq2_ls3, &q8k);
    assert!(
        (r3 - 3.0 * r1).abs() < 0.01,
        "ls=3 should give 3× ls=1: {r3} vs {}",
        3.0 * r1
    );
}

#[test]
fn iq2_xxs_scale_linearity() {
    // Doubling d_q8k should double the result.
    let iq2 = make_iq2_blk(F16_ONE, 0, 0, 1);
    let q8k_1 = make_q8k_blk(1.0, &[1i8; 256]);
    let q8k_2 = make_q8k_blk(2.0, &[1i8; 256]);
    let r1 = iq2_xxs_q8k_dot(&iq2, &q8k_1);
    let r2 = iq2_xxs_q8k_dot(&iq2, &q8k_2);
    assert!(
        (r2 - 2.0 * r1).abs() < 0.01,
        "2× q8k_d should give 2× result: {r2} vs {}",
        2.0 * r1
    );
}

// ── Q2_K tests ───────────────────────────────────────────────────────────────

/// Build a Q2_K block (84 bytes) with uniform values.
fn make_q2k_blk(d: f32, dmin: f32, scale_lo: u8, scale_hi: u8, qs_byte: u8) -> Vec<u8> {
    let mut blk = vec![0u8; 84];
    // scales[0..16]: each byte encodes (hi<<4)|lo nibbles
    for i in 0..16 {
        blk[i] = (scale_hi << 4) | (scale_lo & 0xf);
    }
    // qs[0..64]
    for i in 16..80 {
        blk[i] = qs_byte;
    }
    // d at 80-81
    let d_bits = half::f16::from_f32(d).to_bits();
    blk[80] = (d_bits & 0xff) as u8;
    blk[81] = (d_bits >> 8) as u8;
    // dmin at 82-83
    let dmin_bits = half::f16::from_f32(dmin).to_bits();
    blk[82] = (dmin_bits & 0xff) as u8;
    blk[83] = (dmin_bits >> 8) as u8;
    blk
}

#[test]
fn q2k_zero_qs_gives_minus_dmin_term() {
    // Q2_K qs=0 → isum=0 → result = -dmin * summs
    // With dmin=0, result=0 regardless.
    let q2k = make_q2k_blk(1.0, 0.0, 1, 1, 0x00);
    let q8k = make_q8k_blk(1.0, &[1i8; 256]);
    let result = q2k_q8k_dot(&q2k, &q8k);
    assert!((result - 0.0).abs() < 0.01, "expected 0.0, got {result}");
}

#[test]
fn q2k_all_ones_q2_all_ones_q8k() {
    // Q2_K qs=0x55 (each 2-bit pair = 1), scale_lo=1, scale_hi=1 (dmin factor).
    // Q8_K d=1.0, all q8=1.
    // isum: 16 calls × d=1 × 16 elements × q2v=1 × q8v=1 = 256
    // summs: 16 × bsums[j]=16 × scale_hi=1 = 256; dmin×summs = dmin*256
    // With dmin=0.0: result = dall * 256 = 1.0 * 256 = 256.0
    let q2k = make_q2k_blk(1.0, 0.0, 1, 1, 0x55);
    let q8k = make_q8k_blk(1.0, &[1i8; 256]);
    let result = q2k_q8k_dot(&q2k, &q8k);
    assert!(
        (result - 256.0).abs() < 0.5,
        "expected ~256.0, got {result}"
    );
}

#[test]
fn q2k_dmin_term_subtracts() {
    // With nonzero dmin, the dmin*summs term reduces the result.
    let q2k_no_dmin = make_q2k_blk(1.0, 0.0, 1, 0, 0x55);
    let q2k_yes_dmin = make_q2k_blk(1.0, 1.0, 1, 0, 0x55);
    let q8k = make_q8k_blk(1.0, &[1i8; 256]);
    let r0 = q2k_q8k_dot(&q2k_no_dmin, &q8k);
    let r1 = q2k_q8k_dot(&q2k_yes_dmin, &q8k);
    // dmin term: yd * dmin * summs; with scale_hi=0: summs=0, so r1==r0
    assert!(
        (r0 - r1).abs() < 0.01,
        "scale_hi=0 → summs=0 → dmin has no effect"
    );
}

// ── Router math tests ─────────────────────────────────────────────────────────

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0f32 + x.exp()).ln()
    }
}

fn router_score(logit: f32) -> f32 {
    softplus(logit).sqrt()
}

fn router_normalize(scores: &[f32]) -> Vec<f32> {
    let sum: f32 = scores.iter().sum();
    scores.iter().map(|s| s * 1.5 / sum).collect()
}

#[test]
fn softplus_positive_large_is_identity() {
    assert!((softplus(100.0) - 100.0).abs() < 0.01);
}

#[test]
fn softplus_negative_large_is_near_zero() {
    assert!(softplus(-100.0) < 1e-30);
}

#[test]
fn softplus_zero_is_ln2() {
    assert!((softplus(0.0) - 2.0f32.ln()).abs() < 1e-5);
}

#[test]
fn router_score_at_zero_is_sqrt_ln2() {
    assert!((router_score(0.0) - 2.0f32.ln().sqrt()).abs() < 1e-5);
}

#[test]
fn router_normalize_sums_to_1_5() {
    let scores = vec![1.0f32, 2.0, 0.5, 1.5, 0.8, 1.2]; // 6 experts
    let weights = router_normalize(&scores);
    let sum: f32 = weights.iter().sum();
    assert!(
        (sum - 1.5).abs() < 1e-5,
        "weights should sum to 1.5, got {sum}"
    );
}

#[test]
fn router_normalize_preserves_ratios() {
    let scores = vec![2.0f32, 1.0];
    let weights = router_normalize(&scores);
    // score[0] is 2× score[1], so weight[0] should be 2× weight[1]
    assert!((weights[0] / weights[1] - 2.0).abs() < 1e-5);
}

#[test]
fn router_score_matches_ds4_bos_top_expert() {
    // For the BOS token (pos=0) at layer 0, ds4 reports logit[222] ≈ 3.8282
    // and prob[222] ≈ 1.9621.  Verify our formula matches within rounding.
    let logit = 3.8282f32;
    let score = router_score(logit);
    assert!(
        (score - 1.9621).abs() < 0.002,
        "score={score}, expected ~1.9621"
    );
}
