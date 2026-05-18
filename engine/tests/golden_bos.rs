use deepseek_engine::{engine::Engine, session::Session};
/// Golden output test: full 284B forward pass for the BOS token.
///
/// Verifies end-to-end numerical correctness against the ds4 reference output.
/// This test catches any regression that survives unit and smoke tests — wrong
/// kernel output, misconfigured routing, wrong weight type assumptions, etc.
///
/// Requires: model file at `DEEPSEEK_RS_MODEL_PATH` env var.
/// Skips silently when the env var is not set.
///
/// To run:
///   DEEPSEEK_RS_MODEL_PATH=/path/to/ds4flash.gguf cargo test --test golden_bos
use std::path::PathBuf;

const MODEL_ENV: &str = "DEEPSEEK_RS_MODEL_PATH";

/// ds4 reference top-5 for BOS (token 0, pos 0), verified on 2026-05-18.
const EXPECTED_TOP5: [usize; 5] = [5, 1897, 372, 201, 7249];
/// Minimum acceptable logit for the top token (ds4 reference: 16.78; -0.20 slack).
const TOP1_LOGIT_MIN: f32 = 16.6;

struct GoldenFixture {
    logits_bos: Vec<f32>,
    logits_tok2: Vec<f32>, // pos=1, token=5 (predicted top after BOS)
}

fn run_golden() -> Option<GoldenFixture> {
    let path = std::env::var(MODEL_ENV).ok()?;
    let path = PathBuf::from(&path);
    if !path.exists() {
        eprintln!("golden_bos: {MODEL_ENV} path not found, skipping");
        return None;
    }

    let engine = Engine::open(&path).expect("golden_bos: failed to open model");
    let mut sess = Session::new(&engine, 512).expect("golden_bos: failed to create session");

    let logits_bos = sess.decode(&engine, 0 /*BOS*/, 0).expect("decode pos=0");
    let logits_tok2 = sess.decode(&engine, 5 /*tok*/, 1).expect("decode pos=1");

    Some(GoldenFixture {
        logits_bos,
        logits_tok2,
    })
}

/// All golden assertions in one test so the model is loaded only once.
#[test]
fn bos_golden() {
    let Some(fix) = run_golden() else {
        return;
    };

    // ── BOS top token ─────────────────────────────────────────────────────
    let top1 = fix
        .logits_bos
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(top1, 5, "BOS: top token should be 5, got {top1}");

    // ── Top-5 match ds4 ───────────────────────────────────────────────────
    let mut sorted: Vec<(usize, f32)> = fix.logits_bos.iter().copied().enumerate().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top5: Vec<usize> = sorted[..5].iter().map(|(i, _)| *i).collect();
    assert_eq!(
        top5, EXPECTED_TOP5,
        "BOS top-5 should match ds4 {EXPECTED_TOP5:?}, got {top5:?}"
    );

    // ── Top logit in range ────────────────────────────────────────────────
    let top_logit = fix.logits_bos[5];
    assert!(
        top_logit >= TOP1_LOGIT_MIN,
        "logit[5]={top_logit:.4} below minimum {TOP1_LOGIT_MIN} (ds4: 16.78)"
    );
    assert!(
        top_logit < 22.0,
        "logit[5]={top_logit:.4} suspiciously high"
    );

    // ── Gap between rank-1 and rank-2 ────────────────────────────────────
    // ds4 gap: 16.78 - 15.82 ≈ 0.96; allow down to 0.5.
    let gap = sorted[0].1 - sorted[1].1;
    assert!(
        gap >= 0.5,
        "rank-1 vs rank-2 gap={gap:.4} < 0.5 (distribution looks degenerate)"
    );

    // ── Second token: no NaN/inf, plausible range ─────────────────────────
    let has_nan = fix.logits_tok2.iter().any(|v| v.is_nan());
    let has_inf = fix.logits_tok2.iter().any(|v| v.is_infinite());
    assert!(!has_nan, "pos=1 logits contain NaN");
    assert!(!has_inf, "pos=1 logits contain inf");
    let top1_tok2 = fix
        .logits_tok2
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        top1_tok2 > 0.0 && top1_tok2 < 150.0,
        "pos=1 top logit {top1_tok2} out of plausible range"
    );
}
