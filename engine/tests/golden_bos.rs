/// Golden output tests: full 284B forward pass correctness.
///
/// All assertions share a single model load to avoid OOM on 81 GB VRAM.
/// Loading the model in multiple parallel test fns exhausts VRAM — keep
/// everything in `bos_golden`.
///
/// Requires: model file at `DEEPSEEK_RS_MODEL_PATH` env var.
/// Skips silently when the env var is not set.
///
/// To run:
///   DEEPSEEK_RS_MODEL_PATH=/path/to/ds4flash.gguf cargo test --test golden_bos
use deepseek_engine::{engine::Engine, session::Session};
use std::path::PathBuf;

const MODEL_ENV: &str = "DEEPSEEK_RS_MODEL_PATH";

/// ds4 reference top-5 for BOS (token 0, pos 0), verified on 2026-05-18.
const EXPECTED_TOP5: [usize; 5] = [5, 1897, 372, 201, 7249];
/// Minimum acceptable logit for the top token (ds4 reference: 16.78; -0.20 slack).
const TOP1_LOGIT_MIN: f32 = 16.6;
/// Expected top-1 token after prefill of [BOS] + 199×5 (200 tokens, two chunks),
/// verified on 2026-05-18. Pins correctness of chunked prefill + compressor path.
const EXPECTED_LONG_PREFILL_TOP1: usize = 204;

fn open_engine() -> Option<Engine> {
    let path = std::env::var(MODEL_ENV).ok()?;
    let path = PathBuf::from(&path);
    if !path.exists() {
        eprintln!("golden_bos: {MODEL_ENV} path not found, skipping");
        return None;
    }
    Some(Engine::open(&path).expect("golden_bos: failed to open model"))
}

fn top5(logits: &[f32]) -> Vec<usize> {
    let mut v: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v[..5].iter().map(|(i, _)| *i).collect()
}

fn top1(logits: &[f32]) -> (usize, f32) {
    logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
}

/// All golden assertions in one test so the model is loaded only once.
#[test]
fn bos_golden() {
    let Some(engine) = open_engine() else { return };

    // ── 1. Sequential decode: BOS then token 5 ───────────────────────────────
    // Uses the low-level decode() path with explicit positions.
    {
        let mut sess = Session::new(&engine, 512).expect("session alloc");

        let logits_bos = sess.decode_next(&engine, 0).expect("decode BOS");
        assert_eq!(sess.pos(), 1, "pos after decode_next(BOS)");

        let logits_tok2 = sess.decode_next(&engine, 5).expect("decode tok 5");
        assert_eq!(sess.pos(), 2, "pos after decode_next(5)");

        // BOS top token and top-5.
        let (top1_idx, top1_val) = top1(&logits_bos);
        assert_eq!(top1_idx, 5, "BOS: top token should be 5, got {top1_idx}");
        assert_eq!(
            top5(&logits_bos),
            EXPECTED_TOP5,
            "BOS top-5 mismatch vs ds4 reference"
        );
        assert!(
            top1_val >= TOP1_LOGIT_MIN,
            "BOS logit[5]={top1_val:.4} below minimum {TOP1_LOGIT_MIN} (ds4: 16.78)"
        );
        assert!(top1_val < 22.0, "BOS logit[5]={top1_val:.4} suspiciously high");

        // Rank-1 vs rank-2 gap.
        let sorted = {
            let mut s: Vec<f32> = logits_bos.clone();
            s.sort_by(|a, b| b.partial_cmp(a).unwrap());
            s
        };
        let gap = sorted[0] - sorted[1];
        assert!(gap >= 0.5, "rank-1 vs rank-2 gap={gap:.4} < 0.5 (degenerate)");

        // Second token sanity.
        assert!(!logits_tok2.iter().any(|v| v.is_nan()), "pos=1 logits contain NaN");
        assert!(!logits_tok2.iter().any(|v| v.is_infinite()), "pos=1 logits contain inf");
        let (_, top1_tok2) = top1(&logits_tok2);
        assert!(
            top1_tok2 > 0.0 && top1_tok2 < 150.0,
            "pos=1 top logit {top1_tok2} out of plausible range"
        );
    }

    // ── 2. prefill([BOS]) agrees with sequential decode ───────────────────────
    // The first chunk at abs_start=0 uses prefill_raw for batch attention.
    // Its top-5 must match the decode path.
    {
        let mut sess = Session::new(&engine, 512).expect("session alloc");

        let logits = sess.prefill(&engine, &[0 /*BOS*/]).expect("prefill BOS");
        assert_eq!(sess.pos(), 1, "pos after prefill([BOS])");

        let (top1_idx, top1_val) = top1(&logits);
        assert_eq!(top1_idx, 5, "prefill([BOS]): top token should be 5, got {top1_idx}");
        assert_eq!(
            top5(&logits),
            EXPECTED_TOP5,
            "prefill([BOS]) top-5 must match decode top-5"
        );
        assert!(
            top1_val >= TOP1_LOGIT_MIN,
            "prefill([BOS]) top logit {top1_val:.4} below minimum {TOP1_LOGIT_MIN}"
        );
    }

    // ── 3. Long-prompt prefill: 200 tokens across two chunks ─────────────────
    // Exercises the chunking path (chunk 0 uses prefill_raw, chunk 1 uses
    // per-token decode_mixed).  Verifies no crash, finite logits, correct pos.
    {
        let mut sess = Session::new(&engine, 512).expect("session alloc");

        // [BOS] + 199 × token-5: a valid sequence the model can process.
        let mut prompt = vec![0i32]; // BOS
        prompt.extend(std::iter::repeat(5i32).take(199));
        assert_eq!(prompt.len(), 200);

        let logits = sess.prefill(&engine, &prompt).expect("prefill 200 tokens");
        assert_eq!(sess.pos(), 200, "pos after 200-token prefill");

        assert!(!logits.iter().any(|v| v.is_nan()), "200-token prefill logits contain NaN");
        assert!(!logits.iter().any(|v| v.is_infinite()), "200-token prefill logits contain inf");

        let (top1_idx, top_val) = top1(&logits);
        assert_eq!(
            top1_idx, EXPECTED_LONG_PREFILL_TOP1,
            "200-token prefill top-1 regressed: got {top1_idx}, expected {EXPECTED_LONG_PREFILL_TOP1}"
        );
        assert!(
            top_val > 0.0 && top_val < 150.0,
            "200-token prefill top logit {top_val:.4} out of plausible range"
        );
    }

    // ── 4. prefill then decode_next: positions advance correctly ─────────────
    {
        let mut sess = Session::new(&engine, 512).expect("session alloc");

        sess.prefill(&engine, &[0i32, 5i32]).expect("prefill [BOS, 5]");
        assert_eq!(sess.pos(), 2, "pos after prefill([BOS, 5])");

        sess.decode_next(&engine, 1897).expect("decode_next after prefill");
        assert_eq!(sess.pos(), 3, "pos after decode_next following prefill");
    }
}
