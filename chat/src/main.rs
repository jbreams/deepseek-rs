use anyhow::Result;
use deepseek_engine::{engine::Engine, session::Session, tokenizer::Tokenizer};
use rustyline::{DefaultEditor, error::ReadlineError};
use std::io::Write;
use std::path::PathBuf;

// DeepSeek V4 Flash chat template tokens.
const TOK_USER: i32 = 128803; // <｜User｜>
const TOK_ASST: i32 = 128804; // <｜Assistant｜>
const TOK_THINK: i32 = 128821; // <think>
const TOK_ETHINK: i32 = 128822; // </think>

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/srv/data/work/ds4/ds4flash.gguf"));
    let max_tokens: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let ctx_size: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8192);

    eprintln!("deepseek-chat: loading {}", model_path.display());
    let engine = Engine::open(&model_path)?;
    let tok = Tokenizer::from_gguf(&engine.gguf)?;
    eprintln!(
        "  {} tensors, vocab {}",
        engine.gguf.tensors.len(),
        tok.vocab.len()
    );

    let mut sess = Session::new(&engine, ctx_size)?;
    eprintln!("  session ready (ctx={ctx_size})\n");

    // Seed the KV cache with BOS.
    sess.decode_next(&engine, tok.bos_id as i32)?;

    let mut rl = DefaultEditor::new()?;
    let mut stdout = std::io::stdout();

    loop {
        // ── Read user input ───────────────────────────────────────────────
        let line = match rl.readline("You: ") {
            Ok(l) => l,
            Err(ReadlineError::Eof | ReadlineError::Interrupted) => break,
            Err(e) => return Err(e.into()),
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        rl.add_history_entry(&line)?;

        // ── Encode user turn ──────────────────────────────────────────────
        let mut turn: Vec<i32> = vec![TOK_USER];
        turn.extend(tok.encode(&line).iter().map(|&id| id as i32));
        turn.push(TOK_ASST);

        // Feed user-turn tokens and the <｜Assistant｜> start token.
        // Use decode_next (sequential, correct for multi-turn) so the new
        // tokens attend to everything already in the KV cache.
        let mut logits = {
            let mut last = None;
            for &t in &turn {
                last = Some(sess.decode_next(&engine, t)?);
            }
            last.unwrap()
        };

        // ── Generate response ─────────────────────────────────────────────
        print!("Assistant: ");
        stdout.flush()?;

        let mut in_think = false;
        let mut gen_count = 0usize;

        loop {
            if gen_count >= max_tokens {
                break;
            }
            let next = greedy(&logits) as i32;
            if next == tok.eos_id as i32 {
                break;
            }

            // Track <think>/</think> to dim thinking output.
            if next == TOK_THINK {
                in_think = true;
            }
            if next == TOK_ETHINK {
                in_think = false;
                print!("\n");
            }

            let text = tok.decode_token(next as u32);
            if in_think {
                // Print thinking output dimmed (ANSI dim).
                print!("\x1b[2m{text}\x1b[0m");
            } else {
                print!("{text}");
            }
            stdout.flush()?;

            logits = sess.decode_next(&engine, next)?;
            gen_count += 1;

            // Context full — warn and stop.
            if sess.pos() >= ctx_size - 1 {
                eprintln!("\n[context full at {} tokens]", sess.pos());
                break;
            }
        }
        println!("\n");
    }

    eprintln!("\nSession ended at {} tokens.", sess.pos());
    Ok(())
}

fn greedy(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}
