/// Integration tests for BPE encoding against the Python reference implementation.
///
/// Expected IDs computed with the Python `regex` + manual BPE reference,
/// verified on 2026-05-18 against the ds4flash.gguf vocabulary.
///
/// Requires: DEEPSEEK_RS_MODEL_PATH set to the GGUF file path.
use deepseek_engine::{gguf::GgufFile, tokenizer::Tokenizer};

const MODEL_ENV: &str = "DEEPSEEK_RS_MODEL_PATH";

fn load_tokenizer() -> Option<Tokenizer> {
    let path = std::env::var(MODEL_ENV).ok()?;
    let path = std::path::PathBuf::from(path);
    if !path.exists() {
        return None;
    }
    let gguf = GgufFile::open(&path).ok()?;
    Tokenizer::from_gguf(&gguf).ok()
}

#[test]
fn encode_matches_python_reference() {
    let Some(tok) = load_tokenizer() else {
        return;
    };

    let cases: &[(&str, &[u32])] = &[
        // Python reference: encode('Hello, world!') = [19923, 14, 2058, 3]
        ("Hello, world!", &[19923, 14, 2058, 3]),
        // encode('import os') = [1897, 5688]
        ("import os", &[1897, 5688]),
        // encode("don't") = [20385, 1664]
        ("don't", &[20385, 1664]),
        // encode('def foo():') = [3465, 52735, 24590]
        ("def foo():", &[3465, 52735, 24590]),
        // encode('Write a poem') = [21750, 260, 17261]
        ("Write a poem", &[21750, 260, 17261]),
        // encode('你好世界') = [30594, 3427]
        ("你好世界", &[30594, 3427]),
    ];

    for &(text, expected) in cases {
        let got = tok.encode(text);
        assert_eq!(
            got, expected,
            "encode({text:?}): expected {expected:?}, got {got:?}"
        );
    }
}

#[test]
fn encode_decode_roundtrip() {
    let Some(tok) = load_tokenizer() else {
        return;
    };

    let texts = ["Hello, world!", "import os\nprint('hi')", "Write a haiku"];
    for text in texts {
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(
            decoded, text,
            "roundtrip failed for {text:?}: got {decoded:?}"
        );
    }
}

#[test]
fn special_tokens_not_in_encode_output() {
    let Some(tok) = load_tokenizer() else {
        return;
    };
    // BOS (0) and EOS (1) should never appear from encoding regular text
    let ids = tok.encode("Hello world");
    assert!(!ids.contains(&tok.bos_id), "BOS appeared in encode output");
    assert!(!ids.contains(&tok.eos_id), "EOS appeared in encode output");
}
