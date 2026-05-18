use crate::gguf::GgufFile;
use anyhow::{Context, Result};
use regex::Regex;
/// Vocabulary loading, BPE encoding, and token-to-text decoding.
///
/// The GGUF stores tokens using the GPT-2 byte-to-unicode mapping where each
/// of the 256 possible byte values is assigned a unique Unicode character:
///   - ASCII printable 33-126 → themselves
///   - Latin-1 161-172 and 174-255 → themselves
///   - Remaining bytes (0-32, 127-160, 173) → U+0100..=U+0143
///
/// So space (0x20 = 32) → Ġ (U+0120), newline (0x0A = 10) → Ċ (U+010A), etc.
///
/// Encoding pipeline:
///   1. Pre-tokenize with the GPT-2/cl100k regex (splits on word/space/punct
///      boundaries so spaces attach to the following word)
///   2. Convert each pre-token's raw UTF-8 bytes to GPT-2 Unicode characters
///   3. Apply BPE merges greedily (lowest-rank merge first)
///   4. Map each resulting piece to its vocabulary ID
use std::collections::HashMap;
use std::sync::OnceLock;

// ── Pre-tokenization regex ────────────────────────────────────────────────────
//
// Derived from the GPT-2 / tiktoken cl100k pattern. Splits text into pieces
// where each word naturally carries its leading space (e.g. " hello").
// We drop the negative lookahead `\s+(?!\S)` because the `regex` crate does
// not support lookaheads; for typical prompts this makes no practical difference.
static PRE_TOK_RE: OnceLock<Regex> = OnceLock::new();
fn pre_tok_re() -> &'static Regex {
    PRE_TOK_RE.get_or_init(|| {
        Regex::new(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+"
        ).unwrap()
    })
}

// ── Tokenizer ─────────────────────────────────────────────────────────────────

pub struct Tokenizer {
    pub vocab: Vec<String>,
    pub token_types: Vec<i32>,
    pub bos_id: u32,
    pub eos_id: u32,
    /// token_string → token_id (built from vocab for fast lookup)
    vocab_map: HashMap<String, u32>,
    /// BPE merge ranks: merge_ranks[left][right] = rank (lower = higher priority)
    merge_ranks: HashMap<String, HashMap<String, u32>>,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let bos_id = gguf
            .read_kv_u32("tokenizer.ggml.bos_token_id")
            .context("tokenizer.ggml.bos_token_id not found")?;
        let eos_id = gguf
            .read_kv_u32("tokenizer.ggml.eos_token_id")
            .context("tokenizer.ggml.eos_token_id not found")?;

        let vocab = gguf.read_kv_string_array("tokenizer.ggml.tokens");
        anyhow::ensure!(!vocab.is_empty(), "tokenizer.ggml.tokens empty or missing");
        let token_types = gguf.read_kv_i32_array("tokenizer.ggml.token_type");

        // vocab_map: string → id (exact reverse of vocab array)
        let mut vocab_map = HashMap::with_capacity(vocab.len());
        for (id, s) in vocab.iter().enumerate() {
            vocab_map.insert(s.clone(), id as u32);
        }

        // merge_ranks: parse "left right" strings, indexed by left token
        let merges_raw = gguf.read_kv_string_array("tokenizer.ggml.merges");
        let mut merge_ranks: HashMap<String, HashMap<String, u32>> =
            HashMap::with_capacity(merges_raw.len() / 4);
        for (rank, m) in merges_raw.iter().enumerate() {
            if let Some(sp) = m.find(' ') {
                let left = m[..sp].to_string();
                let right = m[sp + 1..].to_string();
                merge_ranks
                    .entry(left)
                    .or_default()
                    .insert(right, rank as u32);
            }
        }

        Ok(Self {
            vocab,
            token_types,
            bos_id,
            eos_id,
            vocab_map,
            merge_ranks,
        })
    }

    /// Encode `text` into token IDs using BPE.  Does not prepend BOS.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();
        for m in pre_tok_re().find_iter(text) {
            result.extend(self.bpe_encode_bytes(m.as_str().as_bytes()));
        }
        result
    }

    /// Decode a single token ID to a UTF-8 string fragment.
    /// Returns empty string for BOS/EOS (they produce no visible text).
    pub fn decode_token(&self, id: u32) -> String {
        if id == self.bos_id || id == self.eos_id {
            return String::new();
        }
        if id as usize >= self.vocab.len() {
            return format!("<{id}>");
        }
        let s = &self.vocab[id as usize];
        let ty = self.token_types.get(id as usize).copied().unwrap_or(1);
        if ty == 3 {
            return s.clone();
        } // control/special: print as-is
        let bytes = gpt2_token_to_bytes(s);
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode a sequence of token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut all_bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if id == self.bos_id || id == self.eos_id {
                continue;
            }
            if id as usize >= self.vocab.len() {
                all_bytes.extend_from_slice(format!("<{id}>").as_bytes());
                continue;
            }
            let ty = self.token_types.get(id as usize).copied().unwrap_or(1);
            if ty == 3 {
                all_bytes.extend_from_slice(self.vocab[id as usize].as_bytes());
            } else {
                all_bytes.extend_from_slice(&gpt2_token_to_bytes(&self.vocab[id as usize]));
            }
        }
        String::from_utf8_lossy(&all_bytes).into_owned()
    }

    // ── BPE internals ──────────────────────────────────────────────────────────

    /// Encode a single pre-tokenized piece (as raw UTF-8 bytes) via BPE.
    fn bpe_encode_bytes(&self, bytes: &[u8]) -> Vec<u32> {
        if bytes.is_empty() {
            return vec![];
        }

        // Each byte → GPT-2 unicode string (single-char tokens initially)
        let mut parts: Vec<String> = bytes
            .iter()
            .map(|&b| byte_to_unicode(b).to_string())
            .collect();

        // Fast path: single byte
        if parts.len() == 1 {
            return match self.vocab_map.get(&parts[0]) {
                Some(&id) => vec![id],
                None => {
                    eprintln!("warn: unknown byte token {:?}", parts[0]);
                    vec![]
                }
            };
        }

        // BPE merge loop: greedily apply lowest-rank (= highest-priority) merge.
        loop {
            let mut best_rank = u32::MAX;
            let mut best_i = usize::MAX;
            for i in 0..parts.len() - 1 {
                if let Some(rank) = self
                    .merge_ranks
                    .get(&parts[i])
                    .and_then(|m| m.get(&parts[i + 1]))
                    .copied()
                {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = i;
                    }
                }
            }
            if best_i == usize::MAX {
                break;
            }
            // Merge parts[best_i] with parts[best_i+1]
            let right = parts.remove(best_i + 1);
            parts[best_i].push_str(&right);
        }

        // Map each resulting piece to its vocab ID
        parts
            .iter()
            .map(|t| {
                match self.vocab_map.get(t) {
                    Some(&id) => id,
                    None => {
                        eprintln!("warn: piece not in vocab: {t:?}");
                        self.eos_id // safe fallback
                    }
                }
            })
            .collect()
    }
}

// ── Byte↔Unicode helpers ──────────────────────────────────────────────────────

/// Map one raw byte to its GPT-2 unicode character.
pub fn byte_to_unicode(b: u8) -> char {
    match b {
        33..=126 => b as char,
        161..=172 | 174..=255 => b as char,
        0..=32 => char::from_u32(0x100 + b as u32).unwrap(), // → U+0100..U+0120
        127..=160 => char::from_u32(0x121 + b as u32 - 127).unwrap(), // → U+0121..U+0142
        173 => '\u{0143}',
    }
}

/// Reverse GPT-2 byte-to-unicode: decode a token string to raw bytes.
/// Each character in the token string was originally one byte.
fn gpt2_token_to_bytes(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    for c in s.chars() {
        let cp = c as u32;
        let byte = if cp >= 33 && cp <= 126 {
            cp as u8
        } else if (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255) {
            cp as u8
        } else if cp >= 0x100 && cp <= 0x120 {
            (cp - 0x100) as u8
        } else if cp >= 0x121 && cp <= 0x142 {
            (cp - 0x121 + 127) as u8
        } else if cp == 0x143 {
            173u8
        } else {
            // Genuine multi-byte Unicode in the vocab string; pass through as UTF-8.
            let mut buf = [0u8; 4];
            let s = c.encode_utf8(&mut buf);
            out.extend_from_slice(s.as_bytes());
            continue;
        };
        out.push(byte);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_to_unicode_space() {
        assert_eq!(byte_to_unicode(32), '\u{0120}'); // Ġ
    }
    #[test]
    fn byte_to_unicode_newline() {
        assert_eq!(byte_to_unicode(10), '\u{010A}'); // Ċ
    }
    #[test]
    fn byte_to_unicode_ascii_letter() {
        assert_eq!(byte_to_unicode(b'A'), 'A');
    }

    #[test]
    fn gpt2_decode_space() {
        assert_eq!(gpt2_token_to_bytes("Ġ"), vec![b' ']);
    }
    #[test]
    fn gpt2_decode_newline() {
        assert_eq!(gpt2_token_to_bytes("Ċ"), vec![b'\n']);
    }
    #[test]
    fn gpt2_decode_word_with_space() {
        assert_eq!(gpt2_token_to_bytes("Ġthe"), b" the");
    }
    #[test]
    fn gpt2_decode_ascii_token() {
        assert_eq!(gpt2_token_to_bytes("import"), b"import");
    }

    #[test]
    fn pre_tokenize_simple() {
        // "Hello, world!" should split into ["Hello", ",", " world", "!"]
        let pieces: Vec<&str> = pre_tok_re()
            .find_iter("Hello, world!")
            .map(|m| m.as_str())
            .collect();
        assert_eq!(pieces, vec!["Hello", ",", " world", "!"]);
    }

    #[test]
    fn pre_tokenize_leading_space() {
        // Leading space attaches to following word
        let pieces: Vec<&str> = pre_tok_re()
            .find_iter(" hello")
            .map(|m| m.as_str())
            .collect();
        assert_eq!(pieces, vec![" hello"]);
    }

    #[test]
    fn pre_tokenize_numbers() {
        // Numbers split into groups of up to 3 digits
        let pieces: Vec<&str> = pre_tok_re()
            .find_iter("12345")
            .map(|m| m.as_str())
            .collect();
        assert_eq!(pieces, vec!["123", "45"]);
    }

    #[test]
    fn pre_tokenize_contractions() {
        let pieces: Vec<&str> = pre_tok_re()
            .find_iter("don't")
            .map(|m| m.as_str())
            .collect();
        assert_eq!(pieces, vec!["don", "'t"]);
    }
}
