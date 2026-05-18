use deepseek_engine::gguf::GgufFile;
use deepseek_engine::model::TensorType;
/// GGUF parser regression tests using synthetic in-memory GGUF binaries.
///
/// The critical thing to catch: the parser must correctly read tensor type IDs
/// so the engine uses the right matmul kernel.  The ffn_gate_inp bug (F16 parsed
/// as Q8_0) would have been caught immediately by a test like this.
use std::io::Write;

// ── GGUF binary builder ───────────────────────────────────────────────────────

struct GgufBuilder {
    buf: Vec<u8>,
}

impl GgufBuilder {
    fn new() -> Self {
        let mut b = Self { buf: Vec::new() };
        b.write_u32(0x46554747); // "GGUF" magic
        b.write_u32(3); // version 3
        b
    }

    #[allow(dead_code)]
    fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn write_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn write_str(&mut self, s: &str) {
        self.write_u64(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
    }

    fn finish_header(&mut self, n_tensors: u64, n_kv: u64) {
        // Insert n_tensors and n_kv after the 8-byte magic+version.
        let counts: Vec<u8> = [n_tensors.to_le_bytes(), n_kv.to_le_bytes()].concat();
        self.buf.splice(8..8, counts);
    }

    fn add_tensor(&mut self, name: &str, type_id: u32, dims: &[u64], offset: u64) {
        self.write_str(name);
        self.write_u32(dims.len() as u32);
        for &d in dims {
            self.write_u64(d);
        }
        self.write_u32(type_id);
        self.write_u64(offset);
    }

    /// Pad to 32-byte alignment for the data section.
    fn align_data(&mut self) -> usize {
        let aligned = (self.buf.len() + 31) & !31;
        self.buf.resize(aligned, 0u8);
        aligned
    }

    fn build(self) -> Vec<u8> {
        self.buf
    }
}

/// Write a synthetic GGUF to a temp file and parse it, returning the GgufFile.
fn parse_synthetic(build_fn: impl FnOnce(&mut GgufBuilder)) -> GgufFile {
    let mut b = GgufBuilder::new();
    build_fn(&mut b);
    let bytes = b.build();

    let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
    tmp.write_all(&bytes).expect("write");
    tmp.flush().expect("flush");
    GgufFile::open(tmp.path()).expect("parse")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_single_f16_tensor() {
    let gguf = parse_synthetic(|b| {
        b.finish_header(1, 0);
        b.add_tensor("blk.0.ffn_gate_inp.weight", 1 /*F16*/, &[4096, 256], 0);
        b.align_data();
        // 4096 * 256 * 2 bytes (f16) of data
        b.buf.extend(vec![0u8; 4096 * 256 * 2]);
    });

    assert_eq!(gguf.tensors.len(), 1);
    let t = &gguf.tensors[0];
    assert_eq!(t.name, "blk.0.ffn_gate_inp.weight");
    assert_eq!(t.ty, TensorType::F16);
    assert_eq!(t.dims, vec![4096u64, 256]);
    assert_eq!(t.offset, 0);
    // byte_size: 4096 * 256 * 2 = 2_097_152
    assert_eq!(t.byte_size(), 4096 * 256 * 2);
}

#[test]
fn parse_single_q8_0_tensor() {
    // Q8_0: 34 bytes per 32-element block.  4 elements = 1 block (rounding up) = 34 bytes.
    // Actually for 32 elements: 1 block = 34 bytes.
    let gguf = parse_synthetic(|b| {
        b.finish_header(1, 0);
        b.add_tensor("blk.0.attn_kv.weight", 8 /*Q8_0*/, &[4096, 512], 0);
        b.align_data();
        let n_blocks = (4096 * 512 + 31) / 32;
        b.buf.extend(vec![0u8; n_blocks * 34]);
    });

    let t = &gguf.tensors[0];
    assert_eq!(t.name, "blk.0.attn_kv.weight");
    assert_eq!(
        t.ty,
        TensorType::Q8_0,
        "Q8_0 type_id=8 must parse as Q8_0, not F16"
    );
    assert_eq!(t.dims, vec![4096u64, 512]);
}

#[test]
fn parse_iq2_xxs_tensor() {
    let gguf = parse_synthetic(|b| {
        b.finish_header(1, 0);
        b.add_tensor(
            "blk.0.ffn_gate_exps.weight",
            16, /*IQ2_XXS*/
            &[4096, 2048],
            0,
        );
        b.align_data();
        // IQ2_XXS: 66 bytes per 256-element block
        let n_elems: usize = 4096 * 2048;
        let n_blocks = (n_elems + 255) / 256;
        b.buf.extend(vec![0u8; n_blocks * 66]);
    });

    let t = &gguf.tensors[0];
    assert_eq!(t.ty, TensorType::Iq2Xxs);
}

#[test]
fn parse_multiple_tensors_with_offsets() {
    // Two tensors: second starts right after first (offset = byte_size of first).
    let size0: u64 = 64; // 32 f32 elements × 2 bytes each (F16)
    let size1: u64 = 128;
    let gguf = parse_synthetic(|b| {
        b.finish_header(2, 0);
        b.add_tensor("tensor0", 1 /*F16*/, &[4, 8], 0); // 4*8*2=64 bytes
        b.add_tensor("tensor1", 1 /*F16*/, &[8, 8], size0); // starts at 64
        b.align_data();
        b.buf.extend(vec![0u8; (size0 + size1) as usize]);
    });

    assert_eq!(gguf.tensors.len(), 2);
    assert_eq!(gguf.tensors[0].offset, 0);
    assert_eq!(gguf.tensors[1].offset, size0);

    // tensor_data slices must be non-overlapping and correct size.
    let d0 = gguf.tensor_data(&gguf.tensors[0]);
    let d1 = gguf.tensor_data(&gguf.tensors[1]);
    assert_eq!(d0.len(), size0 as usize);
    assert_eq!(d1.len(), size1 as usize);
}

#[test]
fn tensor_data_slice_contains_written_bytes() {
    // Write a recognisable pattern into the data section and verify it comes back.
    let gguf = parse_synthetic(|b| {
        b.finish_header(1, 0);
        b.add_tensor("w", 1 /*F16*/, &[1, 4], 0); // 4 f16 = 8 bytes
        b.align_data();
        // f16 1.0 = 0x3C00 → bytes [0x00, 0x3C]
        for _ in 0..4 {
            b.buf.push(0x00);
            b.buf.push(0x3C);
        }
    });

    let data = gguf.tensor_data(&gguf.tensors[0]);
    assert_eq!(data.len(), 8);
    for i in 0..4 {
        assert_eq!(data[i * 2], 0x00, "lo byte of f16 1.0");
        assert_eq!(data[i * 2 + 1], 0x3C, "hi byte of f16 1.0");
    }
}

#[test]
fn find_by_name() {
    let gguf = parse_synthetic(|b| {
        b.finish_header(2, 0);
        b.add_tensor("alpha", 0 /*F32*/, &[4], 0);
        b.add_tensor("beta", 0 /*F32*/, &[4], 16);
        b.align_data();
        b.buf.extend(vec![0u8; 32]);
    });

    assert!(gguf.find("alpha").is_some());
    assert!(gguf.find("beta").is_some());
    assert!(gguf.find("gamma").is_none());
    assert_eq!(gguf.find("beta").unwrap().offset, 16);
}

#[test]
fn type_id_1_is_f16_not_q8_0() {
    // This is the exact bug class that caused wrong expert selection:
    // type_id=1 (F16) must NOT parse as Q8_0 (id=8).
    let gguf = parse_synthetic(|b| {
        b.finish_header(1, 0);
        b.add_tensor("router", 1, &[4096, 256], 0);
        b.align_data();
        b.buf.extend(vec![0u8; 4096 * 256 * 2]);
    });
    let t = &gguf.tensors[0];
    assert_eq!(t.ty, TensorType::F16, "type_id=1 must be F16");
    assert_ne!(t.ty, TensorType::Q8_0, "type_id=1 must NOT be Q8_0");
}
