use crate::model::{TensorMeta, TensorType};
use anyhow::{Context, Result, bail};
use memmap2::Mmap;
/// Minimal read-only GGUF v3 parser.
/// Only parses what deepseek-rs needs: tensor names, types, dims, and offsets.
/// The file is memory-mapped; all tensor data is accessed through slices into
/// the mapping.
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian

pub struct GgufFile {
    pub mmap: Mmap,
    pub tensors: Vec<TensorMeta>,
    /// Byte offset from file start where tensor data begins.
    pub data_offset: u64,
    /// Byte offset of the first KV pair (right after the 24-byte header).
    pub kv_section_start: usize,
    /// Number of KV metadata pairs.
    pub n_kv: usize,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mmap =
            unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
        Self::parse(mmap)
    }

    fn parse(mmap: Mmap) -> Result<Self> {
        let mut cur = 0usize;
        let data = &mmap[..];

        macro_rules! read_u32 {
            () => {{
                let v = u32::from_le_bytes(data[cur..cur + 4].try_into().unwrap());
                cur += 4;
                v
            }};
        }
        macro_rules! read_u64 {
            () => {{
                let v = u64::from_le_bytes(data[cur..cur + 8].try_into().unwrap());
                cur += 8;
                v
            }};
        }

        macro_rules! read_str {
            () => {{
                let len = read_u64!() as usize;
                let s = std::str::from_utf8(&data[cur..cur + len])
                    .context("invalid UTF-8 in GGUF string")?
                    .to_string();
                cur += len;
                s
            }};
        }

        let magic = read_u32!();
        if magic != GGUF_MAGIC {
            bail!("not a GGUF file (bad magic {:08x})", magic);
        }

        let version = read_u32!();
        if version < 2 || version > 3 {
            bail!("unsupported GGUF version {}", version);
        }

        let n_tensors = read_u64!() as usize;
        let n_kv = read_u64!() as usize;
        let kv_section_start = cur;

        // Skip metadata k-v pairs — we use fixed model constants.
        for _ in 0..n_kv {
            let _key = read_str!();
            let vtype = read_u32!();
            skip_value(&mut cur, data, vtype)?;
        }

        // Parse tensor info.
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = read_str!();
            let n_dims = read_u32!() as usize;
            let dims: Vec<u64> = (0..n_dims).map(|_| read_u64!()).collect();
            let ty_raw = read_u32!();
            let offset = read_u64!();
            let ty = TensorType::from_u32(ty_raw)
                .with_context(|| format!("unknown tensor type {} for '{}'", ty_raw, name))?;
            tensors.push(TensorMeta {
                name,
                ty,
                dims,
                offset,
            });
        }

        // Data section starts at the next 32-byte-aligned boundary.
        let data_offset = (cur as u64 + 31) & !31;

        Ok(Self {
            mmap,
            tensors,
            data_offset,
            kv_section_start,
            n_kv,
        })
    }

    /// Return the raw byte slice for a tensor's data.
    pub fn tensor_data(&self, t: &TensorMeta) -> &[u8] {
        let start = (self.data_offset + t.offset) as usize;
        let end = start + t.byte_size();
        &self.mmap[start..end]
    }

    /// Find a tensor by name.
    pub fn find(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Scan KV metadata and return the u32 value for `key`, if present.
    pub fn read_kv_u32(&self, key: &str) -> Option<u32> {
        let data = &self.mmap[..];
        let mut cur = self.kv_section_start;
        for _ in 0..self.n_kv {
            let k = kv_read_str(data, &mut cur);
            let vtype = kv_read_u32(data, &mut cur);
            if k == key && vtype == 4 {
                return Some(kv_read_u32(data, &mut cur));
            }
            skip_value(&mut cur, data, vtype).ok()?;
        }
        None
    }

    /// Scan KV metadata and return a string array value for `key`.
    pub fn read_kv_string_array(&self, key: &str) -> Vec<String> {
        let data = &self.mmap[..];
        let mut cur = self.kv_section_start;
        for _ in 0..self.n_kv {
            let k = kv_read_str(data, &mut cur);
            let vtype = kv_read_u32(data, &mut cur);
            if k == key && vtype == 9 {
                let elem_type = kv_read_u32(data, &mut cur);
                let count = kv_read_u64(data, &mut cur) as usize;
                if elem_type != 8 {
                    return vec![];
                } // not a string array
                let mut out = Vec::with_capacity(count);
                for _ in 0..count {
                    out.push(kv_read_str(data, &mut cur));
                }
                return out;
            }
            if skip_value(&mut cur, data, vtype).is_err() {
                break;
            }
        }
        vec![]
    }

    /// Scan KV metadata and return an i32 array value for `key`.
    pub fn read_kv_i32_array(&self, key: &str) -> Vec<i32> {
        let data = &self.mmap[..];
        let mut cur = self.kv_section_start;
        for _ in 0..self.n_kv {
            let k = kv_read_str(data, &mut cur);
            let vtype = kv_read_u32(data, &mut cur);
            if k == key && vtype == 9 {
                let elem_type = kv_read_u32(data, &mut cur);
                let count = kv_read_u64(data, &mut cur) as usize;
                if elem_type != 5 {
                    return vec![];
                }
                let mut out = Vec::with_capacity(count);
                for _ in 0..count {
                    out.push(i32::from_le_bytes(data[cur..cur + 4].try_into().unwrap()));
                    cur += 4;
                }
                return out;
            }
            if skip_value(&mut cur, data, vtype).is_err() {
                break;
            }
        }
        vec![]
    }
}

fn kv_read_u32(data: &[u8], cur: &mut usize) -> u32 {
    let v = u32::from_le_bytes(data[*cur..*cur + 4].try_into().unwrap());
    *cur += 4;
    v
}
fn kv_read_u64(data: &[u8], cur: &mut usize) -> u64 {
    let v = u64::from_le_bytes(data[*cur..*cur + 8].try_into().unwrap());
    *cur += 8;
    v
}
fn kv_read_str(data: &[u8], cur: &mut usize) -> String {
    let len = kv_read_u64(data, cur) as usize;
    let s = std::str::from_utf8(&data[*cur..*cur + len])
        .unwrap_or("")
        .to_string();
    *cur += len;
    s
}

/// Skip a GGUF metadata value of the given type code.
fn skip_value(cur: &mut usize, data: &[u8], vtype: u32) -> Result<()> {
    match vtype {
        0 => {
            *cur += 1;
        } // uint8
        1 => {
            *cur += 1;
        } // int8
        2 => {
            *cur += 2;
        } // uint16
        3 => {
            *cur += 2;
        } // int16
        4 => {
            *cur += 4;
        } // uint32
        5 => {
            *cur += 4;
        } // int32
        6 => {
            *cur += 4;
        } // float32
        7 => {
            *cur += 1;
        } // bool
        8 => {
            // string
            let len = u64::from_le_bytes(data[*cur..*cur + 8].try_into().unwrap()) as usize;
            *cur += 8 + len;
        }
        9 => {
            // array
            let elem_type = u32::from_le_bytes(data[*cur..*cur + 4].try_into().unwrap());
            *cur += 4;
            let count = u64::from_le_bytes(data[*cur..*cur + 8].try_into().unwrap()) as usize;
            *cur += 8;
            for _ in 0..count {
                skip_value(cur, data, elem_type)?;
            }
        }
        10 => {
            *cur += 8;
        } // uint64
        11 => {
            *cur += 8;
        } // int64
        12 => {
            *cur += 8;
        } // float64
        t => bail!("unknown GGUF metadata type {}", t),
    }
    Ok(())
}
