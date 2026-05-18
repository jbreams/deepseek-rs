use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

// Raw f16 bits ↔ f32 helpers used in device code.
// We represent f16 values as `u16` in kernel signatures to avoid pulling in
// x86 intrinsics from the `half` crate (which cuda-oxide's PTX backend
// cannot lower).  The host side can freely use `half::f16`; its memory
// layout is identical to `u16`.
// TODO(cuda-oxide): when `half::f16` no longer drags in x86 f16c intrinsics
// for device compilation, replace all `u16`-typed f16 parameters with
// `half::f16` directly and remove these conversion helpers.

/// IEEE 754 f16 → f32 conversion (device-side, no x86 intrinsics).
/// Uses `transmute` for the final bit-pattern reinterpret to stay compatible
/// with cuda-oxide's PTX backend.
#[cuda_device::device]
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let h = h as u32;
    let sign = (h & 0x8000) << 16;
    let exp = (h >> 10) & 0x1F;
    let mant = h & 0x3FF;
    let bits: u32 = if exp == 0 {
        sign // Zero or denormal (flush denormals to zero)
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mant << 13) // Inf or NaN
    } else {
        sign | ((exp + 112) << 23) | (mant << 13) // Normal
    };
    f32::from_bits(bits)
}

/// IEEE 754 f32 → f16 conversion (device-side, no x86 intrinsics).
/// Rounds toward zero; flush-to-zero for subnormals.
#[cuda_device::device]
pub fn f32_to_f16_bits(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = (b >> 23) & 0xFF;
    let mant = (b >> 13) & 0x3FF;
    // f16 minimum normal biased f32 exponent = 127 - 14 = 113.
    // Values below this are subnormal in f16; we flush them to zero.
    // (Previously the threshold was 103, causing exponent underflow and garbage large values
    //  for f32 values in the range 2^-24 to 2^-14.)
    if exp == 0 || exp < 113 {
        sign // Flush subnormals and below-subnormal range to zero
    } else if exp > 142 {
        sign | 0x7C00 // Clamp to infinity
    } else {
        sign | (((exp - 112) as u16) << 10) | mant as u16
    }
}

#[cuda_module]
pub mod utils {
    use super::*;

    /// Fill every element of `x` with the constant `v`.
    /// Ported from fill_f32_kernel.
    #[kernel]
    pub fn fill_f32(mut x: DisjointSlice<f32>, v: f32) {
        let idx = thread::index_1d();
        if let Some(elem) = x.get_mut(idx) {
            *elem = v;
        }
    }

    /// Zero every element of `x`.
    /// Ported from zero_kernel.
    #[kernel]
    pub fn zero_f32(mut x: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(elem) = x.get_mut(idx) {
            *elem = 0.0;
        }
    }

    /// Element-wise addition: `out[i] = a[i] + b[i]`.
    /// Ported from add_kernel.
    #[kernel]
    pub fn add_f32(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        // Save raw index before consuming idx via get_mut (ThreadIndex is !Copy).
        // TODO(cuda-oxide): ThreadIndex is !Copy, so idx.get() must be saved before
        // out.get_mut(idx) consumes it. Remove the pre-save when ThreadIndex is Copy.
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = a[i] + b[i];
        }
    }

    /// Copy `n` f32 elements from `src[0..n]` to `dst[dst_offset..dst_offset+n]`.
    ///
    /// Grid: covers `n` threads (1D). Used to write a compressor-emitted row
    /// into the correct offset within the attn_comp_cache.
    #[kernel]
    pub fn copy_f32_at_offset(src: &[f32], mut dst: DisjointSlice<f32>, dst_offset: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        let v = src[i];
        unsafe {
            *dst.get_unchecked_mut(dst_offset as usize + i) = v;
        }
    }

    /// Convert FP32 to FP16 (stored as raw `u16` bits): `out[i] = f16(x[i])`.
    ///
    /// The output buffer holds raw f16 bit patterns; on the host side it is a
    /// `DeviceBuffer<u16>` (same memory layout as `DeviceBuffer<half::f16>`).
    ///
    /// Ported from f32_to_f16_kernel.
    #[kernel]
    pub fn f32_to_f16(x: &[f32], mut out: DisjointSlice<u16>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = super::f32_to_f16_bits(x[i]);
        }
    }

    /// Broadcast a single embedding row across `n_hc` hierarchical-compression
    /// heads: `out[i] = row[i % n_embd]` for i in 0..n_embd*n_hc.
    /// Ported from repeat_hc_kernel.
    #[kernel]
    pub fn repeat_hc(row: &[f32], mut out: DisjointSlice<f32>, n_embd: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = row[i % n_embd as usize];
        }
    }

    /// SwiGLU activation with optional value clamping and output scaling.
    ///
    /// `out[i] = sigmoid(gate[i]) * gate[i] * up[i] * weight`
    ///
    /// When `clamp > 1e-6`:
    ///   - gate values are clamped to `(-∞, clamp]`
    ///   - up   values are clamped to `[-clamp, clamp]`
    ///
    /// Ported from swiglu_kernel.
    #[kernel]
    pub fn swiglu(gate: &[f32], up: &[f32], mut out: DisjointSlice<f32>, clamp: f32, weight: f32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut g = gate[i];
            let mut u = up[i];
            // TODO(cuda-oxide): f32::min/max call intrinsics::minimum_number_nsz_f32
            // which the NVPTX backend cannot lower. Replace with f32::min/max once fixed.
            if clamp > 1.0e-6 {
                if g > clamp {
                    g = clamp;
                }
                if u > clamp {
                    u = clamp;
                }
                if u < -clamp {
                    u = -clamp;
                }
            }
            let s = g / (1.0 + (-g).exp());
            *out_elem = s * u * weight;
        }
    }
}
