use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use deepseek_engine::engine::Kernels;
/// GPU kernel smoke tests using synthetic small data.
///
/// These run the actual compiled CUDA kernels and verify they produce correct
/// output.  The key regression they guard against is the NVPTX alloca bug:
/// using `&slice[i] as *const T` generates a 1-byte alloca instead of a direct
/// GEP, causing the kernel to read garbage past the first byte.  A test that
/// checks numerical output against a CPU reference will fail immediately if
/// that pattern returns.
///
/// To run:  cargo test --test kernels_smoke
/// Skips silently if no GPU is available.
use std::sync::Arc;

// ── CUDA setup ────────────────────────────────────────────────────────────────

struct Ctx {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<cuda_core::CudaStream>,
    kernels: Kernels,
}

/// Try to create a CUDA context + loaded kernels.  Returns None if no GPU.
fn try_setup() -> Option<Ctx> {
    let ctx: Arc<CudaContext> = CudaContext::new(0).ok()?;
    let stream = ctx.new_stream().ok()?;
    let kernels = Kernels::load(&ctx).ok()?;
    Some(Ctx {
        ctx,
        stream,
        kernels,
    })
}

#[allow(dead_code)]
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

// ── matmul_f16 ────────────────────────────────────────────────────────────────

#[test]
fn matmul_f16_all_ones() {
    // W[OUT×IN] = all f16 1.0, x[IN] = all f32 1.0 → each output = IN.
    let Some(c) = try_setup() else {
        return;
    };
    const IN: usize = 64;
    const OUT: usize = 8;
    const F16_ONE: u16 = 0x3C00;

    let w_data: Vec<u16> = vec![F16_ONE; OUT * IN];
    let x_data: Vec<f32> = vec![1.0f32; IN];

    let w_buf = DeviceBuffer::from_host(&c.stream, &w_data).unwrap();
    let x_buf = DeviceBuffer::from_host(&c.stream, &x_data).unwrap();
    let mut out_buf: DeviceBuffer<f32> = DeviceBuffer::zeroed(&c.stream, OUT).unwrap();

    c.kernels
        .matmul
        .matmul_f16(
            &c.stream,
            LaunchConfig {
                grid_dim: (OUT as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            &w_buf,
            &x_buf,
            &mut out_buf,
            IN as u64,
            OUT as u64,
            1,
        )
        .unwrap();

    let result = out_buf.to_host_vec(&c.stream).unwrap();
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - IN as f32).abs() < 0.1,
            "matmul_f16 row {i}: expected {}, got {v}",
            IN
        );
    }
}

#[test]
fn matmul_f16_identity_row() {
    // First row of W is [1,0,0,...], rest are zero → only output[0] = x[0].
    let Some(c) = try_setup() else {
        return;
    };
    const IN: usize = 32;
    const OUT: usize = 4;
    const F16_ONE: u16 = 0x3C00;

    let mut w_data = vec![0u16; OUT * IN];
    w_data[0] = F16_ONE; // W[0, 0] = 1.0, rest = 0
    let x_data: Vec<f32> = (0..IN).map(|i| i as f32).collect();

    let w_buf = DeviceBuffer::from_host(&c.stream, &w_data).unwrap();
    let x_buf = DeviceBuffer::from_host(&c.stream, &x_data).unwrap();
    let mut out_buf: DeviceBuffer<f32> = DeviceBuffer::zeroed(&c.stream, OUT).unwrap();

    c.kernels
        .matmul
        .matmul_f16(
            &c.stream,
            LaunchConfig {
                grid_dim: (OUT as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            &w_buf,
            &x_buf,
            &mut out_buf,
            IN as u64,
            OUT as u64,
            1,
        )
        .unwrap();

    let result = out_buf.to_host_vec(&c.stream).unwrap();
    assert!(
        (result[0] - 0.0).abs() < 0.01,
        "output[0] should be x[0]=0.0, got {}",
        result[0]
    );
    for i in 1..OUT {
        assert!(
            (result[i] - 0.0).abs() < 0.01,
            "output[{i}] should be 0.0, got {}",
            result[i]
        );
    }
}

// ── matmul_q8_0_preq_warp8 ───────────────────────────────────────────────────
//
// This directly exercises the NVPTX alloca-bug-prone code path in the kernel.
// If `&w[blk_off+2] as *const i8` ever returns (wrong: 1-byte alloca instead
// of GEP), the dot product reads garbage and the result diverges from the CPU
// reference computed here.

/// Build a Q8_0 weight matrix (n_rows × 1 block each, 34 bytes/block).
/// All rows identical: f16 scale + all-`qs_val` int8.
fn make_q8_0_weight(n_rows: usize, scale_f16: u16, qs_val: i8) -> Vec<u8> {
    let mut w = vec![0u8; n_rows * 34];
    for r in 0..n_rows {
        let off = r * 34;
        w[off] = (scale_f16 & 0xff) as u8;
        w[off + 1] = (scale_f16 >> 8) as u8;
        for i in 0..32 {
            w[off + 2 + i] = qs_val as u8;
        }
    }
    w
}

#[test]
fn matmul_q8_0_all_ones_single_block() {
    // w_scale=1.0, w_qs=[1]*32; x_scale=1.0, xq=[1]*32.
    // Expected per row: 1.0 * 1.0 * sum(1*1 for 32) = 32.0
    let Some(c) = try_setup() else {
        return;
    };
    const OUT: usize = 8; // output rows; must be multiple of 8 for cfg_warp8
    const IN: usize = 32; // one block of 32 elements
    const BLOCKS: usize = 1;

    let w_data = make_q8_0_weight(OUT, 0x3C00 /*f16 1.0*/, 1i8);
    let xq_data: Vec<i8> = vec![1i8; IN];
    let xs_data: Vec<f32> = vec![1.0f32; BLOCKS];

    let w_buf = DeviceBuffer::from_host(&c.stream, w_data.as_slice()).unwrap();
    let xq_buf = DeviceBuffer::from_host(&c.stream, xq_data.as_slice()).unwrap();
    let xs_buf = DeviceBuffer::from_host(&c.stream, xs_data.as_slice()).unwrap();
    let mut out_buf: DeviceBuffer<f32> = DeviceBuffer::zeroed(&c.stream, OUT).unwrap();

    c.kernels
        .matmul
        .matmul_q8_0_preq_warp8(
            &c.stream,
            LaunchConfig {
                grid_dim: ((OUT as u32 + 7) / 8, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            &w_buf,
            &xq_buf,
            &xs_buf,
            &mut out_buf,
            IN as u64,
            OUT as u64,
            BLOCKS as u64,
        )
        .unwrap();

    let result = out_buf.to_host_vec(&c.stream).unwrap();
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - 32.0).abs() < 0.1,
            "matmul_q8_0 row {i}: expected 32.0, got {v}"
        );
    }
}

#[test]
fn matmul_q8_0_negative_weights_produce_negative_output() {
    let Some(c) = try_setup() else {
        return;
    };
    const OUT: usize = 8;
    const IN: usize = 32;

    // w_scale=1.0, w_qs=[-1]*32; x_scale=1.0, xq=[1]*32 → expected = -32.0
    let w_data = make_q8_0_weight(OUT, 0x3C00, -1i8);
    let xq_buf = DeviceBuffer::from_host(&c.stream, vec![1i8; IN].as_slice()).unwrap();
    let xs_buf = DeviceBuffer::from_host(&c.stream, vec![1.0f32; 1].as_slice()).unwrap();
    let w_buf = DeviceBuffer::from_host(&c.stream, w_data.as_slice()).unwrap();
    let mut out_buf: DeviceBuffer<f32> = DeviceBuffer::zeroed(&c.stream, OUT).unwrap();

    c.kernels
        .matmul
        .matmul_q8_0_preq_warp8(
            &c.stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            &w_buf,
            &xq_buf,
            &xs_buf,
            &mut out_buf,
            IN as u64,
            OUT as u64,
            1,
        )
        .unwrap();

    let result = out_buf.to_host_vec(&c.stream).unwrap();
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - (-32.0)).abs() < 0.1,
            "row {i}: expected -32.0, got {v}"
        );
    }
}

#[test]
fn matmul_q8_0_scale_applied_correctly() {
    // w_scale=0.5, w_qs=[1]*32, x_scale=1.0, xq=[1]*32 → expected = 0.5*32=16.0
    let Some(c) = try_setup() else {
        return;
    };
    const OUT: usize = 8;
    const IN: usize = 32;

    let half_f16: u16 = 0x3800; // f16 0.5
    let w_data = make_q8_0_weight(OUT, half_f16, 1i8);
    let xq_buf = DeviceBuffer::from_host(&c.stream, vec![1i8; IN].as_slice()).unwrap();
    let xs_buf = DeviceBuffer::from_host(&c.stream, vec![1.0f32; 1].as_slice()).unwrap();
    let w_buf = DeviceBuffer::from_host(&c.stream, w_data.as_slice()).unwrap();
    let mut out_buf: DeviceBuffer<f32> = DeviceBuffer::zeroed(&c.stream, OUT).unwrap();

    c.kernels
        .matmul
        .matmul_q8_0_preq_warp8(
            &c.stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            &w_buf,
            &xq_buf,
            &xs_buf,
            &mut out_buf,
            IN as u64,
            OUT as u64,
            1,
        )
        .unwrap();

    let result = out_buf.to_host_vec(&c.stream).unwrap();
    for (i, &v) in result.iter().enumerate() {
        assert!((v - 16.0).abs() < 0.1, "row {i}: expected 16.0, got {v}");
    }
}

// ── rms_norm_plain ────────────────────────────────────────────────────────────

#[test]
fn rms_norm_plain_unit_vector() {
    // Normalising a vector of all-equal values should give all-equal outputs
    // and the RMS of the output should be 1.0.
    let Some(c) = try_setup() else {
        return;
    };
    const N: usize = 256;
    const EPS: f32 = 1e-5;

    let input: Vec<f32> = vec![2.0f32; N];
    let in_buf = DeviceBuffer::from_host(&c.stream, input.as_slice()).unwrap();
    let mut out_buf: DeviceBuffer<f32> = DeviceBuffer::zeroed(&c.stream, N).unwrap();

    c.kernels
        .norm
        .rms_norm_plain(
            &c.stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            &in_buf,
            &mut out_buf,
            N as u32,
            1,
            EPS,
        )
        .unwrap();

    let result = out_buf.to_host_vec(&c.stream).unwrap();
    // All-equal input → all-equal output → RMS = 1.0
    let rms = (result.iter().map(|x| x * x).sum::<f32>() / N as f32).sqrt();
    assert!(
        (rms - 1.0).abs() < 0.001,
        "rms_norm_plain: RMS should be 1.0, got {rms}"
    );
    // All elements should be identical
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - result[0]).abs() < 1e-5,
            "element {i} differs: {v} vs {}",
            result[0]
        );
    }
}

#[test]
fn rms_norm_plain_doubles_rms() {
    // Scaling all inputs by 2 changes norm but output (rms-normalised) stays same.
    // After rms_norm, all-equal input → all-equal output regardless of scale.
    let Some(c) = try_setup() else {
        return;
    };
    const N: usize = 128;

    let input1: Vec<f32> = vec![1.0f32; N];
    let input2: Vec<f32> = vec![3.0f32; N];

    let in1 = DeviceBuffer::from_host(&c.stream, input1.as_slice()).unwrap();
    let in2 = DeviceBuffer::from_host(&c.stream, input2.as_slice()).unwrap();
    let mut out1: DeviceBuffer<f32> = DeviceBuffer::zeroed(&c.stream, N).unwrap();
    let mut out2: DeviceBuffer<f32> = DeviceBuffer::zeroed(&c.stream, N).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    c.kernels
        .norm
        .rms_norm_plain(&c.stream, cfg, &in1, &mut out1, N as u32, 1, 1e-5)
        .unwrap();
    c.kernels
        .norm
        .rms_norm_plain(&c.stream, cfg, &in2, &mut out2, N as u32, 1, 1e-5)
        .unwrap();

    let r1 = out1.to_host_vec(&c.stream).unwrap();
    let r2 = out2.to_host_vec(&c.stream).unwrap();
    assert!(
        (r1[0] - r2[0]).abs() < 1e-4,
        "rms_norm of all-equal inputs should give same output regardless of scale: {} vs {}",
        r1[0],
        r2[0]
    );
}

// ── swiglu ────────────────────────────────────────────────────────────────────

#[test]
fn swiglu_positive_gate_positive_up() {
    // silu(x) = x * sigmoid(x); for x → ∞, silu(x) → x.
    // For gate=10.0, up=2.0: silu(10)*2 ≈ 10*0.9999*2 ≈ 19.999
    let Some(c) = try_setup() else {
        return;
    };
    const N: usize = 32;

    let gate: Vec<f32> = vec![10.0f32; N];
    let up: Vec<f32> = vec![2.0f32; N];
    let gate_buf = DeviceBuffer::from_host(&c.stream, gate.as_slice()).unwrap();
    let up_buf = DeviceBuffer::from_host(&c.stream, up.as_slice()).unwrap();
    let mut out_buf: DeviceBuffer<f32> = DeviceBuffer::zeroed(&c.stream, N).unwrap();

    c.kernels
        .utils
        .swiglu(
            &c.stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            &gate_buf,
            &up_buf,
            &mut out_buf,
            0.0, // no clamp
            1.0, // weight = 1.0
        )
        .unwrap();

    let result = out_buf.to_host_vec(&c.stream).unwrap();
    // silu(10) ≈ 10 * sigmoid(10) ≈ 10 * 0.99995 ≈ 9.9995
    // output ≈ 9.9995 * 2.0 ≈ 19.999
    for (i, &v) in result.iter().enumerate() {
        assert!(v > 19.9 && v < 20.1, "swiglu[{i}]: expected ~20.0, got {v}");
    }
}

#[test]
fn swiglu_zero_gate_gives_zero() {
    // silu(0) = 0 * 0.5 = 0, so output = 0 regardless of up.
    let Some(c) = try_setup() else {
        return;
    };
    const N: usize = 32;

    let gate: Vec<f32> = vec![0.0f32; N];
    let up: Vec<f32> = vec![5.0f32; N];
    let gate_buf = DeviceBuffer::from_host(&c.stream, gate.as_slice()).unwrap();
    let up_buf = DeviceBuffer::from_host(&c.stream, up.as_slice()).unwrap();
    let mut out_buf: DeviceBuffer<f32> = DeviceBuffer::zeroed(&c.stream, N).unwrap();

    c.kernels
        .utils
        .swiglu(
            &c.stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            &gate_buf,
            &up_buf,
            &mut out_buf,
            0.0,
            1.0,
        )
        .unwrap();

    let result = out_buf.to_host_vec(&c.stream).unwrap();
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.abs() < 0.001,
            "swiglu[{i}] with gate=0: expected 0, got {v}"
        );
    }
}

#[test]
fn swiglu_weight_scales_output() {
    // weight=2.0 should double the output vs weight=1.0.
    let Some(c) = try_setup() else {
        return;
    };
    const N: usize = 32;

    let gate: Vec<f32> = vec![1.0f32; N];
    let up: Vec<f32> = vec![1.0f32; N];
    let gate_buf = DeviceBuffer::from_host(&c.stream, gate.as_slice()).unwrap();
    let up_buf = DeviceBuffer::from_host(&c.stream, up.as_slice()).unwrap();
    let mut out1: DeviceBuffer<f32> = DeviceBuffer::zeroed(&c.stream, N).unwrap();
    let mut out2: DeviceBuffer<f32> = DeviceBuffer::zeroed(&c.stream, N).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    c.kernels
        .utils
        .swiglu(&c.stream, cfg, &gate_buf, &up_buf, &mut out1, 0.0, 1.0)
        .unwrap();
    c.kernels
        .utils
        .swiglu(&c.stream, cfg, &gate_buf, &up_buf, &mut out2, 0.0, 2.0)
        .unwrap();

    let r1 = out1.to_host_vec(&c.stream).unwrap();
    let r2 = out2.to_host_vec(&c.stream).unwrap();
    assert!(
        (r2[0] / r1[0] - 2.0).abs() < 0.001,
        "weight=2 should double output: {} / {} ≠ 2",
        r2[0],
        r1[0]
    );
}

// ── quantize_q8_0 ────────────────────────────────────────────────────────────

#[test]
fn quantize_q8_0_all_ones_roundtrip() {
    // Quantize a constant vector and check the scale and qs values.
    // For all-1.0 input: max_abs = 1.0, scale = 1.0/127, qs[i] = 127.
    let Some(c) = try_setup() else {
        return;
    };
    const N: usize = 32; // one block
    const BLOCKS: usize = 1;

    let input: Vec<f32> = vec![1.0f32; N];
    let in_buf = DeviceBuffer::from_host(&c.stream, input.as_slice()).unwrap();
    let mut xq_buf: DeviceBuffer<i8> = DeviceBuffer::zeroed(&c.stream, N).unwrap();
    let mut xs_buf: DeviceBuffer<f32> = DeviceBuffer::zeroed(&c.stream, BLOCKS).unwrap();

    c.kernels
        .quantize
        .quantize_q8_0(
            &c.stream,
            LaunchConfig {
                grid_dim: (BLOCKS as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            &in_buf,
            &mut xq_buf,
            &mut xs_buf,
            N as u64,
            BLOCKS as u64,
        )
        .unwrap();

    let qs = xq_buf.to_host_vec(&c.stream).unwrap();
    let scales = xs_buf.to_host_vec(&c.stream).unwrap();

    // All values should be 127 (max int8)
    for (i, &q) in qs.iter().enumerate() {
        assert_eq!(q, 127i8, "qs[{i}] should be 127, got {q}");
    }
    // Scale should be 1.0/127
    let expected_scale = 1.0f32 / 127.0f32;
    assert!(
        (scales[0] - expected_scale).abs() < 1e-6,
        "scale should be {expected_scale}, got {}",
        scales[0]
    );
}

#[test]
fn quantize_q8_0_dequantizes_correctly() {
    // Quantize then dequantize: q*scale should recover the original values.
    let Some(c) = try_setup() else {
        return;
    };
    const N: usize = 32;

    let input: Vec<f32> = (0..N).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let in_buf = DeviceBuffer::from_host(&c.stream, input.as_slice()).unwrap();
    let mut xq_buf: DeviceBuffer<i8> = DeviceBuffer::zeroed(&c.stream, N).unwrap();
    let mut xs_buf: DeviceBuffer<f32> = DeviceBuffer::zeroed(&c.stream, 1).unwrap();

    c.kernels
        .quantize
        .quantize_q8_0(
            &c.stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            &in_buf,
            &mut xq_buf,
            &mut xs_buf,
            N as u64,
            1,
        )
        .unwrap();

    let qs = xq_buf.to_host_vec(&c.stream).unwrap();
    let scales = xs_buf.to_host_vec(&c.stream).unwrap();

    let max_abs = input.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let expected_scale = max_abs / 127.0;

    for (i, (&q, &orig)) in qs.iter().zip(input.iter()).enumerate() {
        let reconstructed = q as f32 * scales[0];
        let err = (reconstructed - orig).abs();
        assert!(
            err < expected_scale * 1.5,
            "element {i}: orig={orig:.4} reconstructed={reconstructed:.4} err={err:.4}"
        );
    }
}
