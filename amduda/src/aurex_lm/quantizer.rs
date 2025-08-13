//! Quantization and dequantization utilities for Aurex-LM.
//!
//! The routines implemented here perform simple per-tensor symmetric
//! quantization.  INT8 and INT4 quantization return a scale factor that is
//! required for dequantization while BF16 quantization simply truncates the
//! mantissa of `f32` values.

use half::bf16;

/// Quantize a slice of `f32` values into INT8 representation.
///
/// Returns the quantized values and the scaling factor used during
/// quantization.
pub fn quantize_int8(data: &[f32]) -> (Vec<i8>, f32) {
    let max = data
        .iter()
        .fold(0.0_f32, |m, &v| m.max(v.abs()));
    let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
    let quantized = data
        .iter()
        .map(|&v| {
            let q = (v / scale).round();
            q.clamp(-127.0, 127.0) as i8
        })
        .collect();
    (quantized, scale)
}

/// Dequantize INT8 values back into `f32` using the provided scale.
pub fn dequantize_int8(data: &[i8], scale: f32) -> Vec<f32> {
    data.iter().map(|&v| v as f32 * scale).collect()
}

/// Quantize a slice of `f32` values into INT4 representation.
///
/// The returned vector packs two 4-bit signed values per byte.  The caller is
/// expected to track the original length of the data for correct
/// dequantization.
pub fn quantize_int4(data: &[f32]) -> (Vec<u8>, f32) {
    let max = data
        .iter()
        .fold(0.0_f32, |m, &v| m.max(v.abs()));
    let scale = if max == 0.0 { 1.0 } else { max / 7.0 };
    let mut out = Vec::with_capacity((data.len() + 1) / 2);
    for chunk in data.chunks(2) {
        let mut byte = 0u8;
        for (i, &val) in chunk.iter().enumerate() {
            let q = (val / scale).round().clamp(-8.0, 7.0) as i8;
            let nibble = (q as i8 & 0x0F) as u8;
            if i == 0 {
                byte |= nibble;
            } else {
                byte |= nibble << 4;
            }
        }
        out.push(byte);
    }
    (out, scale)
}

/// Dequantize INT4 values back into `f32` using the provided scale and
/// original length of the data.
pub fn dequantize_int4(data: &[u8], scale: f32, len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    for byte in data {
        let low = ((byte & 0x0F) as i8) << 4 >> 4; // sign extend
        let high = ((byte >> 4) as i8) << 4 >> 4;
        out.push(low as f32 * scale);
        if out.len() == len {
            break;
        }
        out.push(high as f32 * scale);
    }
    out
}

/// Quantize a slice of `f32` values into BF16 representation returning raw
/// `u16` bit patterns.
pub fn quantize_bf16(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&v| bf16::from_f32(v).to_bits()).collect()
}

/// Dequantize BF16 values represented as `u16` bit patterns back to `f32`.
pub fn dequantize_bf16(data: &[u16]) -> Vec<f32> {
    data.iter().map(|&v| bf16::from_bits(v).to_f32()).collect()
}

