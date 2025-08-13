use amduda::aurex_lm::quantizer::{
    dequantize_bf16, dequantize_int4, dequantize_int8, quantize_bf16, quantize_int4, quantize_int8,
};

#[test]
fn test_int8_round_trip() {
    let data = [0.0_f32, 0.5, -0.5, 1.0, -1.0];
    let (q, scale) = quantize_int8(&data);
    let deq = dequantize_int8(&q, scale);
    for (a, b) in data.iter().zip(deq.iter()) {
        assert!((a - b).abs() < 1e-2);
    }
}

#[test]
fn test_int4_round_trip() {
    let data = [0.0_f32, 1.0, -1.0, 0.5];
    let (q, scale) = quantize_int4(&data);
    let deq = dequantize_int4(&q, scale, data.len());
    for (a, b) in data.iter().zip(deq.iter()) {
        assert!((a - b).abs() < 1e-1);
    }
}

#[test]
fn test_bf16_round_trip() {
    let data = [0.0_f32, 1.2345, -2.5, 3.75];
    let q = quantize_bf16(&data);
    let deq = dequantize_bf16(&q);
    for (a, b) in data.iter().zip(deq.iter()) {
        assert!((a - b).abs() < 1e-2);
    }
}

