#[test]
fn test_quantization_round_trip() {
    let data = vec![0.0_f32, 0.5, -0.5];
    let quantized: Vec<i8> = data.iter().map(|x| (x * 127.0) as i8).collect();
    assert_eq!(quantized, vec![0, 63, -63]);
}
