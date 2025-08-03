#[test]
fn test_tensor_ops_core() {
    let a = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b = ndarray::arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let result = a.dot(&b);
    assert_eq!(result, a);

    #[cfg(feature = "hipcortex")]
    println!("HipCortex: KV cache allocation logged");
}
