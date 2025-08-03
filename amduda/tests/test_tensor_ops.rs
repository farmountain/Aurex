use amduda::amduda_core::tensor_ops::{tensor_add, tensor_mul};

#[test]
fn test_add_mul() {
    let a = vec![1.0, 2.0];
    let b = vec![3.0, 4.0];
    assert_eq!(tensor_add(&a, &b), vec![4.0, 6.0]);
    assert_eq!(tensor_mul(&a, &b), vec![3.0, 8.0]);
}
