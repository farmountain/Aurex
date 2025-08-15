use aurex_backend::dispatch::{TensorOps, CpuBackend, RocmBackend, SyclBackend, OpenClBackend};
use aurex_backend::VulkanBackend;

fn all_backends() -> Vec<Box<dyn TensorOps>> {
    vec![
        Box::new(CpuBackend),
        Box::new(RocmBackend),
        Box::new(SyclBackend),
        Box::new(OpenClBackend),
        Box::new(VulkanBackend::new()),
    ]
}

#[test]
fn matmul_consistency() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let expected = vec![19.0, 22.0, 43.0, 50.0];
    for backend in all_backends() {
        assert_eq!(backend.matmul(&a, &b, 2, 2, 2), expected);
    }
}

#[test]
fn conv2d_consistency() {
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    let kernel = vec![1.0];
    for backend in all_backends() {
        assert_eq!(backend.conv2d(&input, &kernel, (2, 2), (1, 1)), input);
    }
}

#[test]
fn attention_consistency() {
    let q = vec![1.0, 0.0];
    let k = vec![1.0, 0.0];
    let v = vec![0.5, -0.5];
    let expected = vec![0.25, -0.25];
    for backend in all_backends() {
        assert_eq!(backend.attention(&q, &k, &v, 2), expected);
    }
}

#[test]
fn layer_norm_consistency() {
    let x = vec![1.0, 2.0];
    let gamma = vec![1.0, 1.0];
    let beta = vec![0.0, 0.0];
    for backend in all_backends() {
        let out = backend.layer_norm(&x, &gamma, &beta, 1e-5);
        assert!((out[0] - (-1.0)).abs() < 1e-4);
        assert!((out[1] - 1.0).abs() < 1e-4);
    }
}
