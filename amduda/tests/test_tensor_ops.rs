use amduda::amduda_core::tensor_ops::TensorOps;
use amduda::hal_backends::cpu_simd::CpuSimdBackend;
use amduda::hal_backends::opencl_backend::{DeviceKind, OpenClBackend};
use amduda::hal_backends::rocm_backend::RocmBackend;
use amduda::hal_backends::vulkan_backend::VulkanBackend;

fn run_backend<B: TensorOps>(backend: &B) {
    // MatMul test 2x2 * 2x2
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = backend.matmul(&a, &b, 2, 2, 2);
    assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);

    // Conv2d test 3x3 input with 2x2 kernel
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let kernel = vec![1.0, 0.0, 0.0, 1.0];
    let conv = backend.conv2d(&input, &kernel, (3, 3), (2, 2));
    assert_eq!(conv, vec![6.0, 8.0, 12.0, 14.0]);

    // Attention test with dim=2
    let q = vec![1.0, 0.0];
    let k = vec![1.0, 0.0];
    let v = vec![0.0, 1.0];
    let attn = backend.attention(&q, &k, &v, 2);
    assert_eq!(attn, vec![0.0, 0.5]);

    // LayerNorm test
    let x = vec![1.0, 3.0];
    let gamma = vec![1.0, 1.0];
    let beta = vec![0.0, 0.0];
    let ln = backend.layer_norm(&x, &gamma, &beta, 1e-5);
    for (res, exp) in ln.iter().zip(vec![-1.0, 1.0]) {
        assert!((res - exp).abs() < 1e-3);
    }
}

#[test]
fn cpu_backend_ops() {
    run_backend(&CpuSimdBackend);
}

#[test]
fn rocm_backend_ops() {
    run_backend(&RocmBackend::new());
}

#[test]
fn vulkan_backend_ops() {
    if let Ok(backend) = VulkanBackend::new() {
        run_backend(&backend);
    }
}

#[test]
fn opencl_cpu_backend_ops() {
    let backend = OpenClBackend::new(DeviceKind::Cpu);
    run_backend(&backend);
}

#[test]
fn opencl_gpu_backend_ops() {
    let backend = OpenClBackend::new(DeviceKind::Gpu);
    run_backend(&backend);
}

#[test]
fn opencl_device_enumeration() {
    let devices = OpenClBackend::enumerate();
    assert!(devices.iter().any(|d| d.kind == DeviceKind::Cpu));
    assert!(devices.iter().any(|d| d.kind == DeviceKind::Gpu));
}
