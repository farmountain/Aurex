use amduda::hal_backends::vulkan_backend::VulkanBackend;
use amduda::amduda_core::tensor_ops::TensorOps;

#[test]
fn matmul_kernel_launches() {
    if !VulkanBackend::is_available() {
        eprintln!("Vulkan backend unavailable; skipping test");
        return;
    }
    let backend = VulkanBackend::new().expect("init backend");
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];
    let out = backend.matmul(&a, &b, 2, 2, 2);
    assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
}
