use amduda::hal_backends::vulkan_backend::init;
use amduda::amduda_core::tensor_ops::TensorOps;

fn main() {
    // Initialize Vulkan backend (falls back to CPU if unsupported at runtime).
    let backend = init().expect("Failed to init Vulkan backend");
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![1.0f32, 0.0, 0.0, 1.0];
    let out = backend.matmul(&a, &b, 2, 2, 2);
    println!("Result: {:?}", out);
}
