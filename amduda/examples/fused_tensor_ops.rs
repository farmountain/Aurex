use amduda::amduda_core::tensor_ops::TensorOps;
use amduda::hal_backends::{cpu_simd::CpuSimdBackend, rocm_backend::RocmBackend, vulkan_backend};

fn main() {
    // Instantiate backends. GPU backends fall back to CPU when unavailable.
    let cpu = CpuSimdBackend;
    let rocm = RocmBackend::new();
    let vulkan = vulkan_backend::init().ok();

    // MatMul on ROCm GPU
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];
    let mm = rocm.matmul(&a, &b, 2, 2, 2);

    // Attention on CPU using pieces of matmul output
    let q = vec![mm[0], mm[1]];
    let k = vec![mm[2], mm[3]];
    let v = vec![0.5f32, -0.5];
    let attn = cpu.attention(&q, &k, &v, 2);

    // Conv2d on Vulkan if available, otherwise CPU
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let kernel = vec![1.0, 0.0, 0.0, 1.0];
    let conv = vulkan.map_or_else(
        || cpu.conv2d(&input, &kernel, (3, 3), (2, 2)),
        |vk| vk.conv2d(&input, &kernel, (3, 3), (2, 2)),
    );

    // LayerNorm on CPU
    let gamma = vec![1.0; conv.len()];
    let beta = vec![0.0; conv.len()];
    let norm = cpu.layer_norm(&conv, &gamma, &beta, 1e-5);

    println!("MatMul:   {:?}", mm);
    println!("Attention:{:?}", attn);
    println!("Conv2d:   {:?}", conv);
    println!("LayerNorm:{:?}", norm);
}
