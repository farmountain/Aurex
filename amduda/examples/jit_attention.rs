use amduda::amduda_core::jit_compiler::{compile_kernel_f32, Device};

fn main() {
    let backend = std::env::args().nth(1).unwrap_or_else(|| "cpu".to_string());
    let device = match backend.as_str() {
        "gpu" => Device::GPU,
        _ => Device::CPU,
    };

    let q = vec![1.0, 0.0];
    let k = vec![1.0, 0.0];
    let v = vec![0.0, 1.0];

    let mul = compile_kernel_f32("mul_f32", device).expect("compile kernel");
    let score: f32 = q
        .iter()
        .zip(k.iter())
        .map(|(a, b)| (mul.func)(*a, *b))
        .sum::<f32>()
        / q.len() as f32;
    let out: Vec<f32> = v.iter().map(|x| x * score).collect();

    println!("Attention on {:?}: {:?}", device, out);
}
