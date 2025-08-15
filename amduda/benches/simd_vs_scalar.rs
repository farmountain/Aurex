use amduda::amduda_core::tensor_ops::{CpuFallback, TensorOps};
use amduda::hal_backends::cpu_simd::CpuSimdBackend;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn matmul_bench(c: &mut Criterion) {
    let m = 32;
    let n = 32;
    let k = 32;
    let a: Vec<f32> = (0..m * k).map(|x| x as f32).collect();
    let b: Vec<f32> = (0..k * n).map(|x| x as f32).collect();
    let cpu = CpuFallback;
    let simd = CpuSimdBackend;
    c.bench_function("matmul_scalar", |bench| {
        bench.iter(|| cpu.matmul(black_box(&a), black_box(&b), m, n, k))
    });
    c.bench_function("matmul_simd", |bench| {
        bench.iter(|| simd.matmul(black_box(&a), black_box(&b), m, n, k))
    });
}

fn conv2d_bench(c: &mut Criterion) {
    let input_shape = (32, 32);
    let kernel_shape = (3, 3);
    let input: Vec<f32> = (0..input_shape.0 * input_shape.1).map(|x| x as f32).collect();
    let kernel: Vec<f32> = (0..kernel_shape.0 * kernel_shape.1).map(|x| x as f32).collect();
    let cpu = CpuFallback;
    let simd = CpuSimdBackend;
    c.bench_function("conv2d_scalar", |bench| {
        bench.iter(|| {
            cpu.conv2d(
                black_box(&input),
                black_box(&kernel),
                input_shape,
                kernel_shape,
            )
        })
    });
    c.bench_function("conv2d_simd", |bench| {
        bench.iter(|| {
            simd.conv2d(
                black_box(&input),
                black_box(&kernel),
                input_shape,
                kernel_shape,
            )
        })
    });
}

fn attention_bench(c: &mut Criterion) {
    let dim = 64;
    let q: Vec<f32> = (0..dim).map(|x| x as f32).collect();
    let k: Vec<f32> = (0..dim).map(|x| (x as f32) * 0.5).collect();
    let v: Vec<f32> = (0..dim).map(|x| (x as f32) * 0.25).collect();
    let cpu = CpuFallback;
    let simd = CpuSimdBackend;
    c.bench_function("attention_scalar", |bench| {
        bench.iter(|| cpu.attention(black_box(&q), black_box(&k), black_box(&v), dim))
    });
    c.bench_function("attention_simd", |bench| {
        bench.iter(|| simd.attention(black_box(&q), black_box(&k), black_box(&v), dim))
    });
}

fn layer_norm_bench(c: &mut Criterion) {
    let len = 128;
    let x: Vec<f32> = (0..len).map(|x| x as f32).collect();
    let gamma: Vec<f32> = vec![1.0; len];
    let beta: Vec<f32> = vec![0.0; len];
    let eps = 1e-5;
    let cpu = CpuFallback;
    let simd = CpuSimdBackend;
    c.bench_function("layer_norm_scalar", |bench| {
        bench.iter(|| cpu.layer_norm(black_box(&x), black_box(&gamma), black_box(&beta), eps))
    });
    c.bench_function("layer_norm_simd", |bench| {
        bench.iter(|| simd.layer_norm(black_box(&x), black_box(&gamma), black_box(&beta), eps))
    });
}

criterion_group!(benches, matmul_bench, conv2d_bench, attention_bench, layer_norm_bench);
criterion_main!(benches);
