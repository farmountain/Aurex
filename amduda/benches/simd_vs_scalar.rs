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

criterion_group!(benches, matmul_bench);
criterion_main!(benches);
