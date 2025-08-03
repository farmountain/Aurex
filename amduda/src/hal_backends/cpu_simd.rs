//! CPU backend implementing [`TensorOps`] with naive loops.

use crate::amduda_core::tensor_ops::{CpuFallback, TensorOps};

/// Represents the host CPU using simple SIMD-less operations.
pub struct CpuSimdBackend;

impl TensorOps for CpuSimdBackend {
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let cpu = CpuFallback;
        cpu.matmul(a, b, m, n, k)
    }

    fn conv2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        input_shape: (usize, usize),
        kernel_shape: (usize, usize),
    ) -> Vec<f32> {
        let cpu = CpuFallback;
        cpu.conv2d(input, kernel, input_shape, kernel_shape)
    }

    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
        let cpu = CpuFallback;
        cpu.attention(q, k, v, dim)
    }

    fn layer_norm(&self, x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        let cpu = CpuFallback;
        cpu.layer_norm(x, gamma, beta, eps)
    }
}

/// Initialize the CPU backend.
pub fn init() {
    // No-op for the stubbed backend.
}
