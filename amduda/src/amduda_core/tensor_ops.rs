//! Trait-based tensor operations with a CPU fallback implementation.

/// Common tensor operations used across backends.
pub trait TensorOps {
    /// Matrix multiplication of an `m x k` matrix `a` with a `k x n` matrix `b`.
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32>;

    /// Naive 2D convolution for single channel inputs.
    fn conv2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        input_shape: (usize, usize),
        kernel_shape: (usize, usize),
    ) -> Vec<f32>;

    /// Very small attention primitive returning weighted values.
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], dim: usize) -> Vec<f32>;

    /// Layer normalization applied to a 1D tensor.
    fn layer_norm(&self, x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32>;
}

/// Software fallback used by CPU and emulated by other backends in tests.
pub struct CpuFallback;

impl TensorOps for CpuFallback {
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                out[i * n + j] = sum;
            }
        }
        out
    }

    fn conv2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        input_shape: (usize, usize),
        kernel_shape: (usize, usize),
    ) -> Vec<f32> {
        let (ih, iw) = input_shape;
        let (kh, kw) = kernel_shape;
        let oh = ih - kh + 1;
        let ow = iw - kw + 1;
        let mut out = vec![0.0; oh * ow];
        for i in 0..oh {
            for j in 0..ow {
                let mut sum = 0.0;
                for ki in 0..kh {
                    for kj in 0..kw {
                        sum += input[(i + ki) * iw + (j + kj)] * kernel[ki * kw + kj];
                    }
                }
                out[i * ow + j] = sum;
            }
        }
        out
    }

    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
        let score: f32 = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() / dim as f32;
        v.iter().map(|x| x * score).collect()
    }

    fn layer_norm(&self, x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        let var = x
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / x.len() as f32;
        let denom = (var + eps).sqrt();
        x.iter()
            .zip(gamma)
            .zip(beta)
            .map(|((v, g), b)| ((v - mean) / denom) * g + b)
            .collect()
    }
}
