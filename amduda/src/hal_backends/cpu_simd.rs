//! CPU backend implementing [`TensorOps`] with x86 AVX intrinsics.

use std::arch::x86_64::*;

use crate::amduda_core::tensor_ops::{CpuFallback, TensorOps};

/// Represents the host CPU using SIMD operations when available.
pub struct CpuSimdBackend;

impl TensorOps for CpuSimdBackend {
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        if is_x86_feature_detected!("avx") {
            unsafe { matmul_avx(a, b, m, n, k) }
        } else {
            let cpu = CpuFallback;
            cpu.matmul(a, b, m, n, k)
        }
    }

    fn conv2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        input_shape: (usize, usize),
        kernel_shape: (usize, usize),
    ) -> Vec<f32> {
        if is_x86_feature_detected!("avx") {
            unsafe { conv2d_avx(input, kernel, input_shape, kernel_shape) }
        } else {
            let cpu = CpuFallback;
            cpu.conv2d(input, kernel, input_shape, kernel_shape)
        }
    }

    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
        if is_x86_feature_detected!("avx") {
            unsafe { attention_avx(q, k, v, dim) }
        } else {
            let cpu = CpuFallback;
            cpu.attention(q, k, v, dim)
        }
    }

    fn layer_norm(&self, x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        if is_x86_feature_detected!("avx") {
            unsafe { layer_norm_avx(x, gamma, beta, eps) }
        } else {
            let cpu = CpuFallback;
            cpu.layer_norm(x, gamma, beta, eps)
        }
    }
}

/// Initialize the CPU backend.
pub fn init() {
    // No-op for the stubbed backend.
}

#[target_feature(enable = "avx")]
unsafe fn matmul_avx(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        let row_a = &a[i * k..(i + 1) * k];
        let mut j = 0;
        while j + 8 <= n {
            let mut sum = _mm256_setzero_ps();
            for p in 0..k {
                let a_val = _mm256_set1_ps(row_a[p]);
                let b_ptr = b.as_ptr().add(p * n + j);
                let b_vec = _mm256_loadu_ps(b_ptr);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a_val, b_vec));
            }
            _mm256_storeu_ps(out.as_mut_ptr().add(i * n + j), sum);
            j += 8;
        }
        while j < n {
            let mut s = 0.0;
            for p in 0..k {
                s += row_a[p] * b[p * n + j];
            }
            out[i * n + j] = s;
            j += 1;
        }
    }
    out
}

#[target_feature(enable = "avx")]
unsafe fn conv2d_avx(
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
            let mut acc = _mm256_setzero_ps();
            let mut tail = 0.0;
            for ki in 0..kh {
                let inp_row = (i + ki) * iw + j;
                let ker_row = ki * kw;
                let mut kj = 0;
                while kj + 8 <= kw {
                    let inp = _mm256_loadu_ps(input.as_ptr().add(inp_row + kj));
                    let ker = _mm256_loadu_ps(kernel.as_ptr().add(ker_row + kj));
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(inp, ker));
                    kj += 8;
                }
                while kj < kw {
                    tail += input[inp_row + kj] * kernel[ker_row + kj];
                    kj += 1;
                }
            }
            let mut buf = [0f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), acc);
            let sum = buf.iter().sum::<f32>() + tail;
            out[i * ow + j] = sum;
        }
    }
    out
}

#[target_feature(enable = "avx")]
unsafe fn attention_avx(q: &[f32], k: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
    let mut acc = _mm256_setzero_ps();
    let mut idx = 0;
    while idx + 8 <= dim {
        let qv = _mm256_loadu_ps(q.as_ptr().add(idx));
        let kv = _mm256_loadu_ps(k.as_ptr().add(idx));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(qv, kv));
        idx += 8;
    }
    let mut buf = [0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    let mut score = buf.iter().sum::<f32>();
    while idx < dim {
        score += q[idx] * k[idx];
        idx += 1;
    }
    score /= dim as f32;

    let mut out = vec![0.0; dim];
    let mut idx = 0;
    let score_v = _mm256_set1_ps(score);
    while idx + 8 <= dim {
        let vv = _mm256_loadu_ps(v.as_ptr().add(idx));
        let res = _mm256_mul_ps(vv, score_v);
        _mm256_storeu_ps(out.as_mut_ptr().add(idx), res);
        idx += 8;
    }
    while idx < dim {
        out[idx] = v[idx] * score;
        idx += 1;
    }
    out
}

#[target_feature(enable = "avx")]
unsafe fn layer_norm_avx(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let len = x.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= len {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        sum = _mm256_add_ps(sum, xv);
        i += 8;
    }
    let mut buf = [0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), sum);
    let mut mean = buf.iter().sum::<f32>();
    while i < len {
        mean += x[i];
        i += 1;
    }
    mean /= len as f32;

    let mut var = _mm256_setzero_ps();
    let mut i = 0;
    let mean_v = _mm256_set1_ps(mean);
    while i + 8 <= len {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let diff = _mm256_sub_ps(xv, mean_v);
        var = _mm256_add_ps(var, _mm256_mul_ps(diff, diff));
        i += 8;
    }
    _mm256_storeu_ps(buf.as_mut_ptr(), var);
    let mut variance = buf.iter().sum::<f32>();
    while i < len {
        let d = x[i] - mean;
        variance += d * d;
        i += 1;
    }
    variance /= len as f32;
    let denom = (variance + eps).sqrt();

    let mut out = vec![0.0; len];
    let denom_v = _mm256_set1_ps(denom);
    let mut i = 0;
    while i + 8 <= len {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let gv = _mm256_loadu_ps(gamma.as_ptr().add(i));
        let bv = _mm256_loadu_ps(beta.as_ptr().add(i));
        let norm = _mm256_div_ps(_mm256_sub_ps(xv, mean_v), denom_v);
        let res = _mm256_add_ps(_mm256_mul_ps(norm, gv), bv);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), res);
        i += 8;
    }
    while i < len {
        out[i] = ((x[i] - mean) / denom) * gamma[i] + beta[i];
        i += 1;
    }
    out
}

