//! SYCL backend implemented via oneAPI (DPC++).
//!
//! The real project will interface with oneAPI to JIT and launch kernels on a
//! variety of accelerators.  The containers used for the unit tests do not ship
//! with the oneAPI runtime.  To keep the build portable this module provides a
//! small emulation layer: we model device enumeration, kernel compilation and
//! launch but ultimately fall back to the [`CpuFallback`] implementation.
//!
//! Sample DPC++ kernels are embedded as string constants.  They illustrate how
//! tensor operations could be expressed in SYCL.  During tests the kernels are
//! not compiled; instead we simply record that the compile/launch path was
//! exercised and delegate the actual computation to the CPU fallback.

use crate::amduda_core::tensor_ops::{CpuFallback, TensorOps};

/// Representation of a SYCL device.  In the emulated implementation only an id
/// and name are tracked.
#[derive(Clone, Copy, Debug)]
pub struct SyclDevice {
    pub id: usize,
    pub name: &'static str,
}

/// Minimal SYCL context holding the selected device.
#[derive(Clone, Copy, Debug)]
pub struct SyclContext {
    device: SyclDevice,
}

/// Backend instance used by higher level code.  In a full implementation this
/// would own a `sycl::queue` and compiled kernels.
#[derive(Clone, Copy, Debug)]
pub struct SyclBackend {
    ctx: SyclContext,
}

impl SyclBackend {
    /// Enumerate available SYCL devices.  When the runtime is unavailable we
    /// emulate a single CPU device so that dispatch logic can still be tested.
    pub fn enumerate() -> Vec<SyclDevice> {
        vec![SyclDevice {
            id: 0,
            name: "SYCL CPU",
        }]
    }

    /// Convenience helper mirroring other backends.
    pub fn is_available() -> bool {
        !Self::enumerate().is_empty()
    }

    /// Create a new backend instance selecting the first enumerated device.
    pub fn new() -> Self {
        let device = Self::enumerate().remove(0);
        SyclBackend {
            ctx: SyclContext { device },
        }
    }

    /// Simulate compilation of a DPC++ kernel.  In this portable build the
    /// source is only recorded and no compilation takes place.
    fn compile_kernel(&self, _src: &str, _name: &str) {
        // No-op for the emulated backend.
    }

    /// Launch a previously "compiled" kernel.  The closure is executed on the
    /// host which allows tests to validate the call chain without oneAPI.
    fn launch<F>(&self, f: F)
    where
        F: FnOnce(),
    {
        f();
    }
}

// -----------------------------------------------------------------------------
// Sample DPC++ kernels (not executed during tests)
// -----------------------------------------------------------------------------

/// Matrix multiplication kernel expressed in SYCL/DPC++.
const MATMUL_KERNEL: &str = r#"
#include <sycl/sycl.hpp>
using namespace sycl;
extern "C" void matmul(queue q, const float* a, const float* b, float* c,
                        size_t M, size_t N, size_t K) {
    q.parallel_for(range<2>(M, N), [=](id<2> idx) {
        size_t i = idx[0];
        size_t j = idx[1];
        float sum = 0.0f;
        for (size_t kk = 0; kk < K; ++kk)
            sum += a[i*K + kk] * b[kk*N + j];
        c[i*N + j] = sum;
    }).wait();
}
"#;

/// 2D convolution for single channel inputs.
const CONV2D_KERNEL: &str = r#"
#include <sycl/sycl.hpp>
using namespace sycl;
extern "C" void conv2d(queue q, const float* input, const float* kernel,
                        float* out, size_t ih, size_t iw,
                        size_t kh, size_t kw) {
    size_t oh = ih - kh + 1;
    size_t ow = iw - kw + 1;
    q.parallel_for(range<2>(oh, ow), [=](id<2> idx) {
        size_t i = idx[0];
        size_t j = idx[1];
        float sum = 0.0f;
        for (size_t ki = 0; ki < kh; ++ki)
            for (size_t kj = 0; kj < kw; ++kj)
                sum += input[(i+ki)*iw + (j+kj)] * kernel[ki*kw + kj];
        out[i*ow + j] = sum;
    }).wait();
}
"#;

/// Simple attention primitive.
const ATTENTION_KERNEL: &str = r#"
#include <sycl/sycl.hpp>
using namespace sycl;
extern "C" void attention(queue q, const float* qv, const float* kv,
                           const float* vv, float* out, size_t dim) {
    q.single_task([=]() {
        float score = 0.0f;
        for (size_t i = 0; i < dim; ++i)
            score += qv[i] * kv[i];
        score /= dim;
        for (size_t i = 0; i < dim; ++i)
            out[i] = vv[i] * score;
    }).wait();
}
"#;

/// Layer normalization kernel.
const LAYERNORM_KERNEL: &str = r#"
#include <sycl/sycl.hpp>
using namespace sycl;
extern "C" void layer_norm(queue q, const float* x, const float* gamma,
                             const float* beta, float* out, size_t len,
                             float eps) {
    q.single_task([=]() {
        float mean = 0.0f;
        for (size_t i = 0; i < len; ++i)
            mean += x[i];
        mean /= len;
        float var = 0.0f;
        for (size_t i = 0; i < len; ++i) {
            float d = x[i] - mean;
            var += d * d;
        }
        var /= len;
        float denom = sycl::sqrt(var + eps);
        for (size_t i = 0; i < len; ++i)
            out[i] = ((x[i] - mean) / denom) * gamma[i] + beta[i];
    }).wait();
}
"#;

// -----------------------------------------------------------------------------
// TensorOps implementation
// -----------------------------------------------------------------------------

impl TensorOps for SyclBackend {
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let cpu = CpuFallback;
        self.compile_kernel(MATMUL_KERNEL, "matmul");
        self.launch(|| {});
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
        self.compile_kernel(CONV2D_KERNEL, "conv2d");
        self.launch(|| {});
        cpu.conv2d(input, kernel, input_shape, kernel_shape)
    }

    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
        let cpu = CpuFallback;
        self.compile_kernel(ATTENTION_KERNEL, "attention");
        self.launch(|| {});
        cpu.attention(q, k, v, dim)
    }

    fn layer_norm(&self, x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        let cpu = CpuFallback;
        self.compile_kernel(LAYERNORM_KERNEL, "layer_norm");
        self.launch(|| {});
        cpu.layer_norm(x, gamma, beta, eps)
    }
}

/// Initialize the SYCL backend by probing available devices.
pub fn init() {
    let _ = SyclBackend::new();
}

