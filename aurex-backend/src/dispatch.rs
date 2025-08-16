//! Backend dispatch layer selecting the best available implementation.
//!
//! The dispatcher exposes a [`TensorOps`] trait implemented by several backend
//! stubs.  Backends can be enabled or disabled via environment variables and are
//! chosen based on user preference or workload characteristics.

/// Common tensor operations.
pub trait TensorOps {
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32>;
    fn conv2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        input_shape: (usize, usize),
        kernel_shape: (usize, usize),
    ) -> Vec<f32>;
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], dim: usize) -> Vec<f32>;
    fn layer_norm(&self, x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32>;
}

/// CPU fallback implementing all tensor operations in software.
#[derive(Clone, Copy, Debug)]
pub struct CpuBackend;

impl TensorOps for CpuBackend {
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

use crate::vulkan_backend::VulkanBackend;
pub use crate::sycl_backend::SyclBackend;

/// Placeholder GPU backends delegating to the CPU implementation.
#[derive(Clone, Copy, Debug)]
pub struct RocmBackend;
#[derive(Clone, Copy, Debug)]
pub struct OpenClBackend;

impl TensorOps for RocmBackend {
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        CpuBackend.matmul(a, b, m, n, k)
    }
    fn conv2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        input_shape: (usize, usize),
        kernel_shape: (usize, usize),
    ) -> Vec<f32> {
        CpuBackend.conv2d(input, kernel, input_shape, kernel_shape)
    }
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
        CpuBackend.attention(q, k, v, dim)
    }
    fn layer_norm(&self, x: &[f32], g: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
        CpuBackend.layer_norm(x, g, b, eps)
    }
}


impl TensorOps for OpenClBackend {
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        CpuBackend.matmul(a, b, m, n, k)
    }
    fn conv2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        input_shape: (usize, usize),
        kernel_shape: (usize, usize),
    ) -> Vec<f32> {
        CpuBackend.conv2d(input, kernel, input_shape, kernel_shape)
    }
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
        CpuBackend.attention(q, k, v, dim)
    }
    fn layer_norm(&self, x: &[f32], g: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
        CpuBackend.layer_norm(x, g, b, eps)
    }
}

/// Available compute backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Rocm,
    Sycl,
    OpenCl,
    Vulkan,
}

/// Simplified workload descriptor used by the dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workload {
    Light,
    Heavy,
}

/// Dispatcher wrapping a [`TensorOps`] implementation selected at runtime.
pub struct Dispatcher {
    backend: Backend,
    ops: Box<dyn TensorOps + Send + Sync>,
}

impl Dispatcher {
    /// Create a new dispatcher selecting a backend based on user preference,
    /// availability and workload characteristics.  If [`preferred`] is `None`
    /// the dispatcher consults the `AUREX_BACKEND` environment variable.  When
    /// the requested backend is unavailable it falls back to an automatically
    /// selected implementation for the provided [`Workload`].
    pub fn new(preferred: Option<Backend>, workload: Workload) -> Self {
        let preferred = preferred.or_else(Self::backend_from_env);
        let backend = match preferred {
            Some(b) if Self::is_available(b) => b,
            _ => Self::select_backend(workload),
        };
        let ops = Self::backend_ops(backend);
        Self { backend, ops }
    }

    /// Convenience wrapper constructing the dispatcher solely from environment
    /// variables and workload description.
    pub fn from_env(workload: Workload) -> Self {
        Self::new(None, workload)
    }

    /// Return the backend currently used by the dispatcher.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Determine if a backend is available.  Availability can be overridden via
    /// `AUREX_DISABLE_*` environment variables for testing purposes.
    fn is_available(backend: Backend) -> bool {
        match backend {
            Backend::Cpu => true,
            Backend::Rocm => std::env::var("AUREX_DISABLE_ROCM").is_err(),
            Backend::Sycl => {
                std::env::var("AUREX_DISABLE_SYCL").is_err() && SyclBackend::is_available()
            },
            Backend::OpenCl => std::env::var("AUREX_DISABLE_OPENCL").is_err(),
            Backend::Vulkan => {
                std::env::var("AUREX_DISABLE_VULKAN").is_err() && VulkanBackend::is_available()
            }
        }
    }

    /// Automatically select the most appropriate backend for the workload.
    fn select_backend(workload: Workload) -> Backend {
        if matches!(workload, Workload::Heavy) {
            for candidate in [Backend::Rocm, Backend::OpenCl, Backend::Sycl, Backend::Vulkan] {
                if Self::is_available(candidate) {
                    return candidate;
                }
            }
        }
        Backend::Cpu
    }

    /// Parse the `AUREX_BACKEND` environment variable into a [`Backend`]
    /// value.  Unknown strings are ignored and yield `None`.
    fn backend_from_env() -> Option<Backend> {
        std::env::var("AUREX_BACKEND")
            .ok()
            .and_then(|v| match v.to_lowercase().as_str() {
                "cpu" => Some(Backend::Cpu),
                "rocm" => Some(Backend::Rocm),
                "sycl" => Some(Backend::Sycl),
                "opencl" => Some(Backend::OpenCl),
                "vulkan" => Some(Backend::Vulkan),
                _ => None,
            })
    }

    fn backend_ops(backend: Backend) -> Box<dyn TensorOps + Send + Sync> {
        match backend {
            Backend::Cpu => Box::new(CpuBackend),
            Backend::Rocm => Box::new(RocmBackend),
            Backend::Sycl => Box::new(SyclBackend::new()),
            Backend::OpenCl => Box::new(OpenClBackend),
            Backend::Vulkan => Box::new(VulkanBackend::new()),
        }
    }
}

impl TensorOps for Dispatcher {
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        self.ops.matmul(a, b, m, n, k)
    }

    fn conv2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        input_shape: (usize, usize),
        kernel_shape: (usize, usize),
    ) -> Vec<f32> {
        self.ops.conv2d(input, kernel, input_shape, kernel_shape)
    }

    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
        self.ops.attention(q, k, v, dim)
    }

    fn layer_norm(&self, x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        self.ops.layer_norm(x, gamma, beta, eps)
    }
}
