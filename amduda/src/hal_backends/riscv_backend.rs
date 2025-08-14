use crate::amduda_core::tensor_ops::{CpuFallback, TensorOps};

/// Backend representing execution on a RISC-V target.  On non RISC-V
/// platforms the implementation simply forwards all operations to the
/// [`CpuFallback`] which allows higher level logic to exercise dispatch
/// paths without requiring a cross compiled target.
#[derive(Clone, Copy, Debug)]
pub struct RiscvBackend;

impl RiscvBackend {
    /// Check whether a RISC-V target is available.  For the purposes of the
    /// tests we assume availability even when running on other architectures so
    /// that backend selection can be validated.
    pub fn is_available() -> bool {
        // A real implementation would probe the hardware; the stub always
        // reports availability.
        true
    }

    /// Create a new backend instance.
    pub fn new() -> Self {
        RiscvBackend
    }
}

impl TensorOps for RiscvBackend {
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

/// Initialize the RISC-V backend.  In this stubbed implementation there is
/// no setup required.
pub fn init() {}
