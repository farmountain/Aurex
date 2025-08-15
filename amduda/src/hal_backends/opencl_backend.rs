//! OpenCL backend providing a very small emulation of OpenCL style device
//! management.  The real project would use the `opencl3` crate (or similar) to
//! interface with the OpenCL runtime.  The containers used for the unit tests do
//! not ship with an OpenCL implementation, therefore this module provides a
//! lightweight stub that models platform and device enumeration, basic context
//! management and kernel compilation/launch paths.  Tensor operations simply
//! fall back to the [`CpuFallback`] implementation which allows higher level
//! code to exercise the dispatch and backend selection logic without requiring a
//! functional OpenCL stack.

use crate::amduda_core::tensor_ops::{CpuFallback, TensorOps};

/// Type of an emulated OpenCL device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    /// A device executing on the host CPU.
    Cpu,
    /// A device representing a discrete GPU.
    Gpu,
}

/// Representation of an OpenCL device.  Only an identifier and the device kind
/// are tracked which is sufficient for tests that validate dispatch paths.
#[derive(Clone, Copy, Debug)]
pub struct OpenClDevice {
    pub id: usize,
    pub kind: DeviceKind,
    pub name: &'static str,
}

/// Minimal context holding the selected device.
#[derive(Clone, Copy, Debug)]
pub struct OpenClContext {
    device: OpenClDevice,
}

/// Backend instance used by tests.  In a fully fledged implementation this
/// would manage command queues, compiled kernels and device memory.
#[derive(Clone, Copy, Debug)]
pub struct OpenClBackend {
    ctx: OpenClContext,
}

impl OpenClBackend {
    /// Enumerate available OpenCL devices.  When the real runtime is not
    /// available we emulate a single CPU and GPU device so that the rest of the
    /// stack can exercise device selection and dispatch logic.
    pub fn enumerate() -> Vec<OpenClDevice> {
        vec![
            OpenClDevice {
                id: 0,
                kind: DeviceKind::Cpu,
                name: "OpenCL CPU",
            },
            OpenClDevice {
                id: 1,
                kind: DeviceKind::Gpu,
                name: "OpenCL GPU",
            },
        ]
    }

    /// Convenience helper mirroring other backends.
    pub fn is_available() -> bool {
        !Self::enumerate().is_empty()
    }

    /// Create a new backend for the requested device kind.  When the requested
    /// device is not present we fall back to the CPU device.
    pub fn new(kind: DeviceKind) -> Self {
        let device = Self::enumerate()
            .into_iter()
            .find(|d| d.kind == kind)
            .unwrap_or(OpenClDevice {
                id: 0,
                kind: DeviceKind::Cpu,
                name: "OpenCL CPU",
            });
        OpenClBackend {
            ctx: OpenClContext { device },
        }
    }

    /// Simulate kernel compilation.  In the stub implementation this is a
    /// no-op, but it models the call that a real backend would have to perform.
    fn compile_kernel(&self, _src: &str, _name: &str) {
        // No-op for the emulated backend.
    }

    /// Launch a previously "compiled" kernel.  When OpenCL is unavailable we
    /// simply execute the supplied closure on the host and return its result.
    fn launch<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        // A real implementation would enqueue the kernel on a command queue and
        // wait for completion.  For the stub we directly execute the closure.
        f()
    }
}

impl TensorOps for OpenClBackend {
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let cpu = CpuFallback;
        self.compile_kernel("", "matmul");
        self.launch(|| cpu.matmul(a, b, m, n, k))
    }

    fn conv2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        input_shape: (usize, usize),
        kernel_shape: (usize, usize),
    ) -> Vec<f32> {
        let cpu = CpuFallback;
        self.compile_kernel("", "conv2d");
        self.launch(|| cpu.conv2d(input, kernel, input_shape, kernel_shape))
    }

    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
        let cpu = CpuFallback;
        self.compile_kernel("", "attention");
        self.launch(|| cpu.attention(q, k, v, dim))
    }

    fn layer_norm(&self, x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        let cpu = CpuFallback;
        self.compile_kernel("", "layer_norm");
        self.launch(|| cpu.layer_norm(x, gamma, beta, eps))
    }
}

/// Initialize the OpenCL backend by probing available devices.
pub fn init() {
    let _ = OpenClBackend::enumerate();
}
