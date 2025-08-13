//! Hardware abstraction layer backends.

pub mod cpu_simd;
pub mod opencl_backend;
pub mod rocm_backend;
pub mod vulkan_backend;
pub mod sycl_backend;

/// Enumeration of the available backend types.  This is used by tests to ensure
/// that the correct backend is selected from environment configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Portable SIMD implementation on the host CPU.
    CpuSimd,
    /// AMD ROCm GPU backend.
    Rocm,
    /// Vulkan compute backend.
    Vulkan,
    /// OpenCL backend (CPU or GPU devices).
    OpenCl,
    /// oneAPI SYCL backend.
    Sycl,
}

/// Select a backend based on the `AUREX_BACKEND` environment variable.  If the
/// requested backend is unavailable we fall back to the CPU implementation.
pub fn select_backend() -> BackendKind {
    match std::env::var("AUREX_BACKEND")
        .unwrap_or_default()
        .to_lowercase()
        .as_str()
    {
        "rocm" if rocm_backend::RocmBackend::is_available() => BackendKind::Rocm,
        "vulkan" if vulkan_backend::VulkanBackend::is_available() => BackendKind::Vulkan,
        "opencl" if opencl_backend::OpenClBackend::is_available() => BackendKind::OpenCl,
        "sycl" if sycl_backend::SyclBackend::is_available() => BackendKind::Sycl,
        _ => BackendKind::CpuSimd,
    }
}
