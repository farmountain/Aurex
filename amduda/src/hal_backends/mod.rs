//! Hardware abstraction layer backends.

pub mod vulkan_backend;
pub mod rocm_backend;
pub mod opencl_backend;
pub mod cpu_simd;

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
        "vulkan" => BackendKind::Vulkan,
        _ => BackendKind::CpuSimd,
    }
}
