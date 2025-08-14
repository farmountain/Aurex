use amduda::hal_backends::{self, BackendKind};
use serial_test::serial;

// Helper to run selection with environment variable.
fn with_backend_var<F: FnOnce()>(val: Option<&str>, f: F) {
    use std::env;
    const KEY: &str = "AUREX_BACKEND";
    let original = env::var(KEY).ok();
    match val {
        Some(v) => env::set_var(KEY, v),
        None => env::remove_var(KEY),
    }
    f();
    match original {
        Some(v) => env::set_var(KEY, v),
        None => env::remove_var(KEY),
    }
}

#[test]
#[serial]
fn default_falls_back_to_cpu() {
    with_backend_var(None, || {
        assert_eq!(hal_backends::select_backend(), BackendKind::CpuSimd);
    });
}

#[test]
#[serial]
fn selects_rocm_when_requested() {
    with_backend_var(Some("rocm"), || {
        assert_eq!(hal_backends::select_backend(), BackendKind::Rocm);
    });
}

#[test]
#[serial]
fn selects_vulkan_when_requested() {
    with_backend_var(Some("vulkan"), || {
        if hal_backends::vulkan_backend::VulkanBackend::is_available() {
            assert_eq!(hal_backends::select_backend(), BackendKind::Vulkan);
        } else {
            assert_eq!(hal_backends::select_backend(), BackendKind::CpuSimd);
        }
    });
}

#[test]
#[serial]
fn selects_opencl_when_requested() {
    with_backend_var(Some("opencl"), || {
        assert_eq!(hal_backends::select_backend(), BackendKind::OpenCl);
    });
}

#[test]
#[serial]
fn selects_sycl_when_requested() {
    with_backend_var(Some("sycl"), || {
        assert_eq!(hal_backends::select_backend(), BackendKind::Sycl);
    });
}

#[test]
#[serial]
fn selects_riscv_when_requested() {
    with_backend_var(Some("riscv"), || {
        assert_eq!(hal_backends::select_backend(), BackendKind::Riscv);
    });
}
