use aurex_backend::{Backend, Dispatcher, Workload};
use serial_test::serial;

fn reset_env() {
    std::env::remove_var("AUREX_DISABLE_ROCM");
    std::env::remove_var("AUREX_DISABLE_OPENCL");
    std::env::remove_var("AUREX_DISABLE_SYCL");
}

#[test]
#[serial]
fn honors_user_preference() {
    reset_env();
    let d = Dispatcher::new(Some(Backend::Sycl), Workload::Light);
    assert_eq!(d.backend(), Backend::Sycl);
}

#[test]
#[serial]
fn falls_back_when_preferred_unavailable() {
    reset_env();
    std::env::set_var("AUREX_DISABLE_SYCL", "1");
    let d = Dispatcher::new(Some(Backend::Sycl), Workload::Light);
    assert_eq!(d.backend(), Backend::Cpu);
}

#[test]
#[serial]
fn heavy_workload_prefers_gpu() {
    reset_env();
    let d = Dispatcher::new(None, Workload::Heavy);
    assert_ne!(d.backend(), Backend::Cpu);
}

#[test]
#[serial]
fn heavy_workload_cpu_when_gpu_unavailable() {
    reset_env();
    std::env::set_var("AUREX_DISABLE_ROCM", "1");
    std::env::set_var("AUREX_DISABLE_OPENCL", "1");
    std::env::set_var("AUREX_DISABLE_SYCL", "1");
    let d = Dispatcher::new(None, Workload::Heavy);
    assert_eq!(d.backend(), Backend::Cpu);
}

#[test]
#[serial]
fn light_workload_uses_cpu() {
    reset_env();
    let d = Dispatcher::new(None, Workload::Light);
    assert_eq!(d.backend(), Backend::Cpu);
}
