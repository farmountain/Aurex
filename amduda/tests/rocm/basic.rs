use amduda::hal_backends::rocm_backend::RocmBackend;
use std::ffi::c_void;

#[test]
fn device_enumeration_reports_at_least_one_device() {
    let devices = RocmBackend::enumerate();
    assert!(!devices.is_empty());
}

#[test]
fn memory_roundtrip_via_backend() {
    let backend = RocmBackend::new();
    unsafe {
        let ptr = backend.alloc(4 * std::mem::size_of::<u32>());
        let host = [1u32, 2, 3, 4];
        backend.memcpy_htod(ptr, host.as_ptr() as *const c_void, 16);
        let mut out = [0u32; 4];
        backend.memcpy_dtoh(out.as_mut_ptr() as *mut c_void, ptr, 16);
        backend.free(ptr);
        assert_eq!(host, out);
    }
}

#[test]
fn kernel_launch_executes_closure() {
    let backend = RocmBackend::new();
    let mut executed = false;
    backend.launch(|| {
        executed = true;
    });
    assert!(executed);
}
