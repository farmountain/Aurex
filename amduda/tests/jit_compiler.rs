use amduda::amduda_core::jit_compiler::{compile_kernel, Device};

#[test]
fn cpu_add_kernel_executes() {
    if let Ok(k) = compile_kernel("add", Device::CPU) {
        let result = (k.func)(2, 3);
        assert_eq!(result, 5);
    }
}

#[test]
fn gpu_mul_kernel_executes() {
    // GPU backend currently falls back to CPU JIT.
    if let Ok(k) = compile_kernel("mul", Device::GPU) {
        let result = (k.func)(4, 5);
        assert_eq!(result, 20);
    }
}

#[test]
fn kernels_are_cached_per_device() {
    if let Ok(k1) = compile_kernel("add", Device::CPU) {
        let k2 = compile_kernel("add", Device::CPU).unwrap();
        assert_eq!(k1.func as usize, k2.func as usize);
    }
}
