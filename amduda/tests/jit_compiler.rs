use amduda::amduda_core::jit_compiler::{compile_kernel, Device};

#[test]
fn cpu_add_kernel_executes() {
    let k = compile_kernel("add", Device::CPU).expect("compile");
    let result = (k.func)(2, 3);
    assert_eq!(result, 5);
}

#[test]
fn gpu_mul_kernel_executes() {
    // GPU backend currently falls back to CPU JIT.
    let k = compile_kernel("mul", Device::GPU).expect("compile");
    let result = (k.func)(4, 5);
    assert_eq!(result, 20);
}

#[test]
fn kernels_are_cached_per_device() {
    let k1 = compile_kernel("add", Device::CPU).unwrap();
    let k2 = compile_kernel("add", Device::CPU).unwrap();
    assert_eq!(k1.func as usize, k2.func as usize);
}

