#[cfg(feature = "jit")]
use amduda::amduda_core::jit_compiler::{compile_kernel, Device};

fn main() {
    #[cfg(feature = "jit")]
    {
        // Compile a simple "add" kernel and execute it.
        let kernel = compile_kernel("add", Device::CPU).expect("compile kernel");
        let result = (kernel.func)(2, 3);
        println!("2 + 3 = {}", result);
    }

    #[cfg(not(feature = "jit"))]
    {
        // Example available only when JIT feature is enabled.
        eprintln!("The `jit` feature is not enabled. Enable it to run this example.");
    }
}
