use amduda::amduda_core::tensor_ops::TensorOps;
use amduda::hal_backends::{self, BackendKind};
use aurex_runtime::BackendPlugin;

fn main() {
    // Select backend based on environment variable, defaulting to CPU.
    let kind = hal_backends::select_backend();
    match kind {
        BackendKind::Riscv => {
            let backend = amduda::hal_backends::riscv_backend::RiscvBackend::new();
            let res = backend.matmul(&[1.0], &[1.0], 1, 1, 1);
            println!("RISC-V backend result: {:?}", res);
        }
        _ => {
            let backend = amduda::hal_backends::cpu_simd::CpuSimdBackend;
            let res = backend.matmul(&[1.0], &[1.0], 1, 1, 1);
            println!("CPU backend result: {:?}", res);
        }
    }

    // Demonstrate usage of the FPGA/NPU plugin in an edge scenario.
    let plugin: Box<dyn BackendPlugin> = unsafe { Box::from_raw(fpga_npu::create_plugin()) };
    plugin.initialize();
    plugin.execute();
}
