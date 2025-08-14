use amduda::amduda_core::tensor_ops::TensorOps;
use amduda::hal_backends::{self, BackendKind};
use aurex_runtime::BackendPlugin;
use fpga_npu::{create_plugin, driver};
use std::sync::atomic::Ordering;

#[test]
fn edge_deployment_flow() {
    // Force selection of the RISC-V backend for the test.
    std::env::set_var("AUREX_BACKEND", "riscv");
    match hal_backends::select_backend() {
        BackendKind::Riscv => {
            let backend = amduda::hal_backends::riscv_backend::RiscvBackend::new();
            assert_eq!(backend.matmul(&[1.0], &[1.0], 1, 1, 1), vec![1.0]);
        }
        other => panic!("unexpected backend selected: {:?}", other),
    }

    let plugin: Box<dyn BackendPlugin> = unsafe { Box::from_raw(create_plugin()) };
    plugin.initialize();
    assert!(driver::INITIALIZED.load(Ordering::SeqCst));
    plugin.execute();
    assert!(driver::EXECUTED.load(Ordering::SeqCst));
}
