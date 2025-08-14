#![allow(improper_ctypes_definitions)]

use aurex_runtime::BackendPlugin;

/// Simple FPGA/NPU backend plugin example.
pub struct FpgaNpuPlugin;

impl BackendPlugin for FpgaNpuPlugin {
    fn name(&self) -> &'static str {
        "fpga_npu"
    }

    fn initialize(&self) {
        println!("FPGA/NPU plugin initialized");
    }

    fn execute(&self) {
        println!("FPGA/NPU plugin executed");
    }
}

/// Exported constructor called by the runtime to instantiate the plugin.
#[no_mangle]
pub extern "C" fn create_plugin() -> *mut dyn BackendPlugin {
    Box::into_raw(Box::new(FpgaNpuPlugin))
}
