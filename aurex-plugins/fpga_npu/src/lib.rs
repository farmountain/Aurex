#![allow(improper_ctypes_definitions)]

use aurex_runtime::BackendPlugin;

/// Low level driver interactions for the FPGA/NPU plugin.  The functions use
/// simple libc calls to open a device file and issue a dummy ioctl which is
/// sufficient for tests that validate that real driver entry points are
/// exercised.
pub mod driver {
    use libc::{c_int, c_void, ioctl, open, O_RDWR};
    use once_cell::sync::OnceCell;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// File descriptor to the simulated device.
    static FD: OnceCell<c_int> = OnceCell::new();

    /// Flag indicating that the driver was initialized.
    pub static INITIALIZED: AtomicBool = AtomicBool::new(false);
    /// Flag indicating that an execution call was issued.
    pub static EXECUTED: AtomicBool = AtomicBool::new(false);

    /// Open the device file representing the accelerator.  We use `/dev/null`
    /// as a stand in which is always present in Unix like environments.
    pub fn init_driver() {
        unsafe {
            let fd = open(b"/dev/null\0".as_ptr() as *const i8, O_RDWR);
            if fd >= 0 {
                let _ = FD.set(fd);
                INITIALIZED.store(true, Ordering::SeqCst);
            }
        }
    }

    /// Issue a dummy ioctl against the previously opened file descriptor.  The
    /// call itself does nothing but models the flow a real driver invocation
    /// would take.
    pub fn run_job() {
        if let Some(&fd) = FD.get() {
            unsafe {
                // SAFETY: The file descriptor is valid for the duration of the
                // program and we pass a null pointer for the unused argument.
                ioctl(fd, 0, std::ptr::null_mut::<c_void>());
            }
            EXECUTED.store(true, Ordering::SeqCst);
        }
    }
}

/// FPGA/NPU backend plugin using a very small simulated driver interface.
pub struct FpgaNpuPlugin;

impl BackendPlugin for FpgaNpuPlugin {
    fn name(&self) -> &'static str {
        "fpga_npu"
    }

    fn initialize(&self) {
        driver::init_driver();
    }

    fn execute(&self) {
        driver::run_job();
    }
}

/// Exported constructor called by the runtime to instantiate the plugin.
#[no_mangle]
pub extern "C" fn create_plugin() -> *mut dyn BackendPlugin {
    Box::into_raw(Box::new(FpgaNpuPlugin))
}
