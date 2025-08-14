//! ROCm backend exposing a small subset of the HIP runtime API.
//!
//! The real project will offload tensor operations to AMD GPUs via HIP.  The
//! container used for the unit tests does not ship with ROCm, therefore the
//! implementation below emulates the public API when the `rocm` feature is not
//! enabled.  This allows higher level code and tests to exercise the device
//! discovery, memory allocation and kernel launch pathways without requiring a
//! GPU.

use crate::amduda_core::tensor_ops::{CpuFallback, TensorOps};
use std::ffi::c_void;
use std::ptr;

#[cfg(feature = "rocm")]
use hip_runtime_sys as hip;

/// Representation of a ROCm device.  Only the device identifier is tracked for
/// now.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmDevice {
    pub id: i32,
}

/// Backend instance.  In a full implementation this would manage HIP streams,
/// modules and other runtime state.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmBackend {
    device: RocmDevice,
}

impl RocmBackend {
    /// Enumerate the ROCm devices visible to the process.  When the HIP
    /// runtime is not available we fall back to a single virtual device so
    /// higher level code can exercise dispatch paths.
    pub fn enumerate() -> Vec<RocmDevice> {
        #[cfg(feature = "rocm")]
        unsafe {
            let mut count: i32 = 0;
            if hip::hipGetDeviceCount(&mut count) == hip::hipError_t::hipSuccess as i32 {
                return (0..count).map(|id| RocmDevice { id }).collect();
            }
            Vec::new()
        }

        #[cfg(not(feature = "rocm"))]
        {
            vec![RocmDevice { id: 0 }]
        }
    }

    /// Create a new backend selecting the first available device.
    pub fn new() -> Self {
        #[cfg(feature = "rocm")]
        unsafe {
            let _ = hip::hipInit(0);
        }
        let device = Self::enumerate().into_iter().next().unwrap_or_default();
        RocmBackend { device }
    }

    /// Return the number of ROCm devices visible to the process.
    pub fn device_count() -> usize {
        Self::enumerate().len()
    }

    /// Convenience helper used by backend selection tests.
    pub fn is_available() -> bool {
        Self::device_count() > 0
    }

    /// Allocate memory on the device.  When HIP is unavailable a host allocation
    /// is leaked to simulate a device pointer.
    pub unsafe fn alloc(&self, bytes: usize) -> *mut c_void {
        #[cfg(feature = "rocm")]
        {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let _ = hip::hipMalloc(&mut ptr, bytes);
            ptr
        }
        #[cfg(not(feature = "rocm"))]
        {
            let mut v = vec![0u8; bytes];
            let ptr = v.as_mut_ptr() as *mut c_void;
            std::mem::forget(v);
            ptr
        }
    }

    /// Free previously allocated memory.
    pub unsafe fn free(&self, ptr: *mut c_void) {
        #[cfg(feature = "rocm")]
        {
            let _ = hip::hipFree(ptr);
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = Vec::from_raw_parts(ptr as *mut u8, 0, 0);
        }
    }

    /// Copy memory from host to device.
    pub unsafe fn memcpy_htod(&self, dst: *mut c_void, src: *const c_void, bytes: usize) {
        #[cfg(feature = "rocm")]
        {
            let _ = hip::hipMemcpy(dst, src, bytes, hip::hipMemcpyKind::hipMemcpyHostToDevice as u32);
        }
        #[cfg(not(feature = "rocm"))]
        {
            ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, bytes);
        }
    }

    /// Copy memory from device to host.
    pub unsafe fn memcpy_dtoh(&self, dst: *mut c_void, src: *const c_void, bytes: usize) {
        #[cfg(feature = "rocm")]
        {
            let _ = hip::hipMemcpy(dst, src, bytes, hip::hipMemcpyKind::hipMemcpyDeviceToHost as u32);
        }
        #[cfg(not(feature = "rocm"))]
        {
            ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, bytes);
        }
    }

    /// Launch a kernel.  For the emulated path we simply execute the provided
    /// closure on the host.  The ROCm enabled build would load and launch a HIP
    /// kernel using `hipModuleLaunchKernel` or similar APIs.
    pub fn launch<F>(&self, f: F)
    where
        F: FnOnce(),
    {
        #[cfg(feature = "rocm")]
        unsafe {
            // A real implementation would actually launch a compiled kernel.
            // We simply synchronise to validate the call chain.
            let _ = hip::hipDeviceSynchronize();
        }
        #[cfg(not(feature = "rocm"))]
        {
            f();
        }
    }
}

impl TensorOps for RocmBackend {
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let cpu = CpuFallback;
        let mut out = Vec::new();
        self.launch(|| {
            out = cpu.matmul(a, b, m, n, k);
        });
        out
    }

    fn conv2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        input_shape: (usize, usize),
        kernel_shape: (usize, usize),
    ) -> Vec<f32> {
        let cpu = CpuFallback;
        let mut out = Vec::new();
        self.launch(|| {
            out = cpu.conv2d(input, kernel, input_shape, kernel_shape);
        });
        out
    }

    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
        let cpu = CpuFallback;
        let mut out = Vec::new();
        self.launch(|| {
            out = cpu.attention(q, k, v, dim);
        });
        out
    }

    fn layer_norm(&self, x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        let cpu = CpuFallback;
        let mut out = Vec::new();
        self.launch(|| {
            out = cpu.layer_norm(x, gamma, beta, eps);
        });
        out
    }
}

/// Initialize the ROCm backend by probing devices.
pub fn init() {
    let _ = RocmBackend::new();
}
