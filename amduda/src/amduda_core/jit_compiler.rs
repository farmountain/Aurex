//! JIT compilation support using LLVM.

use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    ptr,
    sync::Mutex,
};

use llvm_sys::{core::*, execution_engine::*, target::*};

use once_cell::sync::Lazy;

/// Target device for kernel execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    /// Host CPU backend.
    CPU,
    /// GPU backend (currently shares CPU implementation).
    GPU,
}

/// Handle to a compiled kernel.
#[derive(Clone, Copy)]
pub struct CompiledKernel {
    /// Function pointer to the JIT-compiled kernel.
    pub func: extern "C" fn(i32, i32) -> i32,
}

/// Global cache of compiled kernels keyed by source and device.
static KERNEL_CACHE: Lazy<Mutex<HashMap<String, CompiledKernel>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Handle to a compiled `f32` kernel.
#[derive(Clone, Copy)]
pub struct CompiledKernelF32 {
    /// Function pointer to the JIT-compiled kernel operating on `f32` values.
    pub func: extern "C" fn(f32, f32) -> f32,
}

/// Cache for `f32` kernels keyed by source and device.
static KERNEL_CACHE_F32: Lazy<Mutex<HashMap<String, CompiledKernelF32>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn c_string(s: &str) -> CString {
    CString::new(s).expect("CString conversion failed")
}

/// Compile a simple kernel described by `source` for the given `device`.
///
/// Supported `source` strings: "add" and "mul".
pub fn compile_kernel(source: &str, device: Device) -> Result<CompiledKernel, String> {
    let key = format!("{}::{:?}", source, device);
    if let Some(cached) = KERNEL_CACHE.lock().unwrap().get(&key).copied() {
        return Ok(cached);
    }

    unsafe {
        // Initialise LLVM for JIT usage.
        LLVMLinkInMCJIT();
        LLVM_InitializeNativeTarget();
        LLVM_InitializeNativeAsmPrinter();

        let context = LLVMContextCreate();
        let module_name = c_string("kernel_module");
        let module = LLVMModuleCreateWithName(module_name.as_ptr());
        let builder = LLVMCreateBuilderInContext(context);

        // i32 function: (i32, i32) -> i32
        let i32_type = LLVMInt32TypeInContext(context);
        let mut arg_types = [i32_type, i32_type];
        let fn_type = LLVMFunctionType(i32_type, arg_types.as_mut_ptr(), 2, 0);
        let fn_name = c_string("kernel");
        let function = LLVMAddFunction(module, fn_name.as_ptr(), fn_type);
        let entry_name = c_string("entry");
        let entry = LLVMAppendBasicBlockInContext(context, function, entry_name.as_ptr());
        LLVMPositionBuilderAtEnd(builder, entry);

        let a = LLVMGetParam(function, 0);
        let b = LLVMGetParam(function, 1);
        let tmp_name = c_string("tmp");
        let result = match source {
            "add" => LLVMBuildAdd(builder, a, b, tmp_name.as_ptr()),
            "mul" => LLVMBuildMul(builder, a, b, tmp_name.as_ptr()),
            _ => {
                LLVMDisposeBuilder(builder);
                LLVMContextDispose(context);
                return Err(format!("unsupported kernel: {}", source));
            }
        };

        LLVMBuildRet(builder, result);

        let mut engine: LLVMExecutionEngineRef = ptr::null_mut();
        let mut error = ptr::null_mut();
        if LLVMCreateExecutionEngineForModule(&mut engine, module, &mut error) != 0 {
            let msg = CStr::from_ptr(error).to_string_lossy().into_owned();
            LLVMDisposeMessage(error);
            return Err(msg);
        }

        let addr = LLVMGetFunctionAddress(engine, fn_name.as_ptr());
        let func = std::mem::transmute::<u64, extern "C" fn(i32, i32) -> i32>(addr);

        LLVMDisposeBuilder(builder);
        // Note: module and context intentionally leaked for simplicity.

        let compiled = CompiledKernel { func };
        KERNEL_CACHE.lock().unwrap().insert(key, compiled);
        Ok(compiled)
    }
}

/// Compile a simple floating point kernel described by `source` for the given `device`.
///
/// Supported `source` strings: "add_f32" and "mul_f32".
pub fn compile_kernel_f32(source: &str, device: Device) -> Result<CompiledKernelF32, String> {
    let key = format!("{}::{:?}", source, device);
    if let Some(cached) = KERNEL_CACHE_F32.lock().unwrap().get(&key).copied() {
        return Ok(cached);
    }

    unsafe {
        // Initialise LLVM for JIT usage.
        LLVMLinkInMCJIT();
        LLVM_InitializeNativeTarget();
        LLVM_InitializeNativeAsmPrinter();

        let context = LLVMContextCreate();
        let module_name = c_string("kernel_module_f32");
        let module = LLVMModuleCreateWithName(module_name.as_ptr());
        let builder = LLVMCreateBuilderInContext(context);

        // f32 function: (f32, f32) -> f32
        let f32_type = LLVMFloatTypeInContext(context);
        let mut arg_types = [f32_type, f32_type];
        let fn_type = LLVMFunctionType(f32_type, arg_types.as_mut_ptr(), 2, 0);
        let fn_name = c_string("kernel");
        let function = LLVMAddFunction(module, fn_name.as_ptr(), fn_type);
        let entry_name = c_string("entry");
        let entry = LLVMAppendBasicBlockInContext(context, function, entry_name.as_ptr());
        LLVMPositionBuilderAtEnd(builder, entry);

        let a = LLVMGetParam(function, 0);
        let b = LLVMGetParam(function, 1);
        let tmp_name = c_string("tmp");
        let result = match source {
            "add_f32" => LLVMBuildFAdd(builder, a, b, tmp_name.as_ptr()),
            "mul_f32" => LLVMBuildFMul(builder, a, b, tmp_name.as_ptr()),
            _ => {
                LLVMDisposeBuilder(builder);
                LLVMContextDispose(context);
                return Err(format!("unsupported kernel: {}", source));
            }
        };

        LLVMBuildRet(builder, result);

        let mut engine: LLVMExecutionEngineRef = ptr::null_mut();
        let mut error = ptr::null_mut();
        if LLVMCreateExecutionEngineForModule(&mut engine, module, &mut error) != 0 {
            let msg = CStr::from_ptr(error).to_string_lossy().into_owned();
            LLVMDisposeMessage(error);
            return Err(msg);
        }

        let addr = LLVMGetFunctionAddress(engine, fn_name.as_ptr());
        let func = std::mem::transmute::<u64, extern "C" fn(f32, f32) -> f32>(addr);

        LLVMDisposeBuilder(builder);
        // Note: module and context intentionally leaked for simplicity.

        let compiled = CompiledKernelF32 { func };
        KERNEL_CACHE_F32.lock().unwrap().insert(key, compiled);
        Ok(compiled)
    }
}
