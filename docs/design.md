# AUREX Design Principles

## Language: Rust
Rust was chosen for:
- Compile-time memory safety
- Concurrency and async-first architecture
- Trait-based extensibility
- FFI interop with ROCm/SYCL/CUDA

## Core Modules:
- `aurex-runtime`: main agent execution engine
- `aurex-kernel`: tensor and symbolic kernels with quantization
- `aurex-agent`: FSM logic, agent traits, and execution loop
- `aurex-backend`: backend dispatch for ROCm, SYCL, CPU
- `aurex-utils`: profilers, test scaffolds, introspection

## JIT Compilation

`amduda_core` ships with a lightweight JIT compiler built on top of `llvm-sys`. Kernels are
parsed from simple textual descriptions (e.g. `"add"`, `"mul"`) into LLVM IR and compiled at
runtime. Compiled kernels are cached per device backend to avoid redundant work and enable
device-specific optimizations.

Example usage:

```rust
use amduda::amduda_core::jit_compiler::{compile_kernel, Device};
let kernel = compile_kernel("add", Device::CPU).unwrap();
assert_eq!((kernel.func)(1, 2), 3);
```

Floating point kernels can be generated in the same manner using
`compile_kernel_f32`. These kernels are cached per device just like their
integer counterparts and are used by TensorOps to JIT compile operations such
as the attention primitive at runtime:

```rust
use amduda::amduda_core::jit_compiler::{compile_kernel_f32, Device};
let mul = compile_kernel_f32("mul_f32", Device::CPU).unwrap();
let result = (mul.func)(2.0, 4.0); // 8.0
```

## Profiling and Instrumentation

`aurex-utils` exposes a lightweight profiler that captures per-operation timing, memory
consumption, and GPU counters. Instrumentation is done through macros that wrap
TensorOps and runtime code blocks:

```rust
use aurex_utils::profiler::Profiler;
use aurex_utils::{profile_runtime_step, profile_tensor_op};

let mut prof = Profiler::new();

profile_tensor_op!(&mut prof, "matmul", {
    // call into a TensorOps backend
});

profile_runtime_step!(&mut prof, "decode", {
    // execute a runtime step
});

for rec in prof.records() {
    println!("{:?}", rec);
}
```

## Coding Conventions:
- Use `async_trait` for extensible agent behavior
- Never use unsafe unless FFI boundary requires
- All core ops should be testable on CPU with `cargo test`
