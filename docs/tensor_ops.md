# TensorOps Trait and Backend Mapping

Aurex exposes a `TensorOps` trait that provides basic linear algebra primitives:
`matmul`, `conv2d`, `attention` and `layer_norm`. Each hardware backend
implements this trait allowing the same code to run across CPU or GPU devices.

## Selecting Backends

The hardware abstraction layer lives in `amduda::hal_backends`. Backends include:

- `CpuSimdBackend` – portable SIMD implementation on the host
- `RocmBackend` – AMD ROCm GPU backend
- `VulkanBackend` – Vulkan compute backend
- `OpenClBackend`, `SyclBackend`, `RiscvBackend` – additional targets used in
  tests

Applications can choose a backend directly or use `select_backend()` which
consults the `AUREX_BACKEND` environment variable. If the requested backend is
not available, the dispatcher falls back to the CPU implementation.

```rust
use amduda::amduda_core::tensor_ops::TensorOps;
use amduda::hal_backends::{self, cpu_simd::CpuSimdBackend};

fn main() {
    let backend = match hal_backends::select_backend() {
        hal_backends::BackendKind::Rocm => hal_backends::rocm_backend::RocmBackend::new(),
        hal_backends::BackendKind::Vulkan => hal_backends::vulkan_backend::init().unwrap().into(),
        _ => CpuSimdBackend,
    };

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let out = backend.matmul(&a, &b, 2, 2, 2);
    println!("{out:?}");
}
```

## Heterogeneous Execution

Different backends can be used for different operations, enabling heterogeneous
execution. See `examples/fused_tensor_ops.rs` for a complete example that routes
`matmul`, `conv2d`, `attention` and `layer_norm` to separate backends.

