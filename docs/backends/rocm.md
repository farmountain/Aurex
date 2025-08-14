# ROCm Backend

AUREX supports executing tensor kernels on AMD GPUs through the ROCm/HIP runtime. The container used for testing does not ship with ROCm, but the backend emulates the public API so code can still be exercised on machines without GPUs.

## Prerequisites
1. Install **ROCm 6.0 or newer** following the instructions for your distribution at [rocm.docs.amd.com](https://rocm.docs.amd.com).
2. Add `/opt/rocm/bin` to `PATH` and `/opt/rocm/lib` to `LD_LIBRARY_PATH` so that the HIP tools and libraries can be located.
3. Verify the installation with `rocminfo` and `hipcc --version`.

## Building
Enable the backend when running tests or examples by activating the `rocm` feature on the `amduda` crate:

```sh
cargo test -p amduda --features rocm
```

If ROCm is absent the build automatically falls back to a host implementation, allowing development and CI to run without a GPU.

## Using the Backend
Select the backend at runtime via the `AUREX_BACKEND` environment variable. When the ROCm backend is requested but no GPU is present the dispatcher gracefully falls back to the CPU implementation.

```sh
AUREX_BACKEND=rocm cargo run --example fused_tensor_ops
```

This will execute tensor operations on the GPU when available or on the CPU otherwise, ensuring cross-backend compatibility.
