# Backend Setup Guide

This guide explains how to enable each AUREX backend.

## ROCm
1. Install **ROCm 6.0 or newer** following the instructions for your distribution at [rocm.docs.amd.com](https://rocm.docs.amd.com).
2. Ensure the HIP toolchain is available by adding `/opt/rocm/bin` to `PATH` and `/opt/rocm/lib` to `LD_LIBRARY_PATH`.
3. Validate the installation with `rocminfo` and `hipcc --version`.
4. Build AUREX with ROCm support:
   ```sh
   cargo test -p amduda --features rocm
   ```

## SYCL
1. Install the [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html) or another DPC++ compiler.
2. Source the environment setup script, e.g. `source /opt/intel/oneapi/setvars.sh`.
3. Verify the toolchain with `clang++ --version` and `sycl-ls`.
4. Enable the SYCL backend when building:
   ```sh
   cargo test -p amduda --features sycl
   ```

## Vulkan
1. Install the [Vulkan SDK](https://vulkan.lunarg.com/) (1.3 or newer) and set the `VULKAN_SDK` environment variable.
2. Ensure your GPU driver provides Vulkan compute support and validate with `vulkaninfo`.
3. Run the Vulkan demo to check your setup:
   ```sh
   cargo run --example vulkan_demo
   ```

## CPU
No additional dependencies are required. The CPU backend is enabled by default and acts as a fallback when no accelerator is present.
Test the CPU path with:
```sh
cargo test --all --exclude amduda
```
