# Vulkan Backend

AUREX provides experimental support for executing TensorOps kernels through the
Vulkan API. The backend relies on the [`ash`](https://crates.io/crates/ash)
crate for runtime loading and compiles GLSL compute shaders to SPIR‑V at
runtime.

## Prerequisites
1. Install the **Vulkan SDK** for your platform from
   [vulkan.lunarg.com](https://vulkan.lunarg.com/).
2. Ensure the Vulkan loader and GPU drivers are available on your system. On
   Linux this typically means the `vulkan-loader` and driver packages from your
   distribution.
3. Verify that `vulkaninfo` reports at least one physical device capable of
   compute operations.

## Building
No special Cargo features are required. Simply run the tests or examples that
use the backend:

```sh
cargo test -p amduda tests::vulkan
```

If the Vulkan runtime or a suitable GPU is not present the backend automatically
falls back to CPU execution so development and CI can proceed without graphics
hardware.

## Using the Backend
Select the Vulkan backend at runtime via the dispatcher or initialize it
directly. The `examples/vulkan_demo.rs` program shows a minimal end‑to‑end
setup:

```sh
cargo run --example vulkan_demo
```

The example multiplies two matrices by dispatching a SPIR‑V compute shader on
the first available Vulkan device and prints the result.
