# API Reference

AUREX is composed of several Rust crates. Generate local documentation with:

```sh
cargo doc --all --open
```

Key crates:
- `aurex-runtime`: host-side runtime and scheduler.
- `aurex-kernel`: tensor, symbolic, and procedural kernels.
- `aurex-agent`: agent abstractions and utilities.
- `aurex-cli`: command line interface for compiling and running models.
- `aurex-utils`: shared helpers and data structures.
- `amduda`: ROCm, SYCL, Vulkan, and CPU backend implementations.

Each crate contains module-level documentation describing its public API. Refer to the generated `target/doc` directory after running `cargo doc` for details.
