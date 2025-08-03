# Contributing to AUREX

Thank you for your interest in contributing! The project is organized as a Cargo workspace. Each crate under `*/src` contains Rust modules for different runtime components. Before opening a pull request, please:

1. Run `cargo test -p aurex-kernel` to ensure tensor primitives compile.
2. Follow the coding style described in `docs/design.md` (use `async_trait`, avoid `unsafe`).
3. Keep examples runnable with `--target=cpu` when possible.

We welcome issues and pull requests for new backends, kernels, and agent features.
