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

## Coding Conventions:
- Use `async_trait` for extensible agent behavior
- Never use unsafe unless FFI boundary requires
- All core ops should be testable on CPU with `cargo test`
