# Module Prompt: amduda_core/hal_backends.rs

**Task:**
Implement hardware abstraction layer backends for GPU and CPU execution.

**Requirements:**
1. Core:
   - Define a `Backend` trait with a `compute` method.
   - Provide `GpuBackend` and `CpuBackend` structs implementing the trait.
2. Optional:
   - `qse` feature adds a `quantum_hint` method on the trait.
3. Unit Test:
   - Instantiate both backends and ensure `compute` is called.

