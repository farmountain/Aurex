# Aurex Unified Device Architecture

Aurex-AMDUDA aims to provide an AI-native compute runtime that unifies tensor operations, procedural scheduling, and memory management across heterogeneous hardware.

## Core Functionalities
1. **Unified TensorOps API**
   - `tensor_mul()`, `tensor_add()`, `kv_cache_allocate()`
   - Works across Vulkan, ROCm, OpenCL, and CPU SIMD backends.
2. **Procedural Kernel Scheduler (FSM + JIT)**
   - Converts compute requests into procedural kernel executions.
   - Supports Int4/Int8/FP16 with optional 1-bit regenerative paths.
3. **Multi-Tier Memory Manager**
   - GPU Shared → CPU DRAM → NVMe paging.
4. **Aurex-LM Core**
   - LLM inference for 7B–70B models with quantization.
   - Operates standalone without AUREUS or HipCortex dependencies.
5. **Optional Integrations**
   - HipCortex symbolic memory.
   - AUREUS agent runtime.
   - Quantum Seed Engine for regenerative inference.
