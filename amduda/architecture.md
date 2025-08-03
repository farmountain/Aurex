# Architecture

## Core Functionalities
1. **Unified TensorOps API**
   - `tensor_mul()`, `tensor_add()`, `kv_cache_allocate()`
   - Cross-backend: Vulkan / ROCm / OpenCL / CPU SIMD

2. **Procedural Kernel Scheduler (FSM + JIT)**
   - Converts compute requests into procedural kernel executions
   - Supports Int4 / Int8 / FP16 with optional 1-bit regeneration

3. **Multi-Tier Memory Manager**
   - GPU Shared → CPU DRAM → NVMe paging
   - Core function operates standalone

4. **Aurex-LM Core**
   - LLM inference engine for 7B–70B models with quantization
   - Int4/Int8 support for EVO-X2 and AMD iGPUs

5. **Optional Integrations**
   - HipCortex symbolic memory hooks
   - AUREUS agent runtime
   - Quantum Seed Engine for regenerative inference
