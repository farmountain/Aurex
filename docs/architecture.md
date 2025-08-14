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
   - Detects GPU, CPU and NVMe capabilities at runtime.
   - Maintains a caching hierarchy of GPU → CPU → NVMe with automatic migration when tiers are full.
4. **Aurex-LM Core**
   - LLM inference for 7B–70B models with quantization.
   - Operates standalone without AUREUS or HipCortex dependencies.
5. **Optional Integrations**
   - HipCortex symbolic memory.
   - AUREUS agent runtime.
   - Quantum Seed Engine for regenerative inference.

## Memory Tiering Strategy

AMDUDA detects available accelerators through environment variables and
allocates memory on the fastest tier. GPU memory acts as a cache for hot data;
when it fills, blocks are migrated to CPU DRAM and, if necessary, spilled to
NVMe storage. Cold blocks are always evicted before hot blocks, and callers may
explicitly tag data as hot or cold to influence caching decisions. The
`MemoryManager` API also exposes manual migration routines to promote or
demote data between tiers.

### Configuration

The following environment variables control detection and sizing of memory
tiers:

| Variable | Description |
|----------|-------------|
| `AMDUDA_HAS_GPU` | Set to `1` if a GPU allocator is available. |
| `AMDUDA_HAS_NVME` | Set to `1` when NVMe storage can be used for spilling. |
| `AMDUDA_GPU_MEM` | Total GPU memory in bytes available for allocations. |
| `AMDUDA_CPU_MEM` | Total CPU memory in bytes managed by the tiering system. |
| `AMDUDA_NVME_MEM` | Amount of NVMe space in bytes usable as the slowest tier. |

Unset variables fall back to conservative defaults. These knobs allow tests and
deployments to emulate a wide range of hardware setups.
