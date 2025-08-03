# Module Prompt: amduda_core/tensor_ops.rs

**Task:**  
Implement TensorOps core API with multi-tier memory.

**Requirements:**
1. Core:
   - tensor_mul(a, b)
   - tensor_add(a, b)
   - kv_cache_allocate(size, tier="gpu/cpu/nvme")
2. Memory fallback: GPU → CPU → NVMe
3. Optional:
   - HipCortex: log KV allocations if enabled
4. CoT Steps:
   - Step 1: Allocate tensor
   - Step 2: Dispatch to backend
   - Step 3: Fallback to CPU if no GPU
5. Unit Test:
   - Multiply 2x2 matrices
   - Allocate KV cache
   - Verify result core-only

