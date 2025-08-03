# Aurex – AMDUDA Codex Prompt Library (Optional Integrations)

> Purpose:
Enable OpenAI Codex / GitHub Copilot to generate core runtime code first, with optional hooks for AUREUS, HipCortex, and QSE for agentic reasoning, symbolic memory, and 1 -bit regenerative inference.

---

## 1⃣ Prompt SOP (Prompt Folding)

**Template:**

```
[Role] You are an AI software engineer focused on high-performance AI runtimes.
[Context] We are building Aurex-AMDUDA: a vendor-agnostic, AI-native CUDA alternative.
[Task] Generate <module_name> with:
   - Core functionality first (TensorOps / FSM / HAL)
   - Optional integration hooks for AUREUS, HipCortex, and QSE
[Guidelines]
   1. Core code must work standalone
   2. Optional blocks should be wrapped with `#[cfg(feature = "aureus")]` or equivalent
   3. Inline CoT (Chain-of-Thought) comments for reasoning steps
   4. Include unit tests in /tests, independent of optional features
```

**Prompt Folding Layers for Codex:**

1. Context Layer → Core runtime first, optional AUREUS/HipCortex/QSE hooks
2. Task Layer → Specific module or functionality
3. Reasoning Layer (CoT) → Step-by-step logic for memory tiering, FSM transitions, or tensor ops
4. Validation Layer → Test instructions (unit → SIT → optional UAT)
5. Output Layer → Code + inline reasoning comments

---

## 2⃣ Module Development Prompts

### A. TensorOps Core (`amduda_core/tensor_ops.rs`)

You are developing the TensorOps API for Aurex-AMDUDA.

1. Implement standalone functions:
   - `tensor_mul(a, b)`
   - `tensor_add(a, b)`
   - `kv_cache_allocate(size, tier="gpu/cpu/nvme")`
2. Multi-tier memory (core requirement):
   - GPU → CPU → NVMe fallback
3. Optional HipCortex integration:
   - Only log KV allocations if `feature = "hipcortex"`
4. CoT steps:
   - Step 1: Allocate tensor
   - Step 2: Dispatch to available backend
   - Step 3: Fallback to CPU if no GPU
5. Unit Test:
   - Multiply 2x2 matrix
   - Allocate KV cache
   - Verify output works without optional features

### B. Procedural FSM Scheduler (`amduda_core/procedural_fsm.rs`)

Task: Implement token streaming FSM for LLM inference.

FSM States:
- FetchToken
- KVCacheUpdate
- ComputeAttention
- OutputToken

**Requirements:**
1. Core FSM logic must run without AUREUS
2. Optional:
   - HipCortex logging for KVCacheUpdate
   - QSE 1-bit regeneration fallback for ComputeAttention
3. CoT inline:
   - Document state transitions and fallback logic
4. Unit Test:
   - Simulate 10 tokens
   - Assert state transitions complete without any optional feature
   - If features enabled, verify Reflexion log is created

### C. HAL Backends (Vulkan / ROCm / OpenCL / CPU SIMD)

You are creating `hal_backends/vulkan_backend.rs` for Aurex-AMDUDA.

1. Functions:
   - `init_vulkan_device()`
   - `dispatch_tensor_op(op, tensor_handle)`
   - `sync_memory(tensor_handle)`
2. Standalone first:
   - Vulkan → CPU fallback
3. Optional HipCortex:
   - Log tensor dispatch to symbolic trace if feature enabled
4. Unit Test:
   - Allocate 2x2 tensor
   - Perform `tensor_add()`
   - Verify backend auto-selection

### D. Aurex-LM Core (`aurex_lm/model_loader.rs` + `paged_attention.rs`)

Task: Implement standalone LLM inference engine.

**Core:**
1. Load quantized LLaMA model (Int4/Int8)
2. Stream tokens using Procedural FSM
3. Multi-tier memory support

**Optional:**
1. HipCortex → KV graph logging
2. QSE → 1-bit procedural kernel execution path
3. AUREUS → Reflexion task graphs

**Unit Test:**
- Load mock 7B Int4 model
- Generate 5 tokens
- Verify output works standalone
- If features enabled, verify KV logging and Reflexion hooks

### E. Aurex-Graph Runtime (Optional Agentic Execution)

Task: Implement `aurex_graph/task_graph.rs`

**Core:**
1. Map tasks to FSM kernels without agent runtime dependency
2. Execute sequentially and output token stream

**Optional:**
1. AUREUS integration → Reflexion-aware multi-agent graph
2. HipCortex → Symbolic memory graph logging
3. QSE → Execute procedural 1-bit subgraphs

**Unit Test:**
- Core: Execute 2 simple tasks sequentially
- Optional: If features enabled, log multi-agent Reflexion path

---

## 3⃣ Testing SOP

### Unit Tests (Core-Only)

Task: Create `tests/test_tensor_ops.rs`

1. Multiply 2x2 matrices
2. Allocate KV cache in GPU tier (simulated)
3. Verify output without optional features
4. If optional features enabled, log symbolic events

### System Integration Test (SIT)

Task: Create SIT test for 7B Int4 LLM inference

1. Stream 50 tokens using FSM
2. Record token/sec and memory usage
3. Core must work without AUREUS/HipCortex
4. If features enabled:
   - Verify Reflexion logs and KV graph snapshots

### User Acceptance Test (UAT, Optional Features)

Task: Execute 2-agent Reflexion scenario (only if AUREUS feature enabled)

1. Execute task graph with 2 mock agents
2. Validate:
   - Output responses correct
   - Optional HipCortex symbolic memory updated
   - Optional Reflexion logs captured

---

## 4⃣ End-to-End Prompt Example

You are generating the complete `amduda_core/procedural_fsm.rs`.

1. FSM states:
   - FetchToken → KVCacheUpdate → ComputeAttention → OutputToken
2. Core logic:
   - Stream tokens with Int4 LLM
3. Optional logic (guarded by features):
   - HipCortex: KV cache trace
   - AUREUS: Reflexion log
   - QSE: 1-bit kernel path
4. CoT steps:
   - Document fallback flow and memory tiering
5. Output:
   - Rust module with inline reasoning comments
   - Unit test: simulate 10 tokens core-only
   - Feature-enabled test: verify logs and optional hooks

---

## 5⃣ Developer Workflow with Optional Integrations

1. Start with Core Module Prompt
   - Confirm standalone functionality works first

2. Add Optional Feature Flags
   - `--features aureus,hipcortex,qse`

3. Test Progression
   - Run Unit tests → Core-only
   - Run SIT → Core-only & feature-enabled
   - Run UAT → Optional AUREUS/HipCortex/QSE

4. Iterate with Reflexion Logs
   - Use feature-enabled mode for symbolic trace validation

---

This updated prompt library ensures:

✅ Core runtime is self-contained and always testable
✅ AUREUS/HipCortex/QSE integrations are feature-gated and optional
✅ Codex/Copilot can generate modular code and tests step-by-step
