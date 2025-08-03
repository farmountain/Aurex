# Aurex Roadmap

## Phase 1: Core Runtime MVP
- TensorOps API
- Procedural FSM kernel for LLM token streaming
- Vulkan / ROCm backend
- Unit + SIT testing
- No mandatory integration with AUREUS/HipCortex

## Phase 2: Multi-Backend & LLM Ready
- OpenCL / CPU SIMD fallback
- Int4/Int8/FP16 LLM inference
- SIT + UAT for 7B → 30B models

## Phase 3: Optional Integration
- HipCortex symbolic KV caching
- AUREUS agent task graph support
- Procedural quantization & 1-bit regen kernels

## Phase 4: Extended Hardware
- FPGA backend + RISC-V edge testing
- Optional QSE integration for experimental regenerative inference
