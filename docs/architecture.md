# AUREX Architecture

AUREX is structured to unify tensor execution, symbolic reasoning, and procedural scheduling under a hardware-agnostic agent-first runtime. It is based on the AUREUS framework.

## Key Components:
- **Runtime Engine**: Orchestrates execution with EffortEvaluator, ReflexionLoop, ConfidenceRegulator
- **Kernel Layer**: Tensor ops (GEMM, Conv, Attn), symbolic ops (FSM), procedural ops (goal/subgoal)
- **Backends**: ROCm, SYCL, Vulkan, CPU
- **Dispatch Layer**: Trait-based hardware routing with plugin support
- **Agent Execution Loop**: perception → reasoning → action → reflexion

## Data Flow:
Agent → Scheduler → Kernel → Dispatcher → Backend
↳ feedback to ReflexionLoop and HypothesisManager for correction and retry
