# 🧠 AUREX: Agentic Unified Runtime for Execution

> An open-source, cross-platform AI acceleration framework that unifies **tensor**, **symbolic**, and **procedural** execution for **AI agents**, built to benchmark CUDA performance and democratize access to advanced compute across AMD, Intel, and heterogeneous systems.

---

## 🚀 Why AUREX?

Most frameworks—like CUDA—optimize solely for matrix math and NVIDIA GPUs. But the next generation of intelligence demands:

- Reasoning agents, not just neural nets
- Symbolic control flow alongside tensor math
- Reflexive adaptation, energy-efficiency, and hardware portability

**AUREX** is designed from the ground up to support **AI agents**, **LLMs**, **procedural planners**, and **symbolic memory systems**, while running on **AMD**, **Intel**, **CPU**, **GPU**, **NPU**, and even edge devices.

---

## 🧠 Agent-Native Execution Stack

[ AUREX Runtime (AUREUS Steering: EffortEvaluator, ReflexionLoop, ConfidenceRegulator) ]
              ↓             ↓             ↓               ↓
 [ Tensor Kernels ] [ Symbolic FSM Ops ] [ Procedural Ops ] [ Agent Scheduler ]
              ↓
   [ LLVM / MLIR IR Compiler with Passes ]
              ↓
   [ ROCm | SYCL | Vulkan | CPU Backend ]

---

## 🧩 Core Modules

### 🔢 Tensor Kernels
- MatMul, Attention, Convolution, LayerNorm
- Quantization (int8, bf16, fp16)
- Kernel fusion & dynamic graph optimization

### 🧠 Symbolic + Procedural Execution
- Finite State Machine (FSM) execution for rule-based agents
- Procedural reasoning & goal-subgoal graphs
- Temporal memory buffer + hypothesis branching

### ⚛️ Agent Runtime (AUREUS Core)
- `EffortEvaluator`: energy/task efficiency routing
- `ConfidenceRegulator`: dynamic precision scaling
- `ReflexionLoop`: runtime graph rewrites
- `HypothesisManager`: plan exploration & rollback

### 🔌 Plugin + Backend Abstraction
- Multi-device execution via:
  - AMD ROCm
  - Intel SYCL (OneAPI)
  - Vulkan compute
  - CPU (fallback)
- Plugin system for custom ops, NPU drivers, edge runtimes

### Aurex-AMDUDA Core Runtime

The `amduda` crate provides the core CUDA-alternative runtime targeting AMD/Intel GPUs and CPU SIMD. It includes a unified TensorOps API, procedural kernel scheduler, multi-tier memory manager, and standalone LLM inference engine with optional integrations.

---

## 🧪 Example: LLM Agent on AMD ROCm

```python
from aurex.runtime import Agent, load_model

class MyAgent(Agent):
    def perceive(self, text):
        return self.tokenizer.encode(text)
    
    def reason(self, tokens):
        return self.model.forward(tokens)  # backend auto-routed

    def act(self, output):
        return self.tokenizer.decode(output)

agent = MyAgent(model=load_model("phi-3", backend="rocm"))
agent.run("Plan a trip to Kyoto with hidden gems.")


---

🎯 Competitive Edge vs CUDA

Feature	NVIDIA CUDA	AUREX

Open Hardware	❌ NVIDIA-only	✅ AMD, Intel, CPU, NPU
Agent Loop	❌ No	✅ Yes (Reflexion, FSM, Reasoning)
Symbolic Support	❌	✅ FSM, Rule Engines
Procedural Ops	❌	✅ Temporal, Stack Ops
Runtime Steering	❌	✅ AUREUS EffortEvaluator
Mobile/Edge	⚠️ Limited	✅ Targeted
Licensing	Proprietary	MIT/BSD Open Source



---

📦 Installation (Coming Soon)

pip install aurex-runtime
aurex-cli compile my_model.py --target=rocm
aurex-cli run my_model.aurex --target=cpu

> Note: ROCm 6.0+ or SYCL-enabled devices required. Vulkan backend needs the Vulkan SDK 1.3+ and a driver with compute support. CPU fallback is always available.

### ROCm Setup

To build and run the ROCm backend you need a working installation of the AMD
ROCm stack.

1. Install **ROCm 6.0 or newer** following the instructions for your
   distribution on [rocm.docs.amd.com](https://rocm.docs.amd.com).
2. Ensure the HIP toolchain is available by adding `\/opt\/rocm/bin` to your
   `PATH` and `\/opt\/rocm/lib` to `LD_LIBRARY_PATH`.
3. Validate the installation with `rocminfo` and `hipcc --version`.
4. Enable the backend when building AUREX:

   ```sh
   cargo test -p amduda --features rocm
   ```

The build falls back to a CPU implementation when ROCm is not present so the
project can still be compiled and tested on systems without AMD GPUs.

If LLVM 15 is not available on your system, skip the `amduda` crate during
tests to avoid the `llvm-sys` build:

```sh
cargo test --all --exclude amduda
```

### Vulkan Setup

To enable the Vulkan compute backend you must have:

1. A Vulkan 1.3 capable GPU and driver.
2. The [Vulkan SDK](https://vulkan.lunarg.com/) installed with `VULKAN_SDK` set.
3. Validation with `vulkaninfo` from the SDK.
4. Run the demo with:

   ```sh
   cargo run --example vulkan_demo
   ```





---

🔬 Roadmap

Phase	Milestones

Phase 1	Tensor core kernels (MatMul, Conv, Attn), ROCm + SYCL backend, CLI
Phase 2	Agent runtime, symbolic kernel support, profiling tools
Phase 3	Edge deployment (WASM, ARM), plugin marketplace, full AI agent ecosystem



---

🧑‍💻 Contributing

We welcome contributors! Please see CONTRIBUTING.md to get started.

Join the mission to free AI from vendor lock-in and build truly intelligent, self-reasoning agents that run everywhere.


---

🧭 Philosophy

> "The future of AI is not just tensors — it’s decisions, reasoning, memory, and intent.
AUREX is built to serve the next wave of AI: agents that think, reflect, and act across any platform."



— AUREX Maintainers


---

📄 License

MIT License © 2025 AUREX Project Contributors

---
