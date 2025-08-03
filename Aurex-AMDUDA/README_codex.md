# Aurex-AMDUDA Codex Prompt Library

This folder provides **OpenAI Codex prompts** to generate modular, testable code for **Aurex-AMDUDA**, a vendor-agnostic CUDA-alternative runtime.

---

## 🎯 Key Points

- **Core runtime first** (TensorOps, FSM, HAL)  
- **Optional features:** AUREUS, HipCortex, Quantum Seed Engine (QSE)  
- **No IDE dependency** — works with **Codex prompts** directly  
- **Unit → SIT → UAT** test progression for modular development  

---

## 🚀 Development Workflow

1. **Pick a prompt** from `codex_prompts/`  
2. **Use OpenAI Codex**: Paste prompt into Codex or Copilot Chat  
3. **Generate code** → Save in `src/<module>.rs`  
4. **Run tests** using Cargo (core-only first)  
5. **Optional:** Enable feature flags for AUREUS/HipCortex/QSE

---

## 🔧 Build & Test Commands

**Core Runtime Only (Standalone)**

```
cargo build
cargo test
```

**Enable AUREUS + HipCortex (Agentic)**

```
cargo build --features agentic
cargo test --features agentic
```

**Full Runtime with QSE**

```
cargo build --features full
cargo test --features full
```

---

## ✅ Test Progression

1. **Unit Tests** – Module correctness  
2. **SIT** – Integrated LLM inference  
3. **UAT** – Optional Reflexion/agentic scenarios

---

This library standardizes **prompt folding** for Codex-based AI‑assisted development.

