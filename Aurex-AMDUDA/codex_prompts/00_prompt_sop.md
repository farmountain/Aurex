# Prompt Folding SOP for Aurex-AMDUDA

**Purpose:**  
Ensure all Codex prompts generate **modular, testable Rust code** with optional features.

**Template Prompt:**
```
[Role] You are an AI software engineer.
[Context] We are building Aurex-AMDUDA: a CUDA-alternative core runtime with optional features.
[Task] Generate <module_name> with:
   - Core functionality first (standalone)
   - Optional blocks for AUREUS, HipCortex, and QSE
[Guidelines]
   1. Core code compiles without features
   2. Optional code wrapped with feature flags
   3. Include inline CoT comments
   4. Include a matching unit test
```

**Prompt Folding Layers:**
1. Context – What module does  
2. Task – Functions or features required  
3. Reasoning – Step-by-step CoT planning  
4. Validation – Unit/SIT/UAT instructions  
5. Output – Code + comments + tests

