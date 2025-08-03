# Reflexion & Reasoning Prompts

**Task:**  
Integrate Reflexion logging optionally.

**Core:**  
- FSM works without Reflexion

**Optional:**  
1. Log FSM transitions to file if AUREUS enabled  
2. Log KV cache states if HipCortex enabled  
3. Capture 1-bit regeneration events if QSE enabled

**Validation:**  
- Core: Unit test runs without any features  
- Optional: Reflexion log verified in feature-enabled SIT

