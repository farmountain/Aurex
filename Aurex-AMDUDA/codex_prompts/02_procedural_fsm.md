# Module Prompt: amduda_core/procedural_fsm.rs

**Task:**  
Implement FSM for LLM token streaming.

**FSM States:**  
- FetchToken  
- KVCacheUpdate  
- ComputeAttention  
- OutputToken

**Core:**  
- Works standalone for Int4 token streaming

**Optional:**  
- HipCortex: KV trace logs  
- AUREUS: Reflexion logging  
- QSE: 1-bit regeneration fallback

**Unit Test:**  
- Simulate 10 tokens core-only  
- Optional: Verify log output if features enabled

