# Module Prompt: amduda_core/aurex_lm.rs

**Task:**
Create a minimal language model interface.

**Requirements:**
1. Core:
   - Loader for model parameters.
   - `forward` function producing logits.
2. Optional:
   - `aureus` feature adds Reflexion hooks.
   - `hipcortex` logs key/value cache usage.
   - `qse` enables quantization fallback.
3. Unit Test:
   - Call `forward` with dummy tensors and check shape of output.

