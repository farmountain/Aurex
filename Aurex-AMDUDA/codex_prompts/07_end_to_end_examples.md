# End-to-End Examples Prompt

**Goal:**
Combine TensorOps and the procedural FSM to demonstrate a minimal inference pipeline.

**Steps:**
1. Load mock tensors for model weights.
2. Stream tokens through the FSM using `tensor_mul` and `tensor_add`.
3. Emit final output tokens.

**Validation:**
- Core example runs without any optional features.
- Feature-gated examples may log additional information.

