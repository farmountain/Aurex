# Testing SOP

**Purpose:**
Outline standardized testing flow for Aurex-AMDUDA modules.

**Steps:**
1. Run `cargo test` for core functionality.
2. Run `cargo test --features agentic` to include AUREUS and HipCortex.
3. Run `cargo test --features full` to enable QSE.

**Progression:**
- Unit Tests → integration tests (SIT) → user acceptance tests (UAT).

