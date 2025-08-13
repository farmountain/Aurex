//! Core runtime components: tensor ops, procedural FSM, and memory tiering.

#[cfg(feature = "jit")]
pub mod jit_compiler;
pub mod memory_tiering;
pub mod procedural_fsm;
pub mod tensor_ops;
