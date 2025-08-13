//! AUREX runtime orchestrates agent execution and dispatches operations to the appropriate backend.
use async_trait::async_trait;

/// Events emitted by the runtime to drive higher level state machines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeEvent {
    /// A token was fetched from the model. `cache_hit` indicates whether the KV
    /// cache already contains the necessary values and the update phase can be
    /// skipped.
    TokenFetched { cache_hit: bool },
    /// The KV cache has been updated and is ready for attention computation.
    CacheUpdated,
    /// Attention computation has completed.
    AttentionComputed,
    /// A token has been emitted as output.
    TokenEmitted,
    /// An unrecoverable error occurred.
    Error,
}

/// Numeric precision used by the runtime for model execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    Bf16,
    Int8,
    Int4,
}

pub struct Runtime {
    precision: Precision,
}

impl Default for Runtime {
    fn default() -> Self {
        Self {
            precision: Precision::F32,
        }
    }
}

impl Runtime {
    /// Perform a single runtime step and return the resulting event. In this
    /// prototype implementation the event is simply echoed back, allowing tests
    /// and higher level logic to simulate runtime behaviour.
    pub async fn step(&self, event: RuntimeEvent) -> RuntimeEvent {
        println!("runtime step: {:?}", event);
        event
    }

    /// Update the runtime's numeric precision. This allows dynamic precision
    /// scaling based on model or system requirements.
    pub fn set_precision(&mut self, precision: Precision) {
        self.precision = precision;
    }

    /// Return the currently configured numeric precision.
    pub fn precision(&self) -> Precision {
        self.precision
    }
}

#[async_trait]
pub trait Executor {
    async fn execute(&self);
}
