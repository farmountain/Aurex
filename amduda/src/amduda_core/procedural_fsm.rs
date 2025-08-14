//! Runtime-driven finite state machine coordinating token processing.

use aurex_runtime::{Runtime, RuntimeEvent};

/// Possible states in the token processing pipeline.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum State {
    /// Fetch the next token from the model.
    FetchToken,
    /// Update the key/value cache if needed.
    KVCacheUpdate,
    /// Compute attention using the current token and cache.
    ComputeAttention,
    /// Emit the processed token as output.
    OutputToken,
    /// An unrecoverable error state.
    Error,
}

/// A simple procedural state machine driven by [`RuntimeEvent`]s.
pub struct ProceduralFsm {
    state: State,
}

impl ProceduralFsm {
    /// Create a new FSM starting in [`State::FetchToken`].
    pub fn new() -> Self {
        Self {
            state: State::FetchToken,
        }
    }

    /// Get the current state.
    pub fn state(&self) -> State {
        self.state
    }

    /// Advance the FSM based on a runtime event.
    pub fn on_event(&mut self, event: RuntimeEvent) -> State {
        self.state = match (self.state, event) {
            (State::FetchToken, RuntimeEvent::TokenFetched { cache_hit }) => {
                if cache_hit {
                    State::ComputeAttention
                } else {
                    State::KVCacheUpdate
                }
            }
            (State::KVCacheUpdate, RuntimeEvent::CacheUpdated) => State::ComputeAttention,
            (State::ComputeAttention, RuntimeEvent::AttentionComputed) => State::OutputToken,
            (State::OutputToken, RuntimeEvent::TokenEmitted) => State::FetchToken,
            (_, RuntimeEvent::Rollback) => State::FetchToken,
            (_, RuntimeEvent::Error) => State::Error,
            // Unexpected events leave the state unchanged.
            (s, _) => s,
        };
        self.state
    }

    /// Convenience helper that applies an event to the FSM and returns the next
    /// runtime event scheduled by [`Runtime::step`].
    pub async fn step_with_runtime(
        &mut self,
        runtime: &Runtime,
        event: RuntimeEvent,
    ) -> RuntimeEvent {
        use async_trait::async_trait;
        use aurex_runtime::{
            ConfidenceRegulator, EffortEvaluator, HypothesisManager, ReflexionLoop,
        };

        struct AcceptEvaluator;
        #[async_trait]
        impl EffortEvaluator for AcceptEvaluator {
            async fn evaluate(&self, _event: &RuntimeEvent) -> bool {
                true
            }
        }

        struct EchoRegulator;
        #[async_trait]
        impl ConfidenceRegulator for EchoRegulator {
            async fn regulate(&self, event: &RuntimeEvent) -> RuntimeEvent {
                event.clone()
            }
        }

        struct Reflector;
        #[async_trait]
        impl ReflexionLoop for Reflector {
            async fn reflect(&self, event: &RuntimeEvent) -> RuntimeEvent {
                event.clone()
            }
        }

        struct Manager;
        #[async_trait]
        impl HypothesisManager for Manager {
            async fn manage(&self, event: &RuntimeEvent) -> RuntimeEvent {
                event.clone()
            }
        }

        // Apply the current event to update state.
        self.on_event(event.clone());

        // Let the runtime schedule the next event.
        runtime
            .step(
                event,
                &AcceptEvaluator,
                &EchoRegulator,
                &Reflector,
                &Manager,
            )
            .await
    }
}
