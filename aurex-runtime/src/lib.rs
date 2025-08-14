//! AUREX runtime orchestrates agent execution and dispatches operations to the appropriate backend.
use async_trait::async_trait;

pub mod plugin;
pub use plugin::{BackendPlugin, PluginRegistry};

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
    /// Perform a single runtime step, invoking the evaluation, regulation,
    /// reflexion, and hypothesis components in sequence. If the effort
    /// evaluator rejects the step, an [`RuntimeEvent::Error`] is returned.
    pub async fn step<Ev, Cf, Rx, Hy>(
        &self,
        event: RuntimeEvent,
        evaluator: &Ev,
        regulator: &Cf,
        reflexion: &Rx,
        hypothesis: &Hy,
    ) -> RuntimeEvent
    where
        Ev: EffortEvaluator + Send + Sync,
        Cf: ConfidenceRegulator + Send + Sync,
        Rx: ReflexionLoop + Send + Sync,
        Hy: HypothesisManager + Send + Sync,
    {
        println!("runtime step: {:?}", event);
        if !evaluator.evaluate(&event).await {
            return RuntimeEvent::Error;
        }

        let event = regulator.regulate(&event).await;
        let event = reflexion.reflect(&event).await;
        hypothesis.manage(&event).await
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

pub mod effort_evaluator {
    use super::RuntimeEvent;
    use async_trait::async_trait;

    #[async_trait]
    pub trait EffortEvaluator {
        async fn evaluate(&self, event: &RuntimeEvent) -> bool;
    }
}

pub mod confidence_regulator {
    use super::RuntimeEvent;
    use async_trait::async_trait;

    #[async_trait]
    pub trait ConfidenceRegulator {
        async fn regulate(&self, event: &RuntimeEvent) -> RuntimeEvent;
    }
}

pub mod reflexion_loop {
    use super::RuntimeEvent;
    use async_trait::async_trait;

    #[async_trait]
    pub trait ReflexionLoop {
        async fn reflect(&self, event: &RuntimeEvent) -> RuntimeEvent;
    }
}

pub mod hypothesis_manager {
    use super::RuntimeEvent;
    use async_trait::async_trait;

    #[async_trait]
    pub trait HypothesisManager {
        async fn manage(&self, event: &RuntimeEvent) -> RuntimeEvent;
    }
}

pub use confidence_regulator::ConfidenceRegulator;
pub use effort_evaluator::EffortEvaluator;
pub use hypothesis_manager::HypothesisManager;
pub use reflexion_loop::ReflexionLoop;

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct AcceptEvaluator;
    #[async_trait]
    impl EffortEvaluator for AcceptEvaluator {
        async fn evaluate(&self, _event: &RuntimeEvent) -> bool {
            true
        }
    }

    struct RejectEvaluator;
    #[async_trait]
    impl EffortEvaluator for RejectEvaluator {
        async fn evaluate(&self, _event: &RuntimeEvent) -> bool {
            false
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

    #[tokio::test]
    async fn runtime_step_success() {
        let runtime = Runtime::default();
        let event = RuntimeEvent::TokenFetched { cache_hit: true };
        let evaluator = AcceptEvaluator;
        let regulator = EchoRegulator;
        let reflexion = Reflector;
        let manager = Manager;
        let result = runtime
            .step(event.clone(), &evaluator, &regulator, &reflexion, &manager)
            .await;
        assert_eq!(result, event);
    }

    #[tokio::test]
    async fn runtime_step_evaluation_fails() {
        let runtime = Runtime::default();
        let event = RuntimeEvent::TokenFetched { cache_hit: true };
        let evaluator = RejectEvaluator;
        let regulator = EchoRegulator;
        let reflexion = Reflector;
        let manager = Manager;
        let result = runtime
            .step(event, &evaluator, &regulator, &reflexion, &manager)
            .await;
        assert_eq!(result, RuntimeEvent::Error);
    }
}
