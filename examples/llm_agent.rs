//! Minimal example of an LLM-based agent using the AUREX runtime.
//! This example can be executed with `cargo run --example llm_agent --target=cpu`.

use async_trait::async_trait;
use aurex_agent::agent::Agent;
use aurex_runtime::confidence_regulator::ConfidenceRegulator;
use aurex_runtime::effort_evaluator::EffortEvaluator;
use aurex_runtime::hypothesis_manager::HypothesisManager;
use aurex_runtime::reflexion_loop::ReflexionLoop;
use aurex_runtime::{Runtime, RuntimeEvent};

use amduda::amduda_core::tensor_ops::TensorOps;
use amduda::aurex_lm::model_loader::{load_model, LoadedModel};
use amduda::hal_backends::cpu_simd::CpuSimdBackend;
use amduda::hal_backends::opencl_backend::{DeviceKind, OpenClBackend};
use amduda::hal_backends::rocm_backend::RocmBackend;
use amduda::hal_backends::sycl_backend::SyclBackend;
use amduda::hal_backends::vulkan_backend::VulkanBackend;
use amduda::hal_backends::{self, BackendKind};

pub struct LlmAgent {
    runtime: Runtime,
    backend: Box<dyn TensorOps + Send + Sync>,
    model: LoadedModel,
}

impl LlmAgent {
    pub fn new(
        runtime: Runtime,
        backend: Box<dyn TensorOps + Send + Sync>,
        model: LoadedModel,
    ) -> Self {
        Self {
            runtime,
            backend,
            model,
        }
    }

    pub async fn run(&self, prompt: &str) {
        println!("Running agent with prompt: {}", prompt);
        // Very small matmul as a stand‑in for real model inference.
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![0.5f32, 0.25, 0.125, 0.0625];
        let out = self.backend.matmul(&a, &b, 2, 2, 2);
        println!("matmul result: {:?}", out);

        // Drive the runtime with a single step using dummy components.
        struct Accept;
        #[async_trait]
        impl EffortEvaluator for Accept {
            async fn evaluate(&self, _event: &RuntimeEvent) -> bool {
                true
            }
        }

        struct Echo;
        #[async_trait]
        impl ConfidenceRegulator for Echo {
            async fn regulate(&self, ev: &RuntimeEvent) -> RuntimeEvent {
                ev.clone()
            }
        }

        struct Reflect;
        #[async_trait]
        impl ReflexionLoop for Reflect {
            async fn reflect(&self, ev: &RuntimeEvent) -> RuntimeEvent {
                ev.clone()
            }
        }

        struct Manage;
        #[async_trait]
        impl HypothesisManager for Manage {
            async fn manage(&self, ev: &RuntimeEvent) -> RuntimeEvent {
                ev.clone()
            }
        }

        let event = RuntimeEvent::TokenFetched { cache_hit: true };
        let _ = self
            .runtime
            .step(event, &Accept, &Echo, &Reflect, &Manage)
            .await;
    }
}

#[async_trait]
impl Agent for LlmAgent {
    async fn perceive(&self, input: &str) -> String {
        input.to_string()
    }

    async fn reason(&self, state: &str) -> String {
        state.to_string()
    }

    async fn act(&self, output: &str) {
        println!("Act: {}", output);
    }
}

#[tokio::main]
async fn main() {
    let backend_kind = hal_backends::select_backend();
    let backend: Box<dyn TensorOps + Send + Sync> = match backend_kind {
        BackendKind::Rocm => Box::new(RocmBackend::new()),
        BackendKind::Vulkan => match VulkanBackend::new() {
            Ok(b) => Box::new(b),
            Err(_) => Box::new(CpuSimdBackend),
        },
        BackendKind::OpenCl => Box::new(OpenClBackend::new(DeviceKind::Gpu)),
        BackendKind::Sycl => Box::new(SyclBackend::new()),
        BackendKind::CpuSimd => Box::new(CpuSimdBackend),
    };
    println!("Selected backend: {:?}", backend_kind);

    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/model_config.json".to_string());
    // Create a tiny dummy weight file when using the bundled config so the
    // example stays self-contained without shipping binary artifacts.
    if model_path == "examples/model_config.json" {
        let weight_path = "examples/model_weights.bin";
        if !std::path::Path::new(weight_path).exists() {
            let bytes: Vec<u8> = (0..16).collect();
            if let Err(e) = std::fs::write(weight_path, &bytes) {
                eprintln!("Failed to create dummy weights: {e}");
                return;
            }
        }
    }
    let model = match load_model(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            return;
        }
    };

    let runtime = Runtime::default();
    let agent = LlmAgent::new(runtime, backend, model);
    agent.run("Hello world").await;
}
