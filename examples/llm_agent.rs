//! Minimal example of an LLM-based agent using the AUREX runtime.
//! This example can be executed with `cargo run --example llm_agent --target=cpu`.

use aurex_agent::agent::Agent;
use aurex_runtime::Runtime;

pub struct LlmAgent {
    runtime: Runtime,
}

impl LlmAgent {
    pub fn new(runtime: Runtime) -> Self {
        Self { runtime }
    }

    pub async fn run(&self, prompt: &str) {
        // placeholder: tokens would be generated and executed via runtime
        println!("Running agent with prompt: {}", prompt);
        self.runtime.step().await;
    }
}

#[tokio::main]
async fn main() {
    let runtime = Runtime::default();
    let agent = LlmAgent::new(runtime);
    agent.run("Hello world").await;
}
