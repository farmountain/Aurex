use async_trait::async_trait;
use aurex_runtime::Executor;

#[async_trait]
pub trait Agent: Send + Sync {
    async fn perceive(&self, input: &str) -> String;
    async fn reason(&self, state: &str) -> String;
    async fn act(&self, output: &str);
}

pub struct BasicAgent;

#[async_trait]
impl Agent for BasicAgent {
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

#[async_trait]
impl Executor for BasicAgent {
    async fn execute(&self) {
        let state = self.perceive("hello").await;
        let plan = self.reason(&state).await;
        self.act(&plan).await;
    }
}
