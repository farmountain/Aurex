//! AUREX runtime orchestrates agent execution and dispatches operations to the appropriate backend.
use async_trait::async_trait;

#[derive(Default)]
pub struct Runtime;

impl Runtime {
    pub async fn step(&self) {
        // placeholder step logic
        println!("runtime step");
    }
}

#[async_trait]
pub trait Executor {
    async fn execute(&self);
}
