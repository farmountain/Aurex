use aurex_agent::agent::BasicAgent;
use aurex_runtime::Executor;

#[tokio::test]
async fn basic_agent_executes_loop() {
    let agent = BasicAgent;
    agent.execute().await;
}
