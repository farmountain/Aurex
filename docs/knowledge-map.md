# AUREX Contributor Knowledge Map

## Entry Points:
- `aurex-runtime/src/lib.rs`: ReflexionLoop, dispatch scheduling
- `aurex-agent/src/agent.rs`: AI agent loop and interface
- `aurex-kernel/src/tensor.rs`: all tensor primitives
- `aurex-backend/src/dispatch.rs`: backend routing logic

## System Flow:
Agent → Kernel → Dispatcher → Backend
       ↳ ReflexionLoop if failure or low confidence
       ↳ HypothesisManager to explore new branches

## Dev Tips:
- Run `cargo test -p aurex-kernel` before commits
- Use `RUST_LOG=debug` for tracing
- All examples in `/examples` must run with `--target=cpu` fallback
- Prefer `trait + impl` over monolithic classes
- Use `#[tokio::main]` and `#[async_trait]` for agents
