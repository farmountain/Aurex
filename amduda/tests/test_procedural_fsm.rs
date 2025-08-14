use amduda::amduda_core::procedural_fsm::{ProceduralFsm, State};
use aurex_runtime::{Runtime, RuntimeEvent};

// Helper to run async code in tests without requiring the tokio macros.
fn run_async<F: std::future::Future<Output = ()>>(fut: F) {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(fut);
}

#[test]
fn fsm_branches_and_rolls_back() {
    run_async(async {
        let runtime = Runtime::default();
        let mut fsm = ProceduralFsm::new();

        // Start with a cache hit which goes directly to attention.
        let mut ev = RuntimeEvent::TokenFetched { cache_hit: true };
        ev = fsm.step_with_runtime(&runtime, ev).await;
        assert_eq!(fsm.state(), State::ComputeAttention);

        // Continue through attention and output back to fetch.
        ev = fsm.step_with_runtime(&runtime, ev).await;
        assert_eq!(fsm.state(), State::OutputToken);
        ev = fsm.step_with_runtime(&runtime, ev).await;
        assert_eq!(fsm.state(), State::FetchToken);

        // Returned event now represents the next token fetch. Simulate cache miss.
        ev = fsm.step_with_runtime(&runtime, ev).await;
        assert_eq!(fsm.state(), State::KVCacheUpdate);
        let _ = fsm.step_with_runtime(&runtime, ev).await;
        assert_eq!(fsm.state(), State::ComputeAttention);

        // Roll back the computation and ensure we return to fetching.
        let _ = fsm
            .step_with_runtime(&runtime, RuntimeEvent::Rollback)
            .await;
        assert_eq!(fsm.state(), State::FetchToken);
    });
}

#[test]
fn fsm_handles_errors() {
    run_async(async {
        let runtime = Runtime::default();
        let mut fsm = ProceduralFsm::new();

        // Any error event should transition to the error state regardless of current state.
        let _ev = fsm.step_with_runtime(&runtime, RuntimeEvent::Error).await;
        assert_eq!(fsm.state(), State::Error);

        // After entering Error state, further events should not change the state.
        let _ev = fsm
            .step_with_runtime(&runtime, RuntimeEvent::TokenEmitted)
            .await;
        assert_eq!(fsm.state(), State::Error);
    });
}
