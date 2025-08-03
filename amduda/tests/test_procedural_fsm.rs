use amduda::amduda_core::procedural_fsm::{ProceduralFsm, State};
use aurex_runtime::{Runtime, RuntimeEvent};

// Helper to run async code in tests without requiring the tokio macros.
fn run_async<F: std::future::Future<Output = ()>>(fut: F) {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(fut);
}

#[test]
fn fsm_branches_on_cache_hit() {
    run_async(async {
        let runtime = Runtime::default();
        let mut fsm = ProceduralFsm::new();

        // Cache hit should skip KV cache update and go directly to attention.
        let state = fsm
            .step_with_runtime(&runtime, RuntimeEvent::TokenFetched { cache_hit: true })
            .await;
        assert_eq!(state, State::ComputeAttention);

        // Continue normal flow through attention and output back to fetch.
        let state = fsm
            .step_with_runtime(&runtime, RuntimeEvent::AttentionComputed)
            .await;
        assert_eq!(state, State::OutputToken);
        let state = fsm
            .step_with_runtime(&runtime, RuntimeEvent::TokenEmitted)
            .await;
        assert_eq!(state, State::FetchToken);

        // Cache miss requires KV cache update before attention.
        let state = fsm
            .step_with_runtime(&runtime, RuntimeEvent::TokenFetched { cache_hit: false })
            .await;
        assert_eq!(state, State::KVCacheUpdate);
        let state = fsm
            .step_with_runtime(&runtime, RuntimeEvent::CacheUpdated)
            .await;
        assert_eq!(state, State::ComputeAttention);
    });
}

#[test]
fn fsm_handles_errors() {
    run_async(async {
        let runtime = Runtime::default();
        let mut fsm = ProceduralFsm::new();

        // Any error event should transition to the error state regardless of current state.
        let state = fsm
            .step_with_runtime(&runtime, RuntimeEvent::Error)
            .await;
        assert_eq!(state, State::Error);

        // After entering Error state, further events should not change the state.
        let state = fsm
            .step_with_runtime(&runtime, RuntimeEvent::TokenEmitted)
            .await;
        assert_eq!(state, State::Error);
    });
}

