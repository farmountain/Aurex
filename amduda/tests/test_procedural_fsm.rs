use amduda::amduda_core::procedural_fsm::{run_fsm, State};

#[test]
fn test_fsm_sequence() {
    let seq = run_fsm(1);
    assert_eq!(seq, vec![
        State::FetchToken,
        State::KVCacheUpdate,
        State::ComputeAttention,
        State::OutputToken,
    ]);
}
