//! Simple FSM placeholder for token streaming.

 #[derive(Debug, PartialEq)]
 pub enum State {
    FetchToken,
    KVCacheUpdate,
    ComputeAttention,
    OutputToken,
}

pub fn run_fsm(steps: usize) -> Vec<State> {
    let mut seq = Vec::new();
    for _ in 0..steps {
        seq.push(State::FetchToken);
        seq.push(State::KVCacheUpdate);
        seq.push(State::ComputeAttention);
        seq.push(State::OutputToken);
    }
    seq
}
