//! Symbolic finite state machine utilities for agents.

use std::collections::HashMap;

/// Representation of a symbolic state. Each state has a name and an
/// associated set of key/value pairs representing symbolic knowledge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolicState {
    name: String,
    data: HashMap<String, String>,
}

impl SymbolicState {
    /// Create a new state with the provided name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), data: HashMap::new() }
    }

    /// Name of the state.
    pub fn name(&self) -> &str { &self.name }

    /// Insert or update a symbolic value within the state.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.data.insert(key.into(), value.into());
    }

    /// Retrieve a symbolic value if present.
    pub fn get(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }
}

/// A simple deterministic symbolic finite state machine. Transitions are
/// triggered by symbolic events represented as strings.
pub struct StateMachine {
    current: String,
    states: HashMap<String, SymbolicState>,
    transitions: HashMap<(String, String), String>,
}

impl StateMachine {
    /// Construct a state machine with an initial state.
    pub fn new(initial: SymbolicState) -> Self {
        let current = initial.name.clone();
        let mut states = HashMap::new();
        states.insert(initial.name.clone(), initial);
        Self { current, states, transitions: HashMap::new() }
    }

    /// Add a state to the machine.
    pub fn add_state(&mut self, state: SymbolicState) {
        self.states.insert(state.name.clone(), state);
    }

    /// Define a transition between two states based on an event symbol.
    pub fn add_transition(
        &mut self,
        from: impl Into<String>,
        symbol: impl Into<String>,
        to: impl Into<String>,
    ) {
        self.transitions.insert((from.into(), symbol.into()), to.into());
    }

    /// Advance the machine using the provided symbol. Returns the new state if
    /// the transition exists, otherwise returns `None` and the state remains
    /// unchanged.
    pub fn step(&mut self, symbol: impl AsRef<str>) -> Option<&SymbolicState> {
        let key = (self.current.clone(), symbol.as_ref().to_string());
        if let Some(next) = self.transitions.get(&key).cloned() {
            self.current = next;
            self.states.get(&self.current)
        } else {
            None
        }
    }

    /// Return a reference to the current state.
    pub fn current_state(&self) -> &SymbolicState {
        self.states
            .get(&self.current)
            .expect("current state must exist")
    }

    /// Return a mutable reference to the current state for manipulation.
    pub fn current_state_mut(&mut self) -> &mut SymbolicState {
        self.states
            .get_mut(&self.current)
            .expect("current state must exist")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transitions_and_state_manipulation() {
        let mut sm = StateMachine::new(SymbolicState::new("idle"));
        sm.add_state(SymbolicState::new("working"));
        sm.add_transition("idle", "start", "working");
        sm.add_transition("working", "stop", "idle");

        // initial state
        assert_eq!(sm.current_state().name(), "idle");

        // Start transition
        sm.step("start").expect("transition to working");
        assert_eq!(sm.current_state().name(), "working");

        // Manipulate symbolic data
        sm.current_state_mut().set("counter", "1");
        assert_eq!(sm.current_state().get("counter"), Some(&"1".to_string()));

        // Stop transition
        sm.step("stop").expect("transition back to idle");
        assert_eq!(sm.current_state().name(), "idle");
    }

    #[test]
    fn invalid_transition_returns_none() {
        let mut sm = StateMachine::new(SymbolicState::new("a"));
        sm.add_state(SymbolicState::new("b"));
        sm.add_transition("a", "go", "b");

        assert!(sm.step("missing").is_none());
        assert_eq!(sm.current_state().name(), "a");
    }
}

