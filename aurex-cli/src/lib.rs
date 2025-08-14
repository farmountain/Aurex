//! Command handlers for the `aurex-cli` binary.

/// Compile a model for the given backend target.
pub fn compile_model(model: &str, target: &str) {
    println!("Compiling {model} for {target} backend");
    // TODO: integrate with actual compilation pipeline
}

/// Run inference for a compiled model on the selected backend.
pub fn run_model(model: &str, target: &str) {
    println!("Running {model} on {target} backend");
    // TODO: integrate with runtime execution
}
