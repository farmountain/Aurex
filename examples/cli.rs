//! Example demonstrating programmatic access to the CLI helpers.
//!
//! Run with: `cargo run -p aurex-cli --example cli`

fn main() {
    aurex_cli::compile_model("model.onnx", "cpu");
    aurex_cli::run_model("model.onnx", "cpu");
}
