fn main() {
    aurex_cli::compile_model("model.onnx", "cpu");
    aurex_cli::run_model("model.onnx", "cpu");
}
