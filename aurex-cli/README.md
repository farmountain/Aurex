# aurex-cli

Command line interface for compiling models and running inference with AUREX.

## Usage

```bash
# Compile a model for the ROCm backend
cargo run -p aurex-cli -- compile path/to/model.onnx --target=rocm

# Run inference using the compiled model on the CPU backend
cargo run -p aurex-cli -- run path/to/model.onnx --target=cpu
```

The `--target` flag selects the backend (e.g., `cpu`, `rocm`, `vulkan`).
