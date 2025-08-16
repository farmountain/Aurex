use amduda::amduda_core::memory_tiering::MemoryTier;
use amduda::aurex_lm::model_loader::{load_model, Quantization, Weights};
use amduda::aurex_lm::quantizer::{quantize_int4, quantize_int8};
use serial_test::serial;
use serde_json::json;
use tempfile::tempdir;

fn write_dummy_model(size: usize, quant: &str) -> std::path::PathBuf {
    let dir = tempdir().unwrap();
    let weights_path = dir.path().join("weights.bin");
    std::fs::write(&weights_path, vec![1u8; size]).unwrap();
    let config_path = dir.path().join("config.json");
    let cfg = json!({
        "name": "dummy",
        "weight_path": weights_path,
        "quantization": quant,
    });
    std::fs::write(&config_path, serde_json::to_vec(&cfg).unwrap()).unwrap();
    // Keep directory alive by leaking it; tests are short lived.
    std::mem::forget(dir);
    config_path
}

fn write_quantized_model(data: &[f32], quant: Quantization) -> std::path::PathBuf {
    let dir = tempdir().unwrap();
    let weights_path = dir.path().join("weights.bin");
    let (bytes, scale) = match quant {
        Quantization::Int8 => {
            let (q, s) = quantize_int8(data);
            (q.into_iter().map(|v| v as u8).collect(), Some(s))
        }
        Quantization::Int4 => {
            let (q, s) = quantize_int4(data);
            (q, Some(s))
        }
        Quantization::Bf16 => unreachable!(),
    };
    std::fs::write(&weights_path, bytes).unwrap();
    let config_path = dir.path().join("config.json");
    let cfg = json!({
        "name": "dummy",
        "weight_path": weights_path,
        "quantization": quant,
        "scale": scale,
    });
    std::fs::write(&config_path, serde_json::to_vec(&cfg).unwrap()).unwrap();
    std::mem::forget(dir);
    config_path
}

#[test]
#[serial]
fn test_load_model_from_disk() {
    std::env::set_var("AMDUDA_HAS_GPU", "0");
    std::env::set_var("AMDUDA_HAS_NVME", "0");
    let config = write_dummy_model(4, "int8");
    let model = load_model(config.to_str().unwrap()).unwrap();
    assert_eq!(model.config.quantization, Some(Quantization::Int8));
    assert_eq!(model.tier, MemoryTier::Cpu);
    match model.weights {
        Weights::Memory(data) => assert_eq!(data, vec![1u8; 4]),
        _ => panic!("expected in-memory weights"),
    }
}

#[test]
#[serial]
fn test_load_model_bf16() {
    std::env::set_var("AMDUDA_HAS_GPU", "0");
    std::env::set_var("AMDUDA_HAS_NVME", "0");
    let config = write_dummy_model(4, "bf16");
    let model = load_model(config.to_str().unwrap()).unwrap();
    assert_eq!(model.config.quantization, Some(Quantization::Bf16));
    assert_eq!(model.tier, MemoryTier::Cpu);
}

#[test]
#[serial]
fn test_load_model_mmap() {
    std::env::set_var("AMDUDA_HAS_GPU", "0");
    std::env::set_var("AMDUDA_HAS_NVME", "1");
    let config = write_dummy_model(9000, "int4");
    let model = load_model(config.to_str().unwrap()).unwrap();
    assert_eq!(model.config.quantization, Some(Quantization::Int4));
    assert_eq!(model.tier, MemoryTier::Nvme);
    match model.weights {
        Weights::Mmap(mmap) => assert_eq!(mmap.len(), 9000),
        _ => panic!("expected mmap weights"),
    }
}

#[test]
#[serial]
fn test_load_large_model_gpu_nvme() {
    std::env::set_var("AMDUDA_HAS_GPU", "1");
    std::env::set_var("AMDUDA_HAS_NVME", "1");
    std::env::set_var("AMDUDA_GPU_MEM", "64");
    std::env::set_var("AMDUDA_CPU_MEM", "64");
    std::env::set_var("AMDUDA_NVME_MEM", "256");
    let config = write_dummy_model(200, "int8");
    let model = load_model(config.to_str().unwrap()).unwrap();
    assert_eq!(model.tier, MemoryTier::Nvme);
}

#[test]
#[serial]
fn test_dequantize_int8() {
    std::env::set_var("AMDUDA_HAS_GPU", "0");
    std::env::set_var("AMDUDA_HAS_NVME", "0");
    let data = [1.0_f32, -1.0, 0.5, -0.5];
    let config = write_quantized_model(&data, Quantization::Int8);
    let model = load_model(config.to_str().unwrap()).unwrap();
    let deq = model.dequantized_weights(data.len()).unwrap();
    for (orig, got) in data.iter().zip(deq.iter()) {
        assert!((orig - got).abs() < 0.05);
    }
}

#[test]
#[serial]
fn test_dequantize_int4() {
    std::env::set_var("AMDUDA_HAS_GPU", "0");
    std::env::set_var("AMDUDA_HAS_NVME", "0");
    let data = [0.0_f32, 1.0, -1.0, 0.5];
    let config = write_quantized_model(&data, Quantization::Int4);
    let model = load_model(config.to_str().unwrap()).unwrap();
    let deq = model.dequantized_weights(data.len()).unwrap();
    for (orig, got) in data.iter().zip(deq.iter()) {
        assert!((orig - got).abs() < 0.1);
    }
}
