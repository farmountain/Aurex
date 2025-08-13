use amduda::amduda_core::memory_tiering::MemoryTier;
use amduda::aurex_lm::model_loader::{load_model, Quantization, Weights};
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

#[test]
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
