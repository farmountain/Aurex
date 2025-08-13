//! Basic model loading utilities.
//!
//! The loader reads a JSON configuration describing the location of model
//! weights and optional quantization parameters.  Weights are placed on a
//! memory tier using the simulated [`MemoryManager`] and are either loaded
//! into memory or memory‑mapped when they overflow CPU memory limits.

use crate::amduda_core::memory_tiering::{self, MemoryTier};
use anyhow::Result;
use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};

/// Supported on-disk quantized weight formats.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    Int4,
    Int8,
}

/// Configuration for loading a model from disk.
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub weight_path: String,
    #[serde(default)]
    pub quantization: Option<Quantization>,
}

/// Concrete representation of loaded weights.
#[derive(Debug)]
pub enum Weights {
    /// Weights fully resident in CPU/GPU memory.
    Memory(Vec<u8>),
    /// Weights memory mapped from NVMe storage.
    Mmap(Mmap),
}

/// Fully loaded model including configuration and weights.
#[derive(Debug)]
pub struct LoadedModel {
    pub config: ModelConfig,
    pub weights: Weights,
    pub tier: MemoryTier,
}

/// Load a model configuration and its associated weights.
pub fn load_model(path: &str) -> Result<LoadedModel> {
    // Parse configuration.
    let cfg = fs::read_to_string(path)?;
    let config: ModelConfig = serde_json::from_str(&cfg)?;

    // Determine allocation tier based on weight size.
    let metadata = fs::metadata(&config.weight_path)?;
    let size = metadata.len() as usize;
    let tier = memory_tiering::allocate(size);

    // Load or map weights depending on tier.
    let weights = match tier {
        MemoryTier::Nvme => {
            let file = File::open(&config.weight_path)?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            Weights::Mmap(mmap)
        }
        _ => {
            let data = fs::read(&config.weight_path)?;
            Weights::Memory(data)
        }
    };

    Ok(LoadedModel {
        config,
        weights,
        tier,
    })
}
