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
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    Int4,
    Int8,
    Bf16,
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
    /// Optional scale factor used for INT4/INT8 quantization.
    pub scale: Option<f32>,
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
        scale: None,
    })
}

use super::quantizer::{dequantize_bf16, dequantize_int4, dequantize_int8, quantize_bf16, quantize_int4, quantize_int8};

impl LoadedModel {
    /// Change the precision of the provided `data` slice and update the stored
    /// weights and configuration accordingly. `data` should contain the original
    /// floating point weights.
    pub fn change_precision(&mut self, data: &[f32], target: Quantization) {
        let (bytes, scale) = match target {
            Quantization::Int8 => {
                let (q, s) = quantize_int8(data);
                (q.into_iter().map(|v| v as u8).collect(), Some(s))
            }
            Quantization::Int4 => {
                let (q, s) = quantize_int4(data);
                (q, Some(s))
            }
            Quantization::Bf16 => {
                let q = quantize_bf16(data);
                let bytes = q.iter().flat_map(|v| v.to_le_bytes()).collect();
                (bytes, None)
            }
        };
        self.weights = Weights::Memory(bytes);
        self.config.quantization = Some(target);
        self.scale = scale;
    }

    /// Dequantize the currently stored weights back into `f32` values using the
    /// recorded quantization metadata.
    pub fn dequantized_weights(&self, len: usize) -> Option<Vec<f32>> {
        match (&self.weights, self.config.quantization) {
            (Weights::Memory(bytes), Some(Quantization::Int8)) => {
                let data: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
                self.scale.map(|s| dequantize_int8(&data, s))
            }
            (Weights::Memory(bytes), Some(Quantization::Int4)) => {
                self.scale.map(|s| dequantize_int4(bytes, s, len))
            }
            (Weights::Memory(bytes), Some(Quantization::Bf16)) => {
                let mut u16s = Vec::with_capacity(bytes.len() / 2);
                for chunk in bytes.chunks(2) {
                    u16s.push(u16::from_le_bytes([chunk[0], chunk[1]]));
                }
                Some(dequantize_bf16(&u16s))
            }
            _ => None,
        }
    }
}
