//! Sliding‑window and block‑sparse paged attention leveraging the memory
//! tiering subsystem.
//!
//! The primary entry point is [`PagedAttention`], which owns a
//! [`MemoryManager`] and exposes two reference implementations:
//! `compute` implements a classic sliding window, while
//! `compute_block_sparse` operates on blocks of tokens.  Both methods
//! demonstrate how key and value pages may be allocated across memory tiers
//! and migrated between NVMe and CPU memory to efficiently handle large
//! contexts.

use crate::amduda_core::memory_tiering::{DeviceCapabilities, MemoryManager, MemoryTier};
use std::mem::size_of;

/// Result of a paged attention invocation.
#[derive(Debug)]
pub struct AttentionResult {
    pub output: Vec<f32>,
    pub k_tier: MemoryTier,
    pub v_tier: MemoryTier,
}

/// Paged attention engine owning a [`MemoryManager`].
#[derive(Debug)]
pub struct PagedAttention {
    mgr: MemoryManager,
}

impl PagedAttention {
    /// Create a new engine with default tier limits.
    pub fn new(caps: DeviceCapabilities) -> Self {
        Self {
            mgr: MemoryManager::new(caps),
        }
    }

    /// Create a new engine with explicit tier limits, useful for tests.
    pub fn new_with_limits(
        caps: DeviceCapabilities,
        gpu_limit: usize,
        cpu_limit: usize,
        nvme_limit: usize,
    ) -> Self {
        Self {
            mgr: MemoryManager::new_with_limits(caps, gpu_limit, cpu_limit, nvme_limit),
        }
    }

    /// Expose tier usage for tests.
    pub fn usage(&self) -> (usize, usize, usize) {
        self.mgr.usage()
    }

    /// Compute sliding‑window attention.
    ///
    /// * `q` – flattened query vectors of length `n_q * d`.
    /// * `k` – flattened key vectors of length `n_k * d`.
    /// * `v` – flattened value vectors of length `n_k * d`.
    /// * `d` – feature dimension.
    /// * `window` – number of previous tokens each query attends to.
    pub fn compute(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        d: usize,
        window: usize,
    ) -> AttentionResult {
        assert_eq!(k.len(), v.len());
        assert_eq!(q.len() % d, 0);
        assert_eq!(k.len() % d, 0);

        let n_q = q.len() / d;
        let n_k = k.len() / d;
        assert_eq!(n_q, n_k, "q, k and v must have the same number of tokens");

        // Allocate key/value buffers using tiering.
        let byte_len = k.len() * size_of::<f32>();
        let k_tier = self.mgr.allocate(byte_len);
        let v_tier = self.mgr.allocate(byte_len);

        // Pull the working set for the sliding window into CPU memory if it
        // resides on NVMe.  This models a simple cache for large contexts.
        if matches!(k_tier, MemoryTier::Nvme) {
            let window_bytes = window * d * size_of::<f32>();
            self.mgr
                .migrate(MemoryTier::Nvme, MemoryTier::Cpu, window_bytes);
        }
        if matches!(v_tier, MemoryTier::Nvme) {
            let window_bytes = window * d * size_of::<f32>();
            self.mgr
                .migrate(MemoryTier::Nvme, MemoryTier::Cpu, window_bytes);
        }

        let mut output = vec![0f32; n_q * d];
        for i in 0..n_q {
            let q_i = &q[i * d..(i + 1) * d];
            let start = if i >= window { i + 1 - window } else { 0 };

            // Compute unnormalised scores.
            let mut scores = Vec::new();
            for j in start..=i {
                let k_j = &k[j * d..(j + 1) * d];
                let dot: f32 = q_i.iter().zip(k_j.iter()).map(|(a, b)| a * b).sum();
                scores.push((j, dot));
            }

            // Softmax normalisation.
            let max_score = scores
                .iter()
                .map(|(_, s)| *s)
                .fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0f32;
            let mut weights = Vec::new();
            for &(_, s) in &scores {
                let w = (s - max_score).exp();
                denom += w;
                weights.push(w);
            }

            for ((j, _), w) in scores.iter().zip(weights.iter()) {
                let weight = *w / denom;
                let v_j = &v[*j * d..(*j + 1) * d];
                for (out, val) in output[i * d..(i + 1) * d].iter_mut().zip(v_j.iter()) {
                    *out += weight * val;
                }
            }
        }

        AttentionResult {
            output,
            k_tier,
            v_tier,
        }
    }

    /// Compute block‑sparse attention where each token only attends to a
    /// limited number of previous blocks.
    ///
    /// * `block_size` – number of tokens per block.
    /// * `block_window` – number of previous blocks (including the current
    ///   one) that are visible to each block.
    pub fn compute_block_sparse(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        d: usize,
        block_size: usize,
        block_window: usize,
    ) -> AttentionResult {
        assert_eq!(k.len(), v.len());
        assert_eq!(q.len() % d, 0);
        assert_eq!(k.len() % d, 0);

        let n = q.len() / d;
        assert_eq!(n, k.len() / d, "q, k and v must have the same number of tokens");
        assert!(block_size > 0);
        let n_blocks = (n + block_size - 1) / block_size;

        // Allocate key/value buffers using tiering.
        let byte_len = k.len() * size_of::<f32>();
        let k_tier = self.mgr.allocate(byte_len);
        let v_tier = self.mgr.allocate(byte_len);

        // Bring the initial window of blocks into CPU memory if backing store
        // is NVMe.  This models a simple cache of hot blocks.
        let block_bytes = block_size * d * size_of::<f32>();
        let init_blocks = block_window.min(n_blocks);
        let window_bytes = init_blocks * block_bytes;
        if matches!(k_tier, MemoryTier::Nvme) {
            self.mgr
                .migrate(MemoryTier::Nvme, MemoryTier::Cpu, window_bytes);
            self.mgr.mark_hot(MemoryTier::Cpu, window_bytes);
        }
        if matches!(v_tier, MemoryTier::Nvme) {
            self.mgr
                .migrate(MemoryTier::Nvme, MemoryTier::Cpu, window_bytes);
            self.mgr.mark_hot(MemoryTier::Cpu, window_bytes);
        }

        let mut output = vec![0f32; n * d];
        for b in 0..n_blocks {
            let q_start = b * block_size;
            let q_end = (q_start + block_size).min(n);
            let first_block = b.saturating_sub(block_window - 1);
            let ctx_start = first_block * block_size;

            for i in q_start..q_end {
                let q_i = &q[i * d..(i + 1) * d];
                let mut scores = Vec::new();
                for j in ctx_start..=i {
                    let k_j = &k[j * d..(j + 1) * d];
                    let dot: f32 = q_i.iter().zip(k_j.iter()).map(|(a, b)| a * b).sum();
                    scores.push((j, dot));
                }

                let max_score = scores
                    .iter()
                    .map(|(_, s)| *s)
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0f32;
                let mut weights = Vec::new();
                for &(_, s) in &scores {
                    let w = (s - max_score).exp();
                    denom += w;
                    weights.push(w);
                }

                for ((j, _), w) in scores.iter().zip(weights.iter()) {
                    let weight = *w / denom;
                    let v_j = &v[*j * d..(*j + 1) * d];
                    for (out, val) in output[i * d..(i + 1) * d].iter_mut().zip(v_j.iter()) {
                        *out += weight * val;
                    }
                }
            }

            // Update cache for next block.
            if b + 1 < n_blocks {
                if matches!(k_tier, MemoryTier::Nvme) {
                    if b + 1 >= block_window {
                        self.mgr.mark_cold(MemoryTier::Cpu, block_bytes);
                        self.mgr
                            .migrate(MemoryTier::Cpu, MemoryTier::Nvme, block_bytes);
                    }
                    self.mgr
                        .migrate(MemoryTier::Nvme, MemoryTier::Cpu, block_bytes);
                    self.mgr.mark_hot(MemoryTier::Cpu, block_bytes);
                }
                if matches!(v_tier, MemoryTier::Nvme) {
                    if b + 1 >= block_window {
                        self.mgr.mark_cold(MemoryTier::Cpu, block_bytes);
                        self.mgr
                            .migrate(MemoryTier::Cpu, MemoryTier::Nvme, block_bytes);
                    }
                    self.mgr
                        .migrate(MemoryTier::Nvme, MemoryTier::Cpu, block_bytes);
                    self.mgr.mark_hot(MemoryTier::Cpu, block_bytes);
                }
            }
        }

        AttentionResult {
            output,
            k_tier,
            v_tier,
        }
    }
}
