use amduda::amduda_core::memory_tiering::{DeviceCapabilities, MemoryTier};
use amduda::aurex_lm::paged_attention::PagedAttention;
use serial_test::serial;

fn naive_sliding_window(q: &[f32], k: &[f32], v: &[f32], d: usize, window: usize) -> Vec<f32> {
    assert_eq!(k.len(), v.len());
    let n = q.len() / d;
    let mut out = vec![0f32; n * d];
    for i in 0..n {
        let q_i = &q[i * d..(i + 1) * d];
        let start = if i >= window { i + 1 - window } else { 0 };

        let mut scores = Vec::new();
        for j in start..=i {
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
            for (o, val) in out[i * d..(i + 1) * d].iter_mut().zip(v_j.iter()) {
                *o += weight * val;
            }
        }
    }
    out
}

fn naive_block_sparse(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d: usize,
    block_size: usize,
    block_window: usize,
) -> Vec<f32> {
    assert_eq!(k.len(), v.len());
    let n = q.len() / d;
    let mut out = vec![0f32; n * d];
    let n_blocks = (n + block_size - 1) / block_size;
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
                for (o, val) in out[i * d..(i + 1) * d].iter_mut().zip(v_j.iter()) {
                    *o += weight * val;
                }
            }
        }
    }
    out
}

#[test]
#[serial]
fn sliding_window_attention_correctness_and_tiering() {
    std::env::set_var("AMDUDA_HAS_GPU", "0");
    std::env::set_var("AMDUDA_HAS_NVME", "1");
    let caps = DeviceCapabilities::detect();
    let mut attn = PagedAttention::new_with_limits(caps, 0, 16, 1024);

    let d = 2;
    let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
    let k = q.clone();
    let v = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 1.0];
    let window = 2;

    let result = attn.compute(&q, &k, &v, d, window);
    let expected = naive_sliding_window(&q, &k, &v, d, window);

    assert!(result
        .output
        .iter()
        .zip(expected.iter())
        .all(|(a, b)| (a - b).abs() < 1e-5));
    assert_eq!(result.k_tier, MemoryTier::Nvme);
    assert_eq!(result.v_tier, MemoryTier::Nvme);

    let usage = attn.usage();
    assert_eq!(usage, (0, 16, 48));
}

#[test]
#[serial]
fn block_sparse_attention_correctness_and_tiering() {
    std::env::set_var("AMDUDA_HAS_GPU", "0");
    std::env::set_var("AMDUDA_HAS_NVME", "1");
    let caps = DeviceCapabilities::detect();
    let mut attn = PagedAttention::new_with_limits(caps, 0, 32, 1024);

    let d = 2;
    let q = vec![
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.2, 0.8, 0.8, 0.2, 0.3, 0.7, 0.7, 0.3,
    ];
    let k = q.clone();
    let v = vec![
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.2, 0.8, 0.8, 0.2, 0.3, 0.7, 0.7, 0.3,
    ];
    let block_size = 2;
    let block_window = 1;

    let result = attn.compute_block_sparse(&q, &k, &v, d, block_size, block_window);
    let expected = naive_block_sparse(&q, &k, &v, d, block_size, block_window);

    assert!(result
        .output
        .iter()
        .zip(expected.iter())
        .all(|(a, b)| (a - b).abs() < 1e-5));
    assert_eq!(result.k_tier, MemoryTier::Nvme);
    assert_eq!(result.v_tier, MemoryTier::Nvme);

    let usage = attn.usage();
    assert_eq!(usage, (0, 32, 96));
}
