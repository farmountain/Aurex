//! Kernel fusion and dynamic graph stubs for tensor operations.

/// Fuses a matrix multiplication with a subsequent layer normalization.
pub fn fuse_matmul_layernorm(
    a: &[f32],
    b: &[f32],
    gamma: &[f32],
    beta: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    // Matmul
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }

    // LayerNorm on each row
    let eps = 1e-5;
    for i in 0..m {
        let row = &mut out[i * n..(i + 1) * n];
        let mean = row.iter().sum::<f32>() / n as f32;
        let var = row
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / n as f32;
        let denom = (var + eps).sqrt();
        for j in 0..n {
            row[j] = ((row[j] - mean) / denom) * gamma[j] + beta[j];
        }
    }
    out
}

/// Simple dynamic graph node used for optimization stubs.
#[derive(Clone)]
pub struct GraphNode {
    pub name: &'static str,
}

/// Runs a dummy optimization pass returning the provided graph unmodified.
pub fn optimize_graph(nodes: &[GraphNode]) -> Vec<GraphNode> {
    nodes.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuse_and_optimize() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0];
        let out = fuse_matmul_layernorm(&a, &b, &gamma, &beta, 2, 2, 2);
        assert_eq!(out.len(), 4);
        let graph = vec![GraphNode { name: "matmul" }, GraphNode { name: "ln" }];
        assert_eq!(optimize_graph(&graph).len(), 2);
    }
}
