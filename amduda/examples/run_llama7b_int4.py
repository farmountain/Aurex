"""Example script to run a 7B Int4 quantized model.

The real project loads quantized weights and executes them on a GPU.  This
demo keeps things lightweight by generating a tiny random matrix and performing
a vector multiplication.  If ``torch`` with CUDA support is available the
computation runs on the GPU; otherwise a pure Python implementation is used as
the CPU fallback.
"""

import os
import random


try:  # pragma: no cover - optional dependency
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:  # torch not installed
    torch = None
    DEVICE = "cpu"


def dequantize_int4(values, scale):
    """Dequantize a list of 4‑bit integers using ``scale``."""

    return [v * scale for v in values]


def matmul(a, b, m, n, k):
    """Naive matrix multiplication used for the CPU path."""

    out = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += a[i * k + p] * b[p * n + j]
            out[i * n + j] = s
    return out


def load_dummy_model():
    """Generate a small random INT4 weight matrix."""

    dim = 4
    q_weights = [random.randint(-8, 7) for _ in range(dim * dim)]
    scale = 0.02
    weights = dequantize_int4(q_weights, scale)
    return weights, dim


def run_inference():
    print(f"Running LLAMA 7B Int4 example on {DEVICE.upper()} backend")
    weights, dim = load_dummy_model()
    vec = [1.0] * dim
    if torch is not None and DEVICE == "cuda":
        w = torch.tensor(weights, device=DEVICE).view(dim, dim)
        x = torch.tensor(vec, device=DEVICE).view(1, dim)
        y = torch.matmul(x, w)
        print("Output:", y.cpu().view(-1).tolist())
    else:
        out = matmul(vec, weights, 1, dim, dim)
        print("Output:", out)


if __name__ == "__main__":  # pragma: no cover - demo script
    run_inference()
