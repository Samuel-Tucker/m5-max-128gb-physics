"""Shared utilities, disjoint prompt corpora, SLC flusher, and Metal primitives for Track C."""

from __future__ import annotations

import gc
import math
import pathlib
import statistics
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

import mlx.core as mx
import numpy as np

MODEL_ID = "mlx-community/Qwen3.5-9B-4bit"
REPO_ROOT = pathlib.Path("<REDACTED_LOCAL_PATH>")

# ---------------------------------------------------------------------------
# Strictly disjoint Calibration vs Held-Out Test Prompt Corpora
# Calibration prompts are ONLY used for fitting PCA/subspace bases in Duel 1.
# Held-out prompts are NEVER seen during basis fitting.
# ---------------------------------------------------------------------------

CALIBRATION_PROMPTS: List[Tuple[str, str]] = [
    (
        "code_cal_1",
        "Write a Python implementation of Tarjan's strongly connected components algorithm "
        "on a directed graph represented as an adjacency list, including iterative DFS so it "
        "never hits recursion limits:\n```python\ndef tarjan_scc(graph):\n",
    ),
    (
        "code_cal_2",
        "Implement a lock-free single-producer single-consumer ring buffer in C++20 using "
        "std::atomic with acquire-release memory ordering and cache-line padding:\n```cpp\n",
    ),
    (
        "math_cal_1",
        "Prove using Sylvester's law of inertia and Rayleigh quotients that if A and B are "
        "n x n Hermitian matrices with B positive semidefinite, then the eigenvalues of A + B "
        "dominate those of A componentwise (Weyl's monotonicity theorem). Step 1:",
    ),
    (
        "math_cal_2",
        "Evaluate the contour integral of f(z) = exp(i*a*z) / (z^2 + b^2)^2 along the real "
        "line for positive real parameters a and b using the residue theorem and Jordan's lemma:\n",
    ),
    (
        "prose_cal_1",
        "Explain the thermodynamic and economic trade-offs between high-bypass turbofan "
        "engines and open-rotor propfans for short-haul commercial aviation, focusing on "
        "propulsive efficiency, cabin acoustics, and maintenance intervals:",
    ),
    (
        "prose_cal_2",
        "In 19th-century British industrial history, how did the transition from water-powered "
        "textile mills in Derbyshire to steam-powered cotton factories in Lancashire reshape "
        "urban transport networks, canal finance, and factory legislation?",
    ),
    (
        "code_cal_3",
        "Write a Metal compute shader that performs an in-place radix-2 Cooley-Tukey FFT "
        "over 1024 complex float2 elements in threadgroup memory with bank-conflict avoidance:\n```metal\n",
    ),
    (
        "prose_cal_3",
        "Compare the distributed motor control of the octopus arm nervous system, where "
        "roughly two-thirds of neurons reside in brachial ganglia, with vertebrate spinal "
        "central pattern generators. Focus on bandwidth constraints between central brain and periphery:",
    ),
]

HELD_OUT_PROMPTS: List[Tuple[str, str]] = [
    (
        "code_test_1",
        "Implement an exact suffix automaton in Python that supports online character extension, "
        "longest common substring queries, and counting distinct substrings in O(n) time:\n```python\n",
    ),
    (
        "code_test_2",
        "Write a Rust parser for a minimal JSON subset supporting nested objects, arrays, "
        "escaped UTF-8 strings, and IEEE-754 numbers without external crates:\n```rust\n",
    ),
    (
        "code_test_3",
        "Explain how Apple Silicon unified memory coherency works between GPU L1/L2 caches, "
        "the System Level Cache (SLC), and LPDDR5X channels when running bandwidth-bound "
        "4-bit quantized matrix-vector multiplication:\n",
    ),
    (
        "math_test_1",
        "State and prove the Parseval-Plancherel identity for the Walsh-Hadamard transform "
        "on Z_2^k, then explain why pre-multiplying a heavy-tailed vector by a randomized "
        "Walsh-Hadamard matrix reduces its infinity norm by O(sqrt(log d / d)) with high probability:",
    ),
    (
        "math_test_2",
        "Find all integer solutions (x, y) to the Pell-type Diophantine equation x^2 - 7*y^2 = 1 "
        "using the continued fraction expansion of sqrt(7), and prove the recurrence for the "
        "k-th fundamental solution:\n",
    ),
    (
        "math_test_3",
        "Given a real symmetric positive semidefinite matrix G = K^T K of size d x d, prove "
        "the rank-trace inequality rank(G) >= (tr G)^2 / tr(G^2) and characterize all matrices "
        "for which equality holds:",
    ),
    (
        "prose_test_1",
        "Draft a concise operational memo for a UK industrial PPE distributor evaluating "
        "whether to hold 90 days of powered air-purifying respirator filter inventory ahead "
        "of winter lead-time volatility, weighing working capital cost in GBP against stockout risk:",
    ),
    (
        "prose_test_2",
        "Summarize the archaeological evidence for Bronze Age tin trade routes connecting "
        "Cornwall and Central Asia to Eastern Mediterranean palaces around 1400 BCE, noting "
        "which isotope measurements remain contested among specialists:",
    ),
]


def load_long_context_text() -> str:
    """Assemble a deterministic, non-repetitive real corpus (>3500 tokens) from local repo files."""
    candidates = [
        REPO_ROOT / "docs/council/20261120-rh-m5max/synthesis.md",
        REPO_ROOT / "docs/contracts/reverse-prompt-contracts.md",
        REPO_ROOT / "docs/council/20261120-rh-m5max/chair.md",
        REPO_ROOT / "docs/council/20261120-rh-m5max/brief.md",
        REPO_ROOT / "NORTHSTAR.md",
    ]
    parts: List[str] = []
    for p in candidates:
        if p.exists():
            parts.append(p.read_text(encoding="utf-8", errors="ignore"))
    if not parts:
        parts = [txt for _, txt in HELD_OUT_PROMPTS + CALIBRATION_PROMPTS] * 8
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# SLC Flusher & Synchronized Interleaved A/B Benchmarker
# ---------------------------------------------------------------------------

class SLCFlusher:
    """Evicts Apple M5 Max SLC (~48-64 MB) by streaming >=540 MB through GPU."""

    def __init__(self, mb: int = 540):
        n_floats = (mb * 1024 * 1024) // 4
        # Use a deterministic pattern so Metal does not optimize away zero pages
        self.buf = mx.ones((n_floats,), dtype=mx.float32) * mx.array(1.0001, dtype=mx.float32)
        mx.eval(self.buf)
        mx.synchronize()
        self.mb = mb

    def flush(self) -> float:
        # Read entire >=540 MB buffer and synchronize to guarantee SLC eviction
        s = mx.sum(self.buf)
        mx.eval(s)
        mx.synchronize()
        return float(s.item())


def bench_interleaved(
    arms: Dict[str, Callable[[], Any]],
    flusher: SLCFlusher,
    trials: int = 9,
    warmup: int = 2,
) -> Dict[str, Dict[str, Any]]:
    """Run interleaved A/B trials with SLC flush + mx.synchronize() before every measurement."""
    names = list(arms.keys())
    # Warmup (compile kernels, populate instruction cache)
    for _ in range(warmup):
        for name in names:
            out = arms[name]()
            mx.eval(out)
            mx.synchronize()

    raw: Dict[str, List[float]] = {n: [] for n in names}
    for t in range(trials):
        order = names[t % len(names) :] + names[: t % len(names)]
        for name in order:
            flusher.flush()
            t0 = time.perf_counter()
            out = arms[name]()
            mx.eval(out)
            mx.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            raw[name].append(dt_ms)

    summary: Dict[str, Dict[str, Any]] = {}
    for name in names:
        vals = sorted(raw[name])
        p90_idx = min(len(vals) - 1, int(math.ceil(0.9 * len(vals))) - 1)
        summary[name] = {
            "median_ms": float(statistics.median(vals)),
            "mean_ms": float(statistics.mean(vals)),
            "min_ms": float(vals[0]),
            "max_ms": float(vals[-1]),
            "p90_ms": float(vals[p90_idx]),
            "std_ms": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
            "trials_ms": [round(v, 4) for v in raw[name]],
        }
    return summary


# ---------------------------------------------------------------------------
# Fast Metal 4096-point Normalized Walsh-Hadamard Transform (exact orthogonal)
# Supports arbitrary batch shape (..., 4096) or (..., K*4096) block-4096 FWHT.
# ---------------------------------------------------------------------------

_FWHT_4096_KERNEL = None


def get_fwht_4096_kernel():
    global _FWHT_4096_KERNEL
    if _FWHT_4096_KERNEL is None:
        source = r"""
            uint row = threadgroup_position_in_grid.x;
            uint tid = thread_position_in_threadgroup.x; // 0..511
            threadgroup float buf[4096];

            uint base = row * 4096u;
            // Each of 512 threads loads 8 contiguous elements
            for (uint i = 0u; i < 8u; ++i) {
                uint idx = tid + i * 512u;
                buf[idx] = float(x[base + idx]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 12 butterfly stages (2^12 = 4096). Each thread performs 4 butterflies/stage.
            for (uint stage = 0u; stage < 12u; ++stage) {
                uint half_step = 1u << stage;
                uint mask = half_step - 1u;
                for (uint b = 0u; b < 4u; ++b) {
                    uint pair = tid + b * 512u; // 0..2047
                    uint lo = pair & mask;
                    uint hi = pair >> stage;
                    uint idx0 = (hi << (stage + 1u)) + lo;
                    uint idx1 = idx0 + half_step;
                    float a = buf[idx0];
                    float c = buf[idx1];
                    buf[idx0] = a + c;
                    buf[idx1] = a - c;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Normalize by 1 / sqrt(4096) = 1 / 64.0f
            constexpr float inv_sqrt_n = 0.015625f;
            for (uint i = 0u; i < 8u; ++i) {
                uint idx = tid + i * 512u;
                out[base + idx] = T(buf[idx] * inv_sqrt_n);
            }
        """
        _FWHT_4096_KERNEL = mx.fast.metal_kernel(
            name="fwht_4096_ortho",
            input_names=["x"],
            output_names=["out"],
            source=source,
        )
    return _FWHT_4096_KERNEL


def fwht_block4096(x: mx.array, out_dtype: mx.Dtype | None = None) -> mx.array:
    """Apply exact normalized Walsh-Hadamard transform along last dim (multiple of 4096)."""
    orig_shape = x.shape
    d = orig_shape[-1]
    assert d % 4096 == 0, f"Last dim {d} must be a multiple of 4096"
    dtype = out_dtype if out_dtype is not None else x.dtype
    x_2d = x.reshape(-1, 4096)
    n_rows = x_2d.shape[0]
    k = get_fwht_4096_kernel()
    (y,) = k(
        inputs=[x_2d],
        template=[("T", dtype)],
        grid=(n_rows * 512, 1, 1),
        threadgroup=(512, 1, 1),
        output_shapes=[(n_rows, 4096)],
        output_dtypes=[dtype],
    )
    return y.reshape(orig_shape)


def clear_memory() -> None:
    gc.collect()
    mx.clear_cache()
