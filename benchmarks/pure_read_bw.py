#!/usr/bin/env python3
"""
Track E.1 — Pure-Read DRAM Bandwidth Ceiling & Empirical SLC Capacity Sweep (Apple M5 Max)

Answers GLM 5.3 & Opus 5.5's #1 silicon physics question:
  What is the M5 Max's TRUE pure-read DRAM streaming ceiling (0% write traffic)
  vs R+W copy (~503.2 GB/s) vs 614.4 GB/s LPDDR5X spec, and what is the exact
  empirical System Level Cache (SLC) capacity?

Methodology:
  - Custom Metal reduction kernels via mx.fast.metal_kernel using coalesced
    4x-unrolled uint4 / float4 loads (64 bytes/iter per thread, 2048B/SIMD-group).
  - Tiny output buffer (one uint/float per threadgroup = 4 KB vs 4 GB read -> 0.0001% write).
  - Exact analytical checksum verification to prove zero dead-code elimination.
  - Warm-reuse sweep (4 MB .. 4096 MB) with cyclic address offset (prevents loop hoisting)
    to pinpoint the exact SLC capacity cliff.
  - Cold-SLC-flushed single-pass sweep (64 MB .. 4096 MB) with an independent 768 MB
    SLC flush buffer between every timed trial, plus interleaved A/B R+W copy control.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time

# Auto-reexec with mlx-lm venv python if current python lacks mlx
_MLX_PY = "python3"
try:
    import mlx.core as mx
except ImportError:
    if os.path.exists(_MLX_PY) and os.path.realpath(sys.executable) != os.path.realpath(_MLX_PY):
        os.execv(_MLX_PY, [_MLX_PY] + sys.argv)
    raise

# ---------------------------------------------------------------------------
# Custom Metal Kernels (pure-read uint4, pure-read float4, and matched R+W copy)
# ---------------------------------------------------------------------------

_PURE_READ_UINT4_SRC = r"""
uint tid = thread_position_in_threadgroup.x;
uint tg_id = threadgroup_position_in_grid.x;
uint tg_size = threads_per_threadgroup.x;
uint num_tg = threadgroups_per_grid.x;

uint64_t n_vec = uint64_t(params[0]);
uint reps = params[1];
uint64_t total_threads = uint64_t(num_tg) * uint64_t(tg_size);
uint64_t gid = uint64_t(tg_id) * uint64_t(tg_size) + uint64_t(tid);

const device uint4* src = reinterpret_cast<const device uint4*>(inp);

uint4 acc0 = uint4(0);
uint4 acc1 = uint4(0);
uint4 acc2 = uint4(0);
uint4 acc3 = uint4(0);

uint64_t stride4 = total_threads * 4ULL;

for (uint r = 0; r < reps; ++r) {
    // Non-zero cyclic shift per rep prevents any compiler load hoisting
    // while keeping 100% sequential SIMD-coalesced access across the exact buffer.
    uint64_t shift = (uint64_t(r) * 1024ULL) % n_vec;
    uint64_t i = gid;

    for (; i + 3ULL * total_threads < n_vec; i += stride4) {
        uint64_t idx0 = i + shift; if (idx0 >= n_vec) idx0 -= n_vec;
        uint64_t idx1 = i + total_threads + shift; if (idx1 >= n_vec) idx1 -= n_vec;
        uint64_t idx2 = i + 2ULL * total_threads + shift; if (idx2 >= n_vec) idx2 -= n_vec;
        uint64_t idx3 = i + 3ULL * total_threads + shift; if (idx3 >= n_vec) idx3 -= n_vec;
        acc0 += src[idx0];
        acc1 += src[idx1];
        acc2 += src[idx2];
        acc3 += src[idx3];
    }
    for (; i < n_vec; i += total_threads) {
        uint64_t idx = i + shift; if (idx >= n_vec) idx -= n_vec;
        acc0 += src[idx];
    }
}

uint4 sum4 = (acc0 + acc1) + (acc2 + acc3);
uint local_sum = sum4.x + sum4.y + sum4.z + sum4.w;

// SIMD-group + threadgroup reduction -> 1 uint write per threadgroup
uint sg_sum = simd_sum(local_sum);
threadgroup uint tg_buf[32];
uint simd_lane = thread_index_in_simdgroup;
uint simd_id = simdgroup_index_in_threadgroup;
if (simd_lane == 0) {
    tg_buf[simd_id] = sg_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd_id == 0) {
    uint num_sg = (tg_size + 31) / 32;
    uint val = (simd_lane < num_sg) ? tg_buf[simd_lane] : 0u;
    uint total = simd_sum(val);
    if (simd_lane == 0) {
        out[tg_id] = total;
    }
}
"""

_PURE_READ_FLOAT4_SRC = r"""
uint tid = thread_position_in_threadgroup.x;
uint tg_id = threadgroup_position_in_grid.x;
uint tg_size = threads_per_threadgroup.x;
uint num_tg = threadgroups_per_grid.x;

uint64_t n_vec = uint64_t(params[0]);
uint reps = params[1];
uint64_t total_threads = uint64_t(num_tg) * uint64_t(tg_size);
uint64_t gid = uint64_t(tg_id) * uint64_t(tg_size) + uint64_t(tid);

const device float4* src = reinterpret_cast<const device float4*>(inp);

float4 acc0 = float4(0.0f);
float4 acc1 = float4(0.0f);
float4 acc2 = float4(0.0f);
float4 acc3 = float4(0.0f);

uint64_t stride4 = total_threads * 4ULL;

for (uint r = 0; r < reps; ++r) {
    uint64_t shift = (uint64_t(r) * 1024ULL) % n_vec;
    uint64_t i = gid;

    for (; i + 3ULL * total_threads < n_vec; i += stride4) {
        uint64_t idx0 = i + shift; if (idx0 >= n_vec) idx0 -= n_vec;
        uint64_t idx1 = i + total_threads + shift; if (idx1 >= n_vec) idx1 -= n_vec;
        uint64_t idx2 = i + 2ULL * total_threads + shift; if (idx2 >= n_vec) idx2 -= n_vec;
        uint64_t idx3 = i + 3ULL * total_threads + shift; if (idx3 >= n_vec) idx3 -= n_vec;
        acc0 += src[idx0];
        acc1 += src[idx1];
        acc2 += src[idx2];
        acc3 += src[idx3];
    }
    for (; i < n_vec; i += total_threads) {
        uint64_t idx = i + shift; if (idx >= n_vec) idx -= n_vec;
        acc0 += src[idx];
    }
}

float4 sum4 = (acc0 + acc1) + (acc2 + acc3);
float local_sum = sum4.x + sum4.y + sum4.z + sum4.w;

float sg_sum = simd_sum(local_sum);
threadgroup float tg_buf[32];
uint simd_lane = thread_index_in_simdgroup;
uint simd_id = simdgroup_index_in_threadgroup;
if (simd_lane == 0) {
    tg_buf[simd_id] = sg_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd_id == 0) {
    uint num_sg = (tg_size + 31) / 32;
    float val = (simd_lane < num_sg) ? tg_buf[simd_lane] : 0.0f;
    float total = simd_sum(val);
    if (simd_lane == 0) {
        out[tg_id] = total;
    }
}
"""

_COPY_UINT4_SRC = r"""
uint tid = thread_position_in_threadgroup.x;
uint tg_id = threadgroup_position_in_grid.x;
uint tg_size = threads_per_threadgroup.x;
uint num_tg = threadgroups_per_grid.x;

uint64_t n_vec = uint64_t(params[0]);
uint64_t total_threads = uint64_t(num_tg) * uint64_t(tg_size);
uint64_t gid = uint64_t(tg_id) * uint64_t(tg_size) + uint64_t(tid);

const device uint4* src = reinterpret_cast<const device uint4*>(inp);
device uint4* dst = reinterpret_cast<device uint4*>(out);

uint64_t stride4 = total_threads * 4ULL;
uint64_t i = gid;

for (; i + 3ULL * total_threads < n_vec; i += stride4) {
    uint4 v0 = src[i];
    uint4 v1 = src[i + total_threads];
    uint4 v2 = src[i + 2ULL * total_threads];
    uint4 v3 = src[i + 3ULL * total_threads];
    dst[i] = v0;
    dst[i + total_threads] = v1;
    dst[i + 2ULL * total_threads] = v2;
    dst[i + 3ULL * total_threads] = v3;
}
for (; i < n_vec; i += total_threads) {
    dst[i] = src[i];
}
"""

_PURE_READ_UINT4_KERNEL = mx.fast.metal_kernel(
    name="track_e_pure_read_uint4",
    input_names=["inp", "params"],
    output_names=["out"],
    source=_PURE_READ_UINT4_SRC,
)

_PURE_READ_FLOAT4_KERNEL = mx.fast.metal_kernel(
    name="track_e_pure_read_float4",
    input_names=["inp", "params"],
    output_names=["out"],
    source=_PURE_READ_FLOAT4_SRC,
)

_COPY_UINT4_KERNEL = mx.fast.metal_kernel(
    name="track_e_copy_uint4",
    input_names=["inp", "params"],
    output_names=["out"],
    source=_COPY_UINT4_SRC,
)


_FILL_PRNG_UINT32_SRC = r"""
uint64_t gid = uint64_t(thread_position_in_grid.x);
uint64_t n = uint64_t(params[0]);
uint seed = params[1];
if (gid < n) {
    uint state = uint(gid) * 747796405u + 2891336453u + seed * 2654435761u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    out[gid] = (word >> 22u) ^ word;
}
"""

_FILL_PRNG_FLOAT32_SRC = r"""
uint64_t gid = uint64_t(thread_position_in_grid.x);
uint64_t n = uint64_t(params[0]);
uint seed = params[1];
if (gid < n) {
    uint state = uint(gid) * 747796405u + 2891336453u + seed * 2654435761u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    word = (word >> 22u) ^ word;
    // Exact binary fraction in [-0.5, 0.5) using top 16 bits, zero NaN/Inf, high entropy
    out[gid] = float(int(word & 0xFFFFu) - 32768) * (1.0f / 65536.0f);
}
"""

_FILL_PRNG_UINT32_KERNEL = mx.fast.metal_kernel(
    name="track_e_fill_prng_u32",
    input_names=["params"],
    output_names=["out"],
    source=_FILL_PRNG_UINT32_SRC,
)

_FILL_PRNG_FLOAT32_KERNEL = mx.fast.metal_kernel(
    name="track_e_fill_prng_f32",
    input_names=["params"],
    output_names=["out"],
    source=_FILL_PRNG_FLOAT32_SRC,
)


def make_incompressible_u32(n_elem: int, seed: int = 1) -> mx.array:
    """Fill buffer with full-entropy PCG32 words to defeat hardware lossless DRAM/SLC compression."""
    params = mx.array([n_elem, seed], dtype=mx.uint32)
    (out,) = _FILL_PRNG_UINT32_KERNEL(
        inputs=[params],
        grid=(n_elem, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_elem,)],
        output_dtypes=[mx.uint32],
    )
    mx.eval(out)
    mx.synchronize()
    return out


def make_incompressible_f32(n_elem: int, seed: int = 2) -> mx.array:
    """Fill buffer with full-entropy finite float32 values to defeat hardware lossless compression."""
    params = mx.array([n_elem, seed], dtype=mx.uint32)
    (out,) = _FILL_PRNG_FLOAT32_KERNEL(
        inputs=[params],
        grid=(n_elem, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_elem,)],
        output_dtypes=[mx.float32],
    )
    mx.eval(out)
    mx.synchronize()
    return out


def run_pure_read(
    buf: mx.array,
    *,
    mode: str = "uint4",
    reps: int = 1,
    num_tg: int = 1024,
    tg_size: int = 256,
) -> mx.array:
    n_bytes = buf.size * buf.itemsize
    assert n_bytes % 16 == 0, "Buffer byte size must be divisible by 16"
    n_vec = n_bytes // 16
    params = mx.array([n_vec, reps], dtype=mx.uint32)
    kernel = _PURE_READ_UINT4_KERNEL if mode == "uint4" else _PURE_READ_FLOAT4_KERNEL
    out_dtype = mx.uint32 if mode == "uint4" else mx.float32
    (out,) = kernel(
        inputs=[buf, params],
        grid=(num_tg * tg_size, 1, 1),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[(num_tg,)],
        output_dtypes=[out_dtype],
    )
    return out


def run_copy_rw(
    buf: mx.array,
    *,
    num_tg: int = 1024,
    tg_size: int = 256,
) -> mx.array:
    n_bytes = buf.size * buf.itemsize
    assert n_bytes % 16 == 0
    n_vec = n_bytes // 16
    params = mx.array([n_vec, 1], dtype=mx.uint32)
    (out,) = _COPY_UINT4_KERNEL(
        inputs=[buf, params],
        grid=(num_tg * tg_size, 1, 1),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[buf.shape],
        output_dtypes=[buf.dtype],
    )
    return out


def flush_slc(flush_buf: mx.array) -> None:
    """Evict SLC completely by streaming through an independent >=512 MB buffer."""
    out = run_pure_read(flush_buf, mode="uint4", reps=1, num_tg=1024, tg_size=256)
    mx.eval(out)
    mx.synchronize()


def verify_kernels() -> dict:
    """Strict numerical verification proving zero dead-code elimination."""
    n_u32 = 4 * 1024 * 1024  # 16 MB
    # uint4 test: fill with 3u -> sum over 1 rep = 3 * n_u32 (mod 2^32)
    buf_u = mx.full((n_u32,), 3, dtype=mx.uint32)
    mx.eval(buf_u)
    mx.synchronize()

    out_u1 = run_pure_read(buf_u, mode="uint4", reps=1, num_tg=256, tg_size=256)
    out_u3 = run_pure_read(buf_u, mode="uint4", reps=3, num_tg=256, tg_size=256)
    mx.eval(out_u1, out_u3)
    mx.synchronize()

    got_u1 = int(mx.sum(out_u1.astype(mx.uint64)).item()) & 0xFFFFFFFF
    exp_u1 = (3 * n_u32) & 0xFFFFFFFF
    got_u3 = int(mx.sum(out_u3.astype(mx.uint64)).item()) & 0xFFFFFFFF
    exp_u3 = (3 * 3 * n_u32) & 0xFFFFFFFF
    assert got_u1 == exp_u1, f"uint4 rep=1 mismatch: got {got_u1} != exp {exp_u1}"
    assert got_u3 == exp_u3, f"uint4 rep=3 mismatch: got {got_u3} != exp {exp_u3}"

    # float4 test: fill with 0.25f (exact binary fraction)
    buf_f = mx.full((n_u32,), 0.25, dtype=mx.float32)
    mx.eval(buf_f)
    mx.synchronize()
    out_f = run_pure_read(buf_f, mode="float4", reps=2, num_tg=256, tg_size=256)
    mx.eval(out_f)
    mx.synchronize()
    got_f = float(mx.sum(out_f).item())
    exp_f = 0.25 * n_u32 * 2
    rel_err = abs(got_f - exp_f) / exp_f
    assert rel_err < 1e-5, f"float4 mismatch: got {got_f} vs exp {exp_f}"

    # copy test
    cp = run_copy_rw(buf_u, num_tg=256, tg_size=256)
    mx.eval(cp)
    mx.synchronize()
    assert bool(mx.all(cp == buf_u).item()), "copy_uint4 mismatch!"

    del buf_u, buf_f, out_u1, out_u3, out_f, cp
    gc.collect()
    mx.clear_cache()
    return {
        "uint4_exact_match": True,
        "float4_rel_err": rel_err,
        "copy_exact_match": True,
    }


def bench_pure_read_vs_copy(
    sizes_mb: list[int],
    trials: int,
    flush_mb: int = 768,
) -> dict:
    """Interleaved cold-SLC A/B benchmark: pure-read (uint4 & float4) vs R+W copy
    using incompressible high-entropy PCG32 buffers (defeats Apple lossless compression)."""
    flush_buf = make_incompressible_u32(flush_mb * 1024 * 1024 // 4, seed=99)

    # Grid-size sweep at 1024 MB first to select best threadgroup count
    n_1g = 1024 * 1024 * 1024 // 4
    buf_1g = make_incompressible_u32(n_1g, seed=10)

    tg_sweep = {}
    best_tg = 1024
    best_bw = 0.0
    for num_tg in [256, 512, 1024, 2048, 4096]:
        # warmup
        o = run_pure_read(buf_1g, mode="uint4", reps=1, num_tg=num_tg, tg_size=256)
        mx.eval(o)
        mx.synchronize()
        times = []
        for _ in range(max(3, trials)):
            flush_slc(flush_buf)
            t0 = time.perf_counter()
            o = run_pure_read(buf_1g, mode="uint4", reps=1, num_tg=num_tg, tg_size=256)
            mx.eval(o)
            mx.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt)
        med_s = statistics.median(times)
        gb_s = (1024.0 * 1024.0 * 1024.0 / 1e9) / med_s
        tg_sweep[str(num_tg)] = {
            "median_ms": round(med_s * 1e3, 4),
            "gb_s": round(gb_s, 2),
        }
        if gb_s > best_bw:
            best_bw = gb_s
            best_tg = num_tg

    del buf_1g
    gc.collect()
    mx.clear_cache()

    results = {}
    for size_mb in sizes_mb:
        n_elem = size_mb * 1024 * 1024 // 4
        n_bytes = size_mb * 1024 * 1024
        buf_u = make_incompressible_u32(n_elem, seed=20 + size_mb)
        buf_f = make_incompressible_f32(n_elem, seed=40 + size_mb)

        # Warmup
        for fn in (
            lambda: run_pure_read(buf_u, mode="uint4", reps=1, num_tg=best_tg),
            lambda: run_pure_read(buf_f, mode="float4", reps=1, num_tg=best_tg),
            lambda: run_copy_rw(buf_u, num_tg=best_tg),
        ):
            o = fn()
            mx.eval(o)
            mx.synchronize()

        # Interleaved A/B/C/D trials with SLC flush before EVERY trial
        t_u4, t_f4, t_cp_custom, t_cp_mlx = [], [], [], []
        for _ in range(trials):
            flush_slc(flush_buf)
            t0 = time.perf_counter()
            o = run_pure_read(buf_u, mode="uint4", reps=1, num_tg=best_tg)
            mx.eval(o)
            mx.synchronize()
            t_u4.append(time.perf_counter() - t0)

            flush_slc(flush_buf)
            t0 = time.perf_counter()
            o = run_pure_read(buf_f, mode="float4", reps=1, num_tg=best_tg)
            mx.eval(o)
            mx.synchronize()
            t_f4.append(time.perf_counter() - t0)

            flush_slc(flush_buf)
            t0 = time.perf_counter()
            o = run_copy_rw(buf_u, num_tg=best_tg)
            mx.eval(o)
            mx.synchronize()
            t_cp_custom.append(time.perf_counter() - t0)
            del o

            # Stock MLX copy (buf_u + 1u -> R+W = 2 * n_bytes)
            flush_slc(flush_buf)
            t0 = time.perf_counter()
            o = buf_u + mx.array(1, dtype=mx.uint32)
            mx.eval(o)
            mx.synchronize()
            t_cp_mlx.append(time.perf_counter() - t0)
            del o

        def _stat(ts: list[float], total_bytes: int) -> dict:
            med = statistics.median(ts)
            mn = min(ts)
            return {
                "median_ms": round(med * 1e3, 4),
                "min_ms": round(mn * 1e3, 4),
                "median_gb_s": round((total_bytes / 1e9) / med, 2),
                "peak_gb_s": round((total_bytes / 1e9) / mn, 2),
            }

        results[f"{size_mb}MB"] = {
            "buffer_mb": size_mb,
            "pure_read_uint4": _stat(t_u4, n_bytes),
            "pure_read_float4": _stat(t_f4, n_bytes),
            "copy_rw_custom_uint4": _stat(t_cp_custom, 2 * n_bytes),
            "copy_rw_mlx_add": _stat(t_cp_mlx, 2 * n_bytes),
        }

        del buf_u, buf_f
        gc.collect()
        mx.clear_cache()

    del flush_buf
    gc.collect()
    mx.clear_cache()
    return {
        "best_num_tg": best_tg,
        "tg_sweep_1024MB": tg_sweep,
        "cold_dram_comparison": results,
    }


_PURE_READ_UINT4_CHAIN_SRC = r"""
uint tid = thread_position_in_threadgroup.x;
uint tg_id = threadgroup_position_in_grid.x;
uint tg_size = threads_per_threadgroup.x;
uint num_tg = threadgroups_per_grid.x;

uint64_t n_vec = uint64_t(params[0]);
uint64_t total_threads = uint64_t(num_tg) * uint64_t(tg_size);
uint64_t gid = uint64_t(tg_id) * uint64_t(tg_size) + uint64_t(tid);

const device uint4* src = reinterpret_cast<const device uint4*>(inp);
// Read previous pass output to enforce strict GPU barrier between full-buffer passes
uint prev_salt = prev[0] & 1u;

uint4 acc0 = uint4(prev_salt);
uint4 acc1 = uint4(0);
uint4 acc2 = uint4(0);
uint4 acc3 = uint4(0);

uint64_t stride4 = total_threads * 4ULL;
uint64_t i = gid;

for (; i + 3ULL * total_threads < n_vec; i += stride4) {
    acc0 += src[i];
    acc1 += src[i + total_threads];
    acc2 += src[i + 2ULL * total_threads];
    acc3 += src[i + 3ULL * total_threads];
}
for (; i < n_vec; i += total_threads) {
    acc0 += src[i];
}

uint4 sum4 = (acc0 + acc1) + (acc2 + acc3);
uint local_sum = sum4.x + sum4.y + sum4.z + sum4.w;

uint sg_sum = simd_sum(local_sum);
threadgroup uint tg_buf[32];
uint simd_lane = thread_index_in_simdgroup;
uint simd_id = simdgroup_index_in_threadgroup;
if (simd_lane == 0) {
    tg_buf[simd_id] = sg_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd_id == 0) {
    uint num_sg = (tg_size + 31) / 32;
    uint val = (simd_lane < num_sg) ? tg_buf[simd_lane] : 0u;
    uint total = simd_sum(val);
    if (simd_lane == 0) {
        out[tg_id] = total;
    }
}
"""

_PURE_READ_UINT4_CHAIN_KERNEL = mx.fast.metal_kernel(
    name="track_e_pure_read_uint4_chain",
    input_names=["inp", "params", "prev"],
    output_names=["out"],
    source=_PURE_READ_UINT4_CHAIN_SRC,
)


def run_pure_read_chained(
    buf: mx.array,
    passes: int,
    *,
    num_tg: int = 1024,
    tg_size: int = 256,
) -> mx.array:
    """Execute `passes` strictly sequential full-buffer pure-read kernels in one CB.
    Every pass finishes reading the ENTIRE buffer before the next pass begins,
    eliminating intra-wave SLC reuse artifacts on large buffers."""
    n_bytes = buf.size * buf.itemsize
    assert n_bytes % 16 == 0
    n_vec = n_bytes // 16
    params = mx.array([n_vec], dtype=mx.uint32)
    cur = mx.zeros((num_tg,), dtype=mx.uint32)
    for _ in range(passes):
        (cur,) = _PURE_READ_UINT4_CHAIN_KERNEL(
            inputs=[buf, params, cur],
            grid=(num_tg * tg_size, 1, 1),
            threadgroup=(tg_size, 1, 1),
            output_shapes=[(num_tg,)],
            output_dtypes=[mx.uint32],
        )
    return cur


def bench_slc_sweep(
    sizes_mb: list[int],
    trials: int,
    num_tg: int = 1024,
) -> dict:
    """
    True full-buffer warm-reuse sweep across 4 MB .. 4096 MB to empirically reveal
    M5 Max's System Level Cache (SLC) capacity and SLC bandwidth ceiling.
    Uses strictly sequential full-buffer kernel dispatches so each pass evicts SLC
    if size_mb > SLC capacity (no intra-wave partial-buffer reuse artifact).
    """
    rows = []
    for size_mb in sizes_mb:
        n_elem = size_mb * 1024 * 1024 // 4
        n_bytes = size_mb * 1024 * 1024
        # Multiple sequential full passes so total runtime >> CB commit overhead
        if size_mb <= 16:
            passes = 64
        elif size_mb <= 64:
            passes = 24
        elif size_mb <= 256:
            passes = 8
        elif size_mb <= 512:
            passes = 4
        else:
            passes = 2

        buf = make_incompressible_u32(n_elem, seed=100 + size_mb)

        # Warmup: populate SLC if buffer fits
        for _ in range(2):
            o = run_pure_read_chained(buf, passes=min(passes, 4), num_tg=num_tg, tg_size=256)
            mx.eval(o)
            mx.synchronize()

        times = []
        for _ in range(trials):
            t0 = time.perf_counter()
            o = run_pure_read_chained(buf, passes=passes, num_tg=num_tg, tg_size=256)
            mx.eval(o)
            mx.synchronize()
            times.append(time.perf_counter() - t0)

        med_s = statistics.median(times)
        min_s = min(times)
        total_bytes = n_bytes * passes
        med_gb_s = (total_bytes / 1e9) / med_s
        peak_gb_s = (total_bytes / 1e9) / min_s

        rows.append(
            {
                "size_mb": size_mb,
                "passes": passes,
                "total_read_mb": size_mb * passes,
                "median_ms": round(med_s * 1e3, 4),
                "median_gb_s": round(med_gb_s, 2),
                "peak_gb_s": round(peak_gb_s, 2),
            }
        )

        del buf, o
        gc.collect()
        mx.clear_cache()

    # Estimate empirical SLC knee: largest size with >= 1.15x DRAM floor
    dram_ref = rows[-1]["median_gb_s"]
    slc_peak = max(r["median_gb_s"] for r in rows[:4])
    knee_mb = None
    for r in rows:
        if r["median_gb_s"] >= dram_ref * 1.15:
            knee_mb = r["size_mb"]
    return {
        "num_tg": num_tg,
        "sweep": rows,
        "slc_peak_gb_s": round(slc_peak, 2),
        "dram_floor_gb_s": round(dram_ref, 2),
        "estimated_slc_knee_mb": knee_mb,
    }


def run_all(*, quick: bool = False) -> dict:
    ver = verify_kernels()
    if quick:
        dram_sizes = [512, 2048]
        slc_sizes = [4, 16, 32, 48, 64, 96, 128, 256, 512, 2048]
        trials = 3
    else:
        dram_sizes = [512, 1024, 2048, 4096]
        slc_sizes = [4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096]
        trials = 7

    dram_res = bench_pure_read_vs_copy(dram_sizes, trials=trials)
    slc_res = bench_slc_sweep(slc_sizes, trials=trials, num_tg=dram_res["best_num_tg"])

    # Headline summary
    largest_key = f"{dram_sizes[-1]}MB"
    largest = dram_res["cold_dram_comparison"][largest_key]
    pr_u4 = largest["pure_read_uint4"]["median_gb_s"]
    pr_f4 = largest["pure_read_float4"]["median_gb_s"]
    cp_custom = largest["copy_rw_custom_uint4"]["median_gb_s"]
    cp_mlx = largest["copy_rw_mlx_add"]["median_gb_s"]
    pr_best = max(pr_u4, pr_f4)

    return {
        "verified": ver,
        "headline": {
            "largest_buffer_mb": dram_sizes[-1],
            "pure_read_uint4_gb_s": pr_u4,
            "pure_read_float4_gb_s": pr_f4,
            "pure_read_best_gb_s": pr_best,
            "copy_rw_custom_gb_s": cp_custom,
            "copy_rw_mlx_gb_s": cp_mlx,
            "pure_read_vs_spec_614_4_pct": round(100.0 * pr_best / 614.4, 2),
            "pure_read_vs_copy_ratio": round(pr_best / max(cp_custom, 1e-6), 3),
            "slc_peak_gb_s": slc_res["slc_peak_gb_s"],
            "estimated_slc_knee_mb": slc_res["estimated_slc_knee_mb"],
        },
        "dram": dram_res,
        "slc": slc_res,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Track E.1: M5 Max Pure-Read DRAM BW & SLC Sweep")
    ap.add_argument("--quick", action="store_true", help="Fast smoke test")
    ap.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    ap.add_argument("--stream-seconds", type=float, default=0.0,
                    help="Continuous 2 GB pure-read loop for concurrent CPU/ANE interference tests")
    args = ap.parse_args()

    if args.stream_seconds > 0:
        n_elem = 2048 * 1024 * 1024 // 4
        buf = make_incompressible_u32(n_elem, seed=777)
        # warmup
        o = run_pure_read(buf, mode="uint4", reps=1, num_tg=1024)
        mx.eval(o)
        mx.synchronize()
        iters = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.stream_seconds:
            o = run_pure_read(buf, mode="uint4", reps=1, num_tg=1024)
            mx.eval(o)
            mx.synchronize()
            iters += 1
        dt = time.perf_counter() - t0
        gb_s = (iters * 2048.0 * 1024.0 * 1024.0 / 1e9) / dt
        print(json.dumps({"iters": iters, "elapsed_s": round(dt, 4), "gpu_pure_read_gb_s": round(gb_s, 2)}))
        return

    res = run_all(quick=args.quick)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
    print(json.dumps(res["headline"], indent=2))
    print("\n--- SLC Warm-Reuse Sweep (Full-Buffer Sequential Passes) ---")
    for r in res["slc"]["sweep"]:
        print(f"  {r['size_mb']:5d} MB (passes={r['passes']:2d}): {r['median_gb_s']:7.2f} GB/s (peak {r['peak_gb_s']:7.2f} GB/s)")
    print("\n--- Cold-SLC DRAM Comparison ---")
    for k, v in res["dram"]["cold_dram_comparison"].items():
        print(
            f"  {k:>6s}: pure_u4={v['pure_read_uint4']['median_gb_s']:6.2f} | "
            f"pure_f4={v['pure_read_float4']['median_gb_s']:6.2f} | "
            f"copy_u4={v['copy_rw_custom_uint4']['median_gb_s']:6.2f} | "
            f"copy_mlx={v['copy_rw_mlx_add']['median_gb_s']:6.2f} GB/s"
        )


if __name__ == "__main__":
    main()
