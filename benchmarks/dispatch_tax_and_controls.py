#!/usr/bin/env python3
"""
Track E.3 — Metal Dispatch Tax Audit, Real Qwen Op-Count Breakdown,
            Fable 5.2's x[4, 16, 4096] Reshape Test, Dense FP16 Control, & GLM 5.3 Iso-Byte Control

Settles three major council diagnostics:
  1. Opus 5.5's 2,000-dependent-tiny-kernel Metal dispatch tax test -> exact us/dispatch
     (linear regression across N=200..4000 isolates incremental GPU dispatch + barrier tax
     from fixed CB commit intercept, plus parallel control and host-sync roundtrip).
  2. Exact Metal op / kernel dispatch count per B=1 decode token in live cached
     `Qwen3.5-9B-4bit` and `Qwen3.6-35B-A3B-4bit` via `mx.export_to_dot` -> quantifies
     exact dispatch overhead (ms & % of decode step) vs gather/DRAM bandwidth
     (testing Opus 5.5's `< 3.0 ms` falsifier on 35B-A3B!).
  3. Fable 5.2's `x[4, 16, 4096]` batched reshape test + Dense FP16 control (1024 MB > SLC)
     + GLM 5.3's iso-byte wider-N occupancy control (`4096x4096`, `8192x2048`, `16384x1024`)
     with strict interleaved A/B trials, >=512 MB SLC flushes, and exact numerical checks!
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import io
import json
import os
import re
import statistics
import sys
import time

_MLX_PY = "python3"
try:
    import mlx.core as mx
except ImportError:
    if os.path.exists(_MLX_PY) and os.path.realpath(sys.executable) != os.path.realpath(_MLX_PY):
        os.execv(_MLX_PY, [_MLX_PY] + sys.argv)
    raise


# ---------------------------------------------------------------------------
# Part 1: Opus 5.5's 2,000-Dependent-Tiny-Kernel Metal Dispatch Tax Test
# ---------------------------------------------------------------------------

_TINY_INC_SRC = r"""
uint gid = thread_position_in_grid.x;
if (gid == 0) {
    out[0] = inp[0] + 1.0f;
}
"""

_TINY_INC_KERNEL = mx.fast.metal_kernel(
    name="track_e_tiny_inc",
    input_names=["inp"],
    output_names=["out"],
    source=_TINY_INC_SRC,
)


def _tiny_inc(x: mx.array) -> mx.array:
    (y,) = _TINY_INC_KERNEL(
        inputs=[x],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.float32],
    )
    return y


def bench_metal_dispatch_tax(*, quick: bool = False) -> dict:
    trials = 5 if quick else 11
    chain_lengths = [200, 500, 1000, 2000] if quick else [200, 500, 1000, 2000, 4000]

    # Warmup kernel compile + GPU clock
    w = mx.array([0.0], dtype=mx.float32)
    for _ in range(50):
        w = _tiny_inc(w)
    mx.eval(w)
    mx.synchronize()
    assert abs(float(w.item()) - 50.0) < 1e-4

    # 1. Serial dependent chain inside single command submission across N lengths
    serial_curve = []
    for n in chain_lengths:
        times_us = []
        for _ in range(trials):
            x = mx.array([0.0], dtype=mx.float32)
            mx.eval(x)
            mx.synchronize()
            t_build0 = time.perf_counter()
            y = x
            for _ in range(n):
                y = _tiny_inc(y)
            t_eval0 = time.perf_counter()
            mx.eval(y)
            mx.synchronize()
            t_end = time.perf_counter()
            assert abs(float(y.item()) - float(n)) < 1e-2, f"Chain {n} mismatch: {y.item()}"
            times_us.append((t_end - t_eval0) * 1e6)
        med_us = statistics.median(times_us)
        min_us = min(times_us)
        serial_curve.append(
            {
                "n_dispatches": n,
                "median_eval_ms": round(med_us / 1e3, 4),
                "min_eval_ms": round(min_us / 1e3, 4),
                "us_per_dispatch_gross": round(med_us / n, 3),
                "min_us_per_dispatch_gross": round(min_us / n, 3),
            }
        )

    # Ordinary least squares linear fit: eval_us = intercept_us + slope_us * N
    xs = [float(r["n_dispatches"]) for r in serial_curve]
    ys = [r["median_eval_ms"] * 1e3 for r in serial_curve]
    mx_x = sum(xs) / len(xs)
    mx_y = sum(ys) / len(ys)
    slope_us = sum((x - mx_x) * (y - mx_y) for x, y in zip(xs, ys)) / sum((x - mx_x) ** 2 for x in xs)
    intercept_us = mx_y - slope_us * mx_x

    # 2. Independent (parallel, zero RAW dependency) tiny kernels N=2000
    n_par = 2000
    par_times_us = []
    for _ in range(trials):
        inp = mx.array([1.0], dtype=mx.float32)
        mx.eval(inp)
        mx.synchronize()
        outs = [_tiny_inc(inp) for _ in range(n_par)]
        t0 = time.perf_counter()
        mx.eval(*outs)
        mx.synchronize()
        par_times_us.append((time.perf_counter() - t0) * 1e6)
    par_med_us = statistics.median(par_times_us)

    # 3. Stock MLX dependent tiny op chain (compile disabled) N=2000
    n_stock = 2000
    stock_times_us = []
    mx.disable_compile()
    try:
        one = mx.array([1.0], dtype=mx.float32)
        mx.eval(one)
        mx.synchronize()
        for _ in range(trials):
            y = mx.array([0.0], dtype=mx.float32)
            mx.eval(y)
            mx.synchronize()
            for _ in range(n_stock):
                y = y + one
            t0 = time.perf_counter()
            mx.eval(y)
            mx.synchronize()
            stock_times_us.append((time.perf_counter() - t0) * 1e6)
            assert abs(float(y.item()) - float(n_stock)) < 1e-2
    finally:
        mx.enable_compile()
    stock_med_us = statistics.median(stock_times_us)

    # 4. Per-dispatch host roundtrip sync latency (mx.eval + mx.synchronize per op)
    n_sync = 100 if quick else 250
    y = mx.array([0.0], dtype=mx.float32)
    mx.eval(y)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_sync):
        y = _tiny_inc(y)
        mx.eval(y)
        mx.synchronize()
    sync_us_per_op = ((time.perf_counter() - t0) * 1e6) / n_sync
    assert abs(float(y.item()) - float(n_sync)) < 1e-2

    row_2000 = next(r for r in serial_curve if r["n_dispatches"] == 2000)
    return {
        "serial_dependent_chain": serial_curve,
        "linear_fit_serial_us_per_dispatch": round(slope_us, 3),
        "linear_fit_cb_commit_intercept_us": round(intercept_us, 2),
        "serial_2000_gross_us_per_dispatch": row_2000["us_per_dispatch_gross"],
        "serial_2000_total_ms": row_2000["median_eval_ms"],
        "independent_2000_total_ms": round(par_med_us / 1e3, 4),
        "independent_2000_us_per_dispatch": round(par_med_us / n_par, 3),
        "stock_mlx_uncompiled_2000_total_ms": round(stock_med_us / 1e3, 4),
        "stock_mlx_uncompiled_2000_us_per_dispatch": round(stock_med_us / n_stock, 3),
        "host_sync_roundtrip_us_per_commit": round(sync_us_per_op, 2),
    }


# ---------------------------------------------------------------------------
# Part 2: Exact Metal Op / Dispatch Count per Decode Token on Live Qwen Models
# ---------------------------------------------------------------------------

# MLX primitives that are typically zero-copy view/metadata ops (no GPU kernel dispatch)
_ZERO_COST_VIEW_OPS = {
    "Broadcast",
    "Reshape",
    "Squeeze",
    "ExpandDims",
    "Split",
    "Slice",
    "Transpose",
    "StopGradient",
    "NumberOfElements",
    "Load",
    "Full",
}


def _collect_arrays(obj) -> list[mx.array]:
    out = []
    if isinstance(obj, mx.array):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            out.extend(_collect_arrays(x))
    elif isinstance(obj, dict):
        for x in obj.values():
            out.extend(_collect_arrays(x))
    return out


def _parse_dot_ops(dot_text: str) -> dict:
    # Matches lines like: { 31052432440 [label ="Matmul", shape=rectangle]; }
    labels = re.findall(r'\[label\s*=\s*"([^"]+)",\s*shape=rectangle\]', dot_text)
    counts = Counter(labels)
    total_primitives = len(labels)
    view_ops = sum(c for k, c in counts.items() if k in _ZERO_COST_VIEW_OPS)
    kernel_ops = total_primitives - view_ops
    return {
        "total_mlx_primitives": total_primitives,
        "estimated_gpu_kernel_dispatches": kernel_ops,
        "zero_copy_view_or_meta_ops": view_ops,
        "op_histogram": dict(counts.most_common()),
    }


def audit_model_decode_dispatches(
    model_id: str,
    serial_us_per_dispatch: float,
    indep_us_per_dispatch: float,
    *,
    decode_steps: int = 15,
) -> dict:
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    t_load0 = time.perf_counter()
    model, tokenizer = load(model_id)
    load_s = time.perf_counter() - t_load0

    prompt = (
        "Explain from first principles how unified memory bandwidth, cache line granularity, "
        "and matrix engine arithmetic intensity interact during autoregressive token generation "
        "on modern Apple Silicon hardware."
    )
    prompt_ids = tokenizer.encode(prompt)
    cache = make_prompt_cache(model)

    # Prefill + 2 warmup decode tokens
    x = mx.array([prompt_ids], dtype=mx.int32)
    logits = model(x, cache=cache)
    cache_arrs = _collect_arrays([c.state for c in cache])
    mx.eval(logits, *cache_arrs)
    mx.synchronize()

    next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    for _ in range(2):
        logits = model(next_tok, cache=cache)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        cache_arrs = _collect_arrays([c.state for c in cache])
        mx.eval(logits, next_tok, *cache_arrs)
        mx.synchronize()

    # Capture ONE B=1 decode step graph BEFORE evaluating!
    logits = model(next_tok, cache=cache)
    out_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    cache_arrs = _collect_arrays([c.state for c in cache])

    buf = io.StringIO()
    mx.export_to_dot(buf, out_tok, *cache_arrs)
    dot_text = buf.getvalue()
    graph_stats = _parse_dot_ops(dot_text)

    # Commit the captured step
    mx.eval(out_tok, *cache_arrs)
    mx.synchronize()
    next_tok = out_tok

    # Measure synchronized B=1 decode latency over `decode_steps` tokens
    step_ms = []
    for _ in range(decode_steps):
        t0 = time.perf_counter()
        logits = model(next_tok, cache=cache)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        cache_arrs = _collect_arrays([c.state for c in cache])
        mx.eval(next_tok, *cache_arrs)
        mx.synchronize()
        step_ms.append((time.perf_counter() - t0) * 1e3)

    med_ms = statistics.median(step_ms)
    min_ms = min(step_ms)
    n_k = graph_stats["estimated_gpu_kernel_dispatches"]
    n_tot = graph_stats["total_mlx_primitives"]

    implied_serial_ms = (n_k * serial_us_per_dispatch) / 1e3
    implied_indep_ms = (n_k * indep_us_per_dispatch) / 1e3

    # Clean up aggressively before loading next large model
    del model, tokenizer, cache, logits, next_tok, out_tok, cache_arrs
    gc.collect()
    mx.clear_cache()

    return {
        "model_id": model_id,
        "load_s": round(load_s, 2),
        "decode_b1_median_ms": round(med_ms, 3),
        "decode_b1_min_ms": round(min_ms, 3),
        "decode_b1_tok_s": round(1000.0 / med_ms, 2),
        "graph": graph_stats,
        "implied_dispatch_tax_serial_ms": round(implied_serial_ms, 3),
        "implied_dispatch_tax_serial_pct_of_step": round(100.0 * implied_serial_ms / med_ms, 1),
        "implied_dispatch_tax_indep_ms": round(implied_indep_ms, 3),
        "implied_dispatch_tax_indep_pct_of_step": round(100.0 * implied_indep_ms / med_ms, 1),
        "implied_if_all_primitives_dispatched_ms": round((n_tot * serial_us_per_dispatch) / 1e3, 3),
    }


# ---------------------------------------------------------------------------
# Part 3: Fable 5.2's x[4, 16, 4096] Reshape Test, Dense FP16 Control,
#         & GLM 5.3 Iso-Byte Occupancy Control
# ---------------------------------------------------------------------------

def _flush_slc(flush_buf: mx.array) -> None:
    s = mx.sum(flush_buf)
    mx.eval(s)
    mx.synchronize()


def bench_fable_reshape_and_controls(*, quick: bool = False) -> dict:
    n_layers = 32
    K = 4096
    N = 4096
    trials = 5 if quick else 11

    # 600 MB SLC flush buffer
    flush_buf = mx.ones((600 * 1024 * 1024 // 4,), dtype=mx.uint32)
    mx.eval(flush_buf)
    mx.synchronize()

    # --- 3A. Q4 (gs=64, bits=4) 32-layer 4096x4096 sweep (302.0 MB > SLC) ---
    # Scale weights by ~1/sqrt(K) (0.015) so activation variance stays O(1) across 32 chained layers.
    mx.random.seed(42)
    q4_layers = []
    for _ in range(n_layers):
        wf = (mx.random.normal((N, K)) * 0.015).astype(mx.float16)
        wq, sc, bi = mx.quantize(wf, group_size=64, bits=4)
        q4_layers.append((wq, sc, bi))
    mx.eval(*[t for layer in q4_layers for t in layer])
    mx.synchronize()

    q4_bytes = sum(t.size * t.itemsize for layer in q4_layers for t in layer)

    def run_q4_layers(x: mx.array) -> mx.array:
        y = x
        for wq, sc, bi in q4_layers:
            y = mx.quantized_matmul(y, wq, scales=sc, biases=bi, transpose=True, group_size=64, bits=4)
        return y

    # Strict numerical equivalence check for Fable's reshape tricks & M=65 pad
    x64 = (mx.random.normal((64, K)) * 0.5).astype(mx.float16)
    mx.eval(x64)
    mx.synchronize()
    ref_y64 = run_q4_layers(x64)
    mx.eval(ref_y64)
    mx.synchronize()
    ref_scale = max(float(mx.max(mx.abs(ref_y64.astype(mx.float32))).item()), 1e-6)

    reshape_checks = {}
    for label, shape in [("(4,16,4096)", (4, 16, K)), ("(2,32,4096)", (2, 32, K)), ("(8,8,4096)", (8, 8, K))]:
        yr = run_q4_layers(x64.reshape(shape)).reshape(64, N)
        mx.eval(yr)
        mx.synchronize()
        max_err = float(mx.max(mx.abs(yr.astype(mx.float32) - ref_y64.astype(mx.float32))).item())
        rel_err = max_err / ref_scale
        reshape_checks[label] = {
            "max_abs_err": round(max_err, 6),
            "rel_err": round(rel_err, 6),
            "allclose_1e_2": rel_err < 1e-2,
        }
        assert rel_err < 2e-2, f"Reshape {label} numerical mismatch: abs={max_err}, rel={rel_err}"

    # Pad M=65 equivalence check (first 64 rows match; note M=64 uses qmm_splitk, M=65 uses qmm_nax)
    x65 = mx.concatenate([x64, mx.zeros((1, K), dtype=mx.float16)], axis=0)
    y65 = run_q4_layers(x65)
    mx.eval(y65)
    mx.synchronize()
    err65 = float(mx.max(mx.abs(y65[:64].astype(mx.float32) - ref_y64.astype(mx.float32))).item())
    rel65 = err65 / ref_scale
    reshape_checks["pad_M65_first64"] = {
        "max_abs_err": round(err65, 6),
        "rel_err": round(rel65, 6),
        "allclose_1e_2": rel65 < 1e-2,
    }
    assert rel65 < 2e-2, f"Pad M65 mismatch: abs={err65}, rel={rel65}"

    # Interleaved cold-SLC A/B trials across all Q4 shapes
    q4_configs = [
        ("M1", (1, K), 1),
        ("M16", (16, K), 16),
        ("M32", (32, K), 32),
        ("M64_flat", (64, K), 64),
        ("M64_reshape_4x16", (4, 16, K), 64),
        ("M64_reshape_2x32", (2, 32, K), 64),
        ("M64_reshape_8x8", (8, 8, K), 64),
        ("M65_pad1", (65, K), 65),
        ("M128", (128, K), 128),
        ("M256", (256, K), 256),
    ]
    q4_inputs = {}
    for name, shp, _ in q4_configs:
        inp = (mx.random.normal(shp) * 0.05).astype(mx.float16)
        mx.eval(inp)
        q4_inputs[name] = inp
        # Warmup once
        o = run_q4_layers(inp)
        mx.eval(o)
    mx.synchronize()

    q4_times = {name: [] for name, _, _ in q4_configs}
    for _ in range(trials):
        for name, _, _ in q4_configs:
            _flush_slc(flush_buf)
            t0 = time.perf_counter()
            o = run_q4_layers(q4_inputs[name])
            mx.eval(o)
            mx.synchronize()
            q4_times[name].append((time.perf_counter() - t0) * 1e3)

    q4_summary = {}
    for name, shp, rows in q4_configs:
        med = statistics.median(q4_times[name])
        mn = min(q4_times[name])
        flops = n_layers * 2.0 * rows * N * K
        q4_summary[name] = {
            "shape": list(shp),
            "effective_rows": rows,
            "median_ms": round(med, 4),
            "min_ms": round(mn, 4),
            "effective_gb_s": round((q4_bytes / 1e9) / (med / 1e3), 2),
            "effective_tflops": round((flops / 1e12) / (med / 1e3), 2),
        }

    del q4_layers, q4_inputs, x64, x65, ref_y64, y65
    gc.collect()
    mx.clear_cache()

    # --- 3B. Dense FP16 Control (32 x 4096x4096 FP16 = 1073.7 MB >> SLC) ---
    fp16_layers = [
        (mx.random.normal((K, N)) * 0.015).astype(mx.float16)
        for _ in range(n_layers)
    ]
    mx.eval(*fp16_layers)
    mx.synchronize()
    fp16_bytes = sum(w.size * w.itemsize for w in fp16_layers)

    def run_fp16_layers(x: mx.array) -> mx.array:
        y = x
        for w in fp16_layers:
            y = mx.matmul(y, w)
        return y

    fp16_configs = [
        ("M1", (1, K), 1),
        ("M16", (16, K), 16),
        ("M32", (32, K), 32),
        ("M64_flat", (64, K), 64),
        ("M64_reshape_4x16", (4, 16, K), 64),
        ("M65_pad1", (65, K), 65),
        ("M128", (128, K), 128),
        ("M256", (256, K), 256),
    ]
    fp16_inputs = {}
    for name, shp, _ in fp16_configs:
        inp = (mx.random.normal(shp) * 0.05).astype(mx.float16)
        mx.eval(inp)
        fp16_inputs[name] = inp
        o = run_fp16_layers(inp)
        mx.eval(o)
    mx.synchronize()

    fp16_times = {name: [] for name, _, _ in fp16_configs}
    for _ in range(trials):
        for name, _, _ in fp16_configs:
            _flush_slc(flush_buf)
            t0 = time.perf_counter()
            o = run_fp16_layers(fp16_inputs[name])
            mx.eval(o)
            mx.synchronize()
            fp16_times[name].append((time.perf_counter() - t0) * 1e3)

    fp16_summary = {}
    for name, shp, rows in fp16_configs:
        med = statistics.median(fp16_times[name])
        mn = min(fp16_times[name])
        flops = n_layers * 2.0 * rows * N * K
        fp16_summary[name] = {
            "shape": list(shp),
            "effective_rows": rows,
            "median_ms": round(med, 4),
            "min_ms": round(mn, 4),
            "effective_gb_s": round((fp16_bytes / 1e9) / (med / 1e3), 2),
            "effective_tflops": round((flops / 1e12) / (med / 1e3), 2),
        }

    del fp16_layers, fp16_inputs
    gc.collect()
    mx.clear_cache()

    # --- 3C. GLM 5.3 / Neon Iso-Byte Wider-N Occupancy Control (Q4, ~302 MB each) ---
    iso_byte_results = {}
    for (ni, ki) in [(4096, 4096), (8192, 2048), (16384, 1024)]:
        layers = []
        for _ in range(n_layers):
            wf = (mx.random.normal((ni, ki)) * 0.02).astype(mx.float16)
            layers.append(mx.quantize(wf, group_size=64, bits=4))
        mx.eval(*[t for l in layers for t in l])
        mx.synchronize()
        lbytes = sum(t.size * t.itemsize for l in layers for t in l)

        shape_key = f"N{ni}_K{ki}"
        iso_byte_results[shape_key] = {"weight_mb": round(lbytes / 1e6, 2), "n_tg_bn64": ni // 64, "rows": {}}
        for m in [16, 32, 64, 65, 128]:
            x_in = (mx.random.normal((m, ki)) * 0.05).astype(mx.float16)
            mx.eval(x_in)
            # Run 32 independent layers with same x_in (avoids dim mismatch since N != K)
            def _run_iso():
                outs = [
                    mx.quantized_matmul(x_in, wq, scales=sc, biases=bi, transpose=True, group_size=64, bits=4)
                    for (wq, sc, bi) in layers
                ]
                return outs
            o = _run_iso()
            mx.eval(*o)
            mx.synchronize()
            ts = []
            for _ in range(max(3, trials // 2)):
                _flush_slc(flush_buf)
                t0 = time.perf_counter()
                o = _run_iso()
                mx.eval(*o)
                mx.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            med = statistics.median(ts)
            iso_byte_results[shape_key]["rows"][f"M{m}"] = {
                "median_ms": round(med, 4),
                "effective_gb_s": round((lbytes / 1e9) / (med / 1e3), 2),
                "effective_tflops": round((n_layers * 2.0 * m * ni * ki / 1e12) / (med / 1e3), 2),
            }
        del layers
        gc.collect()
        mx.clear_cache()

    del flush_buf
    gc.collect()
    mx.clear_cache()

    # Fable's exact 30-min falsifier verdict
    m64_flat = q4_summary["M64_flat"]["median_ms"]
    m64_4x16 = q4_summary["M64_reshape_4x16"]["median_ms"]
    m65_pad = q4_summary["M65_pad1"]["median_ms"]
    m128_q4 = q4_summary["M128"]["median_ms"]
    fp16_m64 = fp16_summary["M64_flat"]["median_ms"]
    fp16_m128 = fp16_summary["M128"]["median_ms"]

    return {
        "q4_weight_mb": round(q4_bytes / 1e6, 2),
        "fp16_weight_mb": round(fp16_bytes / 1e6, 2),
        "reshape_equivalence_checks": reshape_checks,
        "q4_32layer_4096": q4_summary,
        "dense_fp16_control_32layer_4096": fp16_summary,
        "glm_iso_byte_wider_N_control": iso_byte_results,
        "verdicts": {
            "fable_m64_flat_ms": m64_flat,
            "fable_m64_4x16_ms": m64_4x16,
            "fable_4x16_speedup_vs_flat": round(m64_flat / m64_4x16, 3),
            "fable_30min_prediction_le_3p8ms": m64_4x16 <= 3.8,
            "m65_pad1_ms": m65_pad,
            "m128_q4_ms": m128_q4,
            "dense_fp16_has_m64_valley": fp16_m64 > fp16_m128 * 1.15,
            "dense_fp16_m64_vs_m128_ms": [fp16_m64, fp16_m128],
        },
    }


def run_all(*, quick: bool = False, skip_models: bool = False) -> dict:
    dispatch = bench_metal_dispatch_tax(quick=quick)
    models = {}
    if not skip_models:
        dsteps = 8 if quick else 20
        for short_name, model_id in [
            ("qwen35_9b_4bit", "mlx-community/Qwen3.5-9B-4bit"),
            ("qwen36_35b_a3b_4bit", "mlx-community/Qwen3.6-35B-A3B-4bit"),
        ]:
            models[short_name] = audit_model_decode_dispatches(
                model_id,
                serial_us_per_dispatch=dispatch["linear_fit_serial_us_per_dispatch"],
                indep_us_per_dispatch=dispatch["independent_2000_us_per_dispatch"],
                decode_steps=dsteps,
            )

    controls = bench_fable_reshape_and_controls(quick=quick)
    return {
        "metal_dispatch_tax": dispatch,
        "model_decode_op_audit": models,
        "fable_reshape_and_controls": controls,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Track E.3: Metal Dispatch Tax, Qwen Op Audit & Controls")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    res = run_all(quick=args.quick, skip_models=args.skip_models)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)

    dt = res["metal_dispatch_tax"]
    print("=== Metal Dispatch Tax ===")
    print(f"  Serial dependent slope:    {dt['linear_fit_serial_us_per_dispatch']:.3f} us/dispatch (intercept {dt['linear_fit_cb_commit_intercept_us']:.1f} us)")
    print(f"  2,000 serial tiny kernels: {dt['serial_2000_total_ms']:.3f} ms ({dt['serial_2000_gross_us_per_dispatch']:.3f} us/dispatch)")
    print(f"  2,000 indep  tiny kernels: {dt['independent_2000_total_ms']:.3f} ms ({dt['independent_2000_us_per_dispatch']:.3f} us/dispatch)")
    print(f"  2,000 stock MLX uncomp:    {dt['stock_mlx_uncompiled_2000_total_ms']:.3f} ms ({dt['stock_mlx_uncompiled_2000_us_per_dispatch']:.3f} us/dispatch)")
    print(f"  Host sync roundtrip:       {dt['host_sync_roundtrip_us_per_commit']:.1f} us/commit")

    if res["model_decode_op_audit"]:
        print("\n=== Live Qwen B=1 Decode Op Audit ===")
        for k, m in res["model_decode_op_audit"].items():
            g = m["graph"]
            print(
                f"  {k}: {m['decode_b1_median_ms']:.2f} ms/tok ({m['decode_b1_tok_s']:.1f} tok/s) | "
                f"total_ops={g['total_mlx_primitives']}, gpu_kernels={g['estimated_gpu_kernel_dispatches']} | "
                f"implied_serial_tax={m['implied_dispatch_tax_serial_ms']:.2f} ms ({m['implied_dispatch_tax_serial_pct_of_step']:.1f}%)"
            )

    c = res["fable_reshape_and_controls"]
    print("\n=== Fable x[4,16,4096] Reshape & Dense FP16 Control ===")
    print(json.dumps(c["verdicts"], indent=2))
    print("\n--- Q4 32x4096^2 ---")
    for k, v in c["q4_32layer_4096"].items():
        print(f"  {k:18s}: {v['median_ms']:6.3f} ms | {v['effective_gb_s']:6.1f} GB/s | {v['effective_tflops']:5.2f} TFLOP/s")
    print("\n--- Dense FP16 32x4096^2 Control ---")
    for k, v in c["dense_fp16_control_32layer_4096"].items():
        print(f"  {k:18s}: {v['median_ms']:6.3f} ms | {v['effective_gb_s']:6.1f} GB/s | {v['effective_tflops']:5.2f} TFLOP/s")


if __name__ == "__main__":
    main()
