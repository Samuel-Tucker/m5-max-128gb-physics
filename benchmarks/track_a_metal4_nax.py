"""
Track A Standalone Benchmark Runner (`experiments/track_a_metal4_nax/run_bench.py`).

Executes the complete Track A Silicon Physics & Metal 4 Hardware NAX contract:
  1. Cold-SLC (512 MB buffer flush), synchronized, interleaved A/B 32-layer Q4 microbenchmarks
     across shapes (4096x4096, 12288x4096, 4096x12288) and M in {1, 4, 8, 16, 32, 64, 128},
     comparing:
       - Stock MLX `mx.quantized_matmul`
       - `u4_direct` (Direct hardware `half x uint4b_format -> float` Split-K NAX, zero TG dequant!)
       - `tg_deq_nax` (128-bit/64-bit vectorized threadgroup-dequant Split-K NAX)
       - `grouped_i8i4` (Dynamic per-group INT8 + hardware `int8_t x int4b_format -> int32_t`)
       - `reg_deq_ctb` (Register-dequant `ct_b` ablation)
       - Zero-code workarounds: `pad65`, Fable's `reshape(4, 16, K)`, and stacked-N `gate+up` (24576x4096).
  2. Hardware Integer NAX Peak TOPS (`4096x4096x4096`) with 0-error exact integer verification:
       - `int8_t x int8_t -> int32_t`
       - `int8_t x int4b_format -> int32_t`
       - FP16 `mx.matmul` reference TFLOP/s
  3. Real `mlx-community/Qwen3.5-9B-4bit` end-to-end Verify K in {1, 4, 8, 16, 32, 64}
     with strict greedy token parity (`max_logit_diff`, `greedy_matches / K`).
  4. Automatic Pass-Mark / Falsifier audit written to `--out results.json`.

Usage:
  python3 \\
      experiments/track_a_metal4_nax/run_bench.py --out experiments/track_a_metal4_nax/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import mlx.core as mx
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
KERNELS_DIR = THIS_DIR.parent / "kernels"
for _d in (str(THIS_DIR), str(KERNELS_DIR)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import metal4_nax_kernels as nk
import patch_qwen9b_nax as pq


def flush_slc(flush_buf: mx.array) -> None:
    """Touch a 512 MB buffer and synchronize to evict M5 Max System Level Cache."""
    f = flush_buf + 1.0
    mx.eval(f)
    mx.synchronize()


def time_interleaved_ab(
    fn_a: Callable[[], None],
    fn_b: Callable[[], None],
    flush_buf: mx.array,
    iters: int = 5,
    warmup: int = 2,
) -> Tuple[List[float], List[float]]:
    """Run interleaved A/B trials (AB, BA, ...) with 512 MB SLC flush before every timed call."""
    for _ in range(warmup):
        fn_a()
        fn_b()
    mx.synchronize()

    a_ms: List[float] = []
    b_ms: List[float] = []
    for it in range(iters):
        order = ("a", "b") if (it % 2 == 0) else ("b", "a")
        for arm in order:
            flush_slc(flush_buf)
            t0 = time.perf_counter()
            if arm == "a":
                fn_a()
                mx.synchronize()
                a_ms.append((time.perf_counter() - t0) * 1e3)
            else:
                fn_b()
                mx.synchronize()
                b_ms.append((time.perf_counter() - t0) * 1e3)
    return a_ms, b_ms


def _med(xs: List[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


def bench_integer_nax_tops(iters: int = 10, warmup: int = 3) -> Dict[str, Any]:
    """Verify 0-error exactness and measure peak hardware INT8xINT8 & INT8xINT4 NAX TOPS."""
    # 1. Exact 0-error verification on non-trivial shape
    rng = np.random.default_rng(2026)
    M_v, N_v, K_v = 128, 128, 256
    a_v = rng.integers(-60, 60, size=(M_v, K_v), dtype=np.int8)
    b8_v = rng.integers(-60, 60, size=(N_v, K_v), dtype=np.int8)
    b4_v = rng.integers(-8, 8, size=(N_v, K_v), dtype=np.int8)
    b4_u = (b4_v & 0xF).astype(np.uint8)
    b4_u32 = (b4_u[:, 0::2] | (b4_u[:, 1::2] << 4)).astype(np.uint8).view(np.uint32)

    out88 = nk.nax_int8_int8_exact(mx.array(a_v), mx.array(b8_v))
    out84 = nk.nax_int8_int4_exact(mx.array(a_v), mx.array(b4_u32))
    mx.eval(out88, out84)
    err88 = int(np.max(np.abs(np.array(out88) - (a_v.astype(np.int32) @ b8_v.astype(np.int32).T))))
    err84 = int(np.max(np.abs(np.array(out84) - (a_v.astype(np.int32) @ b4_v.astype(np.int32).T))))
    assert err88 == 0, f"int8xint8 exact error {err88} != 0"
    assert err84 == 0, f"int8xint4 exact error {err84} != 0"

    # 2. Compute-bound peak TOPS on 4096 x 4096 x 4096
    M = N = K = 4096
    flops = 2.0 * M * N * K
    a_i8 = mx.random.randint(-60, 60, (M, K)).astype(mx.int8)
    b_i8 = mx.random.randint(-60, 60, (N, K)).astype(mx.int8)
    b_i4 = mx.random.randint(0, 2**31 - 1, (N, K // 8)).astype(mx.uint32)
    a_f16 = mx.random.normal((M, K)).astype(mx.float16)
    b_f16 = mx.random.normal((K, N)).astype(mx.float16)
    mx.eval(a_i8, b_i8, b_i4, a_f16, b_f16)
    mx.synchronize()

    def _measure(fn):
        for _ in range(warmup):
            out = fn()
            mx.eval(out)
        mx.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = fn()
            mx.eval(out)
            mx.synchronize()
            ts.append(time.perf_counter() - t0)
        med = _med(ts)
        mn = min(ts)
        return {
            "ms_median": round(med * 1e3, 3),
            "ms_min": round(mn * 1e3, 3),
            "tops_median": round((flops / med) / 1e12, 2),
            "tops_peak": round((flops / mn) / 1e12, 2),
        }

    # Sweep BK=64 vs BK=128 for peak integer TOPS
    res_i8i8_64 = _measure(lambda: nk.nax_int8_int8_exact(a_i8, b_i8, BM=64, BN=64, BK=64, NSG=4))
    res_i8i4_64 = _measure(lambda: nk.nax_int8_int4_exact(a_i8, b_i4, BM=64, BN=64, BK=64, NSG=4))
    res_i8i4_128 = _measure(lambda: nk.nax_int8_int4_exact(a_i8, b_i4, BM=64, BN=64, BK=128, NSG=4))
    res_fp16 = _measure(lambda: mx.matmul(a_f16, b_f16))

    return {
        "shape": [M, N, K],
        "exact_int8_int8_max_err": err88,
        "exact_int8_int4_max_err": err84,
        "int8_x_int8_bk64": res_i8i8_64,
        "int8_x_int4_bk64": res_i8i4_64,
        "int8_x_int4_bk128": res_i8i4_128,
        "fp16_mlx_matmul_ref": res_fp16,
    }


def bench_cold_32layer_micro(
    flush_buf: mx.array,
    shapes: List[Tuple[int, int]],
    m_values: List[int],
    dtype: mx.Dtype = mx.float16,
    n_layers: int = 32,
    iters: int = 5,
    warmup: int = 1,
) -> Dict[str, Any]:
    """
    Cold-SLC 32-layer Q4 microbenchmarks across shapes (N, K) and M values.
    Verifies max_abs_err vs FP32 golden reference for every single variant before timing!
    """
    mx.random.seed(42)
    out: Dict[str, Any] = {
        "n_layers": n_layers,
        "dtype": str(dtype),
        "iters": iters,
        "shapes": {},
    }

    for N, K in shapes:
        shape_key = f"{N}x{K}"
        # Create n_layers independent Q4 weight matrices (cold DRAM streaming!)
        layers = []
        layers_i4 = []
        for _ in range(n_layers):
            wf = mx.random.normal((N, K)).astype(dtype) * 0.04
            wq, sc, bi = mx.quantize(wf, group_size=64, bits=4, mode="affine")
            layers.append((wq, sc, bi))
            layers_i4.append(nk.convert_mlx_u4_to_i4(wq, sc, bi))
        mx.eval(layers, layers_i4)

        # Reference FP32 weight for layer 0 accuracy verification
        w0_deq = mx.dequantize(layers[0][0], layers[0][1], layers[0][2], group_size=64, bits=4, mode="affine").astype(mx.float32)
        mx.eval(w0_deq)

        # Bytes per 32-layer pass (packed 4-bit weights + FP16/BF16 scales & biases)
        bytes_per_layer = (N * K // 2) + 2 * (N * (K // 64) * 2)
        total_weight_mb = (bytes_per_layer * n_layers) / (1024 * 1024)

        shape_res: Dict[str, Any] = {
            "N": N,
            "K": K,
            "weight_bytes_32L_MB": round(total_weight_mb, 2),
            "by_M": {},
        }

        for M in m_values:
            x = mx.random.normal((M, K)).astype(dtype) * 0.2
            ref0 = x.astype(mx.float32) @ w0_deq.T
            mx.eval(x, ref0)

            def _run_stock():
                outs = [
                    mx.quantized_matmul(x, wq, scales=sc, biases=bi, transpose=True, group_size=64, bits=4, mode="affine")
                    for wq, sc, bi in layers
                ]
                mx.eval(*outs)

            # Build candidate variants for this (M, N, K)
            BM = 8 if M <= 8 else (16 if M <= 16 else (32 if M <= 32 else 64))
            WM = 1 if BM <= 16 else (2 if BM == 32 else 4)

            # Sweep best (BN, split_k) candidates cleanly
            cand_specs: List[Tuple[str, Callable[[mx.array, Tuple], mx.array]]] = []

            # 1. Direct hardware uint4b_format NAX (`u4_direct`)
            sk_u4_list = [4] if N == 4096 else ([2] if N >= 12288 else [4])
            if M == 64 and N == 4096:
                sk_u4_list = [2, 4]
            elif M == 16 and N == 4096:
                sk_u4_list = [4]
            for sk in sk_u4_list:
                cand_specs.append((
                    f"u4_direct_bn32_sk{sk}",
                    lambda inp, ly, _BM=BM, _WM=WM, _sk=sk: nk.qmm_u4_direct(
                        inp, ly[0], ly[1], ly[2], BM=_BM, BN=32, WM=_WM, WN=1, split_k=_sk, atomic=False
                    ),
                ))
            if M in (16, 32, 64) and N == 4096:
                cand_specs.append((
                    "u4_direct_atomic_bn32_sk4",
                    lambda inp, ly, _BM=BM, _WM=WM: nk.qmm_u4_direct(
                        inp, ly[0], ly[1], ly[2], BM=_BM, BN=32, WM=_WM, WN=1, split_k=4, atomic=True
                    ),
                ))

            # 2. Vectorized threadgroup-dequant Split-K NAX (`tg_deq_nax`)
            tg_configs = []
            if N == 4096 and K == 4096:
                if M <= 16:
                    tg_configs = [(16, 64, 2, False), (32, 64, 4, False)]
                elif M == 32:
                    tg_configs = [(32, 64, 2, True), (32, 64, 4, False)]
                elif M == 64:
                    tg_configs = [(32, 64, 2, False), (64, 64, 2, False)]
                else:
                    tg_configs = [(32, 64, 2, False), (64, 128, 1, False)]
            elif N == 4096 and K >= 12288:
                if M <= 16:
                    tg_configs = [(32, 64, 4, False)]
                elif M == 32:
                    tg_configs = [(64, 64, 2, False), (32, 64, 4, False)]
                else:
                    tg_configs = [(64, 128, 1, False), (64, 128, 2, False)]
            else:
                # Wide-N (12288x4096): BN=64 BK=128 W=WMx2 wins!
                tg_configs = [(64, 128, 1, False), (64, 128, 2, False)]

            for bn, bk, sk, db in tg_configs:
                wn = 2 if bn == 64 else 1
                cand_specs.append((
                    f"tg_deq_bn{bn}_bk{bk}_sk{sk}" + ("_db" if db else ""),
                    lambda inp, ly, _BM=BM, _bn=bn, _bk=bk, _WM=WM, _wn=wn, _sk=sk, _db=db: nk.qmm_tg_deq_nax(
                        inp, ly[0], ly[1], ly[2], BM=_BM, BN=_bn, BK=_bk, WM=_WM, WN=_wn, split_k=_sk, atomic=False, double_buf=_db
                    ),
                ))

            # 3. Auto-dispatched winning schedule
            cand_specs.append((
                "auto_nax",
                lambda inp, ly: nk.fast_qmm_nax(inp, ly[0], ly[1], ly[2], backend="auto"),
            ))

            # 4. Zero-code workarounds (pad65 for M <= 64, reshape(4,16,K) at M=64)
            if M <= 64:
                cand_specs.append((
                    "workaround_pad65",
                    lambda inp, ly: nk.qmm_pad65(inp, ly[0], ly[1], ly[2]),
                ))
            if M == 64:
                cand_specs.append((
                    "workaround_reshape_4x16",
                    lambda inp, ly: nk.qmm_reshape_batch(inp, ly[0], ly[1], ly[2], splits=4),
                ))

            # 5. Register-dequant ct_b ablation (at M=32, 64 on 4096^2)
            if M in (32, 64) and (N, K) == (4096, 4096):
                cand_specs.append((
                    "ablation_reg_ctb_bn32_sk2",
                    lambda inp, ly, _BM=BM, _WM=WM: nk.qmm_reg_deq_ctb(
                        inp, ly[0], ly[1], ly[2], BM=_BM, BN=32, WM=_WM, WN=1, split_k=2
                    ),
                ))

            m_res: Dict[str, Any] = {"variants": {}}

            # Verify & benchmark each candidate interleaved against stock MLX
            for cname, cfn in cand_specs:
                y0 = cfn(x, layers[0])
                mx.eval(y0)
                err = float(mx.max(mx.abs(y0.astype(mx.float32) - ref0)).item())

                def _run_cand(_cfn=cfn):
                    outs = [_cfn(x, ly) for ly in layers]
                    mx.eval(*outs)

                s_ts, c_ts = time_interleaved_ab(_run_stock, _run_cand, flush_buf, iters=iters, warmup=warmup)
                s_med, c_med = _med(s_ts), _med(c_ts)
                s_min, c_min = min(s_ts), min(c_ts)
                gbps = (bytes_per_layer * n_layers / (c_med * 1e-3)) / 1e9
                status = "PASS" if err <= 2e-2 else "KILLED_ACCURACY"
                m_res["variants"][cname] = {
                    "max_abs_err_fp32": round(err, 6),
                    "stock_ms_median": round(s_med, 3),
                    "stock_ms_min": round(s_min, 3),
                    "cand_ms_median": round(c_med, 3),
                    "cand_ms_min": round(c_min, 3),
                    "speedup_median": round(s_med / c_med, 3),
                    "speedup_min": round(s_min / c_min, 3),
                    "effective_gbps_median": round(gbps, 1),
                    "status": status,
                }

            # 6. Grouped-Q4 dynamic INT8 x INT4 hardware NAX (on M in {16, 32, 64})
            if M in (16, 32, 64):
                sk_i = 4 if N == 4096 else 2
                y0_i = nk.qmm_grouped_i8i4(x, layers_i4[0][0], layers_i4[0][1], layers_i4[0][2], BM=BM, BN=32, WM=WM, WN=1, split_k=sk_i)
                mx.eval(y0_i)
                err_i = float(mx.max(mx.abs(y0_i.astype(mx.float32) - ref0)).item())

                def _run_i84(_BM=BM, _WM=WM, _sk=sk_i):
                    outs = [
                        nk.qmm_grouped_i8i4(x, ly[0], ly[1], ly[2], BM=_BM, BN=32, WM=_WM, WN=1, split_k=_sk)
                        for ly in layers_i4
                    ]
                    mx.eval(*outs)

                s_ts, c_ts = time_interleaved_ab(_run_stock, _run_i84, flush_buf, iters=iters, warmup=warmup)
                s_med, c_med = _med(s_ts), _med(c_ts)
                m_res["variants"][f"grouped_i8i4_bn32_sk{sk_i}"] = {
                    "max_abs_err_fp32": round(err_i, 6),
                    "stock_ms_median": round(s_med, 3),
                    "cand_ms_median": round(c_med, 3),
                    "cand_ms_min": round(min(c_ts), 3),
                    "speedup_median": round(s_med / c_med, 3),
                    "effective_gbps_median": round((bytes_per_layer * n_layers / (c_med * 1e-3)) / 1e9, 1),
                    "status": "PASS" if err_i <= 2e-2 else "KILLED_ACCURACY",
                }

            # Pick best accurate variant for this (M, N, K)
            valid_v = {k: v for k, v in m_res["variants"].items() if v["status"] == "PASS"}
            best_name = min(valid_v, key=lambda k: valid_v[k]["cand_ms_median"])
            m_res["best_variant"] = best_name
            m_res["best_ms_median"] = valid_v[best_name]["cand_ms_median"]
            m_res["best_speedup_median"] = valid_v[best_name]["speedup_median"]
            shape_res["by_M"][str(M)] = m_res
            print(
                f"  [{shape_key} M={M:3d}] stock={valid_v[best_name]['stock_ms_median']:5.2f} ms | "
                f"best={best_name} ({valid_v[best_name]['cand_ms_median']:5.2f} ms, "
                f"{valid_v[best_name]['speedup_median']:.2f}x, err={valid_v[best_name]['max_abs_err_fp32']:.5f})"
            )

        out["shapes"][shape_key] = shape_res

    # 7. Stacked-N gate+up test (16 layers of 2 x 12288x4096 vs 16 layers of 24576x4096)
    print("\n--- Stacked-N (gate+up 24576x4096 vs 2x 12288x4096, 16 layers = same 32 matrices) ---")
    n_mlp = 16
    gates = [mx.quantize(mx.random.normal((12288, 4096)).astype(dtype) * 0.04, 64, 4, "affine") for _ in range(n_mlp)]
    ups = [mx.quantize(mx.random.normal((12288, 4096)).astype(dtype) * 0.04, 64, 4, "affine") for _ in range(n_mlp)]
    stacked = [
        (mx.concatenate([g[0], u[0]], axis=0), mx.concatenate([g[1], u[1]], axis=0), mx.concatenate([g[2], u[2]], axis=0))
        for g, u in zip(gates, ups)
    ]
    mx.eval(gates, ups, stacked)

    stacked_res: Dict[str, Any] = {}
    for M in [16, 32, 64]:
        x = mx.random.normal((M, 4096)).astype(dtype) * 0.2
        mx.eval(x)

        def _sep_stock():
            outs = []
            for g, u in zip(gates, ups):
                outs.append(mx.quantized_matmul(x, g[0], scales=g[1], biases=g[2], transpose=True, group_size=64, bits=4))
                outs.append(mx.quantized_matmul(x, u[0], scales=u[1], biases=u[2], transpose=True, group_size=64, bits=4))
            mx.eval(*outs)

        def _cat_stock():
            outs = [mx.quantized_matmul(x, s[0], scales=s[1], biases=s[2], transpose=True, group_size=64, bits=4) for s in stacked]
            mx.eval(*outs)

        def _cat_nax():
            outs = [nk.fast_qmm_nax(x, s[0], s[1], s[2], backend="tg_deq") for s in stacked]
            mx.eval(*outs)

        s_ts, cs_ts = time_interleaved_ab(_sep_stock, _cat_stock, flush_buf, iters=iters, warmup=warmup)
        s2_ts, cn_ts = time_interleaved_ab(_sep_stock, _cat_nax, flush_buf, iters=iters, warmup=warmup)
        stacked_res[str(M)] = {
            "sep_stock_ms_median": round(_med(s_ts), 3),
            "stacked_stock_ms_median": round(_med(cs_ts), 3),
            "stacked_stock_speedup": round(_med(s_ts) / _med(cs_ts), 3),
            "stacked_nax_ms_median": round(_med(cn_ts), 3),
            "stacked_nax_speedup": round(_med(s2_ts) / _med(cn_ts), 3),
        }
        print(
            f"  [gate+up M={M:2d}] sep_stock={stacked_res[str(M)]['sep_stock_ms_median']:.2f} ms | "
            f"stacked_stock={stacked_res[str(M)]['stacked_stock_ms_median']:.2f} ms ({stacked_res[str(M)]['stacked_stock_speedup']:.2f}x) | "
            f"stacked_nax={stacked_res[str(M)]['stacked_nax_ms_median']:.2f} ms ({stacked_res[str(M)]['stacked_nax_speedup']:.2f}x)"
        )

    out["stacked_gate_up_16L"] = stacked_res
    return out


def compute_verdicts(results: Dict[str, Any]) -> Dict[str, Any]:
    """Audit all Track A contract pass marks honestly."""
    micro = results["cold_32layer_micro"]["shapes"]
    m32 = micro.get("4096x4096", {}).get("by_M", {}).get("32", {})
    m64 = micro.get("4096x4096", {}).get("by_M", {}).get("64", {})
    qwen = results.get("qwen35_9b_verify", {}).get("variants", {})

    best_q32_ms = None
    best_q32_name = None
    best_q32_exact = False
    for vname, vr in qwen.items():
        r32 = vr.get("per_k", {}).get("32")
        if r32 and r32.get("greedy_exact"):
            if best_q32_ms is None or r32["variant_ms_median"] < best_q32_ms:
                best_q32_ms = r32["variant_ms_median"]
                best_q32_name = vname
                best_q32_exact = True

    int_res = results["hardware_integer_nax"]
    verdicts = {
        "4096x4096_M64_le_2.45ms": {
            "target_ms": 2.45,
            "measured_ms": m64.get("best_ms_median"),
            "speedup": m64.get("best_speedup_median"),
            "best_variant": m64.get("best_variant"),
            "pass": (m64.get("best_ms_median", 999) <= 2.45),
        },
        "4096x4096_M32_le_1.58ms": {
            "target_ms": 1.58,
            "measured_ms": m32.get("best_ms_median"),
            "speedup": m32.get("best_speedup_median"),
            "best_variant": m32.get("best_variant"),
            "pass": (m32.get("best_ms_median", 999) <= 1.58),
        },
        "hardware_int8_int4_exact_zero_err": {
            "int8_int8_err": int_res["exact_int8_int8_max_err"],
            "int8_int4_err": int_res["exact_int8_int4_max_err"],
            "int8_int8_peak_tops": int_res["int8_x_int8_bk64"]["tops_peak"],
            "int8_int4_peak_tops": max(
                int_res["int8_x_int4_bk64"]["tops_peak"],
                int_res["int8_x_int4_bk128"]["tops_peak"],
            ),
            "pass": (int_res["exact_int8_int8_max_err"] == 0 and int_res["exact_int8_int4_max_err"] == 0),
        },
        "qwen35_9b_verify_K32_le_25ms_exact_greedy": {
            "target_ms": 25.0,
            "measured_ms": best_q32_ms,
            "best_variant": best_q32_name,
            "greedy_exact": best_q32_exact,
            "pass": (best_q32_ms is not None and best_q32_ms <= 25.0 and best_q32_exact),
        },
    }
    return verdicts


def main():
    parser = argparse.ArgumentParser(description="Track A Metal 4 NAX Benchmark Runner")
    parser.add_argument("--out", default=str(THIS_DIR / "results.json"))
    parser.add_argument("--quick", action="store_true", help="Short smoke run (fewer iters/shapes)")
    parser.add_argument("--skip-qwen", action="store_true", help="Skip Qwen3.5-9B model benchmark")
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    iters = 2 if args.quick else args.iters
    warmup = 1 if args.quick else 2
    shapes = [(4096, 4096)] if args.quick else [(4096, 4096), (12288, 4096), (4096, 12288)]
    m_vals = [16, 32, 64] if args.quick else [1, 4, 8, 16, 32, 64, 128]
    k_vals = (1, 16, 32, 64) if args.quick else (1, 4, 8, 16, 32, 64)

    flush_buf = mx.zeros((128 * 1024 * 1024,), dtype=mx.float32)  # 512 MB SLC flush buffer
    mx.eval(flush_buf)

    t_start = time.perf_counter()
    print("=== [1/3] Hardware Integer NAX Peak TOPS (int8xint8, int8xint4) ===")
    int_res = bench_integer_nax_tops(iters=3 if args.quick else 8, warmup=warmup)
    print(
        f"  int8xint8: {int_res['int8_x_int8_bk64']['tops_peak']} TOPS peak (err={int_res['exact_int8_int8_max_err']})\n"
        f"  int8xint4: bk64={int_res['int8_x_int4_bk64']['tops_peak']} TOPS, "
        f"bk128={int_res['int8_x_int4_bk128']['tops_peak']} TOPS (err={int_res['exact_int8_int4_max_err']})\n"
        f"  fp16 ref : {int_res['fp16_mlx_matmul_ref']['tops_peak']} TFLOP/s peak"
    )

    print("\n=== [2/3] Cold-SLC 32-Layer Q4 Microbenchmarks ===")
    micro_res = bench_cold_32layer_micro(
        flush_buf, shapes=shapes, m_values=m_vals, dtype=mx.float16, n_layers=32, iters=iters, warmup=warmup
    )

    qwen_res: Dict[str, Any] = {}
    if not args.skip_qwen:
        print("\n=== [3/3] Real Qwen3.5-9B-4bit Verify K=1..64 ===")
        q_variants = (
            [
                {"name": "tg_deq_nax", "backend": "tg_deq", "min_m": 8, "max_m": 64, "stack_gate_up": False, "patch_lm_head": True},
                {"name": "u4_direct_nax", "backend": "u4_direct", "min_m": 8, "max_m": 64, "stack_gate_up": False, "patch_lm_head": True},
            ]
            if args.quick
            else None
        )
        qwen_res = pq.benchmark_qwen9b_verify(
            model_id="mlx-community/Qwen3.5-9B-4bit",
            k_values=k_vals,
            variants=q_variants,
            iters=iters,
            warmup=warmup,
            flush_slc=True,
        )
        for vname, vr in qwen_res["variants"].items():
            print(f"\n  Variant: {vname}")
            for K in qwen_res["k_values"]:
                r = vr["per_k"][str(K)]
                print(
                    f"    K={K:2d}: stock={r['stock_ms_median']:6.2f} ms | {vname}={r['variant_ms_median']:6.2f} ms "
                    f"({r['speedup_median']:.2f}x) | max_logit_diff={r['max_logit_diff']:.4f} | "
                    f"greedy={r['greedy_matches']}/{r['greedy_total']}"
                )

    results = {
        "track": "Track A — Silicon Physics & Metal 4 Hardware NAX",
        "device": mx.device_info(),
        "mlx_version": mx.__version__,
        "quick_mode": args.quick,
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "hardware_integer_nax": int_res,
        "cold_32layer_micro": micro_res,
        "qwen35_9b_verify": qwen_res,
    }
    results["verdicts"] = compute_verdicts(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved Track A results to {out_path} ({results['elapsed_sec']}s)")
    print("Verdicts:", json.dumps(results["verdicts"], indent=2))


if __name__ == "__main__":
    main()
