#!/usr/bin/env python3
"""
Track E Master Runner — Wild Silicon Frontier (Apple M5 Max)

Usage:
  python3 experiments/track_e_silicon_frontier/run_bench.py \
      --out experiments/track_e_silicon_frontier/results.json [--quick] [--skip-models]

Executes all 4 Track E silicon characterization modules sequentially on an idle GPU:
  1. Pure-Read DRAM Bandwidth Ceiling (2-4 GB) + Empirical SLC Sweep (4 MB .. 4096 MB)
  2. Native C / ARMv9 SME2 Peak Scaling (1..18 threads), Accelerate/BNNS, CPU DRAM/L2,
     L2-resident Draft Heads, + Concurrent CPU Draft vs GPU 2 GB Pure-Read Interference
  3. Metal Dispatch Tax (2,000 dependent tiny kernels), Live Qwen3.5-9B & Qwen3.6-35B-A3B
     Exact Op/Dispatch Audit, Fable x[4,16,4096] Reshape Test, Dense FP16 & Iso-Byte Controls
  4. Real Apple Neural Engine (ANE) / CoreML Probe via zero-dependency FP16 protobuf +
     MLComputePlan verification + Concurrent ANE vs GPU DRAM Contention
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time

_MLX_PY = "python3"
try:
    import mlx.core as mx
except ImportError:
    if os.path.exists(_MLX_PY) and os.path.realpath(sys.executable) != os.path.realpath(_MLX_PY):
        os.execv(_MLX_PY, [_MLX_PY] + sys.argv)
    raise

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ane_probe  # noqa: E402
import dispatch_tax_and_controls  # noqa: E402
import pure_read_bw  # noqa: E402

KERNELS_DIR = os.path.join(os.path.dirname(HERE), "kernels")
SME2_SRC = os.path.join(KERNELS_DIR, "sme2_probe.c")
SME2_BIN = os.path.join(KERNELS_DIR, "sme2_probe")
PURE_READ_PY = os.path.join(HERE, "pure_read_bw.py")


def ensure_sme2_compiled() -> None:
    if not os.path.exists(SME2_BIN) or os.path.getmtime(SME2_SRC) > os.path.getmtime(SME2_BIN):
        subprocess.run(
            [
                "clang", "-O3", "-mcpu=native", "-DACCELERATE_NEW_LAPACK",
                "-Wno-deprecated-declarations",
                "-framework", "Accelerate", "-lpthread",
                SME2_SRC, "-o", SME2_BIN,
            ],
            check=True,
        )


def run_sme2_suite(*, quick: bool = False) -> dict:
    ensure_sme2_compiled()
    cmd = [SME2_BIN] + (["--quick"] if quick else [])
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(r.stdout)

    # Concurrent CPU L2-resident INT8 draft MLP + GPU 2 GB Pure-Read DRAM streaming test
    sec = 2.0 if quick else 4.0
    cpu_cmd_512 = [SME2_BIN, "--concurrent-draft", str(sec), "--draft-h", "512"]
    cpu_cmd_1024 = [SME2_BIN, "--concurrent-draft", str(sec), "--draft-h", "1024"]
    gpu_cmd = [sys.executable, PURE_READ_PY, "--stream-seconds", str(sec)]

    def _run_conc(cpu_cmd: list[str]) -> dict:
        cpu_solo = json.loads(subprocess.run(cpu_cmd, capture_output=True, text=True, check=True).stdout)
        gpu_solo = json.loads(subprocess.run(gpu_cmd, capture_output=True, text=True, check=True).stdout)
        p_cpu = subprocess.Popen(cpu_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        p_gpu = subprocess.Popen(gpu_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        c_out, c_err = p_cpu.communicate()
        g_out, g_err = p_gpu.communicate()
        cpu_conc = json.loads(c_out) if p_cpu.returncode == 0 else {"error": c_err}
        gpu_conc = json.loads(g_out) if p_gpu.returncode == 0 else {"error": g_err}
        return {
            "cpu_solo": cpu_solo,
            "gpu_solo": gpu_solo,
            "cpu_concurrent": cpu_conc,
            "gpu_concurrent": gpu_conc,
            "gpu_bw_retention_pct": round(
                100.0 * gpu_conc["gpu_pure_read_gb_s"] / gpu_solo["gpu_pure_read_gb_s"], 2
            ) if "gpu_pure_read_gb_s" in gpu_conc else None,
            "cpu_latency_ratio_concurrent_vs_solo": round(
                cpu_conc["mean_us"] / cpu_solo["mean_us"], 3
            ) if "mean_us" in cpu_conc else None,
        }

    data["concurrent_gpu_interference"] = {
        "draft_mlp_4096_512_4096_int8_4p2MB": _run_conc(cpu_cmd_512),
        "draft_mlp_4096_1024_4096_int8_8p4MB": _run_conc(cpu_cmd_1024),
    }
    return data


def build_executive_verdicts(results: dict) -> dict:
    pr = results["pure_read_and_slc"]["headline"]
    sme = results["cpu_sme2_and_l2"]
    dt = results["dispatch_tax_and_controls"]["metal_dispatch_tax"]
    models = results["dispatch_tax_and_controls"]["model_decode_op_audit"]
    fable = results["dispatch_tax_and_controls"]["fable_reshape_and_controls"]["verdicts"]
    ane = results["ane_coreml"]

    sme_1t = sme["sme2_peak_scaling"][0]
    sme_18t = sme["sme2_peak_scaling"][-1]
    ane_draft = ane["models"]["draft_mlp_s1"]["runs"].get("ane", {})
    ane_peak = max(
        (m["runs"].get("ane", {}).get("effective_tflops", 0.0) for m in ane["models"].values()),
        default=0.0,
    )

    verdicts = {
        "pure_read_dram_best_gb_s": pr["pure_read_best_gb_s"],
        "copy_rw_custom_gb_s": pr["copy_rw_custom_gb_s"],
        "pure_read_vs_copy_ratio": pr["pure_read_vs_copy_ratio"],
        "pure_read_pct_of_614p4_spec": pr["pure_read_vs_spec_614_4_pct"],
        "empirical_slc_peak_gb_s": pr["slc_peak_gb_s"],
        "empirical_slc_knee_mb": pr["estimated_slc_knee_mb"],
        "sme2_1t_fp32_fp16_bf16_int8": {
            "fp32_tflops": sme_1t["fp32_tflops"],
            "fp16_za32_tflops": sme_1t["fp16_za32_tflops"],
            "fp16_za16_tflops": sme_1t["fp16_za16_tflops"],
            "bf16_za32_tflops": sme_1t["bf16_za32_tflops"],
            "int8_tops": sme_1t["int8_tops"],
        },
        "sme2_18t_fp32_fp16_bf16_int8": {
            "fp32_tflops": sme_18t["fp32_tflops"],
            "fp16_za32_tflops": sme_18t["fp16_za32_tflops"],
            "fp16_za16_tflops": sme_18t["fp16_za16_tflops"],
            "bf16_za32_tflops": sme_18t["bf16_za32_tflops"],
            "int8_tops": sme_18t["int8_tops"],
        },
        "sme2_18t_over_1t_scaling_ratio": sme["sme2_inferred_unit_ratio_18t_over_1t"],
        "cpu_dram_read_peak_gb_s": max(sme["cpu_dram_read_bw_512MB_gb_s"].values()),
        "metal_dispatch_serial_us": dt["linear_fit_serial_us_per_dispatch"],
        "metal_dispatch_indep_us": dt["independent_2000_us_per_dispatch"],
        "fable_m64_4x16_ms_vs_flat": [fable["fable_m64_4x16_ms"], fable["fable_m64_flat_ms"]],
        "dense_fp16_has_m64_valley": fable["dense_fp16_has_m64_valley"],
        "ane_draft_mlp_s1_median_ms": ane_draft.get("median_ms"),
        "ane_peak_measured_fp16_tflops": ane_peak,
        "ane_dram_134MB_gpu_bw_retention_pct": ane["concurrent_gpu_contention"]["dram_4x4096_134MB_vs_gpu_2GB_read"]["gpu_bw_retention_pct"],
        "cpu_draft_4p2MB_gpu_bw_retention_pct": sme["concurrent_gpu_interference"]["draft_mlp_4096_512_4096_int8_4p2MB"]["gpu_bw_retention_pct"],
    }
    if "qwen36_35b_a3b_4bit" in models:
        m35 = models["qwen36_35b_a3b_4bit"]
        verdicts["qwen36_35b_a3b_gpu_kernels"] = m35["graph"]["estimated_gpu_kernel_dispatches"]
        verdicts["qwen36_35b_a3b_implied_serial_dispatch_ms"] = m35["implied_dispatch_tax_serial_ms"]
        verdicts["opus_35b_dispatch_gt_3ms_passed"] = m35["implied_dispatch_tax_serial_ms"] >= 3.0
    if "qwen35_9b_4bit" in models:
        m9 = models["qwen35_9b_4bit"]
        verdicts["qwen35_9b_gpu_kernels"] = m9["graph"]["estimated_gpu_kernel_dispatches"]
        verdicts["qwen35_9b_implied_serial_dispatch_ms"] = m9["implied_dispatch_tax_serial_ms"]
    return verdicts


def main() -> None:
    ap = argparse.ArgumentParser(description="Track E Master Silicon Frontier Benchmark Suite")
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "results.json"))
    ap.add_argument("--quick", action="store_true", help="Fast smoke run (<15s without models)")
    ap.add_argument("--skip-models", action="store_true", help="Skip loading 9B/35B HF models")
    args = ap.parse_args()

    t0 = time.perf_counter()
    print("[Track E] 1/4 Running Pure-Read DRAM & Empirical SLC Sweep...", flush=True)
    r_pr = pure_read_bw.run_all(quick=args.quick)

    print("[Track E] 2/4 Running Native CPU SME2, Accelerate/BNNS, L2/DRAM & Concurrent Draft...", flush=True)
    r_sme = run_sme2_suite(quick=args.quick)

    print("[Track E] 3/4 Running Metal Dispatch Tax, Qwen Op Audit, Fable Reshape & Controls...", flush=True)
    r_dt = dispatch_tax_and_controls.run_all(quick=args.quick, skip_models=args.skip_models)

    print("[Track E] 4/4 Running Apple Neural Engine (CoreML + MLComputePlan) & Contention Probe...", flush=True)
    r_ane = ane_probe.run_all(quick=args.quick)

    results = {
        "track": "E_silicon_frontier",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "quick_mode": args.quick,
        "skip_models": args.skip_models,
        "elapsed_total_s": round(time.perf_counter() - t0, 2),
        "environment": {
            "python": sys.executable,
            "platform": platform.platform(),
            "mlx_version": mx.__version__,
            "device_info": mx.device_info(),
        },
        "pure_read_and_slc": r_pr,
        "cpu_sme2_and_l2": r_sme,
        "dispatch_tax_and_controls": r_dt,
        "ane_coreml": r_ane,
    }
    results["executive_verdicts"] = build_executive_verdicts(results)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Track E] DONE in {results['elapsed_total_s']:.1f}s -> {args.out}")
    print(json.dumps(results["executive_verdicts"], indent=2))


if __name__ == "__main__":
    main()
