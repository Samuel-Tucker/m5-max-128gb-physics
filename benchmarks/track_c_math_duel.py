"""Standalone Track C runner: executes all 3 Direct Math Transfer Duels cleanly
on REAL `mlx-community/Qwen3.5-9B-4bit` with 540 MB SLC flush & synchronized timers.

Usage:
  python3 experiments/track_c_math_duel/run_bench.py --out experiments/track_c_math_duel/results.json
  python3 experiments/track_c_math_duel/run_bench.py --quick --out experiments/track_c_math_duel/smoke_results.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import platform
import time
from typing import Any, Dict

import mlx.core as mx
from mlx_lm import load

from common import MODEL_ID, SLCFlusher, clear_memory
from lm_head_cert_duel import run_lm_head_duel
from mlp_sign_pruning import run_mlp_duel
from spectral_inertia_kv import run_spectral_duel


def main():
    parser = argparse.ArgumentParser(description="Track C Direct Math Transfer Duel Runner")
    parser.add_argument(
        "--out",
        type=str,
        default="results/track_c_math_duel.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast correctness & smoke mode (reduced states/trials)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "duel1", "duel2", "duel3"],
        help="Run a single duel or all three",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    dev_info = mx.device_info()
    print("=" * 78)
    print(f"Track C: Direct Math Transfer Duel ({'QUICK SMOKE' if args.quick else 'FULL BENCHMARK'})")
    print(f"Model: {MODEL_ID} | MLX: {mx.__version__} | GPU: {dev_info.get('architecture')}")
    print("=" * 78)

    model, tokenizer = load(MODEL_ID)
    flusher = SLCFlusher(mb=540)

    results: Dict[str, Any] = {
        "track": "Track C — Direct Math Transfer Duel",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mode": "quick_smoke" if args.quick else "full_benchmark",
        "hardware": {
            "platform": platform.platform(),
            "mlx_version": mx.__version__,
            "gpu_architecture": dev_info.get("architecture"),
            "memory_gb": round(dev_info.get("memory_size", 0) / (1024 ** 3), 1),
            "slc_flush_mb": flusher.mb,
        },
        "model_id": MODEL_ID,
        "duels": {},
    }

    if args.only in ("all", "duel1"):
        results["duels"]["duel1_lm_head_cert"] = run_lm_head_duel(
            quick=args.quick, model=model, tokenizer=tokenizer, flusher=flusher
        )
        clear_memory()

    if args.only in ("all", "duel2"):
        results["duels"]["duel2_mlp_sign_pruning"] = run_mlp_duel(
            quick=args.quick, model=model, tokenizer=tokenizer, flusher=flusher
        )
        clear_memory()

    if args.only in ("all", "duel3"):
        results["duels"]["duel3_spectral_inertia_kv"] = run_spectral_duel(
            quick=args.quick, model=model, tokenizer=tokenizer, flusher=flusher
        )
        clear_memory()

    results["total_elapsed_sec"] = round(time.perf_counter() - t0, 2)
    results["verdicts"] = {
        k: v.get("verdict") for k, v in results["duels"].items()
    }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("TRACK C VERDICT SUMMARY:")
    for k, v in results["verdicts"].items():
        print(f"  {k:32s}: {v}")
    print(f"Total elapsed: {results['total_elapsed_sec']:.1f}s | Saved: {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
