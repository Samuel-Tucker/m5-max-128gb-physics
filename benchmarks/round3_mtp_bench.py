#!/usr/bin/env python3
"""
Round-3 Full 12-Sealed-Prompt Benchmark Suite:
Native Qwen3.5/3.6 MTP + Zero-Sync GPU Pipelining + True-ASCII Draft LM Head +
Hybrid Suffix+MTP + Hybrid Track A Metal 4 NAX Verifier on Idle Apple M5 Max GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import mlx.core as mx
import mlx_lm
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
KERNELS_DIR = THIS_DIR.parent / "kernels"
for d in (str(THIS_DIR), str(KERNELS_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)

from fast_draft_lm_head import (
    ActiveVocabSubHead,
    ExactDraftLMHead,
    build_true_ascii_vocab_indices,
)
from mtp_module import load_mtp_head
from mtp_spec_engine import (
    DeltaNetHiddenEngine,
    MTPSpecRunner,
    patch_qwen9b_hybrid_nax,
    run_serial_baseline,
)
from prompts_sealed import format_prompt_tokens, get_prompts
from proposers import PromptLookupProposer


def flush_slc(flusher: mx.array) -> None:
    s = mx.sum(flusher)
    mx.eval(s)
    mx.synchronize()


def _get_eos_ids(tokenizer: Any) -> set:
    eos = set()
    if hasattr(tokenizer, "eos_token_ids") and tokenizer.eos_token_ids:
        eos.update(int(x) for x in tokenizer.eos_token_ids)
    elif getattr(tokenizer, "eos_token_id", None) is not None:
        if isinstance(tokenizer.eos_token_id, (list, tuple, set)):
            eos.update(int(x) for x in tokenizer.eos_token_id)
        else:
            eos.add(int(tokenizer.eos_token_id))
    return eos


def summarize_arm(records: List[Dict[str, Any]], baseline_by_id: Dict[str, float]) -> Dict[str, Any]:
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)

    cat_summary: Dict[str, Any] = {}
    for cat, items in sorted(by_cat.items()):
        tps_vals = [x["tok_per_sec"] for x in items]
        sp_vals = [x["tok_per_sec"] / baseline_by_id[x["prompt_id"]] for x in items]
        tau_vals = [x.get("mean_tokens_per_verify", 1.0) for x in items]
        cat_summary[cat] = {
            "mean_tps": round(float(np.mean(tps_vals)), 2),
            "peak_tps": round(float(np.max(tps_vals)), 2),
            "mean_speedup": round(float(np.mean(sp_vals)), 3),
            "peak_speedup": round(float(np.max(sp_vals)), 3),
            "mean_tau": round(float(np.mean(tau_vals)), 3),
            "exact_matches": sum(1 for x in items if x.get("exact_match")),
            "total": len(items),
        }

    all_tps = [x["tok_per_sec"] for x in records]
    all_sp = [x["tok_per_sec"] / baseline_by_id[x["prompt_id"]] for x in records]
    all_tau = [x.get("mean_tokens_per_verify", 1.0) for x in records]
    return {
        "overall_mean_tps": round(float(np.mean(all_tps)), 2),
        "overall_mean_speedup": round(float(np.mean(all_sp)), 3),
        "overall_peak_tps": round(float(np.max(all_tps)), 2),
        "overall_peak_speedup": round(float(np.max(all_sp)), 3),
        "overall_mean_tau": round(float(np.mean(all_tau)), 3),
        "exact_matches": sum(1 for x in records if x.get("exact_match")),
        "total_prompts": len(records),
        "by_category": cat_summary,
    }


def run_suite_for_model(
    model_key: str,
    prompts_per_cat: int = 4,
    max_tokens: int = 128,
) -> Dict[str, Any]:
    if model_key == "35b":
        model_id = "mlx-community/Qwen3.6-35B-A3B-4bit"
        mtp_path = THIS_DIR.parent / "weights/qwen36_35b_mtp_q4.safetensors"
    else:
        model_id = "mlx-community/Qwen3.5-9B-4bit"
        mtp_path = THIS_DIR.parent / "weights/qwen35_9b_mtp_q4.safetensors"

    print(f"\n==================================================================", flush=True)
    print(f"ROUND 3 SUITE: {model_id} ({prompts_per_cat*3} prompts, {max_tokens} tok)", flush=True)
    print(f"==================================================================", flush=True)

    model, tokenizer = mlx_lm.load(model_id)
    mtp = load_mtp_head(model, mtp_path, quantized=True)
    engine = DeltaNetHiddenEngine(model)
    eos_ids = _get_eos_ids(tokenizer)
    V = engine.args.vocab_size

    exact_head = ExactDraftLMHead(model)
    ascii64k_ids = build_true_ascii_vocab_indices(tokenizer, V, max_rows=65536, include_latin1=False)
    ascii64k_head = ActiveVocabSubHead(model, ascii64k_ids, bits=4, label="true_ascii_64k_q4")
    print(
        f"Loaded MTP ({mtp_path.stat().st_size/1e6:.1f} MB) | "
        f"exact_head={exact_head.weight_bytes/1e6:.1f} MB | "
        f"ascii64k_head={ascii64k_head.weight_bytes/1e6:.1f} MB",
        flush=True,
    )

    flusher = mx.ones(((512 * 1024 * 1024) // 4,), dtype=mx.float32)
    mx.eval(flusher)

    prompts = get_prompts(prompts_per_cat=prompts_per_cat)
    prompt_toks_map = {p["id"]: format_prompt_tokens(tokenizer, p["prompt"]) for p in prompts}

    # Warm up Metal shaders (1 short throwaway run)
    _ = run_serial_baseline(engine, prompt_toks_map[prompts[0]["id"]], 8, eos_ids)
    _warm_runner = MTPSpecRunner(engine, mtp, draft_head=ascii64k_head, k_max=3, name="warmup")
    _ = _warm_runner.run_prompt(prompt_toks_map[prompts[0]["id"]], 12, eos_ids)
    mx.synchronize()

    # 1. Baseline serial B=1 across all 12 prompts
    print("\n--- Arm: baseline (serial B=1) ---", flush=True)
    base_records = []
    ref_tokens_by_id: Dict[str, List[int]] = {}
    base_tps_by_id: Dict[str, float] = {}
    for p in prompts:
        flush_slc(flusher)
        r = run_serial_baseline(engine, prompt_toks_map[p["id"]], max_tokens, eos_ids)
        r["prompt_id"] = p["id"]
        r["category"] = p["category"]
        ref_tokens_by_id[p["id"]] = r.pop("generated_tokens")
        base_tps_by_id[p["id"]] = r["tok_per_sec"]
        base_records.append(r)
        print(f"  {p['id']:14s} ({p['category']:10s}): {r['tok_per_sec']:6.2f} tok/s ({r['tokens_generated']} tok)", flush=True)

    results: Dict[str, Any] = {
        "model_id": model_id,
        "mtp_weights_mb": round(mtp_path.stat().st_size / 1e6, 2),
        "exact_lm_head_mb": round(exact_head.weight_bytes / 1e6, 2),
        "ascii64k_lm_head_mb": round(ascii64k_head.weight_bytes / 1e6, 2),
        "max_tokens": max_tokens,
        "prompts_per_cat": prompts_per_cat,
        "arms": {"baseline": {"summary": summarize_arm(base_records, base_tps_by_id), "records": base_records}},
    }

    # Define arms per model
    if model_key == "35b":
        stock_arms = [
            MTPSpecRunner(engine, mtp, draft_head=exact_head, k_max=2, name="mtp_k2_exact"),
            MTPSpecRunner(engine, mtp, draft_head=exact_head, k_max=3, name="mtp_k3_exact"),
            MTPSpecRunner(engine, mtp, draft_head=ascii64k_head, k_max=2, name="mtp_k2_ascii64k"),
            MTPSpecRunner(engine, mtp, draft_head=ascii64k_head, k_max=3, name="mtp_k3_ascii64k"),
            MTPSpecRunner(engine, mtp, draft_head=ascii64k_head, k_max=4, name="mtp_k4_ascii64k"),
            MTPSpecRunner(engine, mtp, draft_head=ascii64k_head, k_max=6, p_min_chain=0.50, name="mtp_conf_ascii64k"),
            MTPSpecRunner(
                engine,
                mtp,
                draft_head=ascii64k_head,
                k_max=3,
                suffix_proposer=PromptLookupProposer(
                    k_max=7, min_order=4, max_order=6, adaptive=True,
                    k_by_order={6: 7, 5: 7, 4: 5},
                ),
                suffix_min_len=5,
                name="hybrid_suffix_o4_mtp_35b",
            ),
        ]
        nax_arms: List[MTPSpecRunner] = []
    else:
        stock_arms = [
            MTPSpecRunner(engine, mtp, draft_head=exact_head, k_max=2, name="mtp_k2_exact"),
            MTPSpecRunner(engine, mtp, draft_head=exact_head, k_max=3, name="mtp_k3_exact"),
            MTPSpecRunner(engine, mtp, draft_head=ascii64k_head, k_max=2, name="mtp_k2_ascii64k"),
            MTPSpecRunner(engine, mtp, draft_head=ascii64k_head, k_max=3, name="mtp_k3_ascii64k"),
            MTPSpecRunner(engine, mtp, draft_head=ascii64k_head, k_max=4, name="mtp_k4_ascii64k"),
            MTPSpecRunner(
                engine,
                mtp,
                draft_head=ascii64k_head,
                k_max=3,
                suffix_proposer=PromptLookupProposer(
                    k_max=7, min_order=4, max_order=6, adaptive=True,
                    k_by_order={6: 7, 5: 7, 4: 5},
                ),
                suffix_min_len=5,
                name="hybrid_suffix_o4_mtp_9b_bitexact",
            ),
        ]
        nax_arms = [
            MTPSpecRunner(engine, mtp, draft_head=ascii64k_head, k_max=5, name="mtp_k5_ascii64k_nax"),
            MTPSpecRunner(engine, mtp, draft_head=ascii64k_head, k_max=7, name="mtp_k7_ascii64k_nax"),
            MTPSpecRunner(engine, mtp, draft_head=ascii64k_head, k_max=8, p_min_chain=0.55, name="mtp_conf_ascii64k_nax"),
            MTPSpecRunner(
                engine,
                mtp,
                draft_head=ascii64k_head,
                k_max=2,
                suffix_proposer=PromptLookupProposer(
                    k_max=15, min_order=4, max_order=6, adaptive=True,
                    k_by_order={6: 15, 5: 15, 4: 11},
                ),
                suffix_min_len=6,
                name="hybrid_suffix_o4_mtp_k2_9b_nax",
            ),
            MTPSpecRunner(
                engine,
                mtp,
                draft_head=ascii64k_head,
                k_max=3,
                suffix_proposer=PromptLookupProposer(
                    k_max=15, min_order=4, max_order=6, adaptive=True,
                    k_by_order={6: 15, 5: 15, 4: 11},
                ),
                suffix_min_len=6,
                name="hybrid_suffix_o4_mtp_k3_9b_nax",
            ),
        ]

    def _run_arm_list(arm_list: List[MTPSpecRunner]) -> None:
        for runner in arm_list:
            print(f"\n--- Arm: {runner.name} ---", flush=True)
            recs = []
            for p in prompts:
                flush_slc(flusher)
                r = runner.run_prompt(
                    prompt_toks_map[p["id"]],
                    max_tokens=max_tokens,
                    eos_ids=eos_ids,
                    ref_tokens=ref_tokens_by_id[p["id"]],
                )
                r.pop("generated_tokens", None)
                r["prompt_id"] = p["id"]
                r["category"] = p["category"]
                sp = r["tok_per_sec"] / base_tps_by_id[p["id"]]
                r["speedup"] = round(sp, 3)
                recs.append(r)
                eq = "EXACT" if r["exact_match"] else f"DIFF@{r['first_mismatch']}"
                print(
                    f"  {p['id']:14s} ({p['category']:10s}): {r['tok_per_sec']:6.2f} tok/s "
                    f"({sp:4.2f}x) | tau={r['mean_tokens_per_verify']:.2f} | "
                    f"acc={r['accept_rate']*100:4.1f}% | mtp={r['mtp_turns']} suf={r['suffix_turns']} | {eq}",
                    flush=True,
                )
            summ = summarize_arm(recs, base_tps_by_id)
            results["arms"][runner.name] = {"summary": summ, "records": recs}
            bc = summ["by_category"]
            print(
                f"  >>> SUMMARY {runner.name}: OVERALL={summ['overall_mean_tps']:.1f} tok/s "
                f"({summ['overall_mean_speedup']:.2f}x) | "
                f"code={bc['code_edit']['mean_tps']:.1f} ({bc['code_edit']['mean_speedup']:.2f}x) | "
                f"ext={bc['extraction']['mean_tps']:.1f} ({bc['extraction']['mean_speedup']:.2f}x) | "
                f"PROSE={bc['free_prose']['mean_tps']:.1f} ({bc['free_prose']['mean_speedup']:.2f}x) | "
                f"exact={summ['exact_matches']}/{summ['total_prompts']}",
                flush=True,
            )

    _run_arm_list(stock_arms)

    if nax_arms:
        patch_info = patch_qwen9b_hybrid_nax(model, min_tg=6, min_u4=16)
        results["hybrid_nax_patch"] = {k: v for k, v in patch_info.items() if k != "restore"}
        print(f"\n[Hybrid Track A NAX patched {patch_info['patched_count']} layers: tg_deq>=6, u4_direct>=16]", flush=True)
        # Warm up NAX shaders (V=6 tg_deq and V=16 u4_direct)
        _ = nax_arms[0].run_prompt(prompt_toks_map[prompts[0]["id"]], 16, eos_ids)
        _ = nax_arms[-1].run_prompt(prompt_toks_map[prompts[0]["id"]], 24, eos_ids)
        mx.synchronize()
        _run_arm_list(nax_arms)
        patch_info["restore"]()

    del model, mtp, exact_head, ascii64k_head, flusher
    gc.collect()
    mx.clear_cache()
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["35b", "9b"], choices=["35b", "9b"])
    ap.add_argument("--prompts-per-cat", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out-dir", type=str, default="results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mk in args.models:
        res = run_suite_for_model(mk, prompts_per_cat=args.prompts_per_cat, max_tokens=args.max_tokens)
        out_path = out_dir / f"results_{mk}_round3.json"
        out_path.write_text(json.dumps(res, indent=2))
        print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
