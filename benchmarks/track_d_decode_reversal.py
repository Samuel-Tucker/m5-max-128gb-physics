#!/usr/bin/env python3
"""
Track D End-to-End Lossless Speculative Decoding Benchmark Suite.

Evaluates hybrid Gated DeltaNet models (Qwen3.5-9B-4bit and Qwen3.6-35B-A3B-4bit)
across the sealed 12-prompt, 3-category suite (Code-Edit, Extraction, Free Prose):
  - Generates max_tokens (default 128) greedy (temp=0) per prompt.
  - Verifies 100% EXACT TOKEN-FOR-TOKEN IDENTITY vs stock serial B=1 on EVERY
    SINGLE PROMPT and EVERY SINGLE ARM (aborts loudly on even 1 mismatch!).
  - Measures inclusive wall-clock committed tok/s (draft + verify + rollback +
    mx.synchronize()), mean accepted length tau, rollback ms, per category + overall.
  - Supports optional --use-track-a hook to benchmark combined Track A Metal 4 NAX
    verifier + Track D speculative proposers!
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx_lm

_THIS_DIR = Path(__file__).resolve().parent
_KERNELS_DIR = _THIS_DIR.parent / "kernels"
for _d in (str(_THIS_DIR), str(_KERNELS_DIR)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from deltanet_rollback import DeltaNetRollbackEngine
from prompts_sealed import format_prompt_tokens, get_prompts
from proposers import (
    BaseProposer,
    HybridProposer,
    LayerSkipProposer,
    LookaheadProposer,
    PromptLookupProposer,
    SmallModelProposer,
    make_moe_shallow_proposer,
)

MODEL_ALIASES: Dict[str, str] = {
    "9b": "mlx-community/Qwen3.5-9B-4bit",
    "35b": "mlx-community/Qwen3.6-35B-A3B-4bit",
    "2b": "mlx-community/Qwen3.5-2B-4bit",
    "0.8b": "mlx-community/Qwen3.5-0.8B-4bit",
}

DEFAULT_ARMS_9B = [
    "baseline",
    "prompt_lookup_adaptive",
    "prompt_lookup_k4",
    "prompt_lookup_k7",
    "lookahead",
    "layer_skip",
    "draft_2b",
    "draft_0_8b",
    "hybrid_suffix_0_8b",
]

DEFAULT_ARMS_35B = [
    "baseline",
    "moe_shallow",
    "prompt_lookup_k3",
    "lookahead",
    "draft_0_8b",
    "hybrid_suffix_0_8b",
]


def flush_slc(mb: int = 512) -> None:
    """Touch a >=512 MB dummy buffer to flush Apple Silicon SLC between runs."""
    n_elems = (mb * 1024 * 1024) // 4
    buf = mx.ones((n_elems,), dtype=mx.float32)
    out = mx.sum(buf)
    mx.eval(out)
    mx.synchronize()
    del buf, out


def try_apply_track_a_patch(model: Any, model_id: str) -> Dict[str, Any]:
    """Attempt to apply Track A's Metal 4 NAX verifier patch if present."""
    candidates = [
        _KERNELS_DIR / "patch_qwen9b_nax.py",
        _THIS_DIR.parent / "track_a_metal4_nax" / "patch_qwen9b.py",
        _THIS_DIR.parent / "track_a_metal4_nax" / "patch_qwen9b_verify.py",
    ]
    track_a_path = next((p for p in candidates if p.exists()), candidates[0])
    if not track_a_path.exists():
        print(f"[Track A hook] {track_a_path} not found yet; running stock MLX verifier.")
        return {"applied": False, "reason": "file_not_found", "path": str(track_a_path)}

    spec = importlib.util.spec_from_file_location("track_a_patch", str(track_a_path))
    if spec is None or spec.loader is None:
        return {"applied": False, "reason": "import_spec_failed"}
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(track_a_path.parent))
    try:
        spec.loader.exec_module(mod)
        for fn_name in (
            "patch_qwen9b_model",
            "patch_qwen_model",
            "patch_model",
            "apply_track_a_patch",
            "patch_qwen9b",
            "patch_linear_layers",
        ):
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                res = fn(model)
                clean_res = {k: v for k, v in res.items() if not callable(v)} if isinstance(res, dict) else str(res)
                print(f"[Track A hook] Applied {fn_name}() from {track_a_path.name}: {clean_res}")
                return {"applied": True, "function": fn_name, "result": clean_res}
        return {"applied": False, "reason": "no_known_patch_function"}
    except Exception as exc:
        print(f"[Track A hook] Failed to apply patch: {exc!r}")
        return {"applied": False, "reason": f"exception: {exc!r}"}


def _get_eos_ids(tokenizer: Any, model: Any) -> set:
    eos = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        if isinstance(tokenizer.eos_token_id, (list, tuple)):
            eos.update(tokenizer.eos_token_id)
        else:
            eos.add(int(tokenizer.eos_token_id))
    # Qwen3.5 / 3.6 stop tokens (<|endoftext|>=248044, <|im_end|>=248046)
    eos.update({248044, 248046})
    return eos


def run_serial_baseline(
    model: Any,
    engine: DeltaNetRollbackEngine,
    prompt_tokens: List[int],
    max_tokens: int,
    eos_ids: set,
) -> Tuple[List[int], Dict[str, Any]]:
    """Stock serial B=1 greedy decode (ground truth for exactness & baseline tok/s)."""
    cache = engine.make_cache()

    t_prefill_0 = time.perf_counter()
    logits0 = engine.prefill(prompt_tokens, cache)
    y = int(mx.argmax(logits0[0, -1]).item())
    mx.synchronize()
    prefill_ms = (time.perf_counter() - t_prefill_0) * 1e3

    generated = [y]
    if y in eos_ids or max_tokens <= 1:
        return generated, {
            "prefill_ms": round(prefill_ms, 2),
            "decode_tokens": 0,
            "decode_wall_ms": 0.0,
            "tok_per_s": 0.0,
            "steps": 0,
            "tau": 1.0,
        }

    # Timed pure decode phase for remaining max_tokens - 1 tokens
    mx.synchronize()
    t0 = time.perf_counter()
    steps = 0
    for _ in range(max_tokens - 1):
        logits = model(mx.array([[y]], dtype=mx.int32), cache=cache)
        y = int(mx.argmax(logits[0, -1]).item())
        steps += 1
        generated.append(y)
        if y in eos_ids:
            break
    mx.synchronize()
    wall_s = time.perf_counter() - t0

    decode_toks = len(generated) - 1
    return generated, {
        "prefill_ms": round(prefill_ms, 2),
        "decode_tokens": decode_toks,
        "decode_wall_ms": round(wall_s * 1e3, 2),
        "tok_per_s": round(decode_toks / wall_s, 3) if wall_s > 0 else 0.0,
        "steps": steps,
        "tau": 1.0,
        "draft_ms": 0.0,
        "verify_ms": round(wall_s * 1e3, 2),
        "rollback_ms": 0.0,
        "rollback_count": 0,
        "proposed_drafts": 0,
        "accepted_drafts": 0,
        "accept_rate": 0.0,
    }


def run_speculative_arm(
    model: Any,
    engine: DeltaNetRollbackEngine,
    proposer: BaseProposer,
    prompt_tokens: List[int],
    max_tokens: int,
    eos_ids: set,
    sync_rollback_timing: bool = False,
    is_moe: bool = False,
    track_a_applied: bool = False,
    margin_guard: float = 0.50,
) -> Tuple[List[int], Dict[str, Any]]:
    """Run lossless speculative decoding using DeltaNetRollbackEngine + proposer.

    Margin-Guarded Exactness Certificate:
      - Dense Qwen3.5-9B at V <= 8 (stock MLX qmv) is 0.000e+00 bit-exact with
        serial B=1 -> zero margin guard needed.
      - When V > 8 (dense qmm), is_moe (gather_qmm), or track_a_applied (custom
        NAX matmul), bfloat16 accumulation differences can flip razor-thin ties
        (margin <= 0.375, e.g. 0.000 exact ties). Checking top1 - top2 margin on
        v_logits and re-verifying ONLY those ~0.2% tie positions via exact B=1
        guarantees 100% token-for-token identity vs serial B=1!
    """
    cache = engine.make_cache()
    if isinstance(proposer, LayerSkipProposer):
        proposer.bind_target_cache(cache)
    elif isinstance(proposer, HybridProposer):
        if isinstance(proposer.primary, LayerSkipProposer):
            proposer.primary.bind_target_cache(cache)
        if isinstance(proposer.fallback, LayerSkipProposer):
            proposer.fallback.bind_target_cache(cache)

    # Prefill target model & proposer (outside decode wall clock)
    t_pre0 = time.perf_counter()
    logits0 = engine.prefill(prompt_tokens, cache)
    y = int(mx.argmax(logits0[0, -1]).item())
    proposer.reset(prompt_tokens)
    mx.synchronize()
    prefill_ms = (time.perf_counter() - t_pre0) * 1e3

    generated = [y]
    committed = list(prompt_tokens) + [y]
    if y in eos_ids or max_tokens <= 1:
        return generated, {
            "prefill_ms": round(prefill_ms, 2),
            "decode_tokens": 0,
            "decode_wall_ms": 0.0,
            "tok_per_s": 0.0,
            "steps": 0,
            "tau": 1.0,
        }

    # Inclusive timed decode loop (draft + verify + rollback + mx.synchronize())
    draft_s = 0.0
    verify_s = 0.0
    rollback_s = 0.0
    steps = 0
    b1_fallback_steps = 0
    proposed_drafts = 0
    accepted_drafts = 0
    rollback_count = 0
    margin_guard_triggers = 0

    mx.synchronize()
    t_wall_start = time.perf_counter()

    while len(generated) < max_tokens:
        remaining = max_tokens - len(generated)
        # We emit m + 1 tokens per step, so never draft more than remaining - 1
        max_k = remaining - 1

        # 1. Draft
        t_d0 = time.perf_counter()
        if max_k > 0:
            draft = proposer.propose(committed, max_k=max_k)
            if len(draft) > max_k:
                draft = draft[:max_k]
        else:
            draft = []
        draft_s += time.perf_counter() - t_d0

        # 2. Verify (or clean B=1 step if K == 0)
        t_v0 = time.perf_counter()
        if not draft:
            b1_fallback_steps += 1
            logits = model(mx.array([[y]], dtype=mx.int32), cache=cache)
            nxt = int(mx.argmax(logits[0, -1]).item())
            verify_s += time.perf_counter() - t_v0

            steps += 1
            proposer.on_verify(committed, [], [nxt], 0)
            generated.append(nxt)
            committed.append(nxt)
            y = nxt
            if y in eos_ids:
                break
            continue

        k = len(draft)
        V = k + 1
        proposed_drafts += k
        verify_seq = [y] + draft
        verify_tokens = mx.array([verify_seq], dtype=mx.int32)
        v_logits = engine.verify_forward(verify_tokens, cache, record=True)

        use_guard = (margin_guard > 0.0) and (
            (is_moe and V > 4) or ((not is_moe) and V > 8) or track_a_applied
        )
        if use_guard:
            v_arg_arr = mx.argmax(v_logits[0], axis=-1)
            top2_vals = mx.topk(v_logits[0].astype(mx.float32), 2, axis=-1)
            margins_arr = mx.abs(top2_vals[:, 0] - top2_vals[:, 1])
            mx.eval(v_arg_arr, margins_arr)
            v_argmax = v_arg_arr.tolist()
            margins = margins_arr.tolist()
        else:
            v_argmax = mx.argmax(v_logits[0], axis=-1).tolist()
            margins = None
        verify_s += time.perf_counter() - t_v0

        # 3. Exact greedy verification (with Margin-Guarded Exactness Certificate!)
        m = 0
        tie_pos: Optional[int] = None
        while m < k:
            if margins is not None and margins[m] < margin_guard:
                tie_pos = m
                break
            if v_argmax[m] != draft[m]:
                break
            m += 1
        if tie_pos is None and margins is not None and margins[m] < margin_guard:
            tie_pos = m

        if tie_pos is not None:
            # Certify the razor-thin tie position with exact serial B=1 (qmv/gather_qmv)!
            margin_guard_triggers += 1
            t_r0 = time.perf_counter()
            engine.rollback(cache, n_keep_tokens=tie_pos, sync_eval=sync_rollback_timing)
            rollback_s += time.perf_counter() - t_r0
            rollback_count += 1

            t_v1 = time.perf_counter()
            exact_lg = model(mx.array([[verify_seq[tie_pos]]], dtype=mx.int32), cache=cache)
            bonus = int(mx.argmax(exact_lg[0, -1]).item())
            verify_s += time.perf_counter() - t_v1

            m = tie_pos
            accepted_drafts += m
            steps += 1
            emitted = draft[:m] + [bonus]
            hit_eos = False
            final_emitted: List[int] = []
            for tok in emitted:
                final_emitted.append(tok)
                if tok in eos_ids or len(generated) + len(final_emitted) >= max_tokens:
                    hit_eos = tok in eos_ids
                    break

            proposer.on_verify(committed, draft, v_argmax, m)
            generated.extend(final_emitted)
            committed.extend(final_emitted)
            y = bonus
            if hit_eos:
                break
            continue

        bonus = int(v_argmax[m])
        accepted_drafts += m
        steps += 1

        # Check if an EOS occurred inside the accepted prefix
        emitted = draft[:m] + [bonus]
        hit_eos = False
        final_emitted = []
        for tok in emitted:
            final_emitted.append(tok)
            if tok in eos_ids or len(generated) + len(final_emitted) >= max_tokens:
                hit_eos = tok in eos_ids
                break

        # 4. Exact hybrid Gated DeltaNet + KVCache rollback if any draft was rejected
        n_keep = m + 1  # y + draft[0..m-1] are kept in cache; bonus is next pending token
        if n_keep < k + 1 and not hit_eos and (len(generated) + len(final_emitted) < max_tokens):
            t_r0 = time.perf_counter()
            engine.rollback(
                cache,
                n_keep_tokens=n_keep,
                async_eval=(not sync_rollback_timing and not proposer.uses_gpu),
                sync_eval=sync_rollback_timing,
            )
            rollback_s += time.perf_counter() - t_r0
            rollback_count += 1
        else:
            engine.clear_checkpoint()

        # Notify proposer BEFORE updating committed with emitted tokens (so committed[-1] == y0)
        proposer.on_verify(committed, draft, v_argmax, m)

        generated.extend(final_emitted)
        committed.extend(final_emitted)
        y = bonus
        if hit_eos:
            break

    mx.synchronize()
    wall_s = time.perf_counter() - t_wall_start

    decode_toks = len(generated) - 1
    return generated, {
        "prefill_ms": round(prefill_ms, 2),
        "decode_tokens": decode_toks,
        "decode_wall_ms": round(wall_s * 1e3, 2),
        "tok_per_s": round(decode_toks / wall_s, 3) if wall_s > 0 else 0.0,
        "steps": steps,
        "b1_fallback_steps": b1_fallback_steps,
        "margin_guard_triggers": margin_guard_triggers,
        "tau": round(decode_toks / steps, 3) if steps > 0 else 1.0,
        "draft_ms": round(draft_s * 1e3, 2),
        "verify_ms": round(verify_s * 1e3, 2),
        "rollback_ms": round(rollback_s * 1e3, 2),
        "rollback_count": rollback_count,
        "avg_rollback_ms": round((rollback_s * 1e3) / rollback_count, 3) if rollback_count > 0 else 0.0,
        "proposed_drafts": proposed_drafts,
        "accepted_drafts": accepted_drafts,
        "accept_rate": round(accepted_drafts / proposed_drafts, 4) if proposed_drafts > 0 else 0.0,
        "proposer_stats": proposer.stats(),
    }


def _check_exact_parity(
    ref_tokens: List[int],
    cand_tokens: List[int],
    prompt_id: str,
    arm_name: str,
    tokenizer: Any,
    allow_mismatch_debug: bool = False,
) -> Dict[str, Any]:
    if cand_tokens == ref_tokens:
        return {"exact_parity": True, "first_mismatch_idx": None}

    first_idx = None
    min_len = min(len(ref_tokens), len(cand_tokens))
    for i in range(min_len):
        if ref_tokens[i] != cand_tokens[i]:
            first_idx = i
            break
    if first_idx is None:
        first_idx = min_len

    ref_tok = ref_tokens[first_idx] if first_idx < len(ref_tokens) else None
    cand_tok = cand_tokens[first_idx] if first_idx < len(cand_tokens) else None
    msg = (
        f"EXACT PARITY FAILURE on prompt={prompt_id}, arm={arm_name} at token index {first_idx}! "
        f"ref_len={len(ref_tokens)} cand_len={len(cand_tokens)} | "
        f"ref_tok={ref_tok} ({tokenizer.decode([ref_tok]) if ref_tok is not None else None!r}) vs "
        f"cand_tok={cand_tok} ({tokenizer.decode([cand_tok]) if cand_tok is not None else None!r})"
    )
    if not allow_mismatch_debug:
        raise AssertionError(msg)
    print(f"WARNING (debug mode): {msg}")
    return {
        "exact_parity": False,
        "first_mismatch_idx": first_idx,
        "ref_tok": ref_tok,
        "cand_tok": cand_tok,
    }


def _aggregate_arm(prompt_records: List[Dict[str, Any]], baseline_records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for rec in prompt_records:
        by_cat.setdefault(rec["category"], []).append(rec)

    cat_summary: Dict[str, Any] = {}
    for cat, recs in by_cat.items():
        tot_tok = sum(r["decode_tokens"] for r in recs)
        tot_ms = sum(r["decode_wall_ms"] for r in recs)
        tot_steps = sum(r["steps"] for r in recs)
        tot_prop = sum(r["proposed_drafts"] for r in recs)
        tot_acc = sum(r["accepted_drafts"] for r in recs)
        tot_rb_ms = sum(r["rollback_ms"] for r in recs)
        tot_rb_cnt = sum(r["rollback_count"] for r in recs)
        tok_s = tot_tok / (tot_ms / 1e3) if tot_ms > 0 else 0.0

        base_tok = sum(baseline_records[r["prompt_id"]]["decode_tokens"] for r in recs)
        base_ms = sum(baseline_records[r["prompt_id"]]["decode_wall_ms"] for r in recs)
        base_tok_s = base_tok / (base_ms / 1e3) if base_ms > 0 else 0.0

        cat_summary[cat] = {
            "prompts": len(recs),
            "decode_tokens": tot_tok,
            "wall_ms": round(tot_ms, 2),
            "tok_per_s": round(tok_s, 2),
            "speedup_vs_b1": round(tok_s / base_tok_s, 3) if base_tok_s > 0 else 1.0,
            "tau": round(tot_tok / tot_steps, 3) if tot_steps > 0 else 1.0,
            "accept_rate": round(tot_acc / tot_prop, 4) if tot_prop > 0 else 0.0,
            "avg_rollback_ms": round(tot_rb_ms / tot_rb_cnt, 3) if tot_rb_cnt > 0 else 0.0,
            "exact_parity_all": all(r["exact_parity"] for r in recs),
        }

    tot_tok = sum(r["decode_tokens"] for r in prompt_records)
    tot_ms = sum(r["decode_wall_ms"] for r in prompt_records)
    tot_steps = sum(r["steps"] for r in prompt_records)
    tot_prop = sum(r["proposed_drafts"] for r in prompt_records)
    tot_acc = sum(r["accepted_drafts"] for r in prompt_records)
    tot_rb_ms = sum(r["rollback_ms"] for r in prompt_records)
    tot_rb_cnt = sum(r["rollback_count"] for r in prompt_records)
    overall_tok_s = tot_tok / (tot_ms / 1e3) if tot_ms > 0 else 0.0

    base_tot_tok = sum(baseline_records[r["prompt_id"]]["decode_tokens"] for r in prompt_records)
    base_tot_ms = sum(baseline_records[r["prompt_id"]]["decode_wall_ms"] for r in prompt_records)
    base_overall_tok_s = base_tot_tok / (base_tot_ms / 1e3) if base_tot_ms > 0 else 0.0

    return {
        "by_category": cat_summary,
        "overall": {
            "prompts": len(prompt_records),
            "decode_tokens": tot_tok,
            "wall_ms": round(tot_ms, 2),
            "tok_per_s": round(overall_tok_s, 2),
            "speedup_vs_b1": round(overall_tok_s / base_overall_tok_s, 3) if base_overall_tok_s > 0 else 1.0,
            "tau": round(tot_tok / tot_steps, 3) if tot_steps > 0 else 1.0,
            "accept_rate": round(tot_acc / tot_prop, 4) if tot_prop > 0 else 0.0,
            "avg_rollback_ms": round(tot_rb_ms / tot_rb_cnt, 3) if tot_rb_cnt > 0 else 0.0,
            "total_rollback_ms": round(tot_rb_ms, 2),
            "rollback_count": tot_rb_cnt,
            "exact_parity_all": all(r["exact_parity"] for r in prompt_records),
        },
    }


def _build_proposer(
    arm_name: str,
    target_model: Any,
    is_moe: bool,
    draft_models_cache: Dict[str, Any],
    track_a_applied: bool = False,
) -> Optional[BaseProposer]:
    if arm_name == "baseline":
        return None
    if arm_name == "prompt_lookup_adaptive":
        if is_moe:
            return PromptLookupProposer(
                k_max=3,
                min_order=2,
                max_order=6,
                adaptive=True,
                k_by_order={6: 3, 5: 3, 4: 3, 3: 3, 2: 2},
                name=arm_name,
            )
        if track_a_applied:
            # With Track A Metal 4 NAX verifier, V=16 is fast!
            return PromptLookupProposer(
                k_max=15,
                min_order=3,
                max_order=6,
                adaptive=True,
                k_by_order={6: 15, 5: 15, 4: 7, 3: 3},
                name=arm_name,
            )
        # Stock MLX Dense 9B: V <= 8 (K <= 7) avoids the V>=9 qmm_splitk cliff
        # AND guarantees 0.000e+00 bit-exact state & token identity with serial B=1!
        return PromptLookupProposer(
            k_max=7,
            min_order=3,
            max_order=6,
            adaptive=True,
            k_by_order={6: 7, 5: 7, 4: 7, 3: 3},
            name=arm_name,
        )
    if arm_name.startswith("prompt_lookup_k"):
        k_val = int(arm_name.split("_k")[-1])
        return PromptLookupProposer(k_max=k_val, adaptive=False, name=arm_name)
    if arm_name == "moe_shallow":
        return make_moe_shallow_proposer(k_max=3, name=arm_name)
    if arm_name == "lookahead":
        return LookaheadProposer(
            k_max=3 if is_moe else 6,
            ngram_max=4,
            include_prompt=True,
            probe_k_on_miss=0 if is_moe else 2,
            adaptive_probe=True,
            name=arm_name,
        )
    if arm_name == "layer_skip":
        return LayerSkipProposer(
            model=target_model,
            k_draft=3,
            skip_mode="skip_middle_mlp",
            name=arm_name,
        )
    if arm_name == "draft_2b":
        if "2b" not in draft_models_cache:
            print("  Loading control draft model mlx-community/Qwen3.5-2B-4bit ...")
            draft_models_cache["2b"], _ = mlx_lm.load(MODEL_ALIASES["2b"])
        return SmallModelProposer(draft_models_cache["2b"], k_draft=3 if is_moe else 4, name=arm_name)
    if arm_name == "draft_0_8b":
        if "0.8b" not in draft_models_cache:
            print("  Loading fast draft model mlx-community/Qwen3.5-0.8B-4bit ...")
            draft_models_cache["0.8b"], _ = mlx_lm.load(MODEL_ALIASES["0.8b"])
        return SmallModelProposer(draft_models_cache["0.8b"], k_draft=3 if is_moe else 5, name=arm_name)
    if arm_name == "hybrid_suffix_0_8b":
        if "0.8b" not in draft_models_cache:
            print("  Loading fast draft model mlx-community/Qwen3.5-0.8B-4bit ...")
            draft_models_cache["0.8b"], _ = mlx_lm.load(MODEL_ALIASES["0.8b"])
        # Only let CPU suffix preempt 0.8B on high-confidence matches!
        # MoE stays in V<=4 (K<=3) bit-exact qmv regime; Dense 9B stays in V<=8 (K<=7) bit-exact qmv regime!
        if is_moe:
            k_suf, k_map, k_sm = 3, {6: 3, 5: 3, 4: 3}, 3
        elif track_a_applied:
            k_suf, k_map, k_sm = 15, {6: 15, 5: 15, 4: 7}, 5
        else:
            k_suf, k_map, k_sm = 7, {6: 7, 5: 7, 4: 7}, 5
        suffix = PromptLookupProposer(
            k_max=k_suf,
            min_order=4,
            max_order=6,
            adaptive=True,
            k_by_order=k_map,
            name="hi_conf_suffix_primary",
        )
        small = SmallModelProposer(
            draft_models_cache["0.8b"],
            k_draft=k_sm,
            name="0_8b_fallback",
        )
        return HybridProposer(primary=suffix, fallback=small, name=arm_name)
    if arm_name == "draft_0_8b_conf":
        if "0.8b" not in draft_models_cache:
            print("  Loading fast draft model mlx-community/Qwen3.5-0.8B-4bit ...")
            draft_models_cache["0.8b"], _ = mlx_lm.load(MODEL_ALIASES["0.8b"])
        return SmallModelProposer(
            draft_models_cache["0.8b"],
            k_draft=3 if is_moe else (7 if track_a_applied else 5),
            p_min=0.55,
            name=arm_name,
        )
    if arm_name == "hybrid_suffix_0_8b_conf":
        if "0.8b" not in draft_models_cache:
            print("  Loading fast draft model mlx-community/Qwen3.5-0.8B-4bit ...")
            draft_models_cache["0.8b"], _ = mlx_lm.load(MODEL_ALIASES["0.8b"])
        if is_moe:
            k_suf, k_map, k_sm = 3, {6: 3, 5: 3, 4: 3}, 3
        elif track_a_applied:
            k_suf, k_map, k_sm = 15, {6: 15, 5: 15, 4: 11, 3: 7}, 7
        else:
            k_suf, k_map, k_sm = 7, {6: 7, 5: 7, 4: 7}, 5
        suffix = PromptLookupProposer(
            k_max=k_suf,
            min_order=3 if track_a_applied else 4,
            max_order=6,
            adaptive=True,
            k_by_order=k_map,
            name="hi_conf_suffix_primary",
        )
        small = SmallModelProposer(
            draft_models_cache["0.8b"],
            k_draft=k_sm,
            p_min=0.55,
            name="0_8b_conf_fallback",
        )
        return HybridProposer(primary=suffix, fallback=small, name=arm_name)

    raise ValueError(f"Unknown arm: {arm_name}")


def benchmark_model(
    model_key: str,
    arms: Optional[List[str]],
    prompts_per_cat: int,
    max_tokens: int,
    use_track_a: bool,
    do_flush_slc: bool,
    sync_rollback_timing: bool,
    allow_mismatch_debug: bool,
    margin_guard: float = 0.50,
) -> Dict[str, Any]:
    model_id = MODEL_ALIASES.get(model_key, model_key)
    is_moe = "35B" in model_id or "moe" in model_id.lower()
    if arms is None:
        arms = list(DEFAULT_ARMS_35B if is_moe else DEFAULT_ARMS_9B)
    if "baseline" not in arms:
        arms = ["baseline"] + list(arms)

    print(f"\n==================================================================")
    print(f"TRACK D BENCHMARK: {model_id} | prompts={prompts_per_cat * 3} | max_tokens={max_tokens}")
    print(f"Arms: {arms}")
    print(f"==================================================================")

    t_load = time.perf_counter()
    model, tokenizer = mlx_lm.load(model_id)
    load_s = time.perf_counter() - t_load
    print(f"Loaded {model_id} in {load_s:.2f}s")

    track_a_info = {"applied": False}
    if use_track_a:
        track_a_info = try_apply_track_a_patch(model, model_id)

    engine = DeltaNetRollbackEngine(model)
    eos_ids = _get_eos_ids(tokenizer, model)
    prompts = get_prompts(prompts_per_cat=prompts_per_cat)
    encoded_prompts = [
        {**p, "tokens": format_prompt_tokens(tokenizer, p["prompt"])}
        for p in prompts
    ]

    # Quick warmup so Metal kernels compile before timed runs
    print("Warming up Metal shaders (B=1 + verify K=4 + replay)...")
    w_cache = engine.make_cache()
    _ = engine.prefill(encoded_prompts[0]["tokens"][:16], w_cache)
    _ = model(mx.array([[100]], dtype=mx.int32), cache=w_cache)
    _ = engine.verify_forward(mx.array([[100, 101, 102, 103, 104]], dtype=mx.int32), w_cache, record=True)
    mx.synchronize()
    engine.rollback(w_cache, n_keep_tokens=2, sync_eval=True)
    del w_cache

    # Step 1: Run stock serial B=1 baseline on all prompts
    print("\n--- Arm: baseline (Stock Serial B=1) ---")
    ref_tokens_by_id: Dict[str, List[int]] = {}
    baseline_recs_by_id: Dict[str, Dict[str, Any]] = {}
    baseline_prompt_records: List[Dict[str, Any]] = []

    for p in encoded_prompts:
        if do_flush_slc:
            flush_slc(512)
        gen, metrics = run_serial_baseline(model, engine, p["tokens"], max_tokens, eos_ids)
        ref_tokens_by_id[p["id"]] = gen
        rec = {
            "prompt_id": p["id"],
            "category": p["category"],
            "prompt_tokens": len(p["tokens"]),
            "generated_tokens": len(gen),
            "exact_parity": True,
            **metrics,
        }
        baseline_recs_by_id[p["id"]] = rec
        baseline_prompt_records.append(rec)
        print(
            f"  {p['id']:14s} [{p['category']:10s}] | "
            f"tok={rec['decode_tokens']:3d} | {rec['tok_per_s']:6.2f} tok/s | "
            f"wall={rec['decode_wall_ms']:7.1f} ms"
        )

    arm_results: Dict[str, Any] = {
        "baseline": {
            "summary": _aggregate_arm(baseline_prompt_records, baseline_recs_by_id),
            "per_prompt": baseline_prompt_records,
        }
    }

    # Step 2: Run each speculative arm and strictly assert 100% token identity
    draft_models_cache: Dict[str, Any] = {}
    for arm_name in arms:
        if arm_name == "baseline":
            continue
        print(f"\n--- Arm: {arm_name} ---")
        proposer = _build_proposer(
            arm_name,
            model,
            is_moe,
            draft_models_cache,
            track_a_applied=bool(track_a_info.get("applied", False)),
        )
        assert proposer is not None

        # Warmup proposer if GPU-backed
        if proposer.uses_gpu:
            w_c = engine.make_cache()
            _ = engine.prefill(encoded_prompts[0]["tokens"][:16], w_c)
            if isinstance(proposer, LayerSkipProposer):
                proposer.bind_target_cache(w_c)
            proposer.reset(encoded_prompts[0]["tokens"][:16])
            _ = proposer.propose(encoded_prompts[0]["tokens"][:16] + [100], max_k=3)
            mx.synchronize()
            del w_c

        prompt_records: List[Dict[str, Any]] = []
        for p in encoded_prompts:
            if do_flush_slc:
                flush_slc(512)
            gen, metrics = run_speculative_arm(
                model,
                engine,
                proposer,
                p["tokens"],
                max_tokens,
                eos_ids,
                sync_rollback_timing=sync_rollback_timing,
                is_moe=is_moe,
                track_a_applied=bool(track_a_info.get("applied", False)),
                margin_guard=margin_guard,
            )
            parity = _check_exact_parity(
                ref_tokens_by_id[p["id"]],
                gen,
                p["id"],
                arm_name,
                tokenizer,
                allow_mismatch_debug=allow_mismatch_debug,
            )
            base_tps = baseline_recs_by_id[p["id"]]["tok_per_s"]
            sp = metrics["tok_per_s"] / base_tps if base_tps > 0 else 1.0
            rec = {
                "prompt_id": p["id"],
                "category": p["category"],
                "prompt_tokens": len(p["tokens"]),
                "generated_tokens": len(gen),
                "speedup_vs_b1": round(sp, 3),
                **parity,
                **metrics,
            }
            prompt_records.append(rec)
            print(
                f"  {p['id']:14s} [{p['category']:10s}] | "
                f"tok={rec['decode_tokens']:3d} | {rec['tok_per_s']:6.2f} tok/s ({sp:5.2f}x) | "
                f"tau={rec['tau']:4.2f} | acc={rec['accept_rate']*100:5.1f}% | "
                f"rb={rec['avg_rollback_ms']:.3f}ms x{rec['rollback_count']:2d} | "
                f"EXACT={'OK' if rec['exact_parity'] else 'FAIL'}"
            )

        summary = _aggregate_arm(prompt_records, baseline_recs_by_id)
        arm_results[arm_name] = {
            "summary": summary,
            "per_prompt": prompt_records,
        }
        ov = summary["overall"]
        cats = summary["by_category"]
        cat_str = " | ".join(
            f"{c}={cats[c]['tok_per_s']:.1f} ({cats[c]['speedup_vs_b1']:.2f}x, tau={cats[c]['tau']:.2f})"
            for c in ["code_edit", "extraction", "free_prose"]
            if c in cats
        )
        print(
            f"  >> SUMMARY {arm_name}: OVERALL={ov['tok_per_s']:.2f} tok/s "
            f"({ov['speedup_vs_b1']:.2f}x, tau={ov['tau']:.2f}) | {cat_str}"
        )

    # Print clean summary table for this model
    print(f"\n=== SUMMARY TABLE: {model_id} ===")
    print(
        f"{'Arm':24s} | {'Overall':>14s} | {'Code Edit':>14s} | "
        f"{'Extraction':>14s} | {'Free Prose':>14s} | {'Tau':>5s} | {'Exact':>5s}"
    )
    print("-" * 105)
    for arm_name, data in arm_results.items():
        ov = data["summary"]["overall"]
        cats = data["summary"]["by_category"]

        def _fmt(d):
            return f"{d['tok_per_s']:5.1f} ({d['speedup_vs_b1']:4.2f}x)"

        print(
            f"{arm_name:24s} | {_fmt(ov):>14s} | "
            f"{_fmt(cats['code_edit']) if 'code_edit' in cats else '-':>14s} | "
            f"{_fmt(cats['extraction']) if 'extraction' in cats else '-':>14s} | "
            f"{_fmt(cats['free_prose']) if 'free_prose' in cats else '-':>14s} | "
            f"{ov['tau']:5.2f} | {'PASS' if ov['exact_parity_all'] else 'FAIL':>5s}"
        )

    del model, draft_models_cache
    gc.collect()
    mx.clear_cache()

    return {
        "model_id": model_id,
        "is_moe": is_moe,
        "load_s": round(load_s, 2),
        "track_a": track_a_info,
        "max_tokens": max_tokens,
        "prompts_per_cat": prompts_per_cat,
        "arms": arm_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Track D Lossless Speculative Decoding Benchmark")
    parser.add_argument(
        "--models",
        type=str,
        default="9b,35b",
        help="Comma-separated models: 9b, 35b, or full HF IDs",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default="all",
        help="Comma-separated arm names, or 'all' for model defaults",
    )
    parser.add_argument("--max-tokens", type=int, default=128, help="Greedy tokens to generate per prompt")
    parser.add_argument("--prompts-per-cat", type=int, default=4, help="Prompts per category (4 -> 12 total; 1 -> smoke)")
    parser.add_argument("--out", type=str, default="results/track_d_decode_reversal.json", help="Output JSON path")
    parser.add_argument("--use-track-a", action="store_true", help="Hook Track A Metal 4 NAX verifier patch if available")
    parser.add_argument("--flush-slc", action="store_true", help="Flush 512MB SLC buffer before each prompt run")
    parser.add_argument("--sync-rollback", action="store_true", help="Synchronize immediately after each rollback to time exact GPU rollback ms")
    parser.add_argument("--margin-guard", type=float, default=0.50, help="Top1-Top2 logit margin threshold for exact B=1 tie certification on qmm/MoE/TrackA")
    parser.add_argument("--allow-mismatch-debug", action="store_true", help="Log mismatches without aborting (debug only)")
    args = parser.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    arm_list = None if args.arms.strip().lower() == "all" else [a.strip() for a in args.arms.split(",") if a.strip()]

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mlx_version": mx.__version__,
        "mlx_lm_version": mlx_lm.__version__,
        "config": vars(args),
        "models": {},
    }

    for mk in model_keys:
        results["models"][mk] = benchmark_model(
            model_key=mk,
            arms=arm_list,
            prompts_per_cat=args.prompts_per_cat,
            max_tokens=args.max_tokens,
            use_track_a=args.use_track_a,
            do_flush_slc=args.flush_slc,
            sync_rollback_timing=args.sync_rollback,
            allow_mismatch_debug=args.allow_mismatch_debug,
            margin_guard=args.margin_guard,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved complete Track D results to {out_path}")


if __name__ == "__main__":
    main()
