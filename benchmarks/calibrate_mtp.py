#!/usr/bin/env python3
"""
Round-3 Step 1: Calibration & Architecture Truth Verification for Native Qwen3.5/3.6 MTP.

Empirically verifies on live `Qwen3.5-9B-4bit` and `Qwen3.6-35B-A3B-4bit`:
  1. Target hidden state polarity: `post_backbone_norm` vs `pre_backbone_norm`
  2. Concatenation order: `cat([emb, h])` vs `cat([h, emb])`
  3. Chaining hidden state for step k>=2: `post_mtp_norm` vs `pre_mtp_norm`
  4. Multi-step native MTP acceptance rates `alpha_1 .. alpha_6` on Free Prose,
     Code Edit, and Extraction (comparing BF16 MTP vs Q4 MTP on 9B!).
  5. Draft-only Fast LM Head head-to-head (exact Q4 vs Q2 vs Q3 vs Latin-49K/65K subvocab
     vs PCA-r64-gather): measures BOTH exact greedy agreement AND cold-SLC latency (`ms/step`).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    PCAGatherDraftLMHead,
    QuantizedDraftLMHead,
    build_latin_code_vocab_indices,
)
from mtp_module import Qwen35MTPHead, load_mtp_head
from prompts_sealed import format_prompt_tokens, get_prompts


def get_target_hidden_and_logits(model, tokens_1d: List[int]) -> Tuple[mx.array, mx.array, mx.array]:
    """Run target backbone on `tokens_1d` and return `(post_norm_h, pre_norm_h, greedy_next_ids)`."""
    tm = model.language_model if hasattr(model, "language_model") else model
    backbone = tm.model
    x = mx.array([tokens_1d], dtype=mx.int32)
    h = backbone.embed_tokens(x)
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    cache = model.make_cache()
    fa_mask = create_attention_mask(h, cache[backbone.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[backbone.ssm_idx])
    for layer, c in zip(backbone.layers, cache):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=c)
    pre_norm = h
    post_norm = backbone.norm(h)
    if tm.args.tie_word_embeddings:
        logits = backbone.embed_tokens.as_linear(post_norm)
    else:
        logits = tm.lm_head(post_norm)
    preds = mx.argmax(logits[0], axis=-1)
    mx.eval(pre_norm, post_norm, preds)
    return post_norm, pre_norm, preds


def generate_greedy_trace(model, prompt_tokens: List[int], n_gen: int = 64) -> Tuple[List[int], mx.array, mx.array]:
    """Generate `n_gen` greedy tokens and capture exact target hidden states at every position."""
    tm = model.language_model if hasattr(model, "language_model") else model
    backbone = tm.model
    cache = model.make_cache()

    # Prefill prompt
    x = mx.array([prompt_tokens], dtype=mx.int32)
    h_post = backbone(x, cache=cache)
    logits = backbone.embed_tokens.as_linear(h_post[:, -1:, :]) if tm.args.tie_word_embeddings else tm.lm_head(h_post[:, -1:, :])
    tok = int(mx.argmax(logits[0, -1]).item())

    all_tokens = list(prompt_tokens) + [tok]
    h_list = [h_post]

    for _ in range(n_gen - 1):
        h_step = backbone(mx.array([[tok]], dtype=mx.int32), cache=cache)
        lg = backbone.embed_tokens.as_linear(h_step) if tm.args.tie_word_embeddings else tm.lm_head(h_step)
        tok = int(mx.argmax(lg[0, -1]).item())
        all_tokens.append(tok)
        h_list.append(h_step)

    full_h_post = mx.concatenate(h_list, axis=1)  # [1, L + n_gen - 1, H]
    mx.eval(full_h_post)

    # Also compute single-pass pre-norm vs post-norm check on full sequence if needed
    return all_tokens, full_h_post, mx.array(all_tokens, dtype=mx.int32)


def eval_mtp_calibration(
    model,
    mtp: Qwen35MTPHead,
    all_tokens: List[int],
    h_post: mx.array,
    prompt_len: int,
    h_pre: mx.array | None = None,
    concat_order: str = "eh",
    use_pre_norm_target: bool = False,
    chain_pre_norm_mtp: bool = False,
    max_k: int = 6,
) -> Dict[str, Any]:
    """Measure exact 1-step .. `max_k`-step MTP top-1 accuracy on generated tokens."""
    tm = model.language_model if hasattr(model, "language_model") else model
    embed = tm.model.embed_tokens
    h_src = h_pre if (use_pre_norm_target and h_pre is not None) else h_post

    # Prefill MTP cache on prompt pairs (h_0..h_{L-2}, tok_1..tok_{L-1})
    mtp_cache = mtp.make_cache()
    L = prompt_len
    n_gen = len(all_tokens) - L
    if L >= 2:
        p_ids = mx.array([all_tokens[1:L]], dtype=mx.int32)
        p_h = h_src[:, : L - 1, :]
        _, _ = mtp.forward_states(p_ids, p_h, embed, cache=mtp_cache, concat_order=concat_order)
        mx.eval(mtp_cache.state)

    hits = [0] * max_k
    totals = [0] * max_k
    consec_accepts = []

    # Step through each generated position t = L .. L + n_gen - 2
    for step_idx in range(n_gen - 1):
        pos = L - 1 + step_idx  # target hidden state index predicting all_tokens[pos + 1]
        base_offset = mtp_cache.offset  # should equal pos!

        # 1. Exact step-0 MTP prediction using real (h_pos, tok_{pos+1})
        cur_tok = all_tokens[pos + 1]
        cur_h = h_src[:, pos : pos + 1, :]

        # Save KV offset BEFORE chaining steps 1..max_k-1, but keep step-0's real KV entry!
        h_post_m, h_pre_m = mtp.forward_states(
            mx.array([[cur_tok]], dtype=mx.int32),
            cur_h,
            embed,
            cache=mtp_cache,
            concat_order=concat_order,
        )
        post_step0_offset = mtp_cache.offset  # real KV now includes pos!

        lg = embed.as_linear(h_post_m) if tm.args.tie_word_embeddings else tm.lm_head(h_post_m)
        pred_tok = int(mx.argmax(lg[0, -1]).item())

        accepted_here = 0
        still_matching = True
        for k in range(max_k):
            target_idx = pos + 2 + k
            if target_idx >= len(all_tokens):
                break
            truth = all_tokens[target_idx]
            totals[k] += 1
            if still_matching and pred_tok == truth:
                hits[k] += 1
                accepted_here += 1
            else:
                still_matching = False

            if k + 1 < max_k and (pos + 2 + k + 1) < len(all_tokens):
                next_h = h_pre_m if chain_pre_norm_mtp else h_post_m
                h_post_m, h_pre_m = mtp.forward_states(
                    mx.array([[pred_tok]], dtype=mx.int32),
                    next_h,
                    embed,
                    cache=mtp_cache,
                    concat_order=concat_order,
                )
                lg = embed.as_linear(h_post_m) if tm.args.tie_word_embeddings else tm.lm_head(h_post_m)
                pred_tok = int(mx.argmax(lg[0, -1]).item())

        consec_accepts.append(accepted_here)
        # Rewind MTP cache back to real pos+1 (discarding chained approximate KV entries)
        mtp_cache.offset = post_step0_offset

    acc_by_k = [round(hits[k] / max(totals[k], 1), 4) for k in range(max_k)]
    return {
        "step_cond_acc": acc_by_k,
        "mean_accepted_draft": round(float(np.mean(consec_accepts)), 3) if consec_accepts else 0.0,
        "mean_tokens_per_verify": round(1.0 + float(np.mean(consec_accepts)), 3) if consec_accepts else 1.0,
        "n_steps": len(consec_accepts),
    }


def benchmark_fast_heads(
    model,
    tokenizer,
    mtp: Qwen35MTPHead,
    sample_h_list: List[mx.array],
    exact_toks: List[int],
    flush_mb: int = 512,
) -> Dict[str, Any]:
    """Compare exact Q4 vs Q2 vs Q3 vs SubVocab vs PCA-Gather on recall AND cold-SLC latency."""
    tm = model.language_model if hasattr(model, "language_model") else model
    V = tm.args.vocab_size
    flusher = mx.ones(((flush_mb * 1024 * 1024) // 4,), dtype=mx.float32)
    mx.eval(flusher)

    def flush():
        s = mx.sum(flusher)
        mx.eval(s)
        mx.synchronize()

    heads = [
        ExactDraftLMHead(model),
        QuantizedDraftLMHead(model, bits=2, group_size=64),
        QuantizedDraftLMHead(model, bits=3, group_size=64),
        ActiveVocabSubHead(model, build_latin_code_vocab_indices(tokenizer, V, max_rows=32768), bits=4, label="latin_32k_q4"),
        ActiveVocabSubHead(model, build_latin_code_vocab_indices(tokenizer, V, max_rows=65536), bits=4, label="latin_64k_q4"),
        ActiveVocabSubHead(model, build_latin_code_vocab_indices(tokenizer, V, max_rows=65536), bits=2, label="latin_64k_q2"),
        PCAGatherDraftLMHead(model, rank=64, top_k=4096),
        PCAGatherDraftLMHead(model, rank=128, top_k=4096),
    ]

    h_probe = sample_h_list[0]
    out: Dict[str, Any] = {}
    for hd in heads:
        # 1. Exact top-1 agreement across all captured MTP hidden states
        agree = 0
        for h_vec, ref_tok in zip(sample_h_list, exact_toks):
            t_pred, _ = hd.sample_greedy_and_prob(h_vec)
            mx.eval(t_pred)
            if int(t_pred.item()) == ref_tok:
                agree += 1
        recall = agree / max(len(exact_toks), 1)

        # 2. Cold-SLC latency (flush 512 MB before every call)
        for _ in range(2):
            t0, p0 = hd.sample_greedy_and_prob(h_probe)
            mx.eval(t0, p0)
        mx.synchronize()

        cold_ms = []
        for _ in range(8):
            flush()
            ts = time.perf_counter()
            t0, p0 = hd.sample_greedy_and_prob(h_probe)
            mx.eval(t0, p0)
            mx.synchronize()
            cold_ms.append((time.perf_counter() - ts) * 1000.0)

        # 3. Warm-burst latency (4 back-to-back calls, simulating K=4 drafting!)
        flush()
        ts = time.perf_counter()
        for _ in range(4):
            t0, p0 = hd.sample_greedy_and_prob(h_probe)
            mx.eval(t0, p0)
        mx.synchronize()
        warm_per_call_ms = ((time.perf_counter() - ts) * 1000.0) / 4.0

        med_cold = float(np.median(cold_ms))
        out[hd.name] = {
            "weight_mb": round(hd.weight_bytes / 1e6, 2),
            "greedy_agreement": round(recall, 4),
            "agree_count": f"{agree}/{len(exact_toks)}",
            "cold_ms": round(med_cold, 3),
            "warm_4step_ms_each": round(warm_per_call_ms, 3),
        }
        print(
            f"  [LMHead] {hd.name:18s} | {hd.weight_bytes/1e6:6.1f} MB | "
            f"agree={recall*100:5.1f}% ({agree}/{len(exact_toks)}) | "
            f"cold={med_cold:.3f} ms | warm4={warm_per_call_ms:.3f} ms",
            flush=True,
        )
        del hd
        gc.collect()
        mx.clear_cache()

    return out


def run_model_calibration(model_key: str) -> Dict[str, Any]:
    if model_key == "9b":
        model_id = "mlx-community/Qwen3.5-9B-4bit"
        q4_path = THIS_DIR.parent / "weights/qwen35_9b_mtp_q4.safetensors"
        bf16_path = THIS_DIR.parent / "weights/qwen35_9b_mtp_bf16.safetensors"
    else:
        model_id = "mlx-community/Qwen3.6-35B-A3B-4bit"
        q4_path = THIS_DIR.parent / "weights/qwen36_35b_mtp_q4.safetensors"
        bf16_path = None

    print(f"\n================ CALIBRATING {model_id} ================", flush=True)
    model, tokenizer = mlx_lm.load(model_id)
    mtp_q4 = load_mtp_head(model, q4_path, quantized=True)

    # Pick 1 prompt from each of the 3 sealed categories
    prompts = get_prompts(prompts_per_cat=1)
    res: Dict[str, Any] = {"model_id": model_id, "polarity_ablation": {}, "categories_q4": {}}

    # Step 1: Polarity & concat-order ablation on the first free_prose prompt!
    prose_p = [p for p in prompts if p["category"] == "free_prose"][0]
    p_toks = format_prompt_tokens(tokenizer, prose_p["prompt"])
    all_toks, h_post_gen, _ = generate_greedy_trace(model, p_toks, n_gen=48)
    # Single forward to also get pre_norm vs post_norm for ablation
    h_post_full, h_pre_full, _ = get_target_hidden_and_logits(model, all_toks[:-1])

    for label, use_pre_tgt, concat_ord, chain_pre_mtp in [
        ("official_post_eh_postchain", False, "eh", False),
        ("ablate_pre_tgt_eh_postchain", True, "eh", False),
        ("ablate_post_he_postchain", False, "he", False),
        ("ablate_post_eh_prechain", False, "eh", True),
    ]:
        m = eval_mtp_calibration(
            model,
            mtp_q4,
            all_toks,
            h_post_full,
            len(p_toks),
            h_pre=h_pre_full,
            concat_order=concat_ord,
            use_pre_norm_target=use_pre_tgt,
            chain_pre_norm_mtp=chain_pre_mtp,
            max_k=5,
        )
        res["polarity_ablation"][label] = m
        print(f"  [Ablation] {label:30s} -> step_acc={m['step_cond_acc']} E[tok]={m['mean_tokens_per_verify']}", flush=True)

    # Step 2: Evaluate Q4 MTP across all 3 categories (+ collect MTP hidden states for Fast LMHead test)
    sample_h_list: List[mx.array] = []
    exact_toks: List[int] = []
    tm = model.language_model if hasattr(model, "language_model") else model
    embed = tm.model.embed_tokens

    for p in prompts:
        pt = format_prompt_tokens(tokenizer, p["prompt"])
        at, hp, _ = generate_greedy_trace(model, pt, n_gen=64)
        m_q4 = eval_mtp_calibration(model, mtp_q4, at, hp, len(pt), max_k=6)
        res["categories_q4"][p["category"]] = m_q4
        print(f"  [Q4 MTP]   {p['category']:12s} -> step_acc={m_q4['step_cond_acc']} E[tok/ver]={m_q4['mean_tokens_per_verify']}", flush=True)

        # Capture 40 real MTP output hidden states per category (120 total) for Fast LMHead eval
        mc = mtp_q4.make_cache()
        if len(pt) >= 2:
            _, _ = mtp_q4.forward_states(mx.array([at[1 : len(pt)]], dtype=mx.int32), hp[:, : len(pt) - 1, :], embed, cache=mc)
        for s_i in range(40):
            pos = len(pt) - 1 + s_i
            h_out, _ = mtp_q4.forward_states(mx.array([[at[pos + 1]]], dtype=mx.int32), hp[:, pos : pos + 1, :], embed, cache=mc)
            lg = embed.as_linear(h_out) if tm.args.tie_word_embeddings else tm.lm_head(h_out)
            tok_ex = int(mx.argmax(lg[0, -1]).item())
            mx.eval(h_out)
            sample_h_list.append(h_out)
            exact_toks.append(tok_ex)

        # Optional BF16 vs Q4 MTP comparison on 9B
        if bf16_path is not None and bf16_path.exists():
            if "categories_bf16" not in res:
                res["categories_bf16"] = {}
                mtp_bf16 = load_mtp_head(model, bf16_path, quantized=False)
            m_bf = eval_mtp_calibration(model, mtp_bf16, at, hp, len(pt), max_k=6)
            res["categories_bf16"][p["category"]] = m_bf
            print(f"  [BF16 MTP] {p['category']:12s} -> step_acc={m_bf['step_cond_acc']} E[tok/ver]={m_bf['mean_tokens_per_verify']}", flush=True)

    # Step 3: Measure pure MTP layer-only latency (without lm_head) vs full MTP step!
    flusher = mx.ones(((512 * 1024 * 1024) // 4,), dtype=mx.float32)
    mx.eval(flusher)
    mc_bench = mtp_q4.make_cache()
    h_one = sample_h_list[0]
    t_one = mx.array([[exact_toks[0]]], dtype=mx.int32)
    for _ in range(3):
        ho, _ = mtp_q4.forward_states(t_one, h_one, embed, cache=mc_bench)
        mx.eval(ho)
        mc_bench.offset = 0
    mx.synchronize()
    cold_layer_ms = []
    for _ in range(8):
        s = mx.sum(flusher); mx.eval(s); mx.synchronize()
        ts = time.perf_counter()
        ho, _ = mtp_q4.forward_states(t_one, h_one, embed, cache=mc_bench)
        mx.eval(ho); mx.synchronize()
        cold_layer_ms.append((time.perf_counter() - ts) * 1000.0)
        mc_bench.offset = 0
    res["mtp_layer_only_cold_ms"] = round(float(np.median(cold_layer_ms)), 3)
    print(f"  [MTP Layer-Only Q4] cold={res['mtp_layer_only_cold_ms']:.3f} ms (no lm_head)", flush=True)

    # Step 4: Fast Draft LM Head head-to-head
    print("  --- Fast Draft LM Head Head-to-Head ---", flush=True)
    res["fast_draft_lm_heads"] = benchmark_fast_heads(model, tokenizer, mtp_q4, sample_h_list, exact_toks)

    del model, mtp_q4
    gc.collect()
    mx.clear_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["9b", "35b"], choices=["9b", "35b"])
    ap.add_argument("--out", type=str, default="results/round3_mtp_calibration.json")
    args = ap.parse_args()

    out: Dict[str, Any] = {}
    for mk in args.models:
        out[mk] = run_model_calibration(mk)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved calibration results to {out_path}")


if __name__ == "__main__":
    main()
