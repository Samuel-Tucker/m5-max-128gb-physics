"""Duel 1: The 5-vs-2 Council Duel — Provably Lossless Sign-Definite Parseval /
Cauchy-Schwarz `lm_head` Pruning on REAL `Qwen3.5-9B-4bit` (`248,320 x 4096` Q4).

Tests ALL THREE council mechanisms against 0 greedy top-1 mismatches:
  (a) Zero-Mode Rank-r SVD/PCA + Exact Orthogonal Norm Bound (r in {32, 64, 128, 256}),
      using an activation-fitted basis P (strictly from disjoint calibration prompts!),
      plus Weight-SVD basis and Astra 6.2's eight 64-coord group projection.
  (b) Opus 5.5's Hadamard-Q2 Sketch + Exact Residual Norm Bound (global Cauchy-Schwarz
      AND tighter block-wise G in {8, 16} Cauchy-Schwarz + L1/Linf bounds), plus Opus's
      exact draft-token 100%-head-skip certificate rate.
  (c) Astra 6.2's Oracle-Threshold Test: given the TRUE winning score z* upfront,
      what exact % of the 248,320 rows are certified skipped?
  (d) Real Metal/MLX Gathered Q4 `lm_head` Rescore Timing Test (cold 540 MB SLC flush,
      interleaved A/B) measuring actual wall-clock ms vs stock `lm_head` AND synthetic
      survivor-rate sweep to pinpoint the hardware break-even threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import numpy as np
from mlx_lm import load

from common import (
    CALIBRATION_PROMPTS,
    HELD_OUT_PROMPTS,
    MODEL_ID,
    SLCFlusher,
    bench_interleaved,
    clear_memory,
    fwht_block4096,
)

# ---------------------------------------------------------------------------
# Custom Metal Kernel: High-Bandwidth Coalesced Gathered Q4 Matvec (D=4096)
# Reads ONLY the surviving rows W[idx], S[idx], B[idx] directly from DRAM.
# ---------------------------------------------------------------------------

_GATHERED_Q4_LMHEAD_KERNEL = None


def get_gathered_q4_lmhead_kernel():
    global _GATHERED_Q4_LMHEAD_KERNEL
    if _GATHERED_Q4_LMHEAD_KERNEL is None:
        source = r"""
            uint tg_id = threadgroup_position_in_grid.x;
            uint num_tgs = threadgroups_per_grid.x;
            uint tid = thread_position_in_threadgroup.x; // 0..255 (8 simdgroups)
            uint sg = tid >> 5u;                         // 0..7
            uint lane = tid & 31u;                       // 0..31

            threadgroup float h_sh[4096];
            for (uint i = tid; i < 4096u; i += 256u) {
                h_sh[i] = float(h[i]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint K = k_count[0];
            uint stride = num_tgs * 8u;
            for (uint t = tg_id * 8u + sg; t < K; t += stride) {
                uint row = indices[t];
                const device uint* w_row = W + ulong(row) * 512ul;
                ulong sb_base = ulong(row) * 64ul;
                float acc = 0.0f;

                // 16 iterations x 32 lanes = 512 uint32 words = 4096 Q4 weights.
                // 32 lanes read 128 contiguous bytes = 1 full Apple GPU cache line.
                for (uint iter = 0u; iter < 16u; ++iter) {
                    uint word_idx = iter * 32u + lane;
                    uint grp = word_idx >> 3u;
                    uint col_base = word_idx << 3u;
                    uint w = w_row[word_idx];
                    float sc = float(S[sb_base + grp]);
                    float bi = float(B[sb_base + grp]);

                    float h0 = h_sh[col_base + 0u];
                    float h1 = h_sh[col_base + 1u];
                    float h2 = h_sh[col_base + 2u];
                    float h3 = h_sh[col_base + 3u];
                    float h4 = h_sh[col_base + 4u];
                    float h5 = h_sh[col_base + 5u];
                    float h6 = h_sh[col_base + 6u];
                    float h7 = h_sh[col_base + 7u];

                    float d = float((w >>  0u) & 0xFu) * h0
                            + float((w >>  4u) & 0xFu) * h1
                            + float((w >>  8u) & 0xFu) * h2
                            + float((w >> 12u) & 0xFu) * h3
                            + float((w >> 16u) & 0xFu) * h4
                            + float((w >> 20u) & 0xFu) * h5
                            + float((w >> 24u) & 0xFu) * h6
                            + float((w >> 28u) & 0xFu) * h7;
                    float hsum = (h0 + h1) + (h2 + h3) + (h4 + h5) + (h6 + h7);
                    acc += fma(d, sc, bi * hsum);
                }
                float total = simd_sum(acc);
                if (lane == 0u) {
                    out[t] = total;
                }
            }
        """
        _GATHERED_Q4_LMHEAD_KERNEL = mx.fast.metal_kernel(
            name="gathered_q4_lmhead_4096",
            input_names=["W", "S", "B", "h", "indices", "k_count"],
            output_names=["out"],
            source=source,
        )
    return _GATHERED_Q4_LMHEAD_KERNEL


def gathered_q4_matvec(
    lm_head,
    h_1d: mx.array,
    indices: mx.array,
    num_tgs: int = 160,
) -> mx.array:
    """Exact gathered Q4 matvec on surviving vocab rows directly from DRAM."""
    k = int(indices.shape[0])
    if k == 0:
        return mx.zeros((0,), dtype=mx.float32)
    k_arr = mx.array([k], dtype=mx.uint32)
    tgs = min(num_tgs, max(1, (k + 7) // 8))
    kernel = get_gathered_q4_lmhead_kernel()
    (out,) = kernel(
        inputs=[
            lm_head.weight,
            lm_head.scales,
            lm_head.biases,
            h_1d,
            indices.astype(mx.uint32),
            k_arr,
        ],
        grid=(tgs * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(k,)],
        output_dtypes=[mx.float32],
    )
    return out


# ---------------------------------------------------------------------------
# Hidden-State Collection (Strictly Disjoint Calibration vs Held-Out Decode)
# ---------------------------------------------------------------------------

def collect_hidden_states(
    model,
    tokenizer,
    decode_steps_per_prompt: int = 32,
) -> Tuple[np.ndarray, np.ndarray, mx.array]:
    """Collect disjoint calibration hidden states (prefill) and held-out decode states."""
    lm = model.language_model

    # 1. Calibration hidden states (prefill on CALIBRATION_PROMPTS ONLY)
    cal_list: List[np.ndarray] = []
    for _, prompt in CALIBRATION_PROMPTS:
        toks = tokenizer.encode(prompt)
        ids = mx.array(toks, dtype=mx.int32)[None, :]
        h = lm.model(ids)  # (1, L, 4096) post-norm bfloat16
        mx.eval(h)
        cal_list.append(np.array(h[0].astype(mx.float32)))
    H_cal = np.concatenate(cal_list, axis=0)  # (N_cal, 4096) float32

    # 2. Held-out real autoregressive decode hidden states (HELD_OUT_PROMPTS ONLY)
    test_bf16_list: List[mx.array] = []
    for _, prompt in HELD_OUT_PROMPTS:
        toks = tokenizer.encode(prompt)
        cache = model.make_cache()
        ids = mx.array(toks, dtype=mx.int32)[None, :]
        h = lm.model(ids, cache=cache)
        h_last = h[:, -1:, :]  # (1, 1, 4096) bfloat16
        logits = lm.lm_head(h_last)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(h_last, next_tok)
        test_bf16_list.append(h_last[0, 0])

        for _ in range(decode_steps_per_prompt - 1):
            inp = next_tok.reshape(1, 1)
            h = lm.model(inp, cache=cache)
            logits = lm.lm_head(h)
            next_tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(h, next_tok)
            test_bf16_list.append(h[0, 0])

    H_test_bf16 = mx.stack(test_bf16_list, axis=0)  # (N_test, 4096) bfloat16
    mx.eval(H_test_bf16)
    H_test_f32 = np.array(H_test_bf16.astype(mx.float32))
    return H_cal, H_test_f32, H_test_bf16


# ---------------------------------------------------------------------------
# Dequantized `lm_head` exact reference & statistics helper
# ---------------------------------------------------------------------------

def dequantize_lm_head_f32(lm_head, chunk_rows: int = 32768) -> np.ndarray:
    """Dequantize 248,320 x 4096 Q4 lm_head to float32 numpy array (4.07 GB)."""
    V = lm_head.weight.shape[0]
    D = lm_head.weight.shape[1] * (32 // lm_head.bits)
    W = np.empty((V, D), dtype=np.float32)
    for start in range(0, V, chunk_rows):
        end = min(V, start + chunk_rows)
        dq = mx.dequantize(
            lm_head.weight[start:end],
            lm_head.scales[start:end],
            lm_head.biases[start:end],
            group_size=lm_head.group_size,
            bits=lm_head.bits,
            mode=lm_head.mode,
        ).astype(mx.float32)
        mx.eval(dq)
        W[start:end] = np.array(dq)
    return W


def evaluate_pruner_stats(
    name: str,
    s_approx: np.ndarray,      # (N_test, V) float32
    bound_margin: np.ndarray,  # (N_test, V) float32, >= |z_exact - s_approx|
    z_exact_f32: np.ndarray,   # (N_test, V) float32
    true_top1_id: np.ndarray,  # (N_test,)
    true_top1_val: np.ndarray, # (N_test,)
    stock_top1_id: np.ndarray, # (N_test,)
    seed_k: int = 16,
    round_eps_f32: float = 1e-3,
    round_eps_bf16: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate exact self-seeded pruning, Oracle pruning, and Opus full-head-skip rate."""
    N, V = s_approx.shape
    U_f32 = s_approx + bound_margin + round_eps_f32

    # Sanity check: verify bound soundness (does U_f32 EVER fall below z_exact_f32?)
    violation = float(np.max(z_exact_f32 - U_f32))
    n_violations = int(np.sum(z_exact_f32 > U_f32))

    # 1. Oracle-Threshold Test (Astra 6.2): given true z* upfront
    oracle_alive_fracs = []
    opus_full_skip_count = 0
    for i in range(N):
        z_star = true_top1_val[i]
        top_id = int(true_top1_id[i])
        alive = U_f32[i] >= z_star
        oracle_alive_fracs.append(float(np.mean(alive)))
        # Opus full-head skip: does z* strictly beat ALL rivals v != top_id?
        u_copy = U_f32[i].copy()
        u_copy[top_id] = -np.inf
        if z_star > float(np.max(u_copy)):
            opus_full_skip_count += 1

    # 2. Self-Seeded Exact 2-Stage Pruning (score top-seed_k by s_approx, then prune)
    seeded_alive_fracs = []
    mismatches_vs_f32 = 0
    mismatches_vs_stock = 0
    for i in range(N):
        # Pick top seed_k candidates by approximate score
        cand_idx = np.argpartition(s_approx[i], -seed_k)[-seed_k:]
        exact_best = float(np.max(z_exact_f32[i, cand_idx]))
        # Survivors needing exact rescore (include seed_k scored candidates)
        surv_mask = U_f32[i] >= exact_best
        surv_mask[cand_idx] = True
        surv_idx = np.flatnonzero(surv_mask)
        seeded_alive_fracs.append(float(len(surv_idx)) / V)

        # Exact rescore over survivors
        best_local = int(np.argmax(z_exact_f32[i, surv_idx]))
        pred_id = int(surv_idx[best_local])
        if pred_id != int(true_top1_id[i]):
            mismatches_vs_f32 += 1
        if pred_id != int(stock_top1_id[i]):
            mismatches_vs_stock += 1

    # Also compute stock-bf16-safe Oracle skip rate (adding bf16 outward rounding margin)
    if round_eps_bf16 > 0.0:
        U_bf16_safe = s_approx + bound_margin + round_eps_bf16
        bf16_oracle_skip = [
            1.0 - float(np.mean(U_bf16_safe[i] >= true_top1_val[i])) for i in range(N)
        ]
        bf16_oracle_skip_mean_pct = 100.0 * float(np.mean(bf16_oracle_skip))
    else:
        bf16_oracle_skip_mean_pct = None

    oracle_skip_pcts = [(1.0 - a) * 100.0 for a in oracle_alive_fracs]
    seeded_skip_pcts = [(1.0 - a) * 100.0 for a in seeded_alive_fracs]

    # Diagnostic: Cauchy-Schwarz looseness ratio = bound_margin / |actual_err|
    actual_err = np.abs(z_exact_f32 - s_approx)
    # Use rows with non-trivial error to avoid division by tiny noise
    looseness = bound_margin / np.maximum(actual_err, 1e-6)
    median_margin = float(np.median(bound_margin))
    median_actual_err = float(np.median(actual_err))
    median_looseness = float(np.median(looseness))

    return {
        "name": name,
        "soundness_max_violation": violation,
        "soundness_num_violations": n_violations,
        "greedy_mismatches_vs_f32": mismatches_vs_f32,
        "greedy_mismatches_vs_stock_bf16": mismatches_vs_stock,
        "oracle_skip_pct_mean": round(float(np.mean(oracle_skip_pcts)), 4),
        "oracle_skip_pct_median": round(float(np.median(oracle_skip_pcts)), 4),
        "oracle_skip_pct_min": round(float(np.min(oracle_skip_pcts)), 4),
        "oracle_skip_pct_max": round(float(np.max(oracle_skip_pcts)), 4),
        "oracle_alive_pct_mean": round(100.0 * float(np.mean(oracle_alive_fracs)), 4),
        "seeded_skip_pct_mean": round(float(np.mean(seeded_skip_pcts)), 4),
        "seeded_skip_pct_median": round(float(np.median(seeded_skip_pcts)), 4),
        "seeded_alive_pct_mean": round(100.0 * float(np.mean(seeded_alive_fracs)), 4),
        "bf16_safe_oracle_skip_pct_mean": (
            round(bf16_oracle_skip_mean_pct, 4)
            if bf16_oracle_skip_mean_pct is not None
            else None
        ),
        "opus_full_head_skip_pct": round(100.0 * opus_full_skip_count / N, 4),
        "median_bound_margin": round(median_margin, 4),
        "median_actual_residual_err": round(median_actual_err, 4),
        "median_cauchy_schwarz_looseness_x": round(median_looseness, 2),
        "astra_falsifier_oracle_alive_gt_54pct_KILLED": bool(
            100.0 * float(np.mean(oracle_alive_fracs)) > 54.0
        ),
    }


# ---------------------------------------------------------------------------
# Main Duel Execution
# ---------------------------------------------------------------------------

def run_lm_head_duel(
    quick: bool = False,
    model=None,
    tokenizer=None,
    flusher: SLCFlusher | None = None,
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    if model is None or tokenizer is None:
        model, tokenizer = load(MODEL_ID)
    if flusher is None:
        flusher = SLCFlusher(mb=540)

    lm_head = model.language_model.lm_head
    V = int(lm_head.weight.shape[0])  # 248320
    D = int(lm_head.weight.shape[1] * (32 // lm_head.bits))  # 4096
    head_bytes = (
        lm_head.weight.nbytes + lm_head.scales.nbytes + lm_head.biases.nbytes
    )

    decode_steps = 8 if quick else 32  # 64 states (quick) vs 256 states (full)
    trials = 5 if quick else 11

    print(
        f"[Duel 1] Collecting hidden states (cal=8 prefill prompts, "
        f"test=8 prompts x {decode_steps} decode steps = {8 * decode_steps} states)..."
    )
    H_cal, H_test, H_test_bf16 = collect_hidden_states(
        model, tokenizer, decode_steps_per_prompt=decode_steps
    )
    N_cal = int(H_cal.shape[0])
    N_test = int(H_test.shape[0])
    print(f"[Duel 1] Collected H_cal={H_cal.shape}, H_test={H_test.shape}")

    # Exact stock MLX lm_head logits on H_test_bf16
    z_stock_mx = lm_head(H_test_bf16)
    mx.eval(z_stock_mx)
    z_stock_f32 = np.array(z_stock_mx.astype(mx.float32))
    stock_top1_id = np.argmax(z_stock_f32, axis=-1)

    # Exact FP32 dequantized lm_head
    print(f"[Duel 1] Dequantizing lm_head ({V} x {D}, {head_bytes / 1e6:.1f} MB Q4)...")
    W = dequantize_lm_head_f32(lm_head)
    W_norm2 = np.sum(W.astype(np.float64) ** 2, axis=1).astype(np.float32)
    W_norm = np.sqrt(W_norm2)

    # Exact FP32 logits z_exact = H_test @ W^T (computed on GPU in chunks for speed)
    z_exact_f32 = np.empty((N_test, V), dtype=np.float32)
    H_test_f32_mx = mx.array(H_test)
    for s in range(0, V, 32768):
        e = min(V, s + 32768)
        w_chunk = mx.array(W[s:e])
        zc = H_test_f32_mx @ w_chunk.T
        mx.eval(zc)
        z_exact_f32[:, s:e] = np.array(zc)
    del H_test_f32_mx

    true_top1_id = np.argmax(z_exact_f32, axis=-1)
    true_top1_val = z_exact_f32[np.arange(N_test), true_top1_id]
    # Top-2 and median logit diagnostics
    top2_vals = np.partition(z_exact_f32, -2, axis=-1)[:, -2]
    logit_margins = true_top1_val - top2_vals
    z_medians = np.median(z_exact_f32, axis=-1)
    z_spread_top_minus_med = true_top1_val - z_medians

    bf16_max_diff = float(np.max(np.abs(z_stock_f32 - z_exact_f32)))
    bf16_top1_agree = int(np.sum(stock_top1_id == true_top1_id))
    h_norms = np.linalg.norm(H_test, axis=1)

    print(
        f"[Duel 1] Logit geometry: median ||h||={np.median(h_norms):.2f}, "
        f"median ||w_i||={np.median(W_norm):.3f}, median z*={np.median(true_top1_val):.2f}, "
        f"median (z* - median_z)={np.median(z_spread_top_minus_med):.2f}, "
        f"median (z1 - z2)={np.median(logit_margins):.2f}, "
        f"stock_bf16 vs f32 max_diff={bf16_max_diff:.4f} (top1 agree {bf16_top1_agree}/{N_test})"
    )

    mechanism_results: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Mechanism (a): Zero-Mode Rank-r Activation-Fitted PCA + Exact Orth Norm
    # Fitted STRICTLY on H_cal (disjoint calibration prompts)
    # -----------------------------------------------------------------------
    print("[Duel 1] Mechanism (a): Activation-fitted PCA basis (strictly from H_cal)...")
    C_cal = (H_cal.astype(np.float64).T @ H_cal.astype(np.float64)) / max(1, N_cal)
    eigvals_cal, eigvecs_cal = np.linalg.eigh(C_cal)
    # Sort descending
    order_cal = np.argsort(eigvals_cal)[::-1]
    eigvals_cal = eigvals_cal[order_cal]
    U_cal = eigvecs_cal[:, order_cal].astype(np.float32)  # (4096, 4096), cols are eigenvectors

    cal_total_energy = float(np.sum(np.maximum(eigvals_cal, 0.0)))
    pca_energy_table: Dict[str, Any] = {}

    ranks = [32, 64, 128, 256]
    best_pca_for_timing: Tuple[int, np.ndarray, np.ndarray, np.ndarray] | None = None
    best_skip_seen = -1.0

    for r in ranks:
        P = U_cal[:, :r].T  # (r, 4096) orthonormal rows
        # Energy captured on calibration vs held-out test
        H_test_par = H_test @ P.T  # (N_test, r)
        H_test_perp = H_test - H_test_par @ P  # (N_test, 4096)
        h_perp_norm = np.linalg.norm(H_test_perp, axis=1)  # (N_test,)
        test_par_energy_frac = float(
            np.mean(np.sum(H_test_par ** 2, axis=1) / np.sum(H_test ** 2, axis=1))
        )
        cal_par_energy_frac = float(np.sum(eigvals_cal[:r]) / cal_total_energy)
        pca_energy_table[f"r={r}"] = {
            "cal_energy_captured_pct": round(100.0 * cal_par_energy_frac, 2),
            "heldout_energy_captured_pct": round(100.0 * test_par_energy_frac, 2),
            "median_h_perp_norm": round(float(np.median(h_perp_norm)), 3),
        }

        # Project W onto P on GPU in chunks to get exact W_par and exact ||W_perp||_2
        P_mx = mx.array(P)
        W_par = np.empty((V, r), dtype=np.float32)
        w_perp_norm = np.empty((V,), dtype=np.float32)
        # For r in {64, 256}, also compute tighter Block-8 and Block-16 residual bounds
        do_blocks = (r in (64, 256))
        if do_blocks:
            w_perp_b8 = np.empty((V, 8), dtype=np.float32)
            w_perp_b16 = np.empty((V, 16), dtype=np.float32)

        for s in range(0, V, 32768):
            e = min(V, s + 32768)
            wc = mx.array(W[s:e])
            wp = wc @ P_mx.T
            w_perp_c = wc - wp @ P_mx
            wn = mx.sqrt(mx.sum(w_perp_c * w_perp_c, axis=1))
            if do_blocks:
                wb8 = mx.sqrt(mx.sum(w_perp_c.reshape(-1, 8, 512) ** 2, axis=2))
                wb16 = mx.sqrt(mx.sum(w_perp_c.reshape(-1, 16, 256) ** 2, axis=2))
                mx.eval(wp, wn, wb8, wb16)
                w_perp_b8[s:e] = np.array(wb8)
                w_perp_b16[s:e] = np.array(wb16)
            else:
                mx.eval(wp, wn)
            W_par[s:e] = np.array(wp)
            w_perp_norm[s:e] = np.array(wn)

        s_par = H_test_par @ W_par.T  # (N_test, V)
        margin_global = h_perp_norm[:, None] * w_perp_norm[None, :]
        res_g = evaluate_pruner_stats(
            f"act_pca_r{r}_global",
            s_par,
            margin_global,
            z_exact_f32,
            true_top1_id,
            true_top1_val,
            stock_top1_id,
            round_eps_bf16=bf16_max_diff,
        )
        res_g["sketch_mb_fp16"] = round((V * r * 2 + V * 4) / 1e6, 2)
        res_g["heldout_h_energy_pct"] = round(100.0 * test_par_energy_frac, 2)
        mechanism_results.append(res_g)
        print(
            f"  act_pca r={r:3d} (global): "
            f"Oracle skip={res_g['oracle_skip_pct_mean']:6.2f}%, "
            f"Seeded skip={res_g['seeded_skip_pct_mean']:6.2f}%, "
            f"Opus full-skip={res_g['opus_full_head_skip_pct']:5.2f}%, "
            f"CS looseness={res_g['median_cauchy_schwarz_looseness_x']:.1f}x, "
            f"mismatches={res_g['greedy_mismatches_vs_f32']}"
        )

        if res_g["seeded_skip_pct_mean"] > best_skip_seen:
            best_skip_seen = res_g["seeded_skip_pct_mean"]
            best_pca_for_timing = (r, P.copy(), W_par.copy(), w_perp_norm.copy())

        if do_blocks:
            h_perp_b8 = np.linalg.norm(H_test_perp.reshape(N_test, 8, 512), axis=2)
            h_perp_b16 = np.linalg.norm(H_test_perp.reshape(N_test, 16, 256), axis=2)
            margin_b8 = h_perp_b8 @ w_perp_b8.T
            margin_b16 = h_perp_b16 @ w_perp_b16.T
            for tag, mb, extra_bytes in [
                (f"act_pca_r{r}_block8", margin_b8, V * 8 * 2),
                (f"act_pca_r{r}_block16", margin_b16, V * 16 * 2),
            ]:
                rb = evaluate_pruner_stats(
                    tag,
                    s_par,
                    mb,
                    z_exact_f32,
                    true_top1_id,
                    true_top1_val,
                    stock_top1_id,
                    round_eps_bf16=bf16_max_diff,
                )
                rb["sketch_mb_fp16"] = round((V * r * 2 + extra_bytes) / 1e6, 2)
                mechanism_results.append(rb)
                print(
                    f"  {tag:22s}: "
                    f"Oracle skip={rb['oracle_skip_pct_mean']:6.2f}%, "
                    f"Seeded skip={rb['seeded_skip_pct_mean']:6.2f}%, "
                    f"CS looseness={rb['median_cauchy_schwarz_looseness_x']:.1f}x, "
                    f"mismatches={rb['greedy_mismatches_vs_f32']}"
                )

    # -----------------------------------------------------------------------
    # Weight-SVD Basis Control (r=64, 256) + Astra 6.2's 8x64 Coord-Group Split
    # -----------------------------------------------------------------------
    print("[Duel 1] Control: Weight-SVD basis (r=64, 256) & Astra 6.2 8x64 coord groups...")
    WtW = np.zeros((D, D), dtype=np.float64)
    for s in range(0, V, 32768):
        e = min(V, s + 32768)
        wc = mx.array(W[s:e])
        g_chunk = wc.T @ wc
        mx.eval(g_chunk)
        WtW += np.array(g_chunk, dtype=np.float64)
    eigvals_w, eigvecs_w = np.linalg.eigh(WtW)
    order_w = np.argsort(eigvals_w)[::-1]
    U_w = eigvecs_w[:, order_w].astype(np.float32)

    for r in [64, 256]:
        Pw = U_w[:, :r].T
        Pw_mx = mx.array(Pw)
        H_par = H_test @ Pw.T
        h_perp_norm = np.linalg.norm(H_test - H_par @ Pw, axis=1)
        W_par = np.empty((V, r), dtype=np.float32)
        w_perp_norm = np.empty((V,), dtype=np.float32)
        for s in range(0, V, 32768):
            e = min(V, s + 32768)
            wc = mx.array(W[s:e])
            wp = wc @ Pw_mx.T
            wn = mx.sqrt(mx.sum((wc - wp @ Pw_mx) ** 2, axis=1))
            mx.eval(wp, wn)
            W_par[s:e] = np.array(wp)
            w_perp_norm[s:e] = np.array(wn)
        s_par = H_par @ W_par.T
        res_w = evaluate_pruner_stats(
            f"weight_svd_r{r}_global",
            s_par,
            h_perp_norm[:, None] * w_perp_norm[None, :],
            z_exact_f32,
            true_top1_id,
            true_top1_val,
            stock_top1_id,
            round_eps_bf16=bf16_max_diff,
        )
        res_w["sketch_mb_fp16"] = round((V * r * 2 + V * 4) / 1e6, 2)
        mechanism_results.append(res_w)
        print(
            f"  weight_svd r={r:3d}      : "
            f"Oracle skip={res_w['oracle_skip_pct_mean']:6.2f}%, "
            f"Seeded skip={res_w['seeded_skip_pct_mean']:6.2f}%, "
            f"CS looseness={res_w['median_cauchy_schwarz_looseness_x']:.1f}x"
        )

    # Astra 6.2's literal 8 x 64-coordinate groups (512 coords aligned to Q4 group_size=64)
    cal_grp_energy = np.sum(H_cal.reshape(N_cal, 64, 64) ** 2, axis=(0, 2))
    top8_grps = np.sort(np.argsort(cal_grp_energy)[-8:])
    keep_mask = np.zeros(64, dtype=bool)
    keep_mask[top8_grps] = True
    keep_cols = np.repeat(keep_mask, 64)
    rest_cols = ~keep_cols

    s_astra = H_test[:, keep_cols] @ W[:, keep_cols].T
    h_rest = H_test[:, rest_cols]  # (N_test, 3584)
    h_rest_norm = np.linalg.norm(h_rest, axis=1)
    w_rest_norm = np.linalg.norm(W[:, rest_cols], axis=1)
    res_astra = evaluate_pruner_stats(
        "astra_8x64_coord_groups_global",
        s_astra,
        h_rest_norm[:, None] * w_rest_norm[None, :],
        z_exact_f32,
        true_top1_id,
        true_top1_val,
        stock_top1_id,
        round_eps_bf16=bf16_max_diff,
    )
    res_astra["sketch_mb_q4"] = round(head_bytes / 8.0 / 1e6, 2)
    mechanism_results.append(res_astra)
    # Block-7 on the remaining 3584 coords (7 blocks of 512)
    h_rest_b7 = np.linalg.norm(h_rest.reshape(N_test, 7, 512), axis=2)
    w_rest_b7 = np.linalg.norm(W[:, rest_cols].reshape(V, 7, 512), axis=2)
    res_astra_b7 = evaluate_pruner_stats(
        "astra_8x64_coord_groups_block7",
        s_astra,
        h_rest_b7 @ w_rest_b7.T,
        z_exact_f32,
        true_top1_id,
        true_top1_val,
        stock_top1_id,
        round_eps_bf16=bf16_max_diff,
    )
    mechanism_results.append(res_astra_b7)
    print(
        f"  astra 8x64 global     : Oracle skip={res_astra['oracle_skip_pct_mean']:6.2f}%, "
        f"Seeded skip={res_astra['seeded_skip_pct_mean']:6.2f}% | "
        f"block7 Oracle skip={res_astra_b7['oracle_skip_pct_mean']:6.2f}%"
    )

    # -----------------------------------------------------------------------
    # Mechanism (b): Opus 5.5's Hadamard-Q2 Sketch + Exact Residual Norm Bound
    # -----------------------------------------------------------------------
    print("[Duel 1] Mechanism (b): Opus 5.5 Hadamard-Q2 Sketch + Residual Norm Bounds...")
    H_test_rot_mx = fwht_block4096(mx.array(H_test), out_dtype=mx.float32)
    mx.eval(H_test_rot_mx)
    H_test_rot = np.array(H_test_rot_mx)

    h_rot_l2 = np.linalg.norm(H_test_rot, axis=1)
    h_rot_l1 = np.sum(np.abs(H_test_rot), axis=1)
    h_rot_b8 = np.linalg.norm(H_test_rot.reshape(N_test, 8, 512), axis=2)
    h_rot_b16 = np.linalg.norm(H_test_rot.reshape(N_test, 16, 256), axis=2)

    s_q2 = np.empty((N_test, V), dtype=np.float32)
    e_l2 = np.empty((V,), dtype=np.float32)
    e_linf = np.empty((V,), dtype=np.float32)
    e_b8 = np.empty((V, 8), dtype=np.float32)
    e_b16 = np.empty((V, 16), dtype=np.float32)

    for s in range(0, V, 32768):
        e = min(V, s + 32768)
        w_rot = fwht_block4096(mx.array(W[s:e]), out_dtype=mx.float32)
        w2_q, s2_q, b2_q = mx.quantize(
            w_rot.astype(mx.bfloat16), group_size=64, bits=2, mode="affine"
        )
        w_hat = mx.dequantize(
            w2_q, s2_q, b2_q, group_size=64, bits=2, mode="affine"
        ).astype(mx.float32)
        err = w_rot - w_hat

        sc_chunk = H_test_rot_mx @ w_hat.T
        el2 = mx.sqrt(mx.sum(err * err, axis=1))
        elinf = mx.max(mx.abs(err), axis=1)
        eb8 = mx.sqrt(mx.sum(err.reshape(-1, 8, 512) ** 2, axis=2))
        eb16 = mx.sqrt(mx.sum(err.reshape(-1, 16, 256) ** 2, axis=2))
        mx.eval(sc_chunk, el2, elinf, eb8, eb16)

        s_q2[:, s:e] = np.array(sc_chunk)
        e_l2[s:e] = np.array(el2)
        e_linf[s:e] = np.array(elinf)
        e_b8[s:e] = np.array(eb8)
        e_b16[s:e] = np.array(eb16)

    q2_bytes = V * (4096 // 16 * 4 + 64 * 4)  # 2-bit weights + bf16 scales+biases
    for tag, margin_mat, extra_b in [
        ("opus_hadamard_q2_global_L2", h_rot_l2[:, None] * e_l2[None, :], V * 4),
        ("opus_hadamard_q2_block8", h_rot_b8 @ e_b8.T, V * 8 * 2),
        ("opus_hadamard_q2_block16", h_rot_b16 @ e_b16.T, V * 16 * 2),
        ("opus_hadamard_q2_L1_Linf", h_rot_l1[:, None] * e_linf[None, :], V * 4),
    ]:
        ro = evaluate_pruner_stats(
            tag,
            s_q2,
            margin_mat,
            z_exact_f32,
            true_top1_id,
            true_top1_val,
            stock_top1_id,
            round_eps_bf16=bf16_max_diff,
        )
        ro["sketch_mb_q2"] = round((q2_bytes + extra_b) / 1e6, 2)
        mechanism_results.append(ro)
        print(
            f"  {tag:28s}: "
            f"Oracle skip={ro['oracle_skip_pct_mean']:6.2f}%, "
            f"Seeded skip={ro['seeded_skip_pct_mean']:6.2f}%, "
            f"Opus full-skip={ro['opus_full_head_skip_pct']:5.2f}%, "
            f"CS looseness={ro['median_cauchy_schwarz_looseness_x']:.1f}x, "
            f"mismatches={ro['greedy_mismatches_vs_f32']}"
        )

    # Free large 4 GB host array before GPU timing benchmarks
    del W, z_exact_f32, s_q2, s_astra
    clear_memory()

    # -----------------------------------------------------------------------
    # Mechanism (d): Real Metal/MLX Gathered Rescore Verification & Timing
    # -----------------------------------------------------------------------
    print("[Duel 1] Mechanism (d): Verifying & benchmarking Metal gathered Q4 lm_head...")
    h_bench = H_test_bf16[0]  # (4096,) bfloat16
    h_bench_2d = h_bench.reshape(1, 4096)

    # Correctness check of gathered_q4_matvec on 2048 random vocab rows
    rng = np.random.default_rng(42)
    check_idx = mx.array(rng.choice(V, size=2048, replace=False).astype(np.uint32))
    ref_full = lm_head(h_bench_2d)[0].astype(mx.float32)
    got_gath = gathered_q4_matvec(lm_head, h_bench, check_idx)
    mx.eval(ref_full, got_gath)
    gather_max_err = float(mx.max(mx.abs(got_gath - ref_full[check_idx])).item())
    print(f"  Metal gathered_q4_matvec correctness (2048 rows vs stock bf16): max_abs_err={gather_max_err:.5f}")
    assert gather_max_err < 0.25, f"Gathered kernel mismatch: {gather_max_err}"

    # Prepare actual best exact 2-stage pruner on GPU for live end-to-end timing
    assert best_pca_for_timing is not None
    r_best, P_best, W_par_best, w_perp_best = best_pca_for_timing
    P_best_mx = mx.array(P_best, dtype=mx.float32)
    W_par_mx = mx.array(W_par_best, dtype=mx.float16)
    # Exact Cauchy-Schwarz allowance for FP16 rounding of W_par
    w_par_qerr = np.linalg.norm(
        W_par_best - np.array(W_par_mx.astype(mx.float32)), axis=1
    ).astype(np.float32)
    w_par_qerr_mx = mx.array(w_par_qerr, dtype=mx.float32)
    w_perp_mx = mx.array(w_perp_best, dtype=mx.float32)
    mx.eval(P_best_mx, W_par_mx, w_par_qerr_mx, w_perp_mx)

    def run_exact_two_stage_pruned(h_vec_bf16: mx.array) -> Tuple[int, int]:
        h_f32 = h_vec_bf16.astype(mx.float32)
        h_par = P_best_mx @ h_f32  # (r,) float32
        h_perp = h_f32 - (h_par @ P_best_mx)
        h_par_n = mx.sqrt(mx.sum(h_par * h_par))
        h_perp_n = mx.sqrt(mx.sum(h_perp * h_perp))
        s_par_vec = (W_par_mx @ h_par.astype(mx.float16)).astype(mx.float32)
        upper = s_par_vec + w_perp_mx * h_perp_n + w_par_qerr_mx * h_par_n + 2e-3

        # Score top-16 candidates exactly
        top16 = mx.argpartition(s_par_vec, -16)[-16:].astype(mx.uint32)
        z_top16 = gathered_q4_matvec(lm_head, h_vec_bf16, top16)
        exact_best = mx.max(z_top16)
        best_seed_idx = int(top16[int(mx.argmax(z_top16).item())].item())
        best_seed_val = float(exact_best.item())

        # Find survivors (dynamic compaction across CPU/GPU boundary is real cost)
        surv_mask = upper >= exact_best
        mx.eval(surv_mask)
        surv_np = np.flatnonzero(np.array(surv_mask)).astype(np.uint32)
        if len(surv_np) == 0:
            return best_seed_idx, 16
        surv_mx = mx.array(surv_np, dtype=mx.uint32)
        z_surv = gathered_q4_matvec(lm_head, h_vec_bf16, surv_mx)
        mx.eval(z_surv)
        best_i = int(mx.argmax(z_surv).item())
        best_surv_val = float(z_surv[best_i].item())
        if best_surv_val >= best_seed_val:
            return int(surv_np[best_i]), int(len(surv_np))
        return best_seed_idx, int(len(surv_np))

    # Verify 0 mismatches on the real GPU 2-stage implementation across first 16 test states
    gpu_verify_n = min(16, N_test)
    gpu_mismatches = 0
    gpu_surv_counts = []
    for i in range(gpu_verify_n):
        pred_id, n_surv = run_exact_two_stage_pruned(H_test_bf16[i])
        gpu_surv_counts.append(n_surv)
        if pred_id != int(true_top1_id[i]):
            gpu_mismatches += 1
    print(
        f"  Live GPU 2-stage pruner (act_pca r={r_best}): "
        f"mismatches={gpu_mismatches}/{gpu_verify_n}, "
        f"mean survivors={np.mean(gpu_surv_counts):.0f}/{V} "
        f"({100.0 * np.mean(gpu_surv_counts) / V:.2f}%)"
    )

    # Pre-generate sorted random index arrays for synthetic survivor-rate break-even curve
    sweep_fracs = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    sweep_indices: Dict[str, mx.array] = {}
    for f in sweep_fracs:
        k_rows = max(1, int(round(V * f)))
        idx_np = np.sort(rng.choice(V, size=k_rows, replace=False)).astype(np.uint32)
        sweep_indices[f"{int(f * 100)}pct"] = mx.array(idx_np)
        mx.eval(sweep_indices[f"{int(f * 100)}pct"])

    # Interleaved cold-SLC timing benchmark
    bench_arms: Dict[str, Any] = {
        "stock_lm_head_full": lambda: lm_head(h_bench_2d),
        f"live_2stage_act_pca_r{r_best}": lambda: mx.array(
            [run_exact_two_stage_pruned(h_bench)[0]], dtype=mx.int32
        ),
    }
    for label, idx_arr in sweep_indices.items():
        bench_arms[f"gathered_only_{label}"] = (
            lambda arr=idx_arr: gathered_q4_matvec(lm_head, h_bench, arr)
        )

    timing = bench_interleaved(bench_arms, flusher=flusher, trials=trials, warmup=2)
    stock_ms = timing["stock_lm_head_full"]["median_ms"]
    live_2stage_ms = timing[f"live_2stage_act_pca_r{r_best}"]["median_ms"]
    live_speedup = stock_ms / live_2stage_ms
    stock_bw_gbs = (head_bytes / 1e9) / (stock_ms / 1e3)

    # Determine overall best Oracle skip and honest verdict
    best_mech = max(mechanism_results, key=lambda d: d["oracle_skip_pct_mean"])
    best_seeded = max(mechanism_results, key=lambda d: d["seeded_skip_pct_mean"])
    passed = (
        live_speedup >= 1.15
        and gpu_mismatches == 0
        and best_seeded["greedy_mismatches_vs_f32"] == 0
    )
    verdict = "WIN" if passed else "KILLED"

    print(
        f"[Duel 1] TIMING (cold 540MB SLC): stock={stock_ms:.3f} ms ({stock_bw_gbs:.1f} GB/s) | "
        f"live_2stage={live_2stage_ms:.3f} ms ({live_speedup:.3f}x) -> VERDICT: {verdict}"
    )
    for f in sweep_fracs:
        lbl = f"gathered_only_{int(f * 100)}pct"
        ms = timing[lbl]["median_ms"]
        print(f"    {lbl:22s}: {ms:.3f} ms ({stock_ms / ms:.2f}x vs stock full)")

    return {
        "duel": "lm_head_cert_duel",
        "verdict": verdict,
        "pass_mark": ">=1.15x wall-clock lm_head speedup AND 0 greedy top-1 mismatches",
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "config": {
            "model_id": MODEL_ID,
            "vocab_size": V,
            "hidden_size": D,
            "lm_head_q4_mb": round(head_bytes / 1e6, 2),
            "n_cal_states": N_cal,
            "n_test_states": N_test,
            "trials": trials,
            "slc_flush_mb": flusher.mb,
        },
        "logit_geometry": {
            "median_h_l2_norm": round(float(np.median(h_norms)), 3),
            "median_w_l2_norm": round(float(np.median(W_norm)), 4),
            "median_z_top1": round(float(np.median(true_top1_val)), 3),
            "median_z_top1_minus_top2": round(float(np.median(logit_margins)), 3),
            "median_z_top1_minus_median_z": round(float(np.median(z_spread_top_minus_med)), 3),
            "stock_bf16_vs_f32_max_abs_diff": round(bf16_max_diff, 5),
            "stock_bf16_vs_f32_top1_agree": f"{bf16_top1_agree}/{N_test}",
        },
        "activation_pca_energy": pca_energy_table,
        "mechanisms": mechanism_results,
        "best_oracle_mechanism": {
            "name": best_mech["name"],
            "oracle_skip_pct_mean": best_mech["oracle_skip_pct_mean"],
            "oracle_alive_pct_mean": best_mech["oracle_alive_pct_mean"],
            "opus_full_head_skip_pct": best_mech["opus_full_head_skip_pct"],
        },
        "best_seeded_mechanism": {
            "name": best_seeded["name"],
            "seeded_skip_pct_mean": best_seeded["seeded_skip_pct_mean"],
            "greedy_mismatches_vs_f32": best_seeded["greedy_mismatches_vs_f32"],
        },
        "metal_gathered_kernel": {
            "correctness_max_abs_err_vs_stock_bf16": round(gather_max_err, 5),
            "live_2stage_gpu_mismatches": f"{gpu_mismatches}/{gpu_verify_n}",
            "live_2stage_mean_survivors": round(float(np.mean(gpu_surv_counts)), 1),
            "stock_lm_head_median_ms": round(stock_ms, 4),
            "stock_lm_head_effective_gbs": round(stock_bw_gbs, 1),
            "live_2stage_median_ms": round(live_2stage_ms, 4),
            "live_2stage_speedup_x": round(live_speedup, 4),
            "timing_details": timing,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    res = run_lm_head_duel(quick=args.quick)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved Duel 1 results to {args.out}")


if __name__ == "__main__":
    main()
