"""Duel 3: Aug 2026 Rank-Trace + Sylvester Inertia & Iterated-Parseval FWHT Audit.

Two concrete empirical audits on REAL `Qwen3.5-9B-4bit` weights & states:
  (a) Real `Qwen3.5-9B` Full-Attention KV Gram Matrices (`K^T K`, `V^T V`) across a
      2048-token real context on all 8 full-attention layers `[3, 7, 11, 15, 19, 23, 27, 31]`:
      - Exact Rank-Trace Lower Bound: `r_eff = (tr G)^2 / tr(G^2)` (zero eigendecomp!)
      - Exact Sylvester Inertia eigenvalue count above `tau` via symmetric indefinite
        pivoted LDL^T (Bunch-Kaufman 1x1 / 2x2 pivots), verified against exact `eigvalsh`
        with 0 discrepancies!
      - Exact attention output error & KL divergence + live end-to-end PPL delta +
        cold-SLC attention speedup at 2x (r=128) and 4x (r=64) certified low-rank KV.
  (b) Real `Qwen3.5-9B` MLP & Attention weight matrices under 2-step Iterated-Parseval
      FWHT + Q2 + low-rank FP16 residual (`W ≈ Q2 + U_r V_r^T`), measuring weight RMSE,
      real-activation error, live patched PPL/KL delta, and real cold-DRAM speed vs Q4!
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
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from common import (
    HELD_OUT_PROMPTS,
    MODEL_ID,
    SLCFlusher,
    bench_interleaved,
    clear_memory,
    fwht_block4096,
    load_long_context_text,
)


# ---------------------------------------------------------------------------
# Exact Pure-NumPy Sylvester Inertia via Bunch-Kaufman Pivoted LDL^T
# Counts (n_pos, n_zero, n_neg) eigenvalues of Hermitian/symmetric A = G - tau*I
# WITHOUT computing full eigendecomposition!
# ---------------------------------------------------------------------------

def sylvester_inertia_ldlt(A: np.ndarray, zero_tol: float = 1e-12) -> Tuple[int, int, int]:
    """Compute exact Sylvester inertia (n_pos, n_zero, n_neg) of real symmetric A via
    complete-pivoted Bunch-Kaufman 1x1 / 2x2 block LDL^T elimination (O(n^3 / 3))."""
    M = np.array(A, dtype=np.float64, copy=True)
    n = M.shape[0]
    n_pos = 0
    n_zero = 0
    n_neg = 0
    alpha = (1.0 + math.sqrt(17.0)) / 8.0  # Bunch-Kaufman threshold ~0.6404

    k = 0
    while k < n:
        # Largest diagonal entry in trailing submatrix M[k:, k:]
        diag_abs = np.abs(np.diag(M[k:, k:]))
        p_rel = int(np.argmax(diag_abs))
        max_diag = float(diag_abs[p_rel])

        if k == n - 1:
            d = float(M[k, k])
            if d > zero_tol:
                n_pos += 1
            elif d < -zero_tol:
                n_neg += 1
            else:
                n_zero += 1
            break

        # Swap largest diagonal to position k
        if p_rel != 0:
            p = k + p_rel
            M[[k, p], :] = M[[p, k], :]
            M[:, [k, p]] = M[:, [p, k]]

        # Find largest off-diagonal in column k below diagonal
        col_abs = np.abs(M[k + 1 :, k])
        r_rel = int(np.argmax(col_abs))
        omega = float(col_abs[r_rel])

        if max(max_diag, omega) <= zero_tol:
            n_zero += n - k
            break

        # Decide 1x1 vs 2x2 pivot
        if max_diag >= alpha * omega:
            # 1x1 pivot at (k, k)
            d = float(M[k, k])
            if d > zero_tol:
                n_pos += 1
            elif d < -zero_tol:
                n_neg += 1
            else:
                n_zero += 1
            if abs(d) > zero_tol:
                v = M[k + 1 :, k]
                M[k + 1 :, k + 1 :] -= np.outer(v, v) / d
            k += 1
        else:
            # Move row/col (k + 1 + r_rel) to (k + 1) for a 2x2 block pivot
            r = k + 1 + r_rel
            if r != k + 1:
                M[[k + 1, r], :] = M[[r, k + 1], :]
                M[:, [k + 1, r]] = M[:, [r, k + 1]]

            a11 = float(M[k, k])
            a21 = float(M[k + 1, k])
            a22 = float(M[k + 1, k + 1])
            det = a11 * a22 - a21 * a21
            tr = a11 + a22
            # Eigenvalues of symmetric 2x2 block [[a11, a21], [a21, a22]]
            disc = math.sqrt(max(0.0, (a11 - a22) ** 2 + 4.0 * a21 * a21))
            lam1 = 0.5 * (tr + disc)
            lam2 = 0.5 * (tr - disc)
            for lam in (lam1, lam2):
                if lam > zero_tol:
                    n_pos += 1
                elif lam < -zero_tol:
                    n_neg += 1
                else:
                    n_zero += 1

            # Schur complement update for 2x2 block
            if k + 2 < n and abs(det) > zero_tol * zero_tol:
                B = M[k + 2 :, k : k + 2]  # (m, 2)
                inv_D = np.array([[a22, -a21], [-a21, a11]], dtype=np.float64) / det
                M[k + 2 :, k + 2 :] -= B @ inv_D @ B.T
            k += 2

    return n_pos, n_zero, n_neg


def rank_trace_lower_bound(G: np.ndarray) -> float:
    """Exact Aug 2026 rank-trace lower bound: r_eff = (tr G)^2 / tr(G^2)."""
    G64 = G.astype(np.float64)
    tr_g = float(np.trace(G64))
    tr_g2 = float(np.sum(G64 * G64))  # Since G is symmetric, tr(G^2) = ||G||_F^2
    return (tr_g * tr_g) / max(tr_g2, 1e-30)


# ---------------------------------------------------------------------------
# Part (a): Real 2048-Token Full-Attention KV Gram & Low-Rank Compression Audit
# ---------------------------------------------------------------------------

def run_kv_spectral_audit(
    model,
    tokenizer,
    flusher: SLCFlusher,
    quick: bool = False,
) -> Dict[str, Any]:
    lm = model.language_model
    layers = lm.layers
    fa_layer_ids = [i for i, l in enumerate(layers) if not l.is_linear]

    ctx_len = 1024 if quick else 2048
    t_eval_q = 64 if quick else 128
    total_len = ctx_len + t_eval_q
    trials = 5 if quick else 11

    corpus = load_long_context_text()
    toks = tokenizer.encode(corpus)
    while len(toks) < total_len + 16:
        toks = toks + toks
    toks = toks[:total_len]
    ids = mx.array(toks, dtype=mx.int32)[None, :]
    print(
        f"[Duel 3a] Prefilling {ctx_len}-tok prefix + {t_eval_q} strictly unseen continuation tokens "
        f"across all {len(fa_layer_ids)} full-attn layers..."
    )

    # Forward pass capturing exact post-RoPE Q, K, V at every full-attention layer
    h = lm.model.embed_tokens(ids)
    fa_mask = create_attention_mask(h, None)
    ssm_mask = create_ssm_mask(h, None)

    captured_qkv: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for l_idx, layer in enumerate(layers):
        norm_x = layer.input_layernorm(h)
        if layer.is_linear:
            r = layer.linear_attn(norm_x, ssm_mask, None)
        else:
            attn = layer.self_attn
            B, L, _ = norm_x.shape
            q_proj_out = attn.q_proj(norm_x)
            queries, gate = mx.split(
                q_proj_out.reshape(B, L, attn.num_attention_heads, -1), 2, axis=-1
            )
            gate = gate.reshape(B, L, -1)
            keys = attn.k_proj(norm_x)
            values = attn.v_proj(norm_x)
            queries = attn.q_norm(queries).transpose(0, 2, 1, 3)  # (1, 16, L, 256)
            keys = attn.k_norm(
                keys.reshape(B, L, attn.num_key_value_heads, -1)
            ).transpose(0, 2, 1, 3)  # (1, 4, L, 256)
            values = values.reshape(
                B, L, attn.num_key_value_heads, -1
            ).transpose(0, 2, 1, 3)  # (1, 4, L, 256)
            queries = attn.rope(queries)
            keys = attn.rope(keys)

            out = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=attn.scale, mask=fa_mask
            )
            out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
            r = attn.o_proj(out * mx.sigmoid(gate))
            mx.eval(r, queries, keys, values)

            # Save STRICTLY unseen continuation queries [ctx_len:] + prefix-only K, V [:ctx_len]
            # (zero leakage of continuation tokens into the Gram matrix / SVD basis!)
            captured_qkv[l_idx] = (
                np.array(queries[0, :, ctx_len:, :].astype(mx.float32)),   # (16, t_eval_q, 256)
                np.array(keys[0, :, :ctx_len, :].astype(mx.float32)),      # (4, ctx_len, 256)
                np.array(values[0, :, :ctx_len, :].astype(mx.float32)),    # (4, ctx_len, 256)
                np.array(keys[0, :, ctx_len:, :].astype(mx.float32)),      # (4, t_eval_q, 256)
                np.array(values[0, :, ctx_len:, :].astype(mx.float32)),    # (4, t_eval_q, 256)
            )
        h_mid = h + r
        h = h_mid + layer.mlp(layer.post_attention_layernorm(h_mid))
        mx.eval(h)

    # Compute baseline logits predicting the t_eval_q unseen continuation tokens
    ref_logits_tail = lm.lm_head(lm.model.norm(h)[:, ctx_len - 1 : total_len - 1, :]).astype(mx.float32)
    targets_tail = ids[0, ctx_len:total_len]
    ref_lp_tail = ref_logits_tail[0] - mx.logsumexp(ref_logits_tail[0], axis=-1, keepdims=True)
    ref_nll_tail = float(-mx.mean(ref_lp_tail[mx.arange(t_eval_q), targets_tail]).item())
    ref_ppl_tail = math.exp(ref_nll_tail)
    mx.eval(ref_logits_tail)
    ref_logits_np = np.array(ref_logits_tail[0])
    ref_top1_np = np.argmax(ref_logits_np, axis=-1)

    # Analyze Gram matrices G_K and G_V across all 8 full-attention layers x 4 KV heads
    ranks = [128, 64, 32]  # 2x, 4x, 8x KV compression (head_dim=256)
    tau_fracs = [1e-1, 1e-2, 1e-3]  # fraction of lambda_max
    sylvester_discrepancies = 0
    sylvester_total_checks = 0

    layer_spectral_rows = []
    # Also store U_k, U_v bases per layer/head (fitted strictly on prefix [:ctx_len])
    kv_bases: Dict[int, Dict[int, Tuple[mx.array, mx.array]]] = {r: {} for r in ranks}
    rank_agg_metrics: Dict[int, Dict[str, List[float]]] = {
        r: {"attn_out_rel_l2": [], "attn_kl_nats": [], "k_energy_pct": [], "v_energy_pct": []}
        for r in ranks
    }

    scale = 256.0 ** -0.5
    # Causal mask for unseen continuation queries [ctx_len..total_len-1] attending to [0..total_len-1]
    q_pos = np.arange(ctx_len, total_len)[:, None]
    k_pos = np.arange(total_len)[None, :]
    causal_mask = np.where(k_pos <= q_pos, 0.0, -1e9).astype(np.float32)

    for l_idx in fa_layer_ids:
        Q_l, K_pre, V_pre, K_cont, V_cont = captured_qkv[l_idx]
        r_eff_k_list, r_eff_v_list = [], []
        sylv_k_counts = {tf: [] for tf in tau_fracs}
        sylv_v_counts = {tf: [] for tf in tau_fracs}
        Uk_heads = {r: [] for r in ranks}
        Uv_heads = {r: [] for r in ranks}

        for h_kv in range(4):
            Kh_pre = K_pre[h_kv]  # (ctx_len, 256)
            Vh_pre = V_pre[h_kv]  # (ctx_len, 256)
            Kh_full = np.concatenate([Kh_pre, K_cont[h_kv]], axis=0)  # (total_len, 256)
            Vh_full = np.concatenate([Vh_pre, V_cont[h_kv]], axis=0)

            Gk = Kh_pre.astype(np.float64).T @ Kh_pre.astype(np.float64)
            Gv = Vh_pre.astype(np.float64).T @ Vh_pre.astype(np.float64)

            r_eff_k_list.append(rank_trace_lower_bound(Gk))
            r_eff_v_list.append(rank_trace_lower_bound(Gv))

            eig_k, vec_k = np.linalg.eigh(Gk)
            eig_v, vec_v = np.linalg.eigh(Gv)
            eig_k = eig_k[::-1]
            vec_k = vec_k[:, ::-1]
            eig_v = eig_v[::-1]
            vec_v = vec_v[:, ::-1]

            # Sylvester inertia verification against exact eigvalsh
            for tf in tau_fracs:
                tau_k = tf * float(eig_k[0])
                tau_v = tf * float(eig_v[0])
                pk, _, _ = sylvester_inertia_ldlt(Gk - tau_k * np.eye(256))
                pv, _, _ = sylvester_inertia_ldlt(Gv - tau_v * np.eye(256))
                exact_pk = int(np.sum(eig_k > tau_k))
                exact_pv = int(np.sum(eig_v > tau_v))
                sylvester_total_checks += 2
                if pk != exact_pk:
                    sylvester_discrepancies += 1
                if pv != exact_pv:
                    sylvester_discrepancies += 1
                sylv_k_counts[tf].append(pk)
                sylv_v_counts[tf].append(pv)

            # Fast BLAS exact attention output for the 4 GQA query heads sharing KV head h_kv
            Q_grp = Q_l[h_kv * 4 : (h_kv + 1) * 4]  # (4, t_eval_q, 256)
            scores_ref = (Q_grp @ Kh_full.T) * scale + causal_mask[None, :, :]
            s_max = np.max(scores_ref, axis=-1, keepdims=True)
            exp_ref = np.exp(scores_ref - s_max)
            p_ref = exp_ref / np.sum(exp_ref, axis=-1, keepdims=True)
            lp_ref = (scores_ref - s_max) - np.log(np.sum(exp_ref, axis=-1, keepdims=True))
            out_ref = p_ref @ Vh_full  # (4, t_eval_q, 256)
            out_ref_norm = float(np.linalg.norm(out_ref))

            sum_ek = max(1e-12, float(np.sum(np.maximum(eig_k, 0.0))))
            sum_ev = max(1e-12, float(np.sum(np.maximum(eig_v, 0.0))))

            for r in ranks:
                Uk = vec_k[:, :r].astype(np.float32)  # (256, r)
                Uv = vec_v[:, :r].astype(np.float32)  # (256, r)
                Uk_heads[r].append(Uk)
                Uv_heads[r].append(Uv)

                # Project prefix KV (and continuation KV onto prefix basis — zero future leakage!)
                K_low = Kh_full @ Uk  # (total_len, r)
                V_low = Vh_full @ Uv  # (total_len, r)
                Q_low = Q_grp @ Uk    # (4, t_eval_q, r)
                scores_low = (Q_low @ K_low.T) * scale + causal_mask[None, :, :]
                sl_max = np.max(scores_low, axis=-1, keepdims=True)
                exp_low = np.exp(scores_low - sl_max)
                p_low = exp_low / np.sum(exp_low, axis=-1, keepdims=True)
                lp_low = (scores_low - sl_max) - np.log(np.sum(exp_low, axis=-1, keepdims=True))
                out_low = (p_low @ V_low) @ Uv.T  # Fast BLAS (4, t_eval_q, r) @ (r, 256)

                rel_err = float(np.linalg.norm(out_low - out_ref)) / max(out_ref_norm, 1e-8)
                kl = float(np.mean(np.sum(p_ref * (lp_ref - lp_low), axis=-1)))
                rank_agg_metrics[r]["attn_out_rel_l2"].append(rel_err)
                rank_agg_metrics[r]["attn_kl_nats"].append(kl)
                rank_agg_metrics[r]["k_energy_pct"].append(100.0 * float(np.sum(eig_k[:r])) / sum_ek)
                rank_agg_metrics[r]["v_energy_pct"].append(100.0 * float(np.sum(eig_v[:r])) / sum_ev)

        for r in ranks:
            kv_bases[r][l_idx] = (
                mx.array(np.stack(Uk_heads[r], axis=0), dtype=mx.bfloat16),  # (4, 256, r)
                mx.array(np.stack(Uv_heads[r], axis=0), dtype=mx.bfloat16),
            )

        layer_spectral_rows.append(
            {
                "layer": l_idx,
                "K_r_eff_mean": round(float(np.mean(r_eff_k_list)), 2),
                "V_r_eff_mean": round(float(np.mean(r_eff_v_list)), 2),
                "K_sylvester_gt_1e-2_lmax": [int(c) for c in sylv_k_counts[1e-2]],
                "V_sylvester_gt_1e-2_lmax": [int(c) for c in sylv_v_counts[1e-2]],
                "K_sylvester_gt_1e-3_lmax": [int(c) for c in sylv_k_counts[1e-3]],
                "V_sylvester_gt_1e-3_lmax": [int(c) for c in sylv_v_counts[1e-3]],
            }
        )

    print(
        f"  Sylvester LDL^T inertia check: {sylvester_total_checks - sylvester_discrepancies}"
        f"/{sylvester_total_checks} EXACT matches vs eigvalsh (discrepancies={sylvester_discrepancies})"
    )
    assert sylvester_discrepancies == 0, f"Sylvester inertia mismatch: {sylvester_discrepancies}"

    # Live end-to-end PPL check when compressing all 8 full-attention KV caches to 2x/4x/8x
    live_kv_ppl: Dict[str, Any] = {
        "stock_tail_ppl": round(ref_ppl_tail, 4),
    }
    ref_mx = np.max(ref_logits_np, axis=-1, keepdims=True)
    ref_lp_np = ref_logits_np - ref_mx - np.log(np.sum(np.exp(ref_logits_np - ref_mx), axis=-1, keepdims=True))
    ref_p_np = np.exp(ref_lp_np)

    for r in [128, 64]:
        h_c = lm.model.embed_tokens(ids)
        for l_idx, layer in enumerate(layers):
            norm_x = layer.input_layernorm(h_c)
            if layer.is_linear:
                res = layer.linear_attn(norm_x, ssm_mask, None)
            else:
                attn = layer.self_attn
                B, L, _ = norm_x.shape
                q_proj_out = attn.q_proj(norm_x)
                queries, gate = mx.split(
                    q_proj_out.reshape(B, L, attn.num_attention_heads, -1), 2, axis=-1
                )
                gate = gate.reshape(B, L, -1)
                keys = attn.k_norm(
                    attn.k_proj(norm_x).reshape(B, L, attn.num_key_value_heads, -1)
                ).transpose(0, 2, 1, 3)
                values = attn.v_proj(norm_x).reshape(
                    B, L, attn.num_key_value_heads, -1
                ).transpose(0, 2, 1, 3)
                queries = attn.rope(attn.q_norm(queries).transpose(0, 2, 1, 3))
                keys = attn.rope(keys)

                # Project K and V onto rank-r Sylvester-certified basis
                Uk_m, Uv_m = kv_bases[r][l_idx]  # (4, 256, r)
                # Compress & reconstruct (equivalent to low-rank KV cache)
                keys_c = (keys[0] @ Uk_m) @ Uk_m.transpose(0, 2, 1)
                vals_c = (values[0] @ Uv_m) @ Uv_m.transpose(0, 2, 1)
                out = mx.fast.scaled_dot_product_attention(
                    queries, keys_c[None], vals_c[None], scale=attn.scale, mask=fa_mask
                )
                out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
                res = attn.o_proj(out * mx.sigmoid(gate))
            h_mid = h_c + res
            h_c = h_mid + layer.mlp(layer.post_attention_layernorm(h_mid))
            mx.eval(h_c)

        lg_c = lm.lm_head(lm.model.norm(h_c)[:, ctx_len - 1 : total_len - 1, :]).astype(mx.float32)
        lp_c = lg_c[0] - mx.logsumexp(lg_c[0], axis=-1, keepdims=True)
        nll_c = float(-mx.mean(lp_c[mx.arange(t_eval_q), targets_tail]).item())
        ppl_c = math.exp(nll_c)
        mx.eval(lg_c)
        lg_c_np = np.array(lg_c[0])
        c_mx = np.max(lg_c_np, axis=-1, keepdims=True)
        c_lp = lg_c_np - c_mx - np.log(np.sum(np.exp(lg_c_np - c_mx), axis=-1, keepdims=True))
        kl_c = float(np.mean(np.sum(ref_p_np * (ref_lp_np - c_lp), axis=-1)))
        agree_c = 100.0 * float(np.mean(np.argmax(lg_c_np, axis=-1) == ref_top1_np))
        live_kv_ppl[f"rank_{r}_{256 // r}x_comp"] = {
            "tail_ppl": round(ppl_c, 4),
            "delta_ppl": round(ppl_c - ref_ppl_tail, 4),
            "next_tok_kl_nats": round(kl_c, 5),
            "greedy_agree_pct": round(agree_c, 2),
        }
        print(
            f"  [KV Live PPL (unseen {t_eval_q} cont. toks)] r={r:3d} ({256 // r}x comp): PPL={ppl_c:.3f} "
            f"(dPPL={ppl_c - ref_ppl_tail:+.4f}), KL={kl_c:.5f}, agree={agree_c:.1f}%"
        )

    # Cold-SLC Decode Attention Timing at L=2048, 4096, 8192 across 8 full-attn layers
    # HONEST END-TO-END: includes Q @ Uk projection and out_low @ Uv^T un-projection!
    kv_timing_summary: Dict[str, Any] = {}
    bench_lens = [2048, 4096] if quick else [2048, 4096, 8192]
    for L_bench in bench_lens:
        q_full = [mx.random.normal((1, 16, 1, 256)).astype(mx.bfloat16) for _ in range(8)]
        k_full = [mx.random.normal((1, 4, L_bench, 256)).astype(mx.bfloat16) for _ in range(8)]
        v_full = [mx.random.normal((1, 4, L_bench, 256)).astype(mx.bfloat16) for _ in range(8)]

        k_128 = [mx.random.normal((1, 4, L_bench, 128)).astype(mx.bfloat16) for _ in range(8)]
        v_128 = [mx.random.normal((1, 4, L_bench, 128)).astype(mx.bfloat16) for _ in range(8)]
        uk_128 = [mx.random.normal((4, 256, 128)).astype(mx.bfloat16) for _ in range(8)]
        uvt_128 = [mx.random.normal((4, 128, 256)).astype(mx.bfloat16) for _ in range(8)]

        k_64 = [mx.random.normal((1, 4, L_bench, 64)).astype(mx.bfloat16) for _ in range(8)]
        v_64 = [mx.random.normal((1, 4, L_bench, 64)).astype(mx.bfloat16) for _ in range(8)]
        uk_64 = [mx.random.normal((4, 256, 64)).astype(mx.bfloat16) for _ in range(8)]
        uvt_64 = [mx.random.normal((4, 64, 256)).astype(mx.bfloat16) for _ in range(8)]
        mx.eval(*q_full, *k_full, *v_full, *k_128, *v_128, *uk_128, *uvt_128, *k_64, *v_64, *uk_64, *uvt_64)

        def run_sdpa_full(qf=q_full, kf=k_full, vf=v_full):
            return [
                mx.fast.scaled_dot_product_attention(q, k, v, scale=256.0 ** -0.5)
                for q, k, v in zip(qf, kf, vf)
            ]

        def run_sdpa_128(qf=q_full, k1=k_128, v1=v_128, uk=uk_128, uvt=uvt_128):
            outs = []
            for q, k, v, pk, pv in zip(qf, k1, v1, uk, uvt):
                # q: (1, 16, 1, 256) -> (4, 4, 256) @ (4, 256, 128) -> (1, 16, 1, 128)
                q_low = (q.reshape(4, 4, 256) @ pk).reshape(1, 16, 1, 128)
                o_low = mx.fast.scaled_dot_product_attention(q_low, k, v, scale=256.0 ** -0.5)
                o_full = (o_low.reshape(4, 4, 128) @ pv).reshape(1, 16, 1, 256)
                outs.append(o_full)
            return outs

        def run_sdpa_64(qf=q_full, k2=k_64, v2=v_64, uk=uk_64, uvt=uvt_64):
            outs = []
            for q, k, v, pk, pv in zip(qf, k2, v2, uk, uvt):
                q_low = (q.reshape(4, 4, 256) @ pk).reshape(1, 16, 1, 64)
                o_low = mx.fast.scaled_dot_product_attention(q_low, k, v, scale=256.0 ** -0.5)
                o_full = (o_low.reshape(4, 4, 64) @ pv).reshape(1, 16, 1, 256)
                outs.append(o_full)
            return outs

        tm = bench_interleaved(
            {
                "sdpa_full_d256": run_sdpa_full,
                "sdpa_lowrank_r128_2x": run_sdpa_128,
                "sdpa_lowrank_r64_4x": run_sdpa_64,
            },
            flusher=flusher,
            trials=trials,
            warmup=2,
        )
        t_full = tm["sdpa_full_d256"]["median_ms"]
        t_128 = tm["sdpa_lowrank_r128_2x"]["median_ms"]
        t_64 = tm["sdpa_lowrank_r64_4x"]["median_ms"]
        kv_mb = 8 * 2 * 4 * L_bench * 256 * 2 / 1e6
        kv_timing_summary[f"L={L_bench}"] = {
            "stock_kv_total_mb_8layers": round(kv_mb, 2),
            "stock_d256_ms": round(t_full, 4),
            "lowrank_r128_2x_ms": round(t_128, 4),
            "lowrank_r128_speedup_x": round(t_full / t_128, 3),
            "lowrank_r64_4x_ms": round(t_64, 4),
            "lowrank_r64_speedup_x": round(t_full / t_64, 3),
        }
        print(
            f"  [KV Timing L={L_bench} (incl. Q@Uk & O@Uv^T)] 8-layer KV={kv_mb:.1f} MB: "
            f"d256={t_full:.3f} ms | r128={t_128:.3f} ms ({t_full/t_128:.2f}x) | "
            f"r64={t_64:.3f} ms ({t_full/t_64:.2f}x)"
        )
        del q_full, k_full, v_full, k_128, v_128, uk_128, uvt_128, k_64, v_64, uk_64, uvt_64
        clear_memory()

    compression_summary = {}
    for r in ranks:
        m = rank_agg_metrics[r]
        compression_summary[f"rank_{r}_{256 // r}x_comp"] = {
            "k_energy_retained_pct_mean": round(float(np.mean(m["k_energy_pct"])), 2),
            "v_energy_retained_pct_mean": round(float(np.mean(m["v_energy_pct"])), 2),
            "attn_out_rel_l2_err_mean": round(float(np.mean(m["attn_out_rel_l2"])), 4),
            "attn_out_rel_l2_err_max": round(float(np.max(m["attn_out_rel_l2"])), 4),
            "attn_weights_kl_nats_mean": round(float(np.mean(m["attn_kl_nats"])), 5),
        }

    # Adjudicate KV part honestly (both PPL delta <= 0.15 AND real speedup incl. projections >= 1.15x)
    r128_dppl = live_kv_ppl["rank_128_2x_comp"]["delta_ppl"]
    longest_key = f"L={bench_lens[-1]}"
    r128_sp = kv_timing_summary[longest_key]["lowrank_r128_speedup_x"]
    kv_verdict = "WIN" if (r128_dppl <= 0.15 and r128_sp >= 1.15) else "KILLED"

    return {
        "verdict": kv_verdict,
        "context_tokens": ctx_len,
        "full_attention_layers": fa_layer_ids,
        "sylvester_ldlt_exact_checks": f"{sylvester_total_checks - sylvester_discrepancies}/{sylvester_total_checks}",
        "sylvester_discrepancies": sylvester_discrepancies,
        "global_K_r_eff_mean": round(float(np.mean([r["K_r_eff_mean"] for r in layer_spectral_rows])), 2),
        "global_V_r_eff_mean": round(float(np.mean([r["V_r_eff_mean"] for r in layer_spectral_rows])), 2),
        "per_layer_spectral": layer_spectral_rows,
        "compression_error_summary": compression_summary,
        "live_end_to_end_ppl": live_kv_ppl,
        "cold_slc_attention_timing": kv_timing_summary,
    }


# ---------------------------------------------------------------------------
# Part (b): Iterated-Parseval FWHT + Q2 + Low-Rank FP16 Residual Weight Audit
# ---------------------------------------------------------------------------

def iterated_parseval_transform(x: mx.array, s1: mx.array, s2: mx.array) -> mx.array:
    """2-step exact orthogonal Iterated-Parseval transform: R(x) = FWHT(s2 * FWHT(s1 * x))."""
    y = fwht_block4096(x * s1, out_dtype=mx.float32)
    return fwht_block4096(y * s2, out_dtype=mx.float32)


def randomized_truncated_svd_mlx(E: mx.array, rank: int, oversample: int = 24, n_iter: int = 2):
    """Fast accurate rank-r truncated SVD of E (M, K) on GPU via subspace iteration."""
    M, K = E.shape
    p = min(K, rank + oversample)
    # Deterministic projection matrix
    mx.random.seed(1234)
    omega = mx.random.normal((K, p), dtype=mx.float32)
    Y = E @ omega
    for _ in range(n_iter):
        # Orthonormalize on CPU (tiny M x p or K x p) for numerical stability
        mx.eval(Y)
        Q_np, _ = np.linalg.qr(np.array(Y), mode="reduced")
        Q = mx.array(Q_np, dtype=mx.float32)
        Z = E.T @ Q
        mx.eval(Z)
        Qz_np, _ = np.linalg.qr(np.array(Z), mode="reduced")
        Qz = mx.array(Qz_np, dtype=mx.float32)
        Y = E @ Qz
    mx.eval(Y)
    Q_np, _ = np.linalg.qr(np.array(Y), mode="reduced")
    Q = mx.array(Q_np, dtype=mx.float32)
    B = Q.T @ E  # (p, K)
    mx.eval(B)
    U_b, S_b, Vt_b = np.linalg.svd(np.array(B), full_matrices=False)
    U_r = (Q_np @ U_b[:, :rank]) * S_b[:rank][None, :]  # (M, rank) absorbs singular values
    V_r = Vt_b[:rank, :]                                # (rank, K)
    return mx.array(U_r, dtype=mx.float16), mx.array(V_r, dtype=mx.float16)


def run_fwht_q2_weight_audit(
    model,
    tokenizer,
    flusher: SLCFlusher,
    quick: bool = False,
) -> Dict[str, Any]:
    lm = model.language_model
    layers = lm.layers
    trials = 5 if quick else 11

    # Verify exact Parseval orthogonality of 2-step iterated FWHT
    rng = np.random.default_rng(2026)
    s1_4096 = mx.array(rng.choice([-1.0, 1.0], size=(4096,)).astype(np.float32))
    s2_4096 = mx.array(rng.choice([-1.0, 1.0], size=(4096,)).astype(np.float32))
    s1_12288 = mx.array(rng.choice([-1.0, 1.0], size=(12288,)).astype(np.float32))
    s2_12288 = mx.array(rng.choice([-1.0, 1.0], size=(12288,)).astype(np.float32))
    mx.eval(s1_4096, s2_4096, s1_12288, s2_12288)

    test_a = mx.random.normal((2, 4096), dtype=mx.float32)
    test_b = mx.random.normal((2, 4096), dtype=mx.float32)
    ra = iterated_parseval_transform(test_a, s1_4096, s2_4096)
    rb = iterated_parseval_transform(test_b, s1_4096, s2_4096)
    dot_orig = mx.sum(test_a * test_b, axis=-1)
    dot_rot = mx.sum(ra * rb, axis=-1)
    mx.eval(dot_orig, dot_rot)
    parseval_err = float(mx.max(mx.abs(dot_orig - dot_rot)).item())
    print(f"[Duel 3b] Iterated-Parseval 2-step FWHT orthogonality dot err: {parseval_err:.2e}")
    assert parseval_err < 1e-3, f"Parseval identity violated: {parseval_err}"

    # Collect real layer activations on 2 held-out prompts for real-activation error check
    eval_prompts = HELD_OUT_PROMPTS[:2] if quick else HELD_OUT_PROMPTS[:4]
    eval_ids = [mx.array(tokenizer.encode(txt), dtype=mx.int32)[None, :] for _, txt in eval_prompts]

    target_mlp_layers = [0, 15, 31] if quick else [0, 11, 23, 31]
    target_attn_layers = [3, 31] if quick else [3, 15, 27, 31]

    # Capture real inputs to those target projections
    real_inputs: Dict[str, mx.array] = {}
    h = lm.model.embed_tokens(eval_ids[0])
    fa_mask = create_attention_mask(h, None)
    ssm_mask = create_ssm_mask(h, None)
    for l_idx, layer in enumerate(layers):
        norm_x = layer.input_layernorm(h)
        if l_idx in target_attn_layers:
            real_inputs[f"L{l_idx}.attn_in"] = norm_x[0].astype(mx.float32)
        if layer.is_linear:
            r = layer.linear_attn(norm_x, ssm_mask, None)
        else:
            r = layer.self_attn(norm_x, fa_mask, None)
        h_mid = h + r
        mlp_in = layer.post_attention_layernorm(h_mid)
        if l_idx in target_mlp_layers:
            real_inputs[f"L{l_idx}.mlp_in"] = mlp_in[0].astype(mx.float32)
            g = layer.mlp.gate_proj(mlp_in)
            u = layer.mlp.up_proj(mlp_in)
            from mlx_lm.models.activations import swiglu
            real_inputs[f"L{l_idx}.down_in"] = swiglu(g, u)[0].astype(mx.float32)
        h = h_mid + layer.mlp(mlp_in)
        mx.eval(h)
    mx.eval(*real_inputs.values())

    # Audit weight matrices: Naive Q2 vs Iterated-Parseval FWHT+Q2 vs FWHT+Q2+rank-r FP16
    ranks = [32, 64, 128]
    matrix_audits: List[Dict[str, Any]] = []
    # Also store factorized replacements for live multi-layer PPL test
    # mode_name -> {(layer_idx, proj_name): callable}
    patched_projs: Dict[str, Dict[Tuple[int, str], Any]] = {
        "naive_q2": {},
        "fwht_q2": {},
        "fwht_q2_r64": {},
        "fwht_q2_r128": {},
    }

    audit_specs = []
    for l_idx in target_mlp_layers:
        mlp = layers[l_idx].mlp
        audit_specs.append((l_idx, "mlp.gate_proj", mlp.gate_proj, real_inputs[f"L{l_idx}.mlp_in"]))
        audit_specs.append((l_idx, "mlp.up_proj", mlp.up_proj, real_inputs[f"L{l_idx}.mlp_in"]))
        audit_specs.append((l_idx, "mlp.down_proj", mlp.down_proj, real_inputs[f"L{l_idx}.down_in"]))
    for l_idx in target_attn_layers:
        attn = layers[l_idx].self_attn
        audit_specs.append((l_idx, "self_attn.q_proj", attn.q_proj, real_inputs[f"L{l_idx}.attn_in"]))

    print(f"[Duel 3b] Factorizing {len(audit_specs)} real Qwen3.5-9B matrices (Q2 / FWHT+Q2 / FWHT+Q2+rank-r)...")
    for l_idx, pname, qlinear, x_real in audit_specs:
        W_q4 = mx.dequantize(
            qlinear.weight, qlinear.scales, qlinear.biases, 64, 4, "affine"
        ).astype(mx.float32)
        M_out, K_in = W_q4.shape
        y_ref = x_real @ W_q4.T
        w_frob = mx.sqrt(mx.sum(W_q4 * W_q4))
        y_norm = mx.sqrt(mx.sum(y_ref * y_ref))
        mx.eval(W_q4, y_ref, w_frob, y_norm)
        w_frob_f = float(w_frob.item())
        y_norm_f = max(float(y_norm.item()), 1e-8)

        s1 = s1_4096 if K_in == 4096 else s1_12288
        s2 = s2_4096 if K_in == 4096 else s2_12288

        # 1. Naive Q2
        w2_n, s2_n, b2_n = mx.quantize(W_q4.astype(mx.bfloat16), group_size=64, bits=2, mode="affine")
        W_n2 = mx.dequantize(w2_n, s2_n, b2_n, 64, 2, "affine").astype(mx.float32)
        err_n2 = W_q4 - W_n2
        y_err_n2 = y_ref - (x_real @ W_n2.T)
        mx.eval(err_n2, y_err_n2)
        naive_w_rel = float(mx.sqrt(mx.sum(err_n2 * err_n2)).item()) / w_frob_f
        naive_y_rel = float(mx.sqrt(mx.sum(y_err_n2 * y_err_n2)).item()) / y_norm_f
        patched_projs["naive_q2"][(l_idx, pname)] = ("q2", w2_n, s2_n, b2_n)

        # 2. Iterated-Parseval FWHT + Q2
        W_rot = iterated_parseval_transform(W_q4, s1, s2)
        x_rot = iterated_parseval_transform(x_real, s1, s2)
        w2_r, s2_r, b2_r = mx.quantize(W_rot.astype(mx.bfloat16), group_size=64, bits=2, mode="affine")
        W_rot_q2 = mx.dequantize(w2_r, s2_r, b2_r, 64, 2, "affine").astype(mx.float32)
        E_rot = W_rot - W_rot_q2
        y_err_r2 = y_ref - (x_rot @ W_rot_q2.T)
        mx.eval(E_rot, y_err_r2)
        fwht_w_rel = float(mx.sqrt(mx.sum(E_rot * E_rot)).item()) / w_frob_f
        fwht_y_rel = float(mx.sqrt(mx.sum(y_err_r2 * y_err_r2)).item()) / y_norm_f
        patched_projs["fwht_q2"][(l_idx, pname)] = ("fwht_q2", w2_r, s2_r, b2_r, s1, s2)

        # 3. Iterated-Parseval FWHT + Q2 + Rank-r FP16 residual (one SVD at max rank=128)
        U_max, V_max = randomized_truncated_svd_mlx(E_rot, rank=max(ranks))
        mx.eval(U_max, V_max)

        row_entry: Dict[str, Any] = {
            "matrix": f"L{l_idx}.{pname}",
            "shape": [M_out, K_in],
            "q4_mb": round((qlinear.weight.nbytes + qlinear.scales.nbytes + qlinear.biases.nbytes) / 1e6, 2),
            "q2_mb": round((w2_r.nbytes + s2_r.nbytes + b2_r.nbytes) / 1e6, 2),
            "naive_q2_w_rel_err": round(naive_w_rel, 4),
            "naive_q2_act_rel_err": round(naive_y_rel, 4),
            "fwht_q2_w_rel_err": round(fwht_w_rel, 4),
            "fwht_q2_act_rel_err": round(fwht_y_rel, 4),
        }
        for r in ranks:
            Ur = U_max[:, :r]
            Vr = V_max[:r, :]
            # Residual approximation error in rotated space (equals original space by Parseval!)
            E_rem = E_rot - (Ur.astype(mx.float32) @ Vr.astype(mx.float32))
            y_res = (x_rot @ Vr.astype(mx.float32).T) @ Ur.astype(mx.float32).T
            y_err_lr = y_err_r2 - y_res
            mx.eval(E_rem, y_err_lr)
            w_rel_r = float(mx.sqrt(mx.sum(E_rem * E_rem)).item()) / w_frob_f
            y_rel_r = float(mx.sqrt(mx.sum(y_err_lr * y_err_lr)).item()) / y_norm_f
            tot_mb = (w2_r.nbytes + s2_r.nbytes + b2_r.nbytes + Ur.nbytes + Vr.nbytes) / 1e6
            row_entry[f"fwht_q2_r{r}_mb"] = round(tot_mb, 2)
            row_entry[f"fwht_q2_r{r}_w_rel_err"] = round(w_rel_r, 4)
            row_entry[f"fwht_q2_r{r}_act_rel_err"] = round(y_rel_r, 4)
            if r == 64:
                patched_projs["fwht_q2_r64"][(l_idx, pname)] = (
                    "fwht_q2_lr", w2_r, s2_r, b2_r, s1, s2, Ur, Vr
                )
            elif r == 128:
                patched_projs["fwht_q2_r128"][(l_idx, pname)] = (
                    "fwht_q2_lr", w2_r, s2_r, b2_r, s1, s2, Ur, Vr
                )

        matrix_audits.append(row_entry)
        print(
            f"  {row_entry['matrix']:22s}: act_err naiveQ2={naive_y_rel:.4f} | "
            f"FWHT+Q2={fwht_y_rel:.4f} | +r64={row_entry['fwht_q2_r64_act_rel_err']:.4f} | "
            f"+r128={row_entry['fwht_q2_r128_act_rel_err']:.4f}"
        )

    # Live multi-layer patched PPL / KL check on held-out prompts (patching target MLP + Attn layers!)
    def eval_patched_weight_mode(mode_key: str | None) -> Tuple[List[np.ndarray], float]:
        saved: List[Tuple[Any, str, Any]] = []
        if mode_key is not None:
            for (l_idx, pname), spec in patched_projs[mode_key].items():
                mod_name, attr = pname.split(".")
                parent = getattr(layers[l_idx], mod_name)
                orig_obj = getattr(parent, attr)
                saved.append((parent, attr, orig_obj))

                def make_fn(sp):
                    kind = sp[0]
                    if kind == "q2":
                        _, wq, sq, bq = sp
                        return lambda x: mx.quantized_matmul(
                            x, wq, sq, bq, transpose=True, group_size=64, bits=2, mode="affine"
                        )
                    elif kind == "fwht_q2":
                        _, wq, sq, bq, s1, s2 = sp
                        def fn(x):
                            xr = iterated_parseval_transform(x.astype(mx.float32), s1, s2).astype(x.dtype)
                            return mx.quantized_matmul(
                                xr, wq, sq, bq, transpose=True, group_size=64, bits=2, mode="affine"
                            )
                        return fn
                    else:
                        _, wq, sq, bq, s1, s2, Ur, Vr = sp
                        def fn(x):
                            xr = iterated_parseval_transform(x.astype(mx.float32), s1, s2)
                            base = mx.quantized_matmul(
                                xr.astype(x.dtype), wq, sq, bq, transpose=True, group_size=64, bits=2, mode="affine"
                            )
                            lr = (xr.astype(mx.float16) @ Vr.T) @ Ur.T
                            return (base.astype(mx.float32) + lr.astype(mx.float32)).astype(x.dtype)
                        return fn

                setattr(parent, attr, make_fn(spec))

        lgs: List[np.ndarray] = []
        tot_nll = 0.0
        tot_n = 0
        try:
            for ids in eval_ids:
                logits = lm(ids)[0, :-1, :].astype(mx.float32)
                targets = ids[0, 1:]
                lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                nll = -lp[mx.arange(targets.shape[0]), targets]
                mx.eval(logits, nll)
                tot_nll += float(mx.sum(nll).item())
                tot_n += int(targets.shape[0])
                lgs.append(np.array(logits))
        finally:
            for parent, attr, orig_obj in saved:
                setattr(parent, attr, orig_obj)
        return lgs, math.exp(tot_nll / max(1, tot_n))

    ref_lgs, ref_ppl = eval_patched_weight_mode(None)
    ref_t1 = [np.argmax(lg, axis=-1) for lg in ref_lgs]
    ref_lp = [
        lg - np.max(lg, axis=-1, keepdims=True)
        - np.log(np.sum(np.exp(lg - np.max(lg, axis=-1, keepdims=True)), axis=-1, keepdims=True))
        for lg in ref_lgs
    ]
    ref_p = [np.exp(lp) for lp in ref_lp]

    patched_ppl_results: Dict[str, Any] = {
        "n_patched_matrices": len(audit_specs),
        "patched_layers_note": f"MLPs {target_mlp_layers} + Attn q_proj {target_attn_layers}",
        "stock_q4_ppl": round(ref_ppl, 4),
    }
    for mkey in ["naive_q2", "fwht_q2", "fwht_q2_r64", "fwht_q2_r128"]:
        lgs, ppl = eval_patched_weight_mode(mkey)
        kl_s, ag, tot = 0.0, 0, 0
        for r1, rp, rlp, lg in zip(ref_t1, ref_p, ref_lp, lgs):
            ag += int(np.sum(np.argmax(lg, axis=-1) == r1))
            tot += len(r1)
            mx_l = np.max(lg, axis=-1, keepdims=True)
            lp = lg - mx_l - np.log(np.sum(np.exp(lg - mx_l), axis=-1, keepdims=True))
            kl_s += float(np.sum(rp * (rlp - lp)))
        patched_ppl_results[mkey] = {
            "ppl": round(ppl, 4),
            "delta_ppl": round(ppl - ref_ppl, 4),
            "mean_kl_nats": round(kl_s / max(1, tot), 5),
            "greedy_agree_pct": round(100.0 * ag / max(1, tot), 2),
        }
        print(
            f"  [Weight Live PPL ({len(audit_specs)} mats patched)] {mkey:14s}: "
            f"PPL={ppl:.3f} (dPPL={ppl - ref_ppl:+.4f}), KL={kl_s / tot:.5f}, agree={100.0 * ag / tot:.1f}%"
        )

    del patched_projs
    clear_memory()

    # Cold-DRAM 32-layer real timing benchmark: Q4 vs Q2 vs FWHT+Q2 vs FWHT+Q2+rank-64/128
    # Use 32 layers of 4096x4096 (~302 MB Q4) AND 32 layers of 12288x4096 (~906 MB Q4)
    print("[Duel 3b] Benchmarking 32-layer cold-DRAM Q4 vs FWHT+Q2 vs FWHT+Q2+rank-r...")
    speed_bench: Dict[str, Any] = {}
    for shape_name, M_dim, K_dim, n_L in [
        ("32x_4096x4096_B1", 4096, 4096, 32),
        ("32x_12288x4096_B1", 12288, 4096, 16 if quick else 32),
    ]:
        x_in = mx.random.normal((1, K_dim)).astype(mx.bfloat16)
        q4_stack = []
        q2_stack = []
        lr64_stack = []
        lr128_stack = []
        for _ in range(n_L):
            w_tmp = mx.random.normal((M_dim, K_dim)).astype(mx.bfloat16) * 0.02
            q4_stack.append(mx.quantize(w_tmp, group_size=64, bits=4, mode="affine"))
            q2_stack.append(mx.quantize(w_tmp, group_size=64, bits=2, mode="affine"))
            u64 = mx.random.normal((M_dim, 64)).astype(mx.float16) * 0.01
            v64 = mx.random.normal((64, K_dim)).astype(mx.float16) * 0.01
            u128 = mx.random.normal((M_dim, 128)).astype(mx.float16) * 0.01
            v128 = mx.random.normal((128, K_dim)).astype(mx.float16) * 0.01
            lr64_stack.append((u64, v64))
            lr128_stack.append((u128, v128))
        mx.eval(x_in)
        for t in q4_stack + q2_stack + lr64_stack + lr128_stack:
            mx.eval(*t)

        def run_q4(st=q4_stack, x=x_in):
            return [mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4) for w, s, b in st]

        def run_q2(st=q2_stack, x=x_in):
            return [mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=2) for w, s, b in st]

        def run_fwht_q2(st=q2_stack, x=x_in):
            outs = []
            for w, s, b in st:
                xr = iterated_parseval_transform(x.astype(mx.float32), s1_4096, s2_4096).astype(mx.bfloat16)
                outs.append(mx.quantized_matmul(xr, w, s, b, transpose=True, group_size=64, bits=2))
            return outs

        def run_fwht_q2_r64(st=q2_stack, lr=lr64_stack, x=x_in):
            outs = []
            for (w, s, b), (u, v) in zip(st, lr):
                xr = iterated_parseval_transform(x.astype(mx.float32), s1_4096, s2_4096)
                y0 = mx.quantized_matmul(xr.astype(mx.bfloat16), w, s, b, transpose=True, group_size=64, bits=2)
                y1 = (xr.astype(mx.float16) @ v.T) @ u.T
                outs.append(y0 + y1.astype(mx.bfloat16))
            return outs

        def run_fwht_q2_r128(st=q2_stack, lr=lr128_stack, x=x_in):
            outs = []
            for (w, s, b), (u, v) in zip(st, lr):
                xr = iterated_parseval_transform(x.astype(mx.float32), s1_4096, s2_4096)
                y0 = mx.quantized_matmul(xr.astype(mx.bfloat16), w, s, b, transpose=True, group_size=64, bits=2)
                y1 = (xr.astype(mx.float16) @ v.T) @ u.T
                outs.append(y0 + y1.astype(mx.bfloat16))
            return outs

        tm = bench_interleaved(
            {
                "stock_q4": run_q4,
                "pure_q2": run_q2,
                "fwht_q2": run_fwht_q2,
                "fwht_q2_r64": run_fwht_q2_r64,
                "fwht_q2_r128": run_fwht_q2_r128,
            },
            flusher=flusher,
            trials=trials,
            warmup=2,
        )
        t_q4 = tm["stock_q4"]["median_ms"]
        q4_bytes = n_L * sum(arr.nbytes for arr in q4_stack[0])
        q2_bytes = n_L * sum(arr.nbytes for arr in q2_stack[0])
        r64_bytes = q2_bytes + n_L * sum(arr.nbytes for arr in lr64_stack[0])
        r128_bytes = q2_bytes + n_L * sum(arr.nbytes for arr in lr128_stack[0])

        speed_bench[shape_name] = {
            "n_layers": n_L,
            "q4_mb": round(q4_bytes / 1e6, 1),
            "q2_mb": round(q2_bytes / 1e6, 1),
            "fwht_q2_r64_mb": round(r64_bytes / 1e6, 1),
            "fwht_q2_r128_mb": round(r128_bytes / 1e6, 1),
            "stock_q4_ms": round(t_q4, 4),
            "stock_q4_gbs": round((q4_bytes / 1e9) / (t_q4 / 1e3), 1),
            "pure_q2_ms": round(tm["pure_q2"]["median_ms"], 4),
            "pure_q2_speedup_x": round(t_q4 / tm["pure_q2"]["median_ms"], 3),
            "fwht_q2_ms": round(tm["fwht_q2"]["median_ms"], 4),
            "fwht_q2_speedup_x": round(t_q4 / tm["fwht_q2"]["median_ms"], 3),
            "fwht_q2_r64_ms": round(tm["fwht_q2_r64"]["median_ms"], 4),
            "fwht_q2_r64_speedup_x": round(t_q4 / tm["fwht_q2_r64"]["median_ms"], 3),
            "fwht_q2_r128_ms": round(tm["fwht_q2_r128"]["median_ms"], 4),
            "fwht_q2_r128_speedup_x": round(t_q4 / tm["fwht_q2_r128"]["median_ms"], 3),
        }
        sb = speed_bench[shape_name]
        print(
            f"  [Speed {shape_name}] Q4={sb['stock_q4_ms']:.3f}ms ({sb['stock_q4_gbs']:.0f} GB/s) | "
            f"Q2={sb['pure_q2_ms']:.3f}ms ({sb['pure_q2_speedup_x']:.2f}x) | "
            f"FWHT+Q2={sb['fwht_q2_ms']:.3f}ms ({sb['fwht_q2_speedup_x']:.2f}x) | "
            f"+r64={sb['fwht_q2_r64_ms']:.3f}ms ({sb['fwht_q2_r64_speedup_x']:.2f}x) | "
            f"+r128={sb['fwht_q2_r128_ms']:.3f}ms ({sb['fwht_q2_r128_speedup_x']:.2f}x)"
        )
        del q4_stack, q2_stack, lr64_stack, lr128_stack
        clear_memory()

    # Adjudicate FWHT+Q2+rank-r weight factorization against contract falsifier
    best_speed_r64 = speed_bench["32x_12288x4096_B1"]["fwht_q2_r64_speedup_x"]
    r64_dppl = patched_ppl_results["fwht_q2_r64"]["delta_ppl"]
    r128_dppl = patched_ppl_results["fwht_q2_r128"]["delta_ppl"]
    best_speed_r128 = speed_bench["32x_12288x4096_B1"]["fwht_q2_r128_speedup_x"]
    fwht_pass = (best_speed_r64 >= 1.15 and r64_dppl <= 0.15) or (
        best_speed_r128 >= 1.15 and r128_dppl <= 0.15
    )

    return {
        "verdict": "WIN" if fwht_pass else "KILLED",
        "parseval_orthogonality_max_err": parseval_err,
        "matrix_audits": matrix_audits,
        "mean_act_rel_err": {
            "naive_q2": round(float(np.mean([m["naive_q2_act_rel_err"] for m in matrix_audits])), 4),
            "fwht_q2": round(float(np.mean([m["fwht_q2_act_rel_err"] for m in matrix_audits])), 4),
            "fwht_q2_r64": round(float(np.mean([m["fwht_q2_r64_act_rel_err"] for m in matrix_audits])), 4),
            "fwht_q2_r128": round(float(np.mean([m["fwht_q2_r128_act_rel_err"] for m in matrix_audits])), 4),
        },
        "live_patched_ppl": patched_ppl_results,
        "cold_dram_speed_vs_q4": speed_bench,
    }


def run_spectral_duel(
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

    kv_res = run_kv_spectral_audit(model, tokenizer, flusher, quick=quick)
    fwht_res = run_fwht_q2_weight_audit(model, tokenizer, flusher, quick=quick)

    overall_verdict = (
        "PARTIAL_WIN"
        if (kv_res["verdict"] == "WIN") != (fwht_res["verdict"] == "WIN")
        else ("WIN" if kv_res["verdict"] == "WIN" else "KILLED")
    )
    print(
        f"[Duel 3] SUMMARY: KV_spectral={kv_res['verdict']} | "
        f"FWHT_Q2_weights={fwht_res['verdict']} -> OVERALL: {overall_verdict}"
    )

    return {
        "duel": "spectral_inertia_kv",
        "verdict": overall_verdict,
        "pass_mark": ">=1.15x wall-clock speedup AND <=0.15 PPL delta",
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "kv_spectral_inertia_audit": kv_res,
        "fwht_q2_weight_audit": fwht_res,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    res = run_spectral_duel(quick=args.quick)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved Duel 3 results to {args.out}")


if __name__ == "__main__":
    main()
