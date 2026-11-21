"""Duel 2: Sign-Definite `gate_proj < 0` Dynamic MLP Sparsity on REAL `Qwen3.5-9B-4bit`.

Measures across all 32 layers on real held-out prompts:
  1. Exact distribution of `g = gate_proj(x)`, `|SiLU(g)|`, and `|SiLU(g)*u|` energy
     vs `g` threshold (`g > 0`, `g > -0.25`, `g > -0.5`, `g > -1.0`, `|SiLU(g)| > tau`)
     and top-k (`k in {1024, 2048, 3072, 4096, 6144}`).
  2. Live end-to-end 32-layer patched `Qwen3.5-9B-4bit` quality on held-out text:
     greedy token agreement %, next-token KL divergence, Perplexity & Delta-PPL
     (plus exact quality check of transposed-Q4 `down_proj_T` layout!).
  3. Real fused Metal/MLX **zero-CPU-sync GPU-compacted dynamic sparse `up_proj + down_proj_T`
     kernel** (`mx.fast.metal_kernel`) that ONLY reads active `up_proj` rows and active
     `down_proj_T` columns from DRAM when `g > tau` (or `|SiLU(g)| > tau`), benchmarked
     cold-DRAM across the real 32-layer MLP stack (~2.72 GB Q4 weights + 540 MB SLC flush!).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any, Callable, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.activations import swiglu

from common import (
    HELD_OUT_PROMPTS,
    MODEL_ID,
    SLCFlusher,
    bench_interleaved,
    clear_memory,
)

# ---------------------------------------------------------------------------
# Custom Metal Kernels for Dynamic Sparse MLP (Zero CPU-GPU Sync!)
#   Kernel 1: compact_active_gates (simdgroup-batched atomic compaction)
#   Kernel 2: fused_sparse_up_down_q4 (registers hold x[16], coalesced uint2
#             reads of W_up[row] & W_dn_T[row], 32 bytes TG mem, 0 inactive reads)
# ---------------------------------------------------------------------------

_COMPACT_GATES_KERNEL = None
_FUSED_SPARSE_UP_DOWN_KERNEL = None


def get_compact_gates_kernel():
    global _COMPACT_GATES_KERNEL
    if _COMPACT_GATES_KERNEL is None:
        source = r"""
            uint gid = thread_position_in_grid.x; // 0..12287
            uint lane = thread_index_in_simdgroup;
            float val = float(g[gid]);
            float sg = val / (1.0f + metal::exp(-val));
            float th = thresh[0];
            uint m = mode[0];
            bool active = (m == 0u) ? (val > th) : (metal::abs(sg) > th);

            uint flag = active ? 1u : 0u;
            uint sg_offset = simd_prefix_exclusive_sum(flag);
            uint sg_total = simd_sum(flag);

            uint base = 0u;
            if (lane == 0u && sg_total > 0u) {
                base = atomic_fetch_add_explicit(&out_cnt[0], sg_total, memory_order_relaxed);
            }
            base = simd_broadcast_first(base);

            if (active) {
                uint pos = base + sg_offset;
                atomic_store_explicit(&out_idx[pos], gid, memory_order_relaxed);
                atomic_store_explicit(&out_silu_bits[pos], as_type<uint>(sg), memory_order_relaxed);
            }
        """
        _COMPACT_GATES_KERNEL = mx.fast.metal_kernel(
            name="compact_active_gates_12288",
            input_names=["g", "thresh", "mode"],
            output_names=["out_idx", "out_silu_bits", "out_cnt"],
            source=source,
            atomic_outputs=True,
        )
    return _COMPACT_GATES_KERNEL


def get_fused_sparse_up_down_kernel():
    global _FUSED_SPARSE_UP_DOWN_KERNEL
    if _FUSED_SPARSE_UP_DOWN_KERNEL is None:
        source = r"""
            uint tg_id = threadgroup_position_in_grid.x;
            uint num_tgs = threadgroups_per_grid.x;
            uint tid = thread_position_in_threadgroup.x; // 0..255
            uint sg = tid >> 5u;                         // 0..7
            uint lane = tid & 31u;                       // 0..31
            uint grp = tid >> 2u;                        // 0..63 (4 threads share 1 Q4 group of 64)
            uint c_base = tid * 16u;

            // Keep this thread's 16 input elements in registers (ZERO TG mem for x!)
            float xr[16];
            float sum_xr = 0.0f;
            for (uint k = 0u; k < 16u; ++k) {
                float v = float(x[c_base + k]);
                xr[k] = v;
                sum_xr += v;
            }

            float acc[16];
            for (uint k = 0u; k < 16u; ++k) {
                acc[k] = 0.0f;
            }

            threadgroup float sg_buf[8];
            uint n_active = active_cnt[0];

            for (uint t = tg_id; t < n_active; t += num_tgs) {
                uint row = active_idx[t];
                float silu_g = as_type<float>(active_silu_bits[t]);

                ulong w_row_off = ulong(row) * 512ul;
                ulong sb_off = ulong(row) * 64ul + ulong(grp);

                // 1. Coalesced load of W_up[row, c_base..c_base+15] (256 threads = 2048 B)
                uint2 wu = ((const device uint2*)(W_up + w_row_off))[tid];
                float sc_u = float(S_up[sb_off]);
                float bi_u = float(B_up[sb_off]);

                float dot_u = 0.0f;
                uint w0 = wu.x;
                uint w1 = wu.y;
                dot_u = fma(float((w0 >>  0u) & 0xFu), xr[0],  dot_u);
                dot_u = fma(float((w0 >>  4u) & 0xFu), xr[1],  dot_u);
                dot_u = fma(float((w0 >>  8u) & 0xFu), xr[2],  dot_u);
                dot_u = fma(float((w0 >> 12u) & 0xFu), xr[3],  dot_u);
                dot_u = fma(float((w0 >> 16u) & 0xFu), xr[4],  dot_u);
                dot_u = fma(float((w0 >> 20u) & 0xFu), xr[5],  dot_u);
                dot_u = fma(float((w0 >> 24u) & 0xFu), xr[6],  dot_u);
                dot_u = fma(float((w0 >> 28u) & 0xFu), xr[7],  dot_u);

                dot_u = fma(float((w1 >>  0u) & 0xFu), xr[8],  dot_u);
                dot_u = fma(float((w1 >>  4u) & 0xFu), xr[9],  dot_u);
                dot_u = fma(float((w1 >>  8u) & 0xFu), xr[10], dot_u);
                dot_u = fma(float((w1 >> 12u) & 0xFu), xr[11], dot_u);
                dot_u = fma(float((w1 >> 16u) & 0xFu), xr[12], dot_u);
                dot_u = fma(float((w1 >> 20u) & 0xFu), xr[13], dot_u);
                dot_u = fma(float((w1 >> 24u) & 0xFu), xr[14], dot_u);
                dot_u = fma(float((w1 >> 28u) & 0xFu), xr[15], dot_u);

                float psum = fma(dot_u, sc_u, bi_u * sum_xr);
                float sg_sum = simd_sum(psum);
                if (lane == 0u) {
                    sg_buf[sg] = sg_sum;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (tid == 0u) {
                    float u_total = (sg_buf[0] + sg_buf[1]) + (sg_buf[2] + sg_buf[3])
                                  + (sg_buf[4] + sg_buf[5]) + (sg_buf[6] + sg_buf[7]);
                    sg_buf[0] = silu_g * u_total;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float h_val = sg_buf[0];

                // 2. Coalesced load of W_down_T[row, c_base..c_base+15]
                uint2 wd = ((const device uint2*)(W_dn_T + w_row_off))[tid];
                float sc_d = float(S_dn_T[sb_off]);
                float bi_d = float(B_dn_T[sb_off]);
                float h_sc = h_val * sc_d;
                float h_bi = h_val * bi_d;

                uint d0 = wd.x;
                uint d1 = wd.y;
                acc[0]  = fma(float((d0 >>  0u) & 0xFu), h_sc, acc[0]  + h_bi);
                acc[1]  = fma(float((d0 >>  4u) & 0xFu), h_sc, acc[1]  + h_bi);
                acc[2]  = fma(float((d0 >>  8u) & 0xFu), h_sc, acc[2]  + h_bi);
                acc[3]  = fma(float((d0 >> 12u) & 0xFu), h_sc, acc[3]  + h_bi);
                acc[4]  = fma(float((d0 >> 16u) & 0xFu), h_sc, acc[4]  + h_bi);
                acc[5]  = fma(float((d0 >> 20u) & 0xFu), h_sc, acc[5]  + h_bi);
                acc[6]  = fma(float((d0 >> 24u) & 0xFu), h_sc, acc[6]  + h_bi);
                acc[7]  = fma(float((d0 >> 28u) & 0xFu), h_sc, acc[7]  + h_bi);

                acc[8]  = fma(float((d1 >>  0u) & 0xFu), h_sc, acc[8]  + h_bi);
                acc[9]  = fma(float((d1 >>  4u) & 0xFu), h_sc, acc[9]  + h_bi);
                acc[10] = fma(float((d1 >>  8u) & 0xFu), h_sc, acc[10] + h_bi);
                acc[11] = fma(float((d1 >> 12u) & 0xFu), h_sc, acc[11] + h_bi);
                acc[12] = fma(float((d1 >> 16u) & 0xFu), h_sc, acc[12] + h_bi);
                acc[13] = fma(float((d1 >> 20u) & 0xFu), h_sc, acc[13] + h_bi);
                acc[14] = fma(float((d1 >> 24u) & 0xFu), h_sc, acc[14] + h_bi);
                acc[15] = fma(float((d1 >> 28u) & 0xFu), h_sc, acc[15] + h_bi);
            }

            ulong out_base = ulong(tg_id) * 4096ul + ulong(c_base);
            for (uint k = 0u; k < 16u; ++k) {
                partials[out_base + k] = acc[k];
            }
        """
        _FUSED_SPARSE_UP_DOWN_KERNEL = mx.fast.metal_kernel(
            name="fused_sparse_up_down_q4_4096",
            input_names=[
                "x",
                "active_idx",
                "active_silu_bits",
                "active_cnt",
                "W_up",
                "S_up",
                "B_up",
                "W_dn_T",
                "S_dn_T",
                "B_dn_T",
            ],
            output_names=["partials"],
            source=source,
        )
    return _FUSED_SPARSE_UP_DOWN_KERNEL


def compact_gates_gpu(
    g_1d: mx.array, thresh: float, mode: int = 0
) -> Tuple[mx.array, mx.array, mx.array]:
    """Compact active MLP gate channels purely on GPU (mode=0: g>thresh, mode=1: |silu(g)|>thresh)."""
    n_inter = int(g_1d.shape[0])
    k = get_compact_gates_kernel()
    th_arr = mx.array([thresh], dtype=mx.float32)
    m_arr = mx.array([mode], dtype=mx.uint32)
    out_idx, out_silu_bits, out_cnt = k(
        inputs=[g_1d, th_arr, m_arr],
        grid=(n_inter, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_inter,), (n_inter,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
        init_value=0,
    )
    return out_idx, out_silu_bits, out_cnt


def fused_sparse_up_down_call(
    x_1d: mx.array,
    active_idx: mx.array,
    active_silu_bits: mx.array,
    active_cnt: mx.array,
    up_w: mx.array,
    up_s: mx.array,
    up_b: mx.array,
    dn_t_w: mx.array,
    dn_t_s: mx.array,
    dn_t_b: mx.array,
    num_tgs: int = 80,
) -> mx.array:
    """Execute fused dynamic sparse up_proj + SwiGLU + down_proj_T and reduce partials."""
    k = get_fused_sparse_up_down_kernel()
    (partials,) = k(
        inputs=[
            x_1d,
            active_idx,
            active_silu_bits,
            active_cnt,
            up_w,
            up_s,
            up_b,
            dn_t_w,
            dn_t_s,
            dn_t_b,
        ],
        grid=(num_tgs * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(num_tgs, 4096)],
        output_dtypes=[mx.float32],
    )
    return mx.sum(partials, axis=0, keepdims=True).astype(x_1d.dtype)


def transpose_quantize_down_proj(down_proj) -> Tuple[mx.array, mx.array, mx.array]:
    """Repack down_proj (4096 x 12288) Q4 -> transposed column-major (12288 x 4096) Q4."""
    w_dq = mx.dequantize(
        down_proj.weight,
        down_proj.scales,
        down_proj.biases,
        group_size=down_proj.group_size,
        bits=down_proj.bits,
        mode=down_proj.mode,
    )  # (4096, 12288) bfloat16
    w_t = mx.contiguous(w_dq.T)  # (12288, 4096) bfloat16
    wt_q, wt_s, wt_b = mx.quantize(
        w_t, group_size=down_proj.group_size, bits=down_proj.bits, mode=down_proj.mode
    )
    mx.eval(wt_q, wt_s, wt_b)
    return wt_q, wt_s, wt_b


# ---------------------------------------------------------------------------
# Part 1: 32-Layer Gate & Energy Distribution Audit on Real Held-Out Prompts
# ---------------------------------------------------------------------------

def profile_all_32_layers(
    model,
    tokenizer,
    prompts: List[Tuple[str, str]],
) -> Tuple[Dict[str, Any], List[mx.array]]:
    """Capture real post-attention-layernorm MLP inputs and gate/up stats across all 32 layers."""
    lm = model.language_model
    n_layers = len(lm.layers)
    inter_dim = int(lm.layers[0].mlp.gate_proj.weight.shape[0])  # 12288

    # Capture MLP inputs across all prompts by tracing forward layer-by-layer
    # (saves memory by accumulating running stats per layer without storing huge tensors)
    thresholds = [
        ("g > -1.0", lambda g, sg: g > -1.0),
        ("g > -0.5", lambda g, sg: g > -0.5),
        ("g > -0.25", lambda g, sg: g > -0.25),
        ("g > 0.0", lambda g, sg: g > 0.0),
        ("g > 0.5", lambda g, sg: g > 0.5),
        ("|SiLU(g)| > 0.05", lambda g, sg: np.abs(sg) > 0.05),
        ("|SiLU(g)| > 0.10", lambda g, sg: np.abs(sg) > 0.10),
        ("|SiLU(g)| > 0.20", lambda g, sg: np.abs(sg) > 0.20),
    ]
    topks = [123, 1024, 2048, 3072, 4096, 6144]  # 123 = 1% of 12288

    layer_stats: List[Dict[str, Any]] = [
        {
            "layer": l,
            "n_tokens": 0,
            "neg_gate_sum": 0.0,
            "thresh_active_sum": {k: 0.0 for k, _ in thresholds},
            "thresh_silu_e_num": {k: 0.0 for k, _ in thresholds},
            "thresh_act_e_num": {k: 0.0 for k, _ in thresholds},
            "topk_g_act_e_num": {k: 0.0 for k in topks},
            "topk_silu_act_e_num": {k: 0.0 for k in topks},
            "total_silu_e": 0.0,
            "total_act_e": 0.0,
        }
        for l in range(n_layers)
    ]
    # Keep 1 representative real MLP input x (1, 4096) bf16 per layer for cold-DRAM bench
    sample_x_per_layer: List[mx.array | None] = [None] * n_layers

    for p_idx, (_, prompt) in enumerate(prompts):
        toks = tokenizer.encode(prompt)
        ids = mx.array(toks, dtype=mx.int32)[None, :]
        h = lm.model.embed_tokens(ids)
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        fa_mask = create_attention_mask(h, None)
        ssm_mask = create_ssm_mask(h, None)

        for l, layer in enumerate(lm.layers):
            mask = ssm_mask if layer.is_linear else fa_mask
            if layer.is_linear:
                r = layer.linear_attn(layer.input_layernorm(h), mask, None)
            else:
                r = layer.self_attn(layer.input_layernorm(h), mask, None)
            h_mid = h + r
            mlp_in = layer.post_attention_layernorm(h_mid)
            g_mx = layer.mlp.gate_proj(mlp_in)
            u_mx = layer.mlp.up_proj(mlp_in)
            a_mx = swiglu(g_mx, u_mx)
            mlp_out = layer.mlp.down_proj(a_mx)
            h = h_mid + mlp_out
            mx.eval(h, g_mx, a_mx, mlp_in)

            if p_idx == 0:
                sample_x_per_layer[l] = mlp_in[0, -1, :]  # (4096,) bf16

            # Stats on last 32 tokens (or all tokens if shorter)
            g_np = np.array(g_mx[0].astype(mx.float32))
            a_np = np.array(a_mx[0].astype(mx.float32))
            sg_np = g_np / (1.0 + np.exp(-np.clip(g_np, -40.0, 40.0)))
            sg2 = sg_np * sg_np
            a2 = a_np * a_np

            st = layer_stats[l]
            T = int(g_np.shape[0])
            st["n_tokens"] += T
            st["neg_gate_sum"] += float(np.mean(g_np < 0.0)) * T
            st["total_silu_e"] += float(np.sum(sg2))
            st["total_act_e"] += float(np.sum(a2))

            for tname, fn in thresholds:
                m = fn(g_np, sg_np)
                st["thresh_active_sum"][tname] += float(np.mean(m)) * T
                st["thresh_silu_e_num"][tname] += float(np.sum(sg2[m]))
                st["thresh_act_e_num"][tname] += float(np.sum(a2[m]))

            # Top-k energy capture (sorted once per token)
            abs_sg = np.abs(sg_np)
            # Sort indices descending along channel axis
            ord_g = np.argsort(g_np, axis=1)
            ord_sg = np.argsort(abs_sg, axis=1)
            rows = np.arange(T)[:, None]
            for k in topks:
                idx_g = ord_g[:, -k:]
                idx_sg = ord_sg[:, -k:]
                st["topk_g_act_e_num"][k] += float(np.sum(a2[rows, idx_g]))
                st["topk_silu_act_e_num"][k] += float(np.sum(a2[rows, idx_sg]))

    # Aggregate per-layer and global summaries
    per_layer_summary = []
    for st in layer_stats:
        nt = max(1, st["n_tokens"])
        te_a = max(1e-12, st["total_act_e"])
        per_layer_summary.append(
            {
                "layer": st["layer"],
                "frac_g_lt_0_pct": round(100.0 * st["neg_gate_sum"] / nt, 2),
                "g_gt_0_active_pct": round(100.0 * st["thresh_active_sum"]["g > 0.0"] / nt, 2),
                "g_gt_0_act_energy_pct": round(100.0 * st["thresh_act_e_num"]["g > 0.0"] / te_a, 2),
                "abs_silu_gt_01_active_pct": round(
                    100.0 * st["thresh_active_sum"]["|SiLU(g)| > 0.10"] / nt, 2
                ),
                "abs_silu_gt_01_act_energy_pct": round(
                    100.0 * st["thresh_act_e_num"]["|SiLU(g)| > 0.10"] / te_a, 2
                ),
                "top1pct_k123_act_energy_pct": round(
                    100.0 * st["topk_silu_act_e_num"][123] / te_a, 2
                ),
                "top2048_silu_act_energy_pct": round(
                    100.0 * st["topk_silu_act_e_num"][2048] / te_a, 2
                ),
            }
        )

    tot_tok = sum(st["n_tokens"] for st in layer_stats)
    tot_silu_e = sum(st["total_silu_e"] for st in layer_stats)
    tot_act_e = sum(st["total_act_e"] for st in layer_stats)

    global_thresholds = {}
    for tname, _ in thresholds:
        act_pct = 100.0 * sum(st["thresh_active_sum"][tname] for st in layer_stats) / tot_tok
        silu_e_pct = 100.0 * sum(st["thresh_silu_e_num"][tname] for st in layer_stats) / tot_silu_e
        act_e_pct = 100.0 * sum(st["thresh_act_e_num"][tname] for st in layer_stats) / tot_act_e
        global_thresholds[tname] = {
            "active_channels_pct": round(act_pct, 2),
            "skipped_channels_pct": round(100.0 - act_pct, 2),
            "silu_energy_kept_pct": round(silu_e_pct, 2),
            "swiglu_energy_kept_pct": round(act_e_pct, 2),
        }

    global_topks = {}
    for k in topks:
        g_e = 100.0 * sum(st["topk_g_act_e_num"][k] for st in layer_stats) / tot_act_e
        sg_e = 100.0 * sum(st["topk_silu_act_e_num"][k] for st in layer_stats) / tot_act_e
        global_topks[f"k={k}"] = {
            "active_channels_pct": round(100.0 * k / inter_dim, 2),
            "topk_by_g_swiglu_energy_pct": round(g_e, 2),
            "topk_by_abs_silu_swiglu_energy_pct": round(sg_e, 2),
        }

    global_neg_pct = 100.0 * sum(st["neg_gate_sum"] for st in layer_stats) / tot_tok
    profile = {
        "n_layers": n_layers,
        "intermediate_size": inter_dim,
        "tokens_profiled_per_layer": layer_stats[0]["n_tokens"],
        "global_frac_g_lt_0_pct": round(global_neg_pct, 2),
        "layer_min_neg_pct": min(d["frac_g_lt_0_pct"] for d in per_layer_summary),
        "layer_max_neg_pct": max(d["frac_g_lt_0_pct"] for d in per_layer_summary),
        "layer31_top1pct_energy_pct": per_layer_summary[-1]["top1pct_k123_act_energy_pct"],
        "global_thresholds": global_thresholds,
        "global_topks": global_topks,
        "per_layer": per_layer_summary,
    }
    return profile, [x for x in sample_x_per_layer if x is not None]


# ---------------------------------------------------------------------------
# Part 2: Live End-to-End 32-Layer Quality Sweep (PPL, KL, Greedy Agreement %)
# ---------------------------------------------------------------------------

def eval_live_mlp_pruning_quality(
    model,
    tokenizer,
    prompts: List[Tuple[str, str]],
    down_t_all_layers: List[Tuple[mx.array, mx.array, mx.array]] | None = None,
    quick: bool = False,
) -> List[Dict[str, Any]]:
    """Patch ALL 32 MLP layers of Qwen3.5-9B-4bit and measure PPL, KL, greedy match."""
    lm = model.language_model
    layers = lm.layers

    # Tokenize held-out prompts
    eval_ids_list = [mx.array(tokenizer.encode(txt), dtype=mx.int32)[None, :] for _, txt in prompts]
    if quick:
        eval_ids_list = eval_ids_list[:4]

    def run_eval_with_mask_fn(
        mask_fn: Callable[[mx.array], mx.array] | None,
        down_t_override: List[Tuple[mx.array, mx.array, mx.array]] | None = None,
    ) -> Tuple[List[np.ndarray], float, float]:
        """Returns list of logits (L-1, V), NLL mean, mean active fraction."""
        orig_calls = []
        active_fracs: List[float] = []

        if mask_fn is not None or down_t_override is not None:
            for idx, layer in enumerate(layers):
                mlp = layer.mlp
                orig_calls.append(mlp)

                class _PatchedMLP:
                    def __init__(self, m_ref, l_idx):
                        self.m_ref = m_ref
                        self.l_idx = l_idx

                    def __call__(self, x):
                        g = self.m_ref.gate_proj(x)
                        u = self.m_ref.up_proj(x)
                        a = swiglu(g, u)
                        if mask_fn is not None:
                            mask = mask_fn(g)
                            active_fracs.append(float(mx.mean(mask.astype(mx.float32)).item()))
                            a = mx.where(mask, a, mx.array(0, dtype=a.dtype))
                        if down_t_override is not None:
                            wt_q, wt_s, wt_b = down_t_override[self.l_idx]
                            return mx.quantized_matmul(
                                a,
                                wt_q,
                                wt_s,
                                wt_b,
                                transpose=False,
                                group_size=64,
                                bits=4,
                                mode="affine",
                            )
                        return self.m_ref.down_proj(a)

                layer.mlp = _PatchedMLP(mlp, idx)

        logits_list: List[np.ndarray] = []
        total_nll = 0.0
        total_toks = 0
        try:
            for ids in eval_ids_list:
                # Forward through layers manually using patched mlp.__call__
                h = lm.model.embed_tokens(ids)
                from mlx_lm.models.base import create_attention_mask, create_ssm_mask

                fa_mask = create_attention_mask(h, None)
                ssm_mask = create_ssm_mask(h, None)
                for layer in layers:
                    mask = ssm_mask if layer.is_linear else fa_mask
                    r = (
                        layer.linear_attn(layer.input_layernorm(h), mask, None)
                        if layer.is_linear
                        else layer.self_attn(layer.input_layernorm(h), mask, None)
                    )
                    h_mid = h + r
                    h = h_mid + layer.mlp(layer.post_attention_layernorm(h_mid))
                logits = lm.lm_head(lm.model.norm(h))[0, :-1, :].astype(mx.float32)
                targets = ids[0, 1:]
                # Cross-entropy NLL
                log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                nll = -log_probs[mx.arange(targets.shape[0]), targets]
                mx.eval(logits, nll)
                total_nll += float(mx.sum(nll).item())
                total_toks += int(targets.shape[0])
                logits_list.append(np.array(logits))
        finally:
            if orig_calls:
                for layer, orig_mlp in zip(layers, orig_calls):
                    layer.mlp = orig_mlp

        mean_act = float(np.mean(active_fracs)) if active_fracs else 1.0
        return logits_list, total_nll / max(1, total_toks), mean_act

    # Stock reference
    ref_logits_list, ref_nll, _ = run_eval_with_mask_fn(None)
    ref_ppl = math.exp(ref_nll)
    ref_top1 = [np.argmax(lg, axis=-1) for lg in ref_logits_list]
    ref_logp = [
        lg - np.log(np.sum(np.exp(lg - np.max(lg, axis=-1, keepdims=True)), axis=-1, keepdims=True))
        - np.max(lg, axis=-1, keepdims=True)
        for lg in ref_logits_list
    ]
    ref_probs = [np.exp(lp) for lp in ref_logp]

    results: List[Dict[str, Any]] = [
        {
            "config": "stock_baseline",
            "active_channels_pct": 100.0,
            "ppl": round(ref_ppl, 4),
            "delta_ppl": 0.0,
            "mean_kl_nats": 0.0,
            "greedy_top1_agree_pct": 100.0,
            "passes_ppl_falsifier_le_0_15": True,
        }
    ]

    def make_topk_g(k: int):
        def fn(g: mx.array) -> mx.array:
            # Threshold at the k-th largest value per token
            n = g.shape[-1]
            part = mx.partition(g, n - k, axis=-1)
            thr = part[..., n - k : n - k + 1]
            return g >= thr
        return fn

    def make_topk_abs_silu(k: int):
        def fn(g: mx.array) -> mx.array:
            sg = mx.abs(nn.silu(g.astype(mx.float32)))
            n = sg.shape[-1]
            part = mx.partition(sg, n - k, axis=-1)
            thr = part[..., n - k : n - k + 1]
            return sg >= thr
        return fn

    configs: List[Tuple[str, Callable[[mx.array], mx.array] | None, Any]] = [
        ("g > -1.0", lambda g: g > -1.0, None),
        ("g > -0.5", lambda g: g > -0.5, None),
        ("g > 0.0", lambda g: g > 0.0, None),
        ("|SiLU(g)| > 0.05", lambda g: mx.abs(nn.silu(g.astype(mx.float32))) > 0.05, None),
        ("|SiLU(g)| > 0.10", lambda g: mx.abs(nn.silu(g.astype(mx.float32))) > 0.10, None),
        ("|SiLU(g)| > 0.20", lambda g: mx.abs(nn.silu(g.astype(mx.float32))) > 0.20, None),
        ("topk_g_6144", make_topk_g(6144), None),
        ("topk_g_4096", make_topk_g(4096), None),
        ("topk_g_3072", make_topk_g(3072), None),
        ("topk_g_2048", make_topk_g(2048), None),
        ("topk_g_1024", make_topk_g(1024), None),
        ("topk_abs_silu_6144", make_topk_abs_silu(6144), None),
        ("topk_abs_silu_4096", make_topk_abs_silu(4096), None),
        ("topk_abs_silu_3072", make_topk_abs_silu(3072), None),
        ("topk_abs_silu_2048", make_topk_abs_silu(2048), None),
        ("topk_abs_silu_1024", make_topk_abs_silu(1024), None),
    ]
    if down_t_all_layers is not None:
        configs.insert(0, ("down_proj_T_repack_unpruned", None, down_t_all_layers))
    if quick:
        keep_set = {
            "down_proj_T_repack_unpruned",
            "g > -0.5",
            "g > 0.0",
            "|SiLU(g)| > 0.05",
            "|SiLU(g)| > 0.10",
            "|SiLU(g)| > 0.20",
            "topk_g_4096",
            "topk_abs_silu_6144",
            "topk_abs_silu_2048",
        }
        configs = [c for c in configs if c[0] in keep_set]

    for cname, mfn, dn_ovr in configs:
        lg_list, nll, act_frac = run_eval_with_mask_fn(mfn, down_t_override=dn_ovr)
        ppl = math.exp(nll)
        delta_ppl = ppl - ref_ppl
        agrees = 0
        total_pos = 0
        kl_sum = 0.0
        for r_t1, r_p, r_lp, lg in zip(ref_top1, ref_probs, ref_logp, lg_list):
            t1 = np.argmax(lg, axis=-1)
            agrees += int(np.sum(t1 == r_t1))
            total_pos += int(len(r_t1))
            mx_lg = np.max(lg, axis=-1, keepdims=True)
            lp = lg - mx_lg - np.log(np.sum(np.exp(lg - mx_lg), axis=-1, keepdims=True))
            kl_sum += float(np.sum(r_p * (r_lp - lp)))
        agree_pct = 100.0 * agrees / max(1, total_pos)
        mean_kl = kl_sum / max(1, total_pos)
        entry = {
            "config": cname,
            "active_channels_pct": round(100.0 * act_frac, 2),
            "ppl": round(ppl, 4),
            "delta_ppl": round(delta_ppl, 4),
            "mean_kl_nats": round(mean_kl, 5),
            "greedy_top1_agree_pct": round(agree_pct, 2),
            "passes_ppl_falsifier_le_0_15": bool(delta_ppl <= 0.15),
        }
        results.append(entry)
        print(
            f"  [Quality] {cname:20s}: active={entry['active_channels_pct']:5.1f}%, "
            f"PPL={entry['ppl']:6.3f} (dPPL={entry['delta_ppl']:+.4f}), "
            f"KL={entry['mean_kl_nats']:.5f}, greedy_agree={entry['greedy_top1_agree_pct']:5.1f}%"
        )

    return results


# ---------------------------------------------------------------------------
# Part 3: Fused Sparse Metal Kernel Verification & Cold-DRAM 32-Layer Timing
# ---------------------------------------------------------------------------

def run_mlp_duel(
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

    lm = model.language_model
    layers = lm.layers
    prompts = HELD_OUT_PROMPTS[:4] if quick else HELD_OUT_PROMPTS
    trials = 5 if quick else 11
    n_bench_layers = 16 if quick else len(layers)  # 32 layers = 2.72 GB cold DRAM!

    print(f"[Duel 2] Profiling all 32 MLP layers on {len(prompts)} real held-out prompts...")
    profile, sample_x_list = profile_all_32_layers(model, tokenizer, prompts)
    print(
        f"[Duel 2] Global gate < 0: {profile['global_frac_g_lt_0_pct']}% "
        f"(layer range {profile['layer_min_neg_pct']}%–{profile['layer_max_neg_pct']}%), "
        f"Layer 31 top-1% energy={profile['layer31_top1pct_energy_pct']}%"
    )

    # Prepare transposed-Q4 down_proj_T across all 32 layers (fast ~0.4s; lets us test repack PPL too!)
    print(f"[Duel 2] Packing transposed-Q4 down_proj_T for all {len(layers)} real layers...")
    dn_t_all_weights: List[Tuple[mx.array, mx.array, mx.array]] = []
    dn_repack_rel_errs: List[float] = []
    for l in range(len(layers)):
        dp = layers[l].mlp.down_proj
        wt_q, wt_s, wt_b = transpose_quantize_down_proj(dp)
        dn_t_all_weights.append((wt_q, wt_s, wt_b))
        if l in (0, 15, len(layers) - 1):
            orig_dq = mx.dequantize(dp.weight, dp.scales, dp.biases, 64, 4, "affine").astype(mx.float32)
            new_dq = mx.dequantize(wt_q, wt_s, wt_b, 64, 4, "affine").astype(mx.float32).T
            rel = float(
                (mx.sqrt(mx.sum((orig_dq - new_dq) ** 2)) / mx.sqrt(mx.sum(orig_dq ** 2))).item()
            )
            dn_repack_rel_errs.append(rel)

    print("[Duel 2] Measuring live 32-layer patched PPL / KL / greedy agreement...")
    quality_curve = eval_live_mlp_pruning_quality(
        model, tokenizer, prompts, down_t_all_layers=dn_t_all_weights, quick=quick
    )
    dn_t_weights = dn_t_all_weights[:n_bench_layers]

    # Verify fused Metal kernel numerical exactness on Layer 0 with real activation x
    l0_mlp = layers[0].mlp
    x0 = sample_x_list[0]  # (4096,) bf16
    g0 = l0_mlp.gate_proj(x0.reshape(1, 4096))[0]
    mx.eval(g0)

    # Test both mode=0 (g > 0.0) and mode=1 (|SiLU(g)| > 0.10)
    kernel_checks: Dict[str, Any] = {}
    wt_q0, wt_s0, wt_b0 = dn_t_weights[0]
    w_dn_t_dq0 = mx.dequantize(wt_q0, wt_s0, wt_b0, 64, 4, "affine").astype(mx.float32)
    mx.eval(w_dn_t_dq0)

    for tag, thresh, mode in [("g_gt_0", 0.0, 0), ("abs_silu_gt_0.1", 0.1, 1)]:
        a_idx, a_silu, a_cnt = compact_gates_gpu(g0, thresh, mode)
        got = fused_sparse_up_down_call(
            x0,
            a_idx,
            a_silu,
            a_cnt,
            l0_mlp.up_proj.weight,
            l0_mlp.up_proj.scales,
            l0_mlp.up_proj.biases,
            wt_q0,
            wt_s0,
            wt_b0,
            num_tgs=80,
        )
        # Exact reference
        g0_f32 = g0.astype(mx.float32)
        sg0 = nn.silu(g0_f32)
        mask0 = (g0_f32 > thresh) if mode == 0 else (mx.abs(sg0) > thresh)
        u0 = l0_mlp.up_proj(x0.reshape(1, 4096))[0].astype(mx.float32)
        ref = (mx.where(mask0, sg0 * u0, 0.0) @ w_dn_t_dq0).reshape(1, 4096)
        mx.eval(got, ref, a_cnt)
        max_abs = float(mx.max(mx.abs(got.astype(mx.float32) - ref)).item())
        rel_l2 = float(
            (mx.sqrt(mx.sum((got.astype(mx.float32) - ref) ** 2))
             / mx.maximum(mx.sqrt(mx.sum(ref ** 2)), mx.array(1e-8))).item()
        )
        cnt_val = int(a_cnt.item())
        exp_cnt = int(mx.sum(mask0).item())
        kernel_checks[tag] = {
            "active_count_gpu": cnt_val,
            "active_count_ref": exp_cnt,
            "max_abs_err": round(max_abs, 5),
            "rel_l2_err": round(rel_l2, 6),
        }
        print(
            f"  [Kernel Verify] {tag}: count={cnt_val}/{exp_cnt}, "
            f"max_abs_err={max_abs:.5f}, rel_l2_err={rel_l2:.6f}"
        )
        assert cnt_val == exp_cnt, f"Compaction count mismatch: {cnt_val} vs {exp_cnt}"
        assert rel_l2 < 5e-3, f"Fused sparse MLP kernel error too high: {rel_l2}"

    del w_dn_t_dq0
    clear_memory()

    # Pre-build benchmark inputs for all n_bench_layers
    bench_mlps = [layers[l].mlp for l in range(n_bench_layers)]
    bench_xs_2d = [sample_x_list[l].reshape(1, 4096) for l in range(n_bench_layers)]
    bench_xs_1d = [sample_x_list[l] for l in range(n_bench_layers)]
    # Precompute real gates for fixed-k sparse kernel sweep
    bench_gs = [m.gate_proj(x)[0] for m, x in zip(bench_mlps, bench_xs_2d)]
    mx.eval(*bench_gs)

    # Precompute dynamic active counts per threshold across bench layers
    dyn_configs = [
        ("dyn_g_gt_0.0", 0.0, 0),
        ("dyn_g_gt_-0.5", -0.5, 0),
        ("dyn_abs_silu_gt_0.05", 0.05, 1),
        ("dyn_abs_silu_gt_0.10", 0.10, 1),
        ("dyn_abs_silu_gt_0.20", 0.20, 1),
    ]
    dyn_mean_active_pct: Dict[str, float] = {}
    for cname, th, md in dyn_configs:
        cnts = []
        for g_vec in bench_gs:
            _, _, c = compact_gates_gpu(g_vec, th, md)
            mx.eval(c)
            cnts.append(int(c.item()))
        dyn_mean_active_pct[cname] = round(100.0 * float(np.mean(cnts)) / 12288.0, 2)

    # Precompute fixed-k index buffers (for exact iso-sparsity hardware scaling curve)
    fixed_ks = [1024, 2048, 3072, 4096, 6144]
    fixed_k_bufs: Dict[int, List[Tuple[mx.array, mx.array, mx.array]]] = {}
    for k_val in fixed_ks:
        per_l = []
        for g_vec in bench_gs:
            sg = nn.silu(g_vec.astype(mx.float32))
            idx = mx.argpartition(mx.abs(sg), 12288 - k_val)[-k_val:].astype(mx.uint32)
            # Sort indices for cleaner DRAM prefetch locality
            idx = mx.sort(idx)
            s_bits = sg[idx].view(mx.uint32)
            cnt = mx.array([k_val], dtype=mx.uint32)
            mx.eval(idx, s_bits, cnt)
            per_l.append((idx, s_bits, cnt))
        fixed_k_bufs[k_val] = per_l

    def run_stock_full_mlp():
        outs = []
        for m, x in zip(bench_mlps, bench_xs_2d):
            outs.append(m(x))
        return outs

    def run_stock_gate_only():
        outs = []
        for m, x in zip(bench_mlps, bench_xs_2d):
            outs.append(m.gate_proj(x))
        return outs

    def make_dyn_sparse_full_mlp(thresh: float, mode: int):
        def fn():
            outs = []
            for m, x2d, x1d, (wt_q, wt_s, wt_b) in zip(
                bench_mlps, bench_xs_2d, bench_xs_1d, dn_t_weights
            ):
                g = m.gate_proj(x2d)[0]
                a_idx, a_silu, a_cnt = compact_gates_gpu(g, thresh, mode)
                y = fused_sparse_up_down_call(
                    x1d,
                    a_idx,
                    a_silu,
                    a_cnt,
                    m.up_proj.weight,
                    m.up_proj.scales,
                    m.up_proj.biases,
                    wt_q,
                    wt_s,
                    wt_b,
                    num_tgs=80,
                )
                outs.append(y)
            return outs
        return fn

    def make_fixed_k_up_down(k_val: int):
        bufs = fixed_k_bufs[k_val]
        def fn():
            outs = []
            for m, x1d, (wt_q, wt_s, wt_b), (a_idx, a_silu, a_cnt) in zip(
                bench_mlps, bench_xs_1d, dn_t_weights, bufs
            ):
                y = fused_sparse_up_down_call(
                    x1d,
                    a_idx,
                    a_silu,
                    a_cnt,
                    m.up_proj.weight,
                    m.up_proj.scales,
                    m.up_proj.biases,
                    wt_q,
                    wt_s,
                    wt_b,
                    num_tgs=80,
                )
                outs.append(y)
            return outs
        return fn

    arms: Dict[str, Callable[[], Any]] = {
        "stock_full_mlp": run_stock_full_mlp,
        "stock_gate_only": run_stock_gate_only,
    }
    for cname, th, md in dyn_configs:
        arms[cname] = make_dyn_sparse_full_mlp(th, md)
    for k_val in fixed_ks:
        arms[f"fused_up_down_only_k{k_val}"] = make_fixed_k_up_down(k_val)

    one_proj_bytes = (
        l0_mlp.up_proj.weight.nbytes
        + l0_mlp.up_proj.scales.nbytes
        + l0_mlp.up_proj.biases.nbytes
    )
    total_mlp_bytes = n_bench_layers * 3 * one_proj_bytes

    print(
        f"[Duel 2] Benchmarking {n_bench_layers}-layer cold-DRAM MLP stack "
        f"({total_mlp_bytes / 1e9:.2f} GB Q4 weights, {trials} interleaved trials, 540MB SLC flush)..."
    )
    timing = bench_interleaved(arms, flusher=flusher, trials=trials, warmup=2)

    stock_full_ms = timing["stock_full_mlp"]["median_ms"]
    stock_gate_ms = timing["stock_gate_only"]["median_ms"]
    stock_up_down_ms = max(1e-6, stock_full_ms - stock_gate_ms)
    stock_bw_gbs = (total_mlp_bytes / 1e9) / (stock_full_ms / 1e3)

    speedup_summary: Dict[str, Any] = {
        "stock_full_mlp_ms": round(stock_full_ms, 4),
        "stock_gate_only_ms": round(stock_gate_ms, 4),
        "stock_implied_up_down_ms": round(stock_up_down_ms, 4),
        "stock_effective_gbs": round(stock_bw_gbs, 1),
    }
    print(
        f"  stock_full_mlp={stock_full_ms:.3f} ms ({stock_bw_gbs:.1f} GB/s), "
        f"stock_gate_only={stock_gate_ms:.3f} ms, stock_up_down={stock_up_down_ms:.3f} ms"
    )

    for cname, _, _ in dyn_configs:
        ms = timing[cname]["median_ms"]
        up_dn_ms = max(1e-6, ms - stock_gate_ms)
        full_sp = stock_full_ms / ms
        up_dn_sp = stock_up_down_ms / up_dn_ms
        # Exact bytes read = gate (1x) + active_frac * (up + down) (2x)
        act_frac = dyn_mean_active_pct[cname] / 100.0
        eff_bytes = n_bench_layers * one_proj_bytes * (1.0 + 2.0 * act_frac)
        eff_gbs = (eff_bytes / 1e9) / (ms / 1e3)
        speedup_summary[cname] = {
            "active_pct": dyn_mean_active_pct[cname],
            "full_mlp_ms": round(ms, 4),
            "full_mlp_speedup_x": round(full_sp, 3),
            "up_down_speedup_x": round(up_dn_sp, 3),
            "effective_gbs": round(eff_gbs, 1),
        }
        print(
            f"  {cname:22s} (act={dyn_mean_active_pct[cname]:5.1f}%): "
            f"{ms:.3f} ms | full MLP={full_sp:.2f}x | up+down={up_dn_sp:.2f}x | {eff_gbs:.1f} GB/s"
        )

    for k_val in fixed_ks:
        lbl = f"fused_up_down_only_k{k_val}"
        ms = timing[lbl]["median_ms"]
        up_dn_sp = stock_up_down_ms / ms
        est_full_ms = stock_gate_ms + ms
        est_full_sp = stock_full_ms / est_full_ms
        eff_bytes = n_bench_layers * 2 * one_proj_bytes * (k_val / 12288.0)
        eff_gbs = (eff_bytes / 1e9) / (ms / 1e3)
        speedup_summary[lbl] = {
            "active_pct": round(100.0 * k_val / 12288.0, 2),
            "up_down_ms": round(ms, 4),
            "up_down_speedup_x": round(up_dn_sp, 3),
            "projected_full_mlp_speedup_x": round(est_full_sp, 3),
            "effective_gbs": round(eff_gbs, 1),
        }
        print(
            f"  {lbl:22s} (act={100.0 * k_val / 12288.0:5.1f}%): "
            f"up+down={ms:.3f} ms ({up_dn_sp:.2f}x) | proj full MLP={est_full_sp:.2f}x | {eff_gbs:.1f} GB/s"
        )

    # Match quality vs speed to adjudicate WIN / KILLED honestly
    # Contract pass mark: >=1.15x wall-clock speedup AND delta_ppl <= 0.15
    qual_by_name = {q["config"]: q for q in quality_curve}
    dyn_to_qual = {
        "dyn_g_gt_0.0": "g > 0.0",
        "dyn_g_gt_-0.5": "g > -0.5",
        "dyn_abs_silu_gt_0.05": "|SiLU(g)| > 0.05",
        "dyn_abs_silu_gt_0.10": "|SiLU(g)| > 0.10",
        "dyn_abs_silu_gt_0.20": "|SiLU(g)| > 0.20",
    }
    passing_arms = []
    for dname, qname in dyn_to_qual.items():
        if qname in qual_by_name:
            q = qual_by_name[qname]
            sp = speedup_summary[dname]["full_mlp_speedup_x"]
            if sp >= 1.15 and q["delta_ppl"] <= 0.15:
                passing_arms.append(
                    {"arm": dname, "speedup_x": sp, "delta_ppl": q["delta_ppl"]}
                )

    verdict = "WIN" if passing_arms else "KILLED"
    print(f"[Duel 2] VERDICT: {verdict} (passing_arms={passing_arms})")

    return {
        "duel": "mlp_sign_pruning",
        "verdict": verdict,
        "passing_arms": passing_arms,
        "pass_mark": ">=1.15x wall-clock MLP speedup AND <=0.15 PPL delta",
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "config": {
            "model_id": MODEL_ID,
            "n_bench_layers": n_bench_layers,
            "total_mlp_q4_gb": round(total_mlp_bytes / 1e9, 3),
            "down_proj_transpose_repack_rel_l2_err": [round(e, 5) for e in dn_repack_rel_errs],
            "trials": trials,
            "slc_flush_mb": flusher.mb,
        },
        "layer_profile": profile,
        "quality_curve": quality_curve,
        "metal_kernel_verification": kernel_checks,
        "cold_dram_timing_summary": speedup_summary,
        "cold_dram_timing_raw": timing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    res = run_mlp_duel(quick=args.quick)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved Duel 2 results to {args.out}")


if __name__ == "__main__":
    main()
