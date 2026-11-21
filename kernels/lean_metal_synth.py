#!/usr/bin/env python3
"""Machine-Native "Lean-Style" Metal 4 Kernel & MoE Synthesizer for Apple M5 Max.

Implements the Alpoge + Claude methodological transfer:
  - Machine-generates 60+ grotesque, shape-specialized, non-human-readable Metal 4 kernels
    across dense Qwen3.5-9B shapes (4096x4096, 12288x4096, 4096x12288), fused SwiGLU
    (gate+up+SiLU removing intermediate DRAM round-trips), and Qwen3.6-35B-A3B B=1 MoE
    8-expert fused gather_qmv+SwiGLU & down+weighted-sum blocks.
  - Exploits Apple M5 Max Metal 4 direct hardware `bfloat x uint4b_format -> float` NAX
    tensor ops (zero threadgroup weight dequant buffer!), zero-barrier per-simdgroup NAX,
    split-K threadgroup-dequant NAX, 128-bit uint4 vectorized B=1 QMV, and zero-K-loop
    K=512 MoE down-projection fusion.
  - EVERY synthesized kernel MUST pass `verifier.py` (exact linear-basis probe + heavy-tailed
    outlier FP32 reference + poisoned init_value=1337.0 fresh-buffer anti-cheat) BEFORE it is
    eligible for timing.
  - Includes an evolutionary mutator that breeds mutated children from top verified parents.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from lean_verifier import (
    _deq_fp32,
    _make_outliers,
    _pack_u4,
    verify_fused_op,
    verify_qmm,
)

# ---------------------------------------------------------------------------
# Common Metal 4 Headers & Shared Helper Kernels
# ---------------------------------------------------------------------------

METAL4_NAX_HEADER = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

METAL_STD_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
"""

_KERNEL_CACHE: Dict[str, Any] = {}


def _compile_cached(
    name: str,
    input_names: List[str],
    output_names: List[str],
    source: str,
    header: str = METAL_STD_HEADER,
    atomic_outputs: bool = False,
) -> Any:
    key = hashlib.sha256(
        (name + "|" + header + "|" + source + "|" + str(atomic_outputs)).encode()
    ).hexdigest()[:16]
    if key not in _KERNEL_CACHE:
        kname = f"{name}_{key}"
        _KERNEL_CACHE[key] = mx.fast.metal_kernel(
            name=kname,
            input_names=input_names,
            output_names=output_names,
            header=header,
            source=source,
            atomic_outputs=atomic_outputs,
        )
    return _KERNEL_CACHE[key]


def _get_splitk_reduce_kernel(split_k: int, total_elems: int, dtype_str: str = "bfloat") -> Any:
    """Vectorized float4 Split-K reduction kernel: [S, M, N] fp32 -> [M, N] dtype."""
    vec4 = (total_elems % 4) == 0
    if vec4:
        n4 = total_elems // 4
        src = f"""
        constexpr int S = {split_k};
        constexpr int N4 = {n4};
        int idx = int(thread_position_in_grid.x);
        if (idx >= N4) return;
        const device float4* p4 = (const device float4*)P;
        float4 acc = p4[idx];
        #pragma clang loop unroll(full)
        for (int s = 1; s < S; ++s) {{
            acc += p4[s * N4 + idx];
        }}
        device {dtype_str}* y_out = Y + idx * 4;
        y_out[0] = {dtype_str}(acc.x);
        y_out[1] = {dtype_str}(acc.y);
        y_out[2] = {dtype_str}(acc.z);
        y_out[3] = {dtype_str}(acc.w);
        """
    else:
        src = f"""
        constexpr int S = {split_k};
        constexpr int TOT = {total_elems};
        int idx = int(thread_position_in_grid.x);
        if (idx >= TOT) return;
        float acc = P[idx];
        #pragma clang loop unroll(full)
        for (int s = 1; s < S; ++s) {{
            acc += P[s * TOT + idx];
        }}
        Y[idx] = {dtype_str}(acc);
        """
    return _compile_cached(f"sk_red_s{split_k}_{total_elems}_{dtype_str}", ["P"], ["Y"], src)


def _get_splitk_swiglu_reduce_kernel(split_k: int, total_elems: int, dtype_str: str = "bfloat") -> Any:
    """Fused Split-K reduction + SwiGLU epilogue: PG[S,M,N], PU[S,M,N] fp32 -> Y[M,N] dtype."""
    vec4 = (total_elems % 4) == 0
    if vec4:
        n4 = total_elems // 4
        src = f"""
        constexpr int S = {split_k};
        constexpr int N4 = {n4};
        int idx = int(thread_position_in_grid.x);
        if (idx >= N4) return;
        const device float4* pg4 = (const device float4*)PG;
        const device float4* pu4 = (const device float4*)PU;
        float4 g = pg4[idx];
        float4 u = pu4[idx];
        #pragma clang loop unroll(full)
        for (int s = 1; s < S; ++s) {{
            g += pg4[s * N4 + idx];
            u += pu4[s * N4 + idx];
        }}
        float4 r = (g / (1.0f + metal::exp(-g))) * u;
        device {dtype_str}* y_out = Y + idx * 4;
        y_out[0] = {dtype_str}(r.x);
        y_out[1] = {dtype_str}(r.y);
        y_out[2] = {dtype_str}(r.z);
        y_out[3] = {dtype_str}(r.w);
        """
    else:
        src = f"""
        constexpr int S = {split_k};
        constexpr int TOT = {total_elems};
        int idx = int(thread_position_in_grid.x);
        if (idx >= TOT) return;
        float g = PG[idx];
        float u = PU[idx];
        #pragma clang loop unroll(full)
        for (int s = 1; s < S; ++s) {{
            g += PG[s * TOT + idx];
            u += PU[s * TOT + idx];
        }}
        Y[idx] = {dtype_str}((g / (1.0f + metal::exp(-g))) * u);
        """
    return _compile_cached(f"sk_swiglu_s{split_k}_{total_elems}_{dtype_str}", ["PG", "PU"], ["Y"], src)


def _dtype_str(dtype: mx.Dtype) -> str:
    return "bfloat" if dtype == mx.bfloat16 else "half"


# ---------------------------------------------------------------------------
# Candidate Specification Dataclass
# ---------------------------------------------------------------------------

@dataclass
class KernelCandidate:
    cid: str
    arena: str  # "dense_qmm", "dense_swiglu", "moe_gate_up", "moe_down", "moe_full"
    family: str
    shape_tag: str
    genome: Dict[str, Any]
    source_lines: int = 0
    compiled: bool = False
    verified: bool = False
    verify_info: Dict[str, Any] = field(default_factory=dict)
    compile_error: Optional[str] = None
    fn: Optional[Callable] = field(default=None, repr=False)

    def to_record(self) -> Dict[str, Any]:
        return {
            "cid": self.cid,
            "arena": self.arena,
            "family": self.family,
            "shape_tag": self.shape_tag,
            "genome": self.genome,
            "source_lines": self.source_lines,
            "compiled": self.compiled,
            "verified": self.verified,
            "verify_info": self.verify_info,
            "compile_error": self.compile_error,
        }


# ---------------------------------------------------------------------------
# ARENA 1A: Dense Q4 Matmul Code Generators (4 Distinct Families)
# ---------------------------------------------------------------------------

def _swizzle_expr(swizzle: int, n_tiles: int, m_tiles: int) -> str:
    if swizzle == 0:
        return "int tx = int(threadgroup_position_in_grid.x); int ty = int(threadgroup_position_in_grid.y);"
    if swizzle == 1:
        # Swapped grid order (M-major inner, N outer)
        return f"""
        int lin = int(threadgroup_position_in_grid.y) * {n_tiles} + int(threadgroup_position_in_grid.x);
        int ty = lin % {m_tiles};
        int tx = lin / {m_tiles};
        """
    # swizzle == 2: 4-wide grouped SLC tile swizzle
    return f"""
    int raw_x = int(threadgroup_position_in_grid.x);
    int raw_y = int(threadgroup_position_in_grid.y);
    constexpr int GRP = 4;
    int lin = raw_y * {n_tiles} + raw_x;
    int tiles_per_grp = GRP * {m_tiles};
    int grp_id = lin / tiles_per_grp;
    int in_grp = lin % tiles_per_grp;
    int grp_w = min(GRP, {n_tiles} - grp_id * GRP);
    int tx = grp_id * GRP + (in_grp % grp_w);
    int ty = in_grp / grp_w;
    """


def build_u4_nax_coop(
    M: int, N: int, K: int, genome: Dict[str, Any], dtype: mx.Dtype = mx.bfloat16
) -> Tuple[Callable, str]:
    """Family 1: Direct hardware bfloat/half x uint4b_format -> float cooperative NAX."""
    BM = int(genome["BM"])
    BN = int(genome["BN"])
    NSG = int(genome["NSG"])
    split_k = int(genome["split_k"])
    k_unroll = int(genome["k_unroll"])
    cache_sb = bool(genome["cache_sb"])
    swizzle = int(genome["swizzle"])
    stage_x = bool(genome.get("stage_x", False))
    relaxed = bool(genome.get("relaxed", False))

    T = _dtype_str(dtype)
    K_G = K // 64
    assert K_G % split_k == 0, f"K_G={K_G} not divisible by split_k={split_k}"
    G_PER_S = K_G // split_k
    assert G_PER_S % k_unroll == 0, f"G_PER_S={G_PER_S} not divisible by k_unroll={k_unroll}"
    assert N % BN == 0, f"N={N} not divisible by BN={BN}"
    N_TILES = N // BN
    M_TILES = (M + BM - 1) // BM
    aligned_m = (M % BM) == 0
    threads = NSG * 32

    # Compute how threads divide row_sum computation across BM rows * 64 cols
    # Each thread handles a stride over (BM * 64) or per-row simd reduction
    if threads >= BM * 8:
        # 8 threads per row (each sums 8 elements, 3 xor shuffles)
        row_sum_code = """
        {
            int r = lid >> 3;
            int l8 = lid & 7;
            if (r < BM) {
                int mr = m0 + r;
                float s_acc = 0.0f;
                if (mr < M) {
                    const device T_IN* xr = X + mr * K + k0 + l8 * 8;
                    #pragma clang loop unroll(full)
                    for (int jj = 0; jj < 8; ++jj) {
                        float xv = float(xr[jj]);
                        s_acc += xv;
                        STAGE_STORE
                    }
                } else {
                    STAGE_ZERO
                }
                s_acc += simd_shuffle_xor(s_acc, 4);
                s_acc += simd_shuffle_xor(s_acc, 2);
                s_acc += simd_shuffle_xor(s_acc, 1);
                if (l8 == 0) row_sum[uu][r] = s_acc;
            }
        }
        """
    elif threads >= BM * 2:
        # 2 threads per row (each sums 32 elements, 1 xor shuffle)
        row_sum_code = """
        {
            int r = lid >> 1;
            int l2 = lid & 1;
            if (r < BM) {
                int mr = m0 + r;
                float s_acc = 0.0f;
                if (mr < M) {
                    const device T_IN* xr = X + mr * K + k0 + l2 * 32;
                    #pragma clang loop unroll(full)
                    for (int jj = 0; jj < 32; ++jj) {
                        float xv = float(xr[jj]);
                        s_acc += xv;
                        STAGE_STORE_32
                    }
                } else {
                    STAGE_ZERO_32
                }
                s_acc += simd_shuffle_xor(s_acc, 1);
                if (l2 == 0) row_sum[uu][r] = s_acc;
            }
        }
        """
    else:
        # 1 thread per row (loop over BM in steps of threads)
        row_sum_code = f"""
        for (int r = lid; r < BM; r += {threads}) {{
            int mr = m0 + r;
            float s_acc = 0.0f;
            if (mr < M) {{
                const device T_IN* xr = X + mr * K + k0;
                #pragma clang loop unroll(full)
                for (int jj = 0; jj < 64; ++jj) {{
                    float xv = float(xr[jj]);
                    s_acc += xv;
                    STAGE_STORE_64
                }}
            }} else {{
                STAGE_ZERO_64
            }}
            row_sum[uu][r] = s_acc;
        }}
        """

    if stage_x:
        row_sum_code = (
            row_sum_code
            .replace("STAGE_STORE_64", "tg_x[uu][r * 64 + jj] = xr[jj];")
            .replace("STAGE_ZERO_64", "for (int jj = 0; jj < 64; ++jj) tg_x[uu][r * 64 + jj] = T_IN(0);")
            .replace("STAGE_STORE_32", "tg_x[uu][r * 64 + l2 * 32 + jj] = xr[jj];")
            .replace("STAGE_ZERO_32", "for (int jj = 0; jj < 32; ++jj) tg_x[uu][r * 64 + l2 * 32 + jj] = T_IN(0);")
            .replace("STAGE_STORE", "tg_x[uu][r * 64 + l8 * 8 + jj] = xr[jj];")
            .replace("STAGE_ZERO", "for (int jj = 0; jj < 8; ++jj) tg_x[uu][r * 64 + l8 * 8 + jj] = T_IN(0);")
        )
    else:
        row_sum_code = (
            row_sum_code
            .replace("STAGE_STORE_64", "")
            .replace("STAGE_ZERO_64", "")
            .replace("STAGE_STORE_32", "")
            .replace("STAGE_ZERO_32", "")
            .replace("STAGE_STORE", "")
            .replace("STAGE_ZERO", "")
        )

    sb_decl = f"threadgroup float tg_s[{k_unroll}][{BN}]; threadgroup float tg_b[{k_unroll}][{BN}];" if cache_sb else ""
    sb_load = (
        f"""
        for (int nn = lid; nn < BN; nn += {threads}) {{
            tg_s[uu][nn] = float(S[(n0 + nn) * K_G + g]);
            tg_b[uu][nn] = float(B[(n0 + nn) * K_G + g]);
        }}
        """
        if cache_sb
        else ""
    )
    stage_x_decl = f"threadgroup {T} tg_x[{k_unroll}][{BM * 64}];" if stage_x else ""

    if stage_x:
        tA0_decl = f"auto tA0 = tensor<threadgroup {T}, dextents<int32_t, 2>, tensor_inline>(tg_x[0], dextents<int32_t, 2>(64, BM));"
        tA_loop = f"auto tA = tensor<threadgroup {T}, dextents<int32_t, 2>, tensor_inline>(tg_x[uu], dextents<int32_t, 2>(64, BM));"
    elif aligned_m:
        tA0_decl = "auto tA0 = fullA.slice<64, BM>(0, m0);"
        tA_loop = "auto tA = fullA.slice<64, BM>(k0, m0);"
    else:
        tA0_decl = "auto tA0 = fullA.slice(0, m0);"
        tA_loop = "auto tA = fullA.slice(k0, m0);"

    if cache_sb:
        sb_use = "float sv = tg_s[uu][int(idx[0])]; float bv = tg_b[uu][int(idx[0])];"
    else:
        sb_use = "int na = n0 + int(idx[0]); float sv = float(S[na * K_G + g]); float bv = float(B[na * K_G + g]);"

    if split_k == 1:
        out_dtype = dtype
        out_shape = (M, N)
        store_stmt = f"""
        int mr = m0 + int(idx[1]);
        if (mr < M) {{
            Y[mr * N + (n0 + int(idx[0]))] = {T}(cAcc[i]);
        }}
        """
    else:
        out_dtype = mx.float32
        out_shape = (split_k, M, N)
        store_stmt = """
        int mr = m0 + int(idx[1]);
        if (mr < M) {
            Y[(s_id * M + mr) * N + (n0 + int(idx[0]))] = cAcc[i];
        }
        """

    relaxed_str = "true" if relaxed else "false"
    source = f"""
    using T_IN = {T};
    constexpr int BM = {BM}, BN = {BN}, NSG = {NSG};
    constexpr int M = {M}, N = {N}, K = {K}, K_G = {K_G};
    constexpr int G_PER_S = {G_PER_S}, UNR = {k_unroll};
    {_swizzle_expr(swizzle, N_TILES, M_TILES)}
    int s_id = int(threadgroup_position_in_grid.z);
    int n0 = tx * BN;
    int m0 = ty * BM;
    int lid = int(thread_index_in_threadgroup);
    int g_start = s_id * G_PER_S;
    int g_end = g_start + G_PER_S;

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, 64, false, true, {relaxed_str}, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    auto fullA = tensor<device {T}, dextents<int32_t, 2>, tensor_inline>((device {T}*)X, dextents<int32_t, 2>(K, M));
    auto fullB = tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline>((device uchar*)W, dextents<int32_t, 2>(K, N));

    threadgroup float row_sum[UNR][BM];
    {sb_decl}
    {stage_x_decl}

    {tA0_decl}
    auto tB0 = fullB.slice<64, BN>(0, n0);
    auto cAcc = op.get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), float>();
    auto cGrp = op.get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cAcc.get_capacity(); ++i) {{
        if (cAcc.is_valid_element(i)) cAcc[i] = 0.0f;
    }}

    for (int g_base = g_start; g_base < g_end; g_base += UNR) {{
        #pragma clang loop unroll(full)
        for (int uu = 0; uu < UNR; ++uu) {{
            int g = g_base + uu;
            int k0 = g * 64;
            {row_sum_code}
            {sb_load}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int uu = 0; uu < UNR; ++uu) {{
            int g = g_base + uu;
            int k0 = g * 64;
            {tA_loop}
            auto tB = fullB.slice<64, BN>(k0, n0);
            op.run(tA, tB, cGrp);
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < cAcc.get_capacity(); ++i) {{
                if (cAcc.is_valid_element(i)) {{
                    auto idx = cAcc.get_multidimensional_index(i);
                    {sb_use}
                    cAcc[i] = fma(cGrp[i], sv, fma(row_sum[uu][int(idx[1])], bv, cAcc[i]));
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cAcc.get_capacity(); ++i) {{
        if (cAcc.is_valid_element(i)) {{
            auto idx = cAcc.get_multidimensional_index(i);
            {store_stmt}
        }}
    }}
    """
    kern = _compile_cached(
        f"u4nax_c_{M}_{N}_{K}_{BM}_{BN}_{NSG}_s{split_k}_u{k_unroll}",
        ["X", "W", "S", "B"],
        ["Y"],
        source,
        header=METAL4_NAX_HEADER,
    )
    red_kern = _get_splitk_reduce_kernel(split_k, M * N, T) if split_k > 1 else None
    grid = (N_TILES * threads, M_TILES, split_k)
    tg = (threads, 1, 1)
    red_threads = ((M * N) // 4) if ((M * N) % 4 == 0) else (M * N)
    red_tg = min(256, max(32, ((red_threads + 31) // 32) * 32))
    red_grid = (((red_threads + red_tg - 1) // red_tg) * red_tg, 1, 1)

    def runner(x: mx.array, w: mx.array, s: mx.array, b: mx.array, init_value: Optional[float] = None) -> mx.array:
        kw = {"init_value": init_value} if init_value is not None else {}
        out = kern(
            inputs=[x, w, s, b],
            grid=grid,
            threadgroup=tg,
            output_shapes=[out_shape],
            output_dtypes=[out_dtype],
            **kw,
        )[0]
        if red_kern is not None:
            out = red_kern(
                inputs=[out],
                grid=red_grid,
                threadgroup=(red_tg, 1, 1),
                output_shapes=[(M, N)],
                output_dtypes=[dtype],
                **kw,
            )[0]
        return out

    return runner, source


def build_u4_nax_sg_nobar(
    M: int, N: int, K: int, genome: Dict[str, Any], dtype: mx.Dtype = mx.bfloat16
) -> Tuple[Callable, str]:
    """Family 2: Zero-barrier independent per-simdgroup direct uint4b_format NAX."""
    SM = int(genome["SM"])
    SN = int(genome["SN"])
    WM = int(genome["WM"])
    WN = int(genome["WN"])
    split_k = int(genome["split_k"])
    k_unroll = int(genome["k_unroll"])
    swizzle = int(genome["swizzle"])
    relaxed = bool(genome.get("relaxed", False))

    T = _dtype_str(dtype)
    BM = SM * WM
    BN = SN * WN
    K_G = K // 64
    assert K_G % split_k == 0
    G_PER_S = K_G // split_k
    assert G_PER_S % k_unroll == 0
    assert N % BN == 0
    N_TILES = N // BN
    M_TILES = (M + BM - 1) // BM
    aligned_m = (M % BM) == 0
    threads = WM * WN * 32

    tA0_decl = "auto tA0 = fullA.slice<64, SM>(0, m0);" if aligned_m else "auto tA0 = fullA.slice(0, m0);"
    tA_loop = "auto tA = fullA.slice<64, SM>(k0, m0);" if aligned_m else "auto tA = fullA.slice(k0, m0);"
    if split_k == 1:
        out_dtype = dtype
        out_shape = (M, N)
        store_stmt = f"int mr = m0 + int(idx[1]); if (mr < M) Y[mr * N + (n0 + int(idx[0]))] = {T}(cAcc[i]);"
    else:
        out_dtype = mx.float32
        out_shape = (split_k, M, N)
        store_stmt = "int mr = m0 + int(idx[1]); if (mr < M) Y[(s_id * M + mr) * N + (n0 + int(idx[0]))] = cAcc[i];"

    relaxed_str = "true" if relaxed else "false"
    source = f"""
    constexpr int SM = {SM}, SN = {SN}, WM = {WM}, WN = {WN};
    constexpr int BM = {BM}, BN = {BN};
    constexpr int M = {M}, N = {N}, K = {K}, K_G = {K_G};
    constexpr int G_PER_S = {G_PER_S}, UNR = {k_unroll};
    {_swizzle_expr(swizzle, N_TILES, M_TILES)}
    int s_id = int(threadgroup_position_in_grid.z);
    int sg = int(simdgroup_index_in_threadgroup);
    int lane = int(thread_index_in_simdgroup);
    int sg_m = sg / WN;
    int sg_n = sg % WN;
    int m0 = ty * BM + sg_m * SM;
    int n0 = tx * BN + sg_n * SN;
    if (m0 >= M) return;

    int g_start = s_id * G_PER_S;
    int g_end = g_start + G_PER_S;

    constexpr auto desc = matmul2d_descriptor(
        SM, SN, 64, false, true, {relaxed_str}, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;

    auto fullA = tensor<device {T}, dextents<int32_t, 2>, tensor_inline>((device {T}*)X, dextents<int32_t, 2>(K, M));
    auto fullB = tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline>((device uchar*)W, dextents<int32_t, 2>(K, N));

    {tA0_decl}
    auto tB0 = fullB.slice<64, SN>(0, n0);
    auto cAcc = op.get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), float>();
    auto cGrp = op.get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cAcc.get_capacity(); ++i) {{
        if (cAcc.is_valid_element(i)) cAcc[i] = 0.0f;
    }}

    float row_sum[SM];
    for (int g_base = g_start; g_base < g_end; g_base += UNR) {{
        #pragma clang loop unroll(full)
        for (int uu = 0; uu < UNR; ++uu) {{
            int g = g_base + uu;
            int k0 = g * 64;
            #pragma clang loop unroll(full)
            for (int r = 0; r < SM; ++r) {{
                int mr = m0 + r;
                float v = 0.0f;
                if (mr < M) {{
                    const device {T}* xr = X + mr * K + k0 + lane * 2;
                    v = float(xr[0]) + float(xr[1]);
                }}
                row_sum[r] = simd_sum(v);
            }}
            {tA_loop}
            auto tB = fullB.slice<64, SN>(k0, n0);
            op.run(tA, tB, cGrp);
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < cAcc.get_capacity(); ++i) {{
                if (cAcc.is_valid_element(i)) {{
                    auto idx = cAcc.get_multidimensional_index(i);
                    int na = n0 + int(idx[0]);
                    cAcc[i] = fma(cGrp[i], float(S[na * K_G + g]), fma(row_sum[int(idx[1])], float(B[na * K_G + g]), cAcc[i]));
                }}
            }}
        }}
    }}

    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cAcc.get_capacity(); ++i) {{
        if (cAcc.is_valid_element(i)) {{
            auto idx = cAcc.get_multidimensional_index(i);
            {store_stmt}
        }}
    }}
    """
    kern = _compile_cached(
        f"u4nax_sg_{M}_{N}_{K}_{SM}_{SN}_{WM}x{WN}_s{split_k}",
        ["X", "W", "S", "B"],
        ["Y"],
        source,
        header=METAL4_NAX_HEADER,
    )
    red_kern = _get_splitk_reduce_kernel(split_k, M * N, T) if split_k > 1 else None
    grid = (N_TILES * threads, M_TILES, split_k)
    tg = (threads, 1, 1)
    red_threads = ((M * N) // 4) if ((M * N) % 4 == 0) else (M * N)
    red_tg = min(256, max(32, ((red_threads + 31) // 32) * 32))
    red_grid = (((red_threads + red_tg - 1) // red_tg) * red_tg, 1, 1)

    def runner(x: mx.array, w: mx.array, s: mx.array, b: mx.array, init_value: Optional[float] = None) -> mx.array:
        kw = {"init_value": init_value} if init_value is not None else {}
        out = kern(
            inputs=[x, w, s, b],
            grid=grid,
            threadgroup=tg,
            output_shapes=[out_shape],
            output_dtypes=[out_dtype],
            **kw,
        )[0]
        if red_kern is not None:
            out = red_kern(
                inputs=[out],
                grid=red_grid,
                threadgroup=(red_tg, 1, 1),
                output_shapes=[(M, N)],
                output_dtypes=[dtype],
                **kw,
            )[0]
        return out

    return runner, source


def build_tgp_deq_nax(
    M: int, N: int, K: int, genome: Dict[str, Any], dtype: mx.Dtype = mx.bfloat16
) -> Tuple[Callable, str]:
    """Family 3: Split-K Threadgroup-Dequantized bfloat/half NAX (multiply_accumulate).

    Supports BK_GROUPS in {1, 2} (64 or 128 K-elements per barrier!) and BN in {32, 64, 128}.
    """
    BM = int(genome["BM"])
    BN = int(genome["BN"])
    NSG = int(genome["NSG"])
    split_k = int(genome["split_k"])
    swizzle = int(genome["swizzle"])
    bk_groups = int(genome.get("bk_groups", 1))  # 1 -> BK=64, 2 -> BK=128

    T = _dtype_str(dtype)
    BK = bk_groups * 64
    K_G = K // 64
    K_W = K // 8  # uint32 words per row
    assert K_G % (split_k * bk_groups) == 0
    G_PER_S = K_G // split_k
    assert N % BN == 0
    N_TILES = N // BN
    M_TILES = (M + BM - 1) // BM
    aligned_m = (M % BM) == 0
    threads = NSG * 32
    # Each row in Ws has BK elems = (BK/32) uint4 loads (32 nibbles each).
    u4_per_row = BK // 32
    total_u4 = BN * u4_per_row

    tA0_decl = f"auto tA0 = fullA.slice<{BK}, BM>(0, m0);" if aligned_m else "auto tA0 = fullA.slice(0, m0);"
    tA_loop = f"auto tA = fullA.slice<{BK}, BM>(k0, m0);" if aligned_m else "auto tA = fullA.slice(k0, m0);"
    if split_k == 1:
        out_dtype = dtype
        out_shape = (M, N)
        store_stmt = f"int mr = m0 + int(idx[1]); if (mr < M) Y[mr * N + (n0 + int(idx[0]))] = {T}(cAcc[i]);"
    else:
        out_dtype = mx.float32
        out_shape = (split_k, M, N)
        store_stmt = "int mr = m0 + int(idx[1]); if (mr < M) Y[(s_id * M + mr) * N + (n0 + int(idx[0]))] = cAcc[i];"

    source = f"""
    constexpr int BM = {BM}, BN = {BN}, BK = {BK}, NSG = {NSG}, BKG = {bk_groups};
    constexpr int M = {M}, N = {N}, K = {K}, K_G = {K_G}, K_W = {K_W};
    constexpr int G_PER_S = {G_PER_S}, U4_ROW = {u4_per_row}, TOTAL_U4 = {total_u4}, THREADS = {threads};
    {_swizzle_expr(swizzle, N_TILES, M_TILES)}
    int s_id = int(threadgroup_position_in_grid.z);
    int n0 = tx * BN;
    int m0 = ty * BM;
    int lid = int(thread_index_in_threadgroup);
    int g_start = s_id * G_PER_S;
    int g_end = g_start + G_PER_S;

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    auto fullA = tensor<device {T}, dextents<int32_t, 2>, tensor_inline>((device {T}*)X, dextents<int32_t, 2>(K, M));
    threadgroup {T} Ws[BN * BK];
    auto tB = tensor<threadgroup {T}, dextents<int32_t, 2>, tensor_inline>(Ws, dextents<int32_t, 2>(BK, BN));

    {tA0_decl}
    auto cAcc = op.get_destination_cooperative_tensor<decltype(tA0), decltype(tB), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cAcc.get_capacity(); ++i) {{
        if (cAcc.is_valid_element(i)) cAcc[i] = 0.0f;
    }}

    for (int g = g_start; g < g_end; g += BKG) {{
        int k0 = g * 64;
        for (int c = lid; c < TOTAL_U4; c += THREADS) {{
            int rn = c / U4_ROW;
            int chunk = c % U4_ROW;
            int sub_g = chunk >> 1;
            int half_idx = chunk & 1;
            int na = n0 + rn;
            int g_cur = g + sub_g;
            float sc = float(S[na * K_G + g_cur]);
            float bi = float(B[na * K_G + g_cur]);
            const device uint4* wp = (const device uint4*)(W + na * K_W + g_cur * 8 + half_idx * 4);
            uint4 pk = wp[0];
            threadgroup {T}* dst = Ws + rn * BK + chunk * 32;
            uint words[4] = {{pk.x, pk.y, pk.z, pk.w}};
            #pragma clang loop unroll(full)
            for (int wi = 0; wi < 4; ++wi) {{
                uint v = words[wi];
                dst[wi * 8 + 0] = {T}(fma(float(v & 0xFu), sc, bi));
                dst[wi * 8 + 1] = {T}(fma(float((v >> 4) & 0xFu), sc, bi));
                dst[wi * 8 + 2] = {T}(fma(float((v >> 8) & 0xFu), sc, bi));
                dst[wi * 8 + 3] = {T}(fma(float((v >> 12) & 0xFu), sc, bi));
                dst[wi * 8 + 4] = {T}(fma(float((v >> 16) & 0xFu), sc, bi));
                dst[wi * 8 + 5] = {T}(fma(float((v >> 20) & 0xFu), sc, bi));
                dst[wi * 8 + 6] = {T}(fma(float((v >> 24) & 0xFu), sc, bi));
                dst[wi * 8 + 7] = {T}(fma(float((v >> 28) & 0xFu), sc, bi));
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {tA_loop}
        op.run(tA, tB, cAcc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cAcc.get_capacity(); ++i) {{
        if (cAcc.is_valid_element(i)) {{
            auto idx = cAcc.get_multidimensional_index(i);
            {store_stmt}
        }}
    }}
    """
    kern = _compile_cached(
        f"tgp_deq_{M}_{N}_{K}_{BM}_{BN}_bk{BK}_{NSG}_s{split_k}",
        ["X", "W", "S", "B"],
        ["Y"],
        source,
        header=METAL4_NAX_HEADER,
    )
    red_kern = _get_splitk_reduce_kernel(split_k, M * N, T) if split_k > 1 else None
    grid = (N_TILES * threads, M_TILES, split_k)
    tg = (threads, 1, 1)
    red_threads = ((M * N) // 4) if ((M * N) % 4 == 0) else (M * N)
    red_tg = min(256, max(32, ((red_threads + 31) // 32) * 32))
    red_grid = (((red_threads + red_tg - 1) // red_tg) * red_tg, 1, 1)

    def runner(x: mx.array, w: mx.array, s: mx.array, b: mx.array, init_value: Optional[float] = None) -> mx.array:
        kw = {"init_value": init_value} if init_value is not None else {}
        out = kern(
            inputs=[x, w, s, b],
            grid=grid,
            threadgroup=tg,
            output_shapes=[out_shape],
            output_dtypes=[out_dtype],
            **kw,
        )[0]
        if red_kern is not None:
            out = red_kern(
                inputs=[out],
                grid=red_grid,
                threadgroup=(red_tg, 1, 1),
                output_shapes=[(M, N)],
                output_dtypes=[dtype],
                **kw,
            )[0]
        return out

    return runner, source


def build_vec_qmv_b1(
    M: int, N: int, K: int, genome: Dict[str, Any], dtype: mx.Dtype = mx.bfloat16
) -> Tuple[Callable, str]:
    """Family 4: Vectorized uint4/uint2/uint register-unpack QMV for M=1..4."""
    NR = int(genome["NR"])
    NSG = int(genome["NSG"])
    vec_bits = int(genome["vec_bits"])  # 32, 64, or 128
    split_k = int(genome["split_k"])
    nibble_mode = str(genome.get("nibble_mode", "shift"))  # "shift" or "mlx_div"

    T = _dtype_str(dtype)
    K_G = K // 64
    K_W = K // 8
    words_per_lane = vec_bits // 32  # 1, 2, or 4 uint32 words = 8, 16, or 32 nibbles
    vals_per_lane = words_per_lane * 8
    step_k = vals_per_lane * 32  # 256, 512, or 1024 elements per simdgroup step
    assert K % (split_k * step_k) == 0, f"K={K} not divisible by split_k*step_k={split_k*step_k}"
    k_per_s = K // split_k
    rows_per_tg = NR * NSG
    assert N % rows_per_tg == 0
    N_TILES = N // rows_per_tg
    threads = NSG * 32

    if nibble_mode == "mlx_div":
        load_x_code = f"""
        float xsum = 0.0f;
        float xv[{vals_per_lane}];
        #pragma clang loop unroll(full)
        for (int i = 0; i < {vals_per_lane}; i += 4) {{
            float a0 = float(xp[i + 0]);
            float a1 = float(xp[i + 1]);
            float a2 = float(xp[i + 2]);
            float a3 = float(xp[i + 3]);
            xsum += (a0 + a1) + (a2 + a3);
            xv[i + 0] = a0;
            xv[i + 1] = a1 * (1.0f / 16.0f);
            xv[i + 2] = a2 * (1.0f / 256.0f);
            xv[i + 3] = a3 * (1.0f / 4096.0f);
        }}
        """
        dot_word_code = """
        uint lo = wword & 0xFFFFu;
        uint hi = wword >> 16;
        d += xv[wi * 8 + 0] * float(lo & 0x000Fu)
           + xv[wi * 8 + 1] * float(lo & 0x00F0u)
           + xv[wi * 8 + 2] * float(lo & 0x0F00u)
           + xv[wi * 8 + 3] * float(lo & 0xF000u)
           + xv[wi * 8 + 4] * float(hi & 0x000Fu)
           + xv[wi * 8 + 5] * float(hi & 0x00F0u)
           + xv[wi * 8 + 6] * float(hi & 0x0F00u)
           + xv[wi * 8 + 7] * float(hi & 0xF000u);
        """
    else:
        load_x_code = f"""
        float xsum = 0.0f;
        float xv[{vals_per_lane}];
        #pragma clang loop unroll(full)
        for (int i = 0; i < {vals_per_lane}; ++i) {{
            float a = float(xp[i]);
            xv[i] = a;
            xsum += a;
        }}
        """
        dot_word_code = """
        d += xv[wi * 8 + 0] * float(wword & 0xFu)
           + xv[wi * 8 + 1] * float((wword >> 4) & 0xFu)
           + xv[wi * 8 + 2] * float((wword >> 8) & 0xFu)
           + xv[wi * 8 + 3] * float((wword >> 12) & 0xFu)
           + xv[wi * 8 + 4] * float((wword >> 16) & 0xFu)
           + xv[wi * 8 + 5] * float((wword >> 20) & 0xFu)
           + xv[wi * 8 + 6] * float((wword >> 24) & 0xFu)
           + xv[wi * 8 + 7] * float((wword >> 28) & 0xFu);
        """

    if words_per_lane == 4:
        w_load_expr = "uint4 pk = *((const device uint4*)(W + r_abs * K_W + (k_pos >> 3))); uint ww[4] = {pk.x, pk.y, pk.z, pk.w};"
    elif words_per_lane == 2:
        w_load_expr = "uint2 pk = *((const device uint2*)(W + r_abs * K_W + (k_pos >> 3))); uint ww[2] = {pk.x, pk.y};"
    else:
        w_load_expr = "uint ww[1] = {W[r_abs * K_W + (k_pos >> 3)]};"

    if split_k == 1:
        out_dtype = dtype
        out_shape = (M, N)
        store_code = f"if (lane == 0) Y[m_id * N + r0 + r] = {T}(res);"
    else:
        out_dtype = mx.float32
        out_shape = (split_k, M, N)
        store_code = "if (lane == 0) Y[(s_id * M + m_id) * N + r0 + r] = res;"

    source = f"""
    constexpr int NR = {NR}, NSG = {NSG}, M = {M}, N = {N}, K = {K};
    constexpr int K_G = {K_G}, K_W = {K_W}, STEP_K = {step_k}, K_PER_S = {k_per_s};
    constexpr int VPL = {vals_per_lane}, WPL = {words_per_lane};
    int sg = int(simdgroup_index_in_threadgroup);
    int lane = int(thread_index_in_simdgroup);
    int r0 = (int(threadgroup_position_in_grid.x) * NSG + sg) * NR;
    int m_id = int(threadgroup_position_in_grid.y);
    int s_id = int(threadgroup_position_in_grid.z);
    int k_start = s_id * K_PER_S;
    int k_end = k_start + K_PER_S;

    float acc[NR];
    #pragma clang loop unroll(full)
    for (int r = 0; r < NR; ++r) acc[r] = 0.0f;

    for (int k = k_start; k < k_end; k += STEP_K) {{
        int k_pos = k + lane * VPL;
        int g_idx = k_pos >> 6;
        const device {T}* xp = X + m_id * K + k_pos;
        {load_x_code}
        #pragma clang loop unroll(full)
        for (int r = 0; r < NR; ++r) {{
            int r_abs = r0 + r;
            {w_load_expr}
            float d = 0.0f;
            #pragma clang loop unroll(full)
            for (int wi = 0; wi < WPL; ++wi) {{
                uint wword = ww[wi];
                {dot_word_code}
            }}
            float sc = float(S[r_abs * K_G + g_idx]);
            float bi = float(B[r_abs * K_G + g_idx]);
            acc[r] = fma(d, sc, fma(xsum, bi, acc[r]));
        }}
    }}
    #pragma clang loop unroll(full)
    for (int r = 0; r < NR; ++r) {{
        float res = simd_sum(acc[r]);
        {store_code}
    }}
    """
    kern = _compile_cached(
        f"vecqmv_{M}_{N}_{K}_r{NR}_sg{NSG}_b{vec_bits}_s{split_k}_{nibble_mode}",
        ["X", "W", "S", "B"],
        ["Y"],
        source,
    )
    red_kern = _get_splitk_reduce_kernel(split_k, M * N, T) if split_k > 1 else None
    grid = (N_TILES * threads, M, split_k)
    tg = (threads, 1, 1)
    red_threads = ((M * N) // 4) if ((M * N) % 4 == 0) else (M * N)
    red_tg = min(256, max(32, ((red_threads + 31) // 32) * 32))
    red_grid = (((red_threads + red_tg - 1) // red_tg) * red_tg, 1, 1)

    def runner(x: mx.array, w: mx.array, s: mx.array, b: mx.array, init_value: Optional[float] = None) -> mx.array:
        kw = {"init_value": init_value} if init_value is not None else {}
        out = kern(
            inputs=[x, w, s, b],
            grid=grid,
            threadgroup=tg,
            output_shapes=[out_shape],
            output_dtypes=[out_dtype],
            **kw,
        )[0]
        if red_kern is not None:
            out = red_kern(
                inputs=[out],
                grid=red_grid,
                threadgroup=(red_tg, 1, 1),
                output_shapes=[(M, N)],
                output_dtypes=[dtype],
                **kw,
            )[0]
        return out

    return runner, source


# ---------------------------------------------------------------------------
# ARENA 1B: Fused Dense gate_proj + up_proj + SiLU SwiGLU (Removes DRAM round-trip!)
# ---------------------------------------------------------------------------

def build_fused_swiglu_u4_nax(
    M: int, N: int, K: int, genome: Dict[str, Any], dtype: mx.Dtype = mx.bfloat16
) -> Tuple[Callable, str]:
    """Family 5: Single-pass direct hardware uint4b_format NAX fused gate+up+SiLU."""
    BM = int(genome["BM"])
    BN = int(genome["BN"])
    NSG = int(genome["NSG"])
    split_k = int(genome["split_k"])
    k_unroll = int(genome["k_unroll"])
    cache_sb = bool(genome["cache_sb"])
    swizzle = int(genome["swizzle"])

    T = _dtype_str(dtype)
    K_G = K // 64
    assert K_G % split_k == 0
    G_PER_S = K_G // split_k
    assert G_PER_S % k_unroll == 0
    assert N % BN == 0
    N_TILES = N // BN
    M_TILES = (M + BM - 1) // BM
    aligned_m = (M % BM) == 0
    threads = NSG * 32

    if threads >= BM * 8:
        row_sum_code = """
        {
            int r = lid >> 3;
            int l8 = lid & 7;
            if (r < BM) {
                int mr = m0 + r;
                float s_acc = 0.0f;
                if (mr < M) {
                    const device T_IN* xr = X + mr * K + k0 + l8 * 8;
                    #pragma clang loop unroll(full)
                    for (int jj = 0; jj < 8; ++jj) s_acc += float(xr[jj]);
                }
                s_acc += simd_shuffle_xor(s_acc, 4);
                s_acc += simd_shuffle_xor(s_acc, 2);
                s_acc += simd_shuffle_xor(s_acc, 1);
                if (l8 == 0) row_sum[uu][r] = s_acc;
            }
        }
        """
    else:
        row_sum_code = f"""
        for (int r = lid; r < BM; r += {threads}) {{
            int mr = m0 + r;
            float s_acc = 0.0f;
            if (mr < M) {{
                const device T_IN* xr = X + mr * K + k0;
                #pragma clang loop unroll(full)
                for (int jj = 0; jj < 64; ++jj) s_acc += float(xr[jj]);
            }}
            row_sum[uu][r] = s_acc;
        }}
        """

    sb_decl = (
        f"threadgroup float tg_sg[{k_unroll}][{BN}], tg_bg[{k_unroll}][{BN}], tg_su[{k_unroll}][{BN}], tg_bu[{k_unroll}][{BN}];"
        if cache_sb
        else ""
    )
    sb_load = (
        f"""
        for (int nn = lid; nn < BN; nn += {threads}) {{
            int na = n0 + nn;
            tg_sg[uu][nn] = float(SG[na * K_G + g]);
            tg_bg[uu][nn] = float(BG[na * K_G + g]);
            tg_su[uu][nn] = float(SU[na * K_G + g]);
            tg_bu[uu][nn] = float(BU[na * K_G + g]);
        }}
        """
        if cache_sb
        else ""
    )
    if cache_sb:
        sb_g = "float svg = tg_sg[uu][int(idx[0])]; float bvg = tg_bg[uu][int(idx[0])];"
        sb_u = "float svu = tg_su[uu][int(idx[0])]; float bvu = tg_bu[uu][int(idx[0])];"
    else:
        sb_g = "int na = n0 + int(idx[0]); float svg = float(SG[na * K_G + g]); float bvg = float(BG[na * K_G + g]);"
        sb_u = "int na = n0 + int(idx[0]); float svu = float(SU[na * K_G + g]); float bvu = float(BU[na * K_G + g]);"

    tA0_decl = "auto tA0 = fullA.slice<64, BM>(0, m0);" if aligned_m else "auto tA0 = fullA.slice(0, m0);"
    tA_loop = "auto tA = fullA.slice<64, BM>(k0, m0);" if aligned_m else "auto tA = fullA.slice(k0, m0);"

    if split_k == 1:
        output_names = ["Y"]
        out_shapes = [(M, N)]
        out_dtypes = [dtype]
        store_code = f"""
        int mr = m0 + int(idx[1]);
        if (mr < M) {{
            float gv = cGate[i];
            float uv = cUp[i];
            Y[mr * N + (n0 + int(idx[0]))] = {T}((gv / (1.0f + metal::exp(-gv))) * uv);
        }}
        """
    else:
        output_names = ["PG", "PU"]
        out_shapes = [(split_k, M, N), (split_k, M, N)]
        out_dtypes = [mx.float32, mx.float32]
        store_code = """
        int mr = m0 + int(idx[1]);
        if (mr < M) {
            int off = (s_id * M + mr) * N + (n0 + int(idx[0]));
            PG[off] = cGate[i];
            PU[off] = cUp[i];
        }
        """

    source = f"""
    using T_IN = {T};
    constexpr int BM = {BM}, BN = {BN}, NSG = {NSG};
    constexpr int M = {M}, N = {N}, K = {K}, K_G = {K_G};
    constexpr int G_PER_S = {G_PER_S}, UNR = {k_unroll};
    {_swizzle_expr(swizzle, N_TILES, M_TILES)}
    int s_id = int(threadgroup_position_in_grid.z);
    int n0 = tx * BN;
    int m0 = ty * BM;
    int lid = int(thread_index_in_threadgroup);
    int g_start = s_id * G_PER_S;
    int g_end = g_start + G_PER_S;

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, 64, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    auto fullA  = tensor<device {T}, dextents<int32_t, 2>, tensor_inline>((device {T}*)X, dextents<int32_t, 2>(K, M));
    auto fullWG = tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline>((device uchar*)WG, dextents<int32_t, 2>(K, N));
    auto fullWU = tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline>((device uchar*)WU, dextents<int32_t, 2>(K, N));

    threadgroup float row_sum[UNR][BM];
    {sb_decl}

    {tA0_decl}
    auto tB0 = fullWG.slice<64, BN>(0, n0);
    auto cGate = op.get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), float>();
    auto cUp   = op.get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), float>();
    auto cTmp  = op.get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cGate.get_capacity(); ++i) {{
        if (cGate.is_valid_element(i)) {{ cGate[i] = 0.0f; cUp[i] = 0.0f; }}
    }}

    for (int g_base = g_start; g_base < g_end; g_base += UNR) {{
        #pragma clang loop unroll(full)
        for (int uu = 0; uu < UNR; ++uu) {{
            int g = g_base + uu;
            int k0 = g * 64;
            {row_sum_code}
            {sb_load}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int uu = 0; uu < UNR; ++uu) {{
            int g = g_base + uu;
            int k0 = g * 64;
            {tA_loop}
            auto tBG = fullWG.slice<64, BN>(k0, n0);
            op.run(tA, tBG, cTmp);
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < cGate.get_capacity(); ++i) {{
                if (cGate.is_valid_element(i)) {{
                    auto idx = cGate.get_multidimensional_index(i);
                    {sb_g}
                    cGate[i] = fma(cTmp[i], svg, fma(row_sum[uu][int(idx[1])], bvg, cGate[i]));
                }}
            }}
            auto tBU = fullWU.slice<64, BN>(k0, n0);
            op.run(tA, tBU, cTmp);
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < cUp.get_capacity(); ++i) {{
                if (cUp.is_valid_element(i)) {{
                    auto idx = cUp.get_multidimensional_index(i);
                    {sb_u}
                    cUp[i] = fma(cTmp[i], svu, fma(row_sum[uu][int(idx[1])], bvu, cUp[i]));
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cGate.get_capacity(); ++i) {{
        if (cGate.is_valid_element(i)) {{
            auto idx = cGate.get_multidimensional_index(i);
            {store_code}
        }}
    }}
    """
    kern = _compile_cached(
        f"fswiglu_nax_{M}_{N}_{K}_{BM}_{BN}_{NSG}_s{split_k}",
        ["X", "WG", "SG", "BG", "WU", "SU", "BU"],
        output_names,
        source,
        header=METAL4_NAX_HEADER,
    )
    red_kern = _get_splitk_swiglu_reduce_kernel(split_k, M * N, T) if split_k > 1 else None
    grid = (N_TILES * threads, M_TILES, split_k)
    tg = (threads, 1, 1)
    red_threads = ((M * N) // 4) if ((M * N) % 4 == 0) else (M * N)
    red_tg = min(256, max(32, ((red_threads + 31) // 32) * 32))
    red_grid = (((red_threads + red_tg - 1) // red_tg) * red_tg, 1, 1)

    def runner(
        x: mx.array, wg: mx.array, sg: mx.array, bg: mx.array,
        wu: mx.array, su: mx.array, bu: mx.array, init_value: Optional[float] = None
    ) -> mx.array:
        kw = {"init_value": init_value} if init_value is not None else {}
        outs = kern(
            inputs=[x, wg, sg, bg, wu, su, bu],
            grid=grid,
            threadgroup=tg,
            output_shapes=out_shapes,
            output_dtypes=out_dtypes,
            **kw,
        )
        if red_kern is not None:
            return red_kern(
                inputs=[outs[0], outs[1]],
                grid=red_grid,
                threadgroup=(red_tg, 1, 1),
                output_shapes=[(M, N)],
                output_dtypes=[dtype],
                **kw,
            )[0]
        return outs[0]

    return runner, source


def build_fused_swiglu_tgp_deq_nax(
    M: int, N: int, K: int, genome: Dict[str, Any], dtype: mx.Dtype = mx.bfloat16
) -> Tuple[Callable, str]:
    """Family 5B: Split-K Threadgroup-Dequantized NAX fused gate+up+SiLU (multiply_accumulate!).

    Reuses ONE Ws buffer sequentially for gate then up, keeping cGate & cUp in hardware NAX
    multiply_accumulate mode with ZERO per-group scale/bias loop and ZERO intermediate DRAM write!
    """
    BM = int(genome["BM"])
    BN = int(genome["BN"])
    NSG = int(genome["NSG"])
    split_k = int(genome["split_k"])
    swizzle = int(genome["swizzle"])
    bk_groups = int(genome.get("bk_groups", 1))

    T = _dtype_str(dtype)
    BK = bk_groups * 64
    K_G = K // 64
    K_W = K // 8
    assert K_G % (split_k * bk_groups) == 0
    G_PER_S = K_G // split_k
    assert N % BN == 0
    N_TILES = N // BN
    M_TILES = (M + BM - 1) // BM
    aligned_m = (M % BM) == 0
    threads = NSG * 32
    u4_per_row = BK // 32
    total_u4 = BN * u4_per_row

    tA0_decl = f"auto tA0 = fullA.slice<{BK}, BM>(0, m0);" if aligned_m else "auto tA0 = fullA.slice(0, m0);"
    tA_loop = f"auto tA = fullA.slice<{BK}, BM>(k0, m0);" if aligned_m else "auto tA = fullA.slice(k0, m0);"

    if split_k == 1:
        output_names = ["Y"]
        out_shapes = [(M, N)]
        out_dtypes = [dtype]
        store_code = f"""
        int mr = m0 + int(idx[1]);
        if (mr < M) {{
            float gv = cGate[i];
            float uv = cUp[i];
            Y[mr * N + (n0 + int(idx[0]))] = {T}((gv / (1.0f + metal::exp(-gv))) * uv);
        }}
        """
    else:
        output_names = ["PG", "PU"]
        out_shapes = [(split_k, M, N), (split_k, M, N)]
        out_dtypes = [mx.float32, mx.float32]
        store_code = """
        int mr = m0 + int(idx[1]);
        if (mr < M) {
            int off = (s_id * M + mr) * N + (n0 + int(idx[0]));
            PG[off] = cGate[i];
            PU[off] = cUp[i];
        }
        """

    source = f"""
    constexpr int BM = {BM}, BN = {BN}, BK = {BK}, NSG = {NSG}, BKG = {bk_groups};
    constexpr int M = {M}, N = {N}, K = {K}, K_G = {K_G}, K_W = {K_W};
    constexpr int G_PER_S = {G_PER_S}, U4_ROW = {u4_per_row}, TOTAL_U4 = {total_u4}, THREADS = {threads};
    {_swizzle_expr(swizzle, N_TILES, M_TILES)}
    int s_id = int(threadgroup_position_in_grid.z);
    int n0 = tx * BN;
    int m0 = ty * BM;
    int lid = int(thread_index_in_threadgroup);
    int g_start = s_id * G_PER_S;
    int g_end = g_start + G_PER_S;

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    auto fullA = tensor<device {T}, dextents<int32_t, 2>, tensor_inline>((device {T}*)X, dextents<int32_t, 2>(K, M));
    threadgroup {T} Ws[BN * BK];
    auto tB = tensor<threadgroup {T}, dextents<int32_t, 2>, tensor_inline>(Ws, dextents<int32_t, 2>(BK, BN));

    {tA0_decl}
    auto cGate = op.get_destination_cooperative_tensor<decltype(tA0), decltype(tB), float>();
    auto cUp   = op.get_destination_cooperative_tensor<decltype(tA0), decltype(tB), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cGate.get_capacity(); ++i) {{
        if (cGate.is_valid_element(i)) {{ cGate[i] = 0.0f; cUp[i] = 0.0f; }}
    }}

    for (int g = g_start; g < g_end; g += BKG) {{
        int k0 = g * 64;
        {tA_loop}
        // Dequant gate tile -> Ws -> cGate
        for (int c = lid; c < TOTAL_U4; c += THREADS) {{
            int rn = c / U4_ROW;
            int chunk = c % U4_ROW;
            int sub_g = chunk >> 1;
            int half_idx = chunk & 1;
            int na = n0 + rn;
            int g_cur = g + sub_g;
            float sc = float(SG[na * K_G + g_cur]);
            float bi = float(BG[na * K_G + g_cur]);
            uint4 pk = *((const device uint4*)(WG + na * K_W + g_cur * 8 + half_idx * 4));
            threadgroup {T}* dst = Ws + rn * BK + chunk * 32;
            uint words[4] = {{pk.x, pk.y, pk.z, pk.w}};
            #pragma clang loop unroll(full)
            for (int wi = 0; wi < 4; ++wi) {{
                uint v = words[wi];
                dst[wi*8+0] = {T}(fma(float(v & 0xFu), sc, bi));
                dst[wi*8+1] = {T}(fma(float((v >> 4) & 0xFu), sc, bi));
                dst[wi*8+2] = {T}(fma(float((v >> 8) & 0xFu), sc, bi));
                dst[wi*8+3] = {T}(fma(float((v >> 12) & 0xFu), sc, bi));
                dst[wi*8+4] = {T}(fma(float((v >> 16) & 0xFu), sc, bi));
                dst[wi*8+5] = {T}(fma(float((v >> 20) & 0xFu), sc, bi));
                dst[wi*8+6] = {T}(fma(float((v >> 24) & 0xFu), sc, bi));
                dst[wi*8+7] = {T}(fma(float((v >> 28) & 0xFu), sc, bi));
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(tA, tB, cGate);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Dequant up tile -> Ws -> cUp (reusing same tA in L1 & same Ws!)
        for (int c = lid; c < TOTAL_U4; c += THREADS) {{
            int rn = c / U4_ROW;
            int chunk = c % U4_ROW;
            int sub_g = chunk >> 1;
            int half_idx = chunk & 1;
            int na = n0 + rn;
            int g_cur = g + sub_g;
            float sc = float(SU[na * K_G + g_cur]);
            float bi = float(BU[na * K_G + g_cur]);
            uint4 pk = *((const device uint4*)(WU + na * K_W + g_cur * 8 + half_idx * 4));
            threadgroup {T}* dst = Ws + rn * BK + chunk * 32;
            uint words[4] = {{pk.x, pk.y, pk.z, pk.w}};
            #pragma clang loop unroll(full)
            for (int wi = 0; wi < 4; ++wi) {{
                uint v = words[wi];
                dst[wi*8+0] = {T}(fma(float(v & 0xFu), sc, bi));
                dst[wi*8+1] = {T}(fma(float((v >> 4) & 0xFu), sc, bi));
                dst[wi*8+2] = {T}(fma(float((v >> 8) & 0xFu), sc, bi));
                dst[wi*8+3] = {T}(fma(float((v >> 12) & 0xFu), sc, bi));
                dst[wi*8+4] = {T}(fma(float((v >> 16) & 0xFu), sc, bi));
                dst[wi*8+5] = {T}(fma(float((v >> 20) & 0xFu), sc, bi));
                dst[wi*8+6] = {T}(fma(float((v >> 24) & 0xFu), sc, bi));
                dst[wi*8+7] = {T}(fma(float((v >> 28) & 0xFu), sc, bi));
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(tA, tB, cUp);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cGate.get_capacity(); ++i) {{
        if (cGate.is_valid_element(i)) {{
            auto idx = cGate.get_multidimensional_index(i);
            {store_code}
        }}
    }}
    """
    kern = _compile_cached(
        f"fswiglu_tgp_{M}_{N}_{K}_{BM}_{BN}_bk{BK}_{NSG}_s{split_k}",
        ["X", "WG", "SG", "BG", "WU", "SU", "BU"],
        output_names,
        source,
        header=METAL4_NAX_HEADER,
    )
    red_kern = _get_splitk_swiglu_reduce_kernel(split_k, M * N, T) if split_k > 1 else None
    grid = (N_TILES * threads, M_TILES, split_k)
    tg = (threads, 1, 1)
    red_threads = ((M * N) // 4) if ((M * N) % 4 == 0) else (M * N)
    red_tg = min(256, max(32, ((red_threads + 31) // 32) * 32))
    red_grid = (((red_threads + red_tg - 1) // red_tg) * red_tg, 1, 1)

    def runner(
        x: mx.array, wg: mx.array, sg: mx.array, bg: mx.array,
        wu: mx.array, su: mx.array, bu: mx.array, init_value: Optional[float] = None
    ) -> mx.array:
        kw = {"init_value": init_value} if init_value is not None else {}
        outs = kern(
            inputs=[x, wg, sg, bg, wu, su, bu],
            grid=grid,
            threadgroup=tg,
            output_shapes=out_shapes,
            output_dtypes=out_dtypes,
            **kw,
        )
        if red_kern is not None:
            return red_kern(
                inputs=[outs[0], outs[1]],
                grid=red_grid,
                threadgroup=(red_tg, 1, 1),
                output_shapes=[(M, N)],
                output_dtypes=[dtype],
                **kw,
            )[0]
        return outs[0]

    return runner, source


def build_fused_swiglu_b1_vec(
    M: int, N: int, K: int, genome: Dict[str, Any], dtype: mx.Dtype = mx.bfloat16
) -> Tuple[Callable, str]:
    """Family 6: B=1 Vectorized fused gate+up+SiLU QMV (1 dispatch, shares X in registers!)."""
    NR = int(genome["NR"])
    NSG = int(genome["NSG"])
    vec_bits = int(genome["vec_bits"])  # 64 or 128
    T = _dtype_str(dtype)
    K_G = K // 64
    K_W = K // 8
    wpl = vec_bits // 32
    vpl = wpl * 8
    step_k = vpl * 32
    assert K % step_k == 0
    rows_per_tg = NR * NSG
    assert N % rows_per_tg == 0
    N_TILES = N // rows_per_tg
    threads = NSG * 32

    if wpl == 4:
        load_pair = """
        uint4 pkg = *((const device uint4*)(WG + r_abs * K_W + (k_pos >> 3)));
        uint4 pku = *((const device uint4*)(WU + r_abs * K_W + (k_pos >> 3)));
        uint wg_w[4] = {pkg.x, pkg.y, pkg.z, pkg.w};
        uint wu_w[4] = {pku.x, pku.y, pku.z, pku.w};
        """
    else:
        load_pair = """
        uint2 pkg = *((const device uint2*)(WG + r_abs * K_W + (k_pos >> 3)));
        uint2 pku = *((const device uint2*)(WU + r_abs * K_W + (k_pos >> 3)));
        uint wg_w[2] = {pkg.x, pkg.y};
        uint wu_w[2] = {pku.x, pku.y};
        """

    source = f"""
    constexpr int NR = {NR}, NSG = {NSG}, N = {N}, K = {K}, K_G = {K_G}, K_W = {K_W};
    constexpr int STEP_K = {step_k}, VPL = {vpl}, WPL = {wpl};
    int sg = int(simdgroup_index_in_threadgroup);
    int lane = int(thread_index_in_simdgroup);
    int r0 = (int(threadgroup_position_in_grid.x) * NSG + sg) * NR;
    int m_id = int(threadgroup_position_in_grid.y);

    float ag[NR], au[NR];
    #pragma clang loop unroll(full)
    for (int r = 0; r < NR; ++r) {{ ag[r] = 0.0f; au[r] = 0.0f; }}

    for (int k = 0; k < K; k += STEP_K) {{
        int k_pos = k + lane * VPL;
        int g_idx = k_pos >> 6;
        const device {T}* xp = X + m_id * K + k_pos;
        float xsum = 0.0f;
        float xv[VPL];
        #pragma clang loop unroll(full)
        for (int i = 0; i < VPL; ++i) {{
            float a = float(xp[i]);
            xv[i] = a;
            xsum += a;
        }}
        #pragma clang loop unroll(full)
        for (int r = 0; r < NR; ++r) {{
            int r_abs = r0 + r;
            {load_pair}
            float dg = 0.0f, du = 0.0f;
            #pragma clang loop unroll(full)
            for (int wi = 0; wi < WPL; ++wi) {{
                uint vg = wg_w[wi];
                uint vu = wu_w[wi];
                dg += xv[wi * 8 + 0] * float(vg & 0xFu) + xv[wi * 8 + 1] * float((vg >> 4) & 0xFu)
                    + xv[wi * 8 + 2] * float((vg >> 8) & 0xFu) + xv[wi * 8 + 3] * float((vg >> 12) & 0xFu)
                    + xv[wi * 8 + 4] * float((vg >> 16) & 0xFu) + xv[wi * 8 + 5] * float((vg >> 20) & 0xFu)
                    + xv[wi * 8 + 6] * float((vg >> 24) & 0xFu) + xv[wi * 8 + 7] * float((vg >> 28) & 0xFu);
                du += xv[wi * 8 + 0] * float(vu & 0xFu) + xv[wi * 8 + 1] * float((vu >> 4) & 0xFu)
                    + xv[wi * 8 + 2] * float((vu >> 8) & 0xFu) + xv[wi * 8 + 3] * float((vu >> 12) & 0xFu)
                    + xv[wi * 8 + 4] * float((vu >> 16) & 0xFu) + xv[wi * 8 + 5] * float((vu >> 20) & 0xFu)
                    + xv[wi * 8 + 6] * float((vu >> 24) & 0xFu) + xv[wi * 8 + 7] * float((vu >> 28) & 0xFu);
            }}
            int s_off = r_abs * K_G + g_idx;
            ag[r] = fma(dg, float(SG[s_off]), fma(xsum, float(BG[s_off]), ag[r]));
            au[r] = fma(du, float(SU[s_off]), fma(xsum, float(BU[s_off]), au[r]));
        }}
    }}
    #pragma clang loop unroll(full)
    for (int r = 0; r < NR; ++r) {{
        float gv = simd_sum(ag[r]);
        float uv = simd_sum(au[r]);
        if (lane == 0) {{
            Y[m_id * N + r0 + r] = {T}((gv / (1.0f + metal::exp(-gv))) * uv);
        }}
    }}
    """
    kern = _compile_cached(
        f"fswiglu_b1_{M}_{N}_{K}_r{NR}_sg{NSG}_b{vec_bits}",
        ["X", "WG", "SG", "BG", "WU", "SU", "BU"],
        ["Y"],
        source,
    )
    grid = (N_TILES * threads, M, 1)
    tg = (threads, 1, 1)

    def runner(
        x: mx.array, wg: mx.array, sg: mx.array, bg: mx.array,
        wu: mx.array, su: mx.array, bu: mx.array, init_value: Optional[float] = None
    ) -> mx.array:
        kw = {"init_value": init_value} if init_value is not None else {}
        return kern(
            inputs=[x, wg, sg, bg, wu, su, bu],
            grid=grid,
            threadgroup=tg,
            output_shapes=[(M, N)],
            output_dtypes=[dtype],
            **kw,
        )[0]

    return runner, source


# ---------------------------------------------------------------------------
# ARENA 2: Qwen3.6-35B-A3B-4bit B=1 MoE Fused 8-Expert Kernels (3 Families)
# Exact shapes: E=256, TOPK=8, H=2048, D_MOE=512, gs=64, bits=4, BF16
# ---------------------------------------------------------------------------

def build_moe_fused_gate_up_swiglu(
    H: int, D_MOE: int, TOPK: int, genome: Dict[str, Any], dtype: mx.Dtype = mx.bfloat16
) -> Tuple[Callable, str]:
    """Family M1: Single-dispatch 8-expert fused gather_qmv(gate+up) + SwiGLU.

    Indexes packed [E, D_MOE, H//8] expert weights in place across all TOPK active experts
    in ONE dispatch, reading X[1, H] ONCE into threadgroup/registers!
    """
    NR = int(genome["NR"])
    NSG = int(genome["NSG"])
    vec_bits = int(genome["vec_bits"])  # 32, 64, or 128
    x_mode = str(genome["x_mode"])      # "tgp_f32", "tgp_bf16", "direct_l1"
    nibble_mode = str(genome.get("nibble_mode", "shift"))
    grid_order = str(genome.get("grid_order", "expert_y"))

    T = _dtype_str(dtype)
    K_G = H // 64
    K_W = H // 8
    wpl = vec_bits // 32
    vpl = wpl * 8
    step_k = vpl * 32
    assert H % step_k == 0
    rows_per_tg = NR * NSG
    assert D_MOE % rows_per_tg == 0
    N_TILES = D_MOE // rows_per_tg
    threads = NSG * 32

    if grid_order == "expert_y":
        idx_expr = "int tile_n = int(threadgroup_position_in_grid.x); int slot = int(threadgroup_position_in_grid.y);"
        grid = (N_TILES * threads, TOPK, 1)
    else:
        idx_expr = "int slot = int(threadgroup_position_in_grid.x); int tile_n = int(threadgroup_position_in_grid.y);"
        grid = (TOPK * threads, N_TILES, 1)

    if x_mode == "tgp_f32":
        tgp_decl = f"threadgroup float x_tg[{H}];"
        tgp_load = f"""
        for (int i = lid; i < {H}; i += {threads}) x_tg[i] = float(X[i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        """
        x_ptr_expr = "const threadgroup float* xp = x_tg + k_pos;"
        x_read_t = "xp[i]"
    elif x_mode == "tgp_bf16":
        tgp_decl = f"threadgroup {T} x_tg[{H}];"
        tgp_load = f"""
        for (int i = lid; i < {H}; i += {threads}) x_tg[i] = X[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        """
        x_ptr_expr = f"const threadgroup {T}* xp = x_tg + k_pos;"
        x_read_t = "float(xp[i])"
    else:
        tgp_decl = ""
        tgp_load = ""
        x_ptr_expr = f"const device {T}* xp = X + k_pos;"
        x_read_t = "float(xp[i])"

    if nibble_mode == "mlx_div":
        x_unpack = f"""
        float xsum = 0.0f;
        float xv[VPL];
        #pragma clang loop unroll(full)
        for (int i = 0; i < VPL; i += 4) {{
            float a0 = {x_read_t.replace('[i]', '[i + 0]')};
            float a1 = {x_read_t.replace('[i]', '[i + 1]')};
            float a2 = {x_read_t.replace('[i]', '[i + 2]')};
            float a3 = {x_read_t.replace('[i]', '[i + 3]')};
            xsum += (a0 + a1) + (a2 + a3);
            xv[i + 0] = a0;
            xv[i + 1] = a1 * (1.0f / 16.0f);
            xv[i + 2] = a2 * (1.0f / 256.0f);
            xv[i + 3] = a3 * (1.0f / 4096.0f);
        }}
        """
        dot_expr = """
        uint glo = vg & 0xFFFFu, ghi = vg >> 16;
        uint ulo = vu & 0xFFFFu, uhi = vu >> 16;
        dg += xv[wi*8+0]*float(glo&0x000Fu) + xv[wi*8+1]*float(glo&0x00F0u)
            + xv[wi*8+2]*float(glo&0x0F00u) + xv[wi*8+3]*float(glo&0xF000u)
            + xv[wi*8+4]*float(ghi&0x000Fu) + xv[wi*8+5]*float(ghi&0x00F0u)
            + xv[wi*8+6]*float(ghi&0x0F00u) + xv[wi*8+7]*float(ghi&0xF000u);
        du += xv[wi*8+0]*float(ulo&0x000Fu) + xv[wi*8+1]*float(ulo&0x00F0u)
            + xv[wi*8+2]*float(ulo&0x0F00u) + xv[wi*8+3]*float(ulo&0xF000u)
            + xv[wi*8+4]*float(uhi&0x000Fu) + xv[wi*8+5]*float(uhi&0x00F0u)
            + xv[wi*8+6]*float(uhi&0x0F00u) + xv[wi*8+7]*float(uhi&0xF000u);
        """
    else:
        x_unpack = f"""
        float xsum = 0.0f;
        float xv[VPL];
        #pragma clang loop unroll(full)
        for (int i = 0; i < VPL; ++i) {{
            float a = {x_read_t};
            xv[i] = a;
            xsum += a;
        }}
        """
        dot_expr = """
        dg += xv[wi*8+0]*float(vg&0xFu) + xv[wi*8+1]*float((vg>>4)&0xFu)
            + xv[wi*8+2]*float((vg>>8)&0xFu) + xv[wi*8+3]*float((vg>>12)&0xFu)
            + xv[wi*8+4]*float((vg>>16)&0xFu) + xv[wi*8+5]*float((vg>>20)&0xFu)
            + xv[wi*8+6]*float((vg>>24)&0xFu) + xv[wi*8+7]*float((vg>>28)&0xFu);
        du += xv[wi*8+0]*float(vu&0xFu) + xv[wi*8+1]*float((vu>>4)&0xFu)
            + xv[wi*8+2]*float((vu>>8)&0xFu) + xv[wi*8+3]*float((vu>>12)&0xFu)
            + xv[wi*8+4]*float((vu>>16)&0xFu) + xv[wi*8+5]*float((vu>>20)&0xFu)
            + xv[wi*8+6]*float((vu>>24)&0xFu) + xv[wi*8+7]*float((vu>>28)&0xFu);
        """

    if wpl == 4:
        w_load = """
        uint4 pkg = *((const device uint4*)(wg_e + r_abs * K_W + (k_pos >> 3)));
        uint4 pku = *((const device uint4*)(wu_e + r_abs * K_W + (k_pos >> 3)));
        uint wg_w[4] = {pkg.x, pkg.y, pkg.z, pkg.w};
        uint wu_w[4] = {pku.x, pku.y, pku.z, pku.w};
        """
    elif wpl == 2:
        w_load = """
        uint2 pkg = *((const device uint2*)(wg_e + r_abs * K_W + (k_pos >> 3)));
        uint2 pku = *((const device uint2*)(wu_e + r_abs * K_W + (k_pos >> 3)));
        uint wg_w[2] = {pkg.x, pkg.y};
        uint wu_w[2] = {pku.x, pku.y};
        """
    else:
        w_load = """
        uint wg_w[1] = {wg_e[r_abs * K_W + (k_pos >> 3)]};
        uint wu_w[1] = {wu_e[r_abs * K_W + (k_pos >> 3)]};
        """

    source = f"""
    constexpr int NR = {NR}, NSG = {NSG}, H = {H}, D_MOE = {D_MOE};
    constexpr int K_G = {K_G}, K_W = {K_W}, STEP_K = {step_k}, VPL = {vpl}, WPL = {wpl};
    constexpr size_t E_STRIDE_W = size_t(D_MOE) * size_t(K_W);
    constexpr size_t E_STRIDE_G = size_t(D_MOE) * size_t(K_G);

    {idx_expr}
    int lid = int(thread_index_in_threadgroup);
    int sg = int(simdgroup_index_in_threadgroup);
    int lane = int(thread_index_in_simdgroup);
    uint eid = IND[slot];
    int r0 = (tile_n * NSG + sg) * NR;

    {tgp_decl}
    {tgp_load}

    const device uint* wg_e = WG + size_t(eid) * E_STRIDE_W;
    const device uint* wu_e = WU + size_t(eid) * E_STRIDE_W;
    const device {T}* sg_e = SG + size_t(eid) * E_STRIDE_G;
    const device {T}* bg_e = BG + size_t(eid) * E_STRIDE_G;
    const device {T}* su_e = SU + size_t(eid) * E_STRIDE_G;
    const device {T}* bu_e = BU + size_t(eid) * E_STRIDE_G;

    float ag[NR], au[NR];
    #pragma clang loop unroll(full)
    for (int r = 0; r < NR; ++r) {{ ag[r] = 0.0f; au[r] = 0.0f; }}

    for (int k = 0; k < H; k += STEP_K) {{
        int k_pos = k + lane * VPL;
        int g_idx = k_pos >> 6;
        {x_ptr_expr}
        {x_unpack}
        #pragma clang loop unroll(full)
        for (int r = 0; r < NR; ++r) {{
            int r_abs = r0 + r;
            {w_load}
            float dg = 0.0f, du = 0.0f;
            #pragma clang loop unroll(full)
            for (int wi = 0; wi < WPL; ++wi) {{
                uint vg = wg_w[wi];
                uint vu = wu_w[wi];
                {dot_expr}
            }}
            int s_off = r_abs * K_G + g_idx;
            ag[r] = fma(dg, float(sg_e[s_off]), fma(xsum, float(bg_e[s_off]), ag[r]));
            au[r] = fma(du, float(su_e[s_off]), fma(xsum, float(bu_e[s_off]), au[r]));
        }}
    }}
    #pragma clang loop unroll(full)
    for (int r = 0; r < NR; ++r) {{
        float gv = simd_sum(ag[r]);
        float uv = simd_sum(au[r]);
        if (lane == 0) {{
            OUT[slot * D_MOE + r0 + r] = {T}((gv / (1.0f + metal::exp(-gv))) * uv);
        }}
    }}
    """
    kern = _compile_cached(
        f"moe_gu_{H}_{D_MOE}_r{NR}_sg{NSG}_b{vec_bits}_{x_mode}_{nibble_mode}_{grid_order}",
        ["X", "IND", "WG", "SG", "BG", "WU", "SU", "BU"],
        ["OUT"],
        source,
    )
    tg = (threads, 1, 1)

    def runner(
        x: mx.array, indices: mx.array,
        wg: mx.array, sg: mx.array, bg: mx.array,
        wu: mx.array, su: mx.array, bu: mx.array,
        init_value: Optional[float] = None,
    ) -> mx.array:
        kw = {"init_value": init_value} if init_value is not None else {}
        return kern(
            inputs=[x, indices, wg, sg, bg, wu, su, bu],
            grid=grid,
            threadgroup=tg,
            output_shapes=[(TOPK, D_MOE)],
            output_dtypes=[dtype],
            **kw,
        )[0]

    return runner, source


def build_moe_fused_down_reduce(
    H: int, D_MOE: int, TOPK: int, genome: Dict[str, Any], dtype: mx.Dtype = mx.bfloat16
) -> Tuple[Callable, str]:
    """Family M2: Single-dispatch 8-expert fused gather_qmv(down_proj) + routing weighted sum.

    Supports both:
      - vec_bits=64 (ZERO K-loop: 32 lanes x 16 nibbles = 512 in ONE uint2 load/lane!), and
      - vec_bits=32 (128-byte exact cache-line coalesced uint32 loads, only 8 regs/thread, 2 K-steps!).
    Supports split_e in {1, 2, 4, 8} (split_e=8 gives 100% contiguous 1-expert-per-TG DRAM streaming!).
    """
    NR = int(genome["NR"])
    NSG = int(genome["NSG"])
    h_mode = str(genome["h_mode"])         # "tgp_scaled_f32", "tgp_scaled_f16", "tgp_bf16", "direct_l1"
    nibble_mode = str(genome.get("nibble_mode", "shift"))
    split_e = int(genome.get("split_e", 1))  # 1, 2, 4, or 8
    vec_bits = int(genome.get("vec_bits", 64))  # 32 or 64

    T = _dtype_str(dtype)
    assert D_MOE == 512, "MoE down kernel specialized for D_MOE=512"
    assert TOPK % split_e == 0
    E_PER_S = TOPK // split_e
    K_G = D_MOE // 64   # 8
    K_W = D_MOE // 8    # 64 uint32 words per row
    rows_per_tg = NR * NSG
    assert H % rows_per_tg == 0
    N_TILES = H // rows_per_tg
    threads = NSG * 32
    wpl = vec_bits // 32  # 1 or 2
    vpl = wpl * 8         # 8 or 16
    step_k = vpl * 32     # 256 or 512

    if h_mode == "tgp_scaled_f32":
        tgp_decl = f"threadgroup float h_tg[{E_PER_S * D_MOE}];"
        tgp_load = f"""
        for (int idx = lid; idx < {E_PER_S * D_MOE}; idx += {threads}) {{
            int ee = idx / {D_MOE};
            int kk = idx % {D_MOE};
            int slot = e_start + ee;
            h_tg[idx] = float(HM[slot * {D_MOE} + kk]) * float(SCORES[slot]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        """
        h_ptr_expr = f"const threadgroup float* hp = h_tg + ee * 512 + k_pos;"
        h_elem = "hp[i]"
    elif h_mode == "tgp_scaled_f16":
        tgp_decl = f"threadgroup half h_tg[{E_PER_S * D_MOE}];"
        tgp_load = f"""
        for (int idx = lid; idx < {E_PER_S * D_MOE}; idx += {threads}) {{
            int ee = idx / {D_MOE};
            int kk = idx % {D_MOE};
            int slot = e_start + ee;
            h_tg[idx] = half(float(HM[slot * {D_MOE} + kk]) * float(SCORES[slot]));
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        """
        h_ptr_expr = "const threadgroup half* hp = h_tg + ee * 512 + k_pos;"
        h_elem = "float(hp[i])"
    elif h_mode == "tgp_bf16":
        tgp_decl = f"threadgroup {T} h_tg[{E_PER_S * D_MOE}];"
        tgp_load = f"""
        for (int idx = lid; idx < {E_PER_S * D_MOE}; idx += {threads}) {{
            int ee = idx / {D_MOE};
            int kk = idx % {D_MOE};
            int slot = e_start + ee;
            h_tg[idx] = HM[slot * {D_MOE} + kk];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        """
        h_ptr_expr = f"const threadgroup {T}* hp = h_tg + ee * 512 + k_pos; float sc_e = float(SCORES[slot]);"
        h_elem = "float(hp[i]) * sc_e"
    else:
        tgp_decl = ""
        tgp_load = ""
        h_ptr_expr = f"const device {T}* hp = HM + slot * 512 + k_pos; float sc_e = float(SCORES[slot]);"
        h_elem = "float(hp[i]) * sc_e"

    if nibble_mode == "mlx_div":
        h_unpack = f"""
        float hsum = 0.0f;
        float hv[{vpl}];
        #pragma clang loop unroll(full)
        for (int i = 0; i < {vpl}; i += 4) {{
            float a0 = {h_elem.replace('[i]', '[i + 0]')};
            float a1 = {h_elem.replace('[i]', '[i + 1]')};
            float a2 = {h_elem.replace('[i]', '[i + 2]')};
            float a3 = {h_elem.replace('[i]', '[i + 3]')};
            hsum += (a0 + a1) + (a2 + a3);
            hv[i + 0] = a0;
            hv[i + 1] = a1 * (1.0f / 16.0f);
            hv[i + 2] = a2 * (1.0f / 256.0f);
            hv[i + 3] = a3 * (1.0f / 4096.0f);
        }}
        """
        dot_word = """
        uint lo = ww & 0xFFFFu, hi = ww >> 16;
        d += hv[wi*8+0]*float(lo&0x000Fu) + hv[wi*8+1]*float(lo&0x00F0u)
           + hv[wi*8+2]*float(lo&0x0F00u) + hv[wi*8+3]*float(lo&0xF000u)
           + hv[wi*8+4]*float(hi&0x000Fu) + hv[wi*8+5]*float(hi&0x00F0u)
           + hv[wi*8+6]*float(hi&0x0F00u) + hv[wi*8+7]*float(hi&0xF000u);
        """
    else:
        h_unpack = f"""
        float hsum = 0.0f;
        float hv[{vpl}];
        #pragma clang loop unroll(full)
        for (int i = 0; i < {vpl}; ++i) {{
            float a = {h_elem};
            hv[i] = a;
            hsum += a;
        }}
        """
        dot_word = """
        d += hv[wi*8+0]*float(ww&0xFu) + hv[wi*8+1]*float((ww>>4)&0xFu)
           + hv[wi*8+2]*float((ww>>8)&0xFu) + hv[wi*8+3]*float((ww>>12)&0xFu)
           + hv[wi*8+4]*float((ww>>16)&0xFu) + hv[wi*8+5]*float((ww>>20)&0xFu)
           + hv[wi*8+6]*float((ww>>24)&0xFu) + hv[wi*8+7]*float((ww>>28)&0xFu);
        """

    if wpl == 2:
        w_load = "uint2 pk = *((const device uint2*)(wd_e + r_abs * K_W + (k_pos >> 3))); uint words[2] = {pk.x, pk.y};"
    else:
        w_load = "uint words[1] = {wd_e[r_abs * K_W + (k_pos >> 3)]};"

    if split_e == 1:
        out_dtype = dtype
        out_shape = (1, H)
        store_code = f"if (lane == 0) Y[r0 + r] = {T}(res);"
    else:
        out_dtype = mx.float32
        out_shape = (split_e, 1, H)
        store_code = "if (lane == 0) Y[s_id * H + r0 + r] = res;"

    source = f"""
    constexpr int NR = {NR}, NSG = {NSG}, H = {H}, D_MOE = {D_MOE};
    constexpr int K_G = {K_G}, K_W = {K_W}, E_PER_S = {E_PER_S};
    constexpr int STEP_K = {step_k}, VPL = {vpl}, WPL = {wpl};
    constexpr size_t E_STRIDE_W = size_t(H) * size_t(K_W);
    constexpr size_t E_STRIDE_G = size_t(H) * size_t(K_G);

    int lid = int(thread_index_in_threadgroup);
    int sg = int(simdgroup_index_in_threadgroup);
    int lane = int(thread_index_in_simdgroup);
    int r0 = (int(threadgroup_position_in_grid.x) * NSG + sg) * NR;
    int s_id = int(threadgroup_position_in_grid.y);
    int e_start = s_id * E_PER_S;

    {tgp_decl}
    {tgp_load}

    float acc[NR];
    #pragma clang loop unroll(full)
    for (int r = 0; r < NR; ++r) acc[r] = 0.0f;

    #pragma clang loop unroll(full)
    for (int ee = 0; ee < E_PER_S; ++ee) {{
        int slot = e_start + ee;
        uint eid = IND[slot];
        const device uint* wd_e = WD + size_t(eid) * E_STRIDE_W;
        const device {T}* sd_e = SD + size_t(eid) * E_STRIDE_G;
        const device {T}* bd_e = BD + size_t(eid) * E_STRIDE_G;

        #pragma clang loop unroll(full)
        for (int k = 0; k < D_MOE; k += STEP_K) {{
            int k_pos = k + lane * VPL;
            int g_idx = k_pos >> 6;
            {h_ptr_expr}
            {h_unpack}

            #pragma clang loop unroll(full)
            for (int r = 0; r < NR; ++r) {{
                int r_abs = r0 + r;
                {w_load}
                float d = 0.0f;
                #pragma clang loop unroll(full)
                for (int wi = 0; wi < WPL; ++wi) {{
                    uint ww = words[wi];
                    {dot_word}
                }}
                int s_off = r_abs * K_G + g_idx;
                acc[r] = fma(d, float(sd_e[s_off]), fma(hsum, float(bd_e[s_off]), acc[r]));
            }}
        }}
    }}

    #pragma clang loop unroll(full)
    for (int r = 0; r < NR; ++r) {{
        float res = simd_sum(acc[r]);
        {store_code}
    }}
    """
    kern = _compile_cached(
        f"moe_dn_{H}_{D_MOE}_r{NR}_sg{NSG}_b{vec_bits}_{h_mode}_{nibble_mode}_se{split_e}",
        ["HM", "IND", "SCORES", "WD", "SD", "BD"],
        ["Y"],
        source,
    )
    red_kern = _get_splitk_reduce_kernel(split_e, H, T) if split_e > 1 else None
    grid = (N_TILES * threads, split_e, 1)
    tg = (threads, 1, 1)
    red_threads = H // 4
    red_tg = min(256, max(32, ((red_threads + 31) // 32) * 32))
    red_grid = (((red_threads + red_tg - 1) // red_tg) * red_tg, 1, 1)

    def runner(
        hm: mx.array, indices: mx.array, scores: mx.array,
        wd: mx.array, sd: mx.array, bd: mx.array,
        init_value: Optional[float] = None,
    ) -> mx.array:
        kw = {"init_value": init_value} if init_value is not None else {}
        out = kern(
            inputs=[hm, indices, scores, wd, sd, bd],
            grid=grid,
            threadgroup=tg,
            output_shapes=[out_shape],
            output_dtypes=[out_dtype],
            **kw,
        )[0]
        if red_kern is not None:
            out = red_kern(
                inputs=[out],
                grid=red_grid,
                threadgroup=(red_tg, 1, 1),
                output_shapes=[(1, H)],
                output_dtypes=[dtype],
                **kw,
            )[0]
        return out

    return runner, source


def build_moe_mega_1dispatch(
    H: int, D_MOE: int, TOPK: int, genome: Dict[str, Any], dtype: mx.Dtype = mx.bfloat16
) -> Tuple[Callable, str]:
    """Family M3: True SINGLE-DISPATCH end-to-end 8-expert SwitchGLU + weighted_sum mega-kernel!

    Stage 1 (inside threadgroup): compute h_tg[512] = silu(x @ W_g[e]^T) * (x @ W_u[e]^T) * scores[e]
    Stage 2 (same threadgroup, ZERO DRAM round-trip for h!): compute y_e = h_tg @ W_d[e]^T -> partial[e, 2048]
    (Uses deterministic partial[TOPK, H] + tiny vectorized reduce OR single-TG per expert -> supports init_value poison!)
    """
    NSG = int(genome["NSG"])  # 8 or 16 simdgroups per expert threadgroup!
    T = _dtype_str(dtype)
    assert H == 2048 and D_MOE == 512
    threads = NSG * 32
    # Each simdgroup handles D_MOE / NSG rows for gate+up (e.g. 512/16=32 rows), and H / NSG rows for down (2048/16=128 rows)
    rows_gu = D_MOE // NSG
    rows_dn = H // NSG
    assert D_MOE % NSG == 0 and H % NSG == 0

    source = f"""
    constexpr int NSG = {NSG}, H = 2048, D_MOE = 512;
    constexpr int K_G_GU = 32, K_W_GU = 256;
    constexpr int K_G_DN = 8,  K_W_DN = 64;
    constexpr size_t STRIDE_W_GU = size_t(D_MOE) * size_t(K_W_GU);
    constexpr size_t STRIDE_G_GU = size_t(D_MOE) * size_t(K_G_GU);
    constexpr size_t STRIDE_W_DN = size_t(H) * size_t(K_W_DN);
    constexpr size_t STRIDE_G_DN = size_t(H) * size_t(K_G_DN);

    int slot = int(threadgroup_position_in_grid.x);
    int lid = int(thread_index_in_threadgroup);
    int sg = int(simdgroup_index_in_threadgroup);
    int lane = int(thread_index_in_simdgroup);
    uint eid = IND[slot];
    float sc_e = float(SCORES[slot]);

    threadgroup float x_tg[H];
    threadgroup float h_tg[D_MOE];
    for (int i = lid; i < H; i += {threads}) x_tg[i] = float(X[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: Fused gate+up+SiLU -> h_tg[512] (pre-scaled by sc_e!)
    const device uint* wg_e = WG + size_t(eid) * STRIDE_W_GU;
    const device uint* wu_e = WU + size_t(eid) * STRIDE_W_GU;
    const device {T}* sg_e = SG + size_t(eid) * STRIDE_G_GU;
    const device {T}* bg_e = BG + size_t(eid) * STRIDE_G_GU;
    const device {T}* su_e = SU + size_t(eid) * STRIDE_G_GU;
    const device {T}* bu_e = BU + size_t(eid) * STRIDE_G_GU;

    int gu_r0 = sg * {rows_gu};
    for (int r = 0; r < {rows_gu}; ++r) {{
        int r_abs = gu_r0 + r;
        float ag = 0.0f, au = 0.0f;
        // 32 lanes * 16 elems/lane = 512 elems/iter -> 4 iters for H=2048
        #pragma clang loop unroll(full)
        for (int k = 0; k < H; k += 512) {{
            int k_pos = k + lane * 16;
            int g_idx = k_pos >> 6;
            const threadgroup float* xp = x_tg + k_pos;
            float xsum = 0.0f;
            float xv[16];
            #pragma clang loop unroll(full)
            for (int i = 0; i < 16; ++i) {{ float a = xp[i]; xv[i] = a; xsum += a; }}
            uint2 pkg = *((const device uint2*)(wg_e + r_abs * K_W_GU + (k_pos >> 3)));
            uint2 pku = *((const device uint2*)(wu_e + r_abs * K_W_GU + (k_pos >> 3)));
            uint g0 = pkg.x, g1 = pkg.y, u0 = pku.x, u1 = pku.y;
            float dg = xv[0]*float(g0&0xFu)+xv[1]*float((g0>>4)&0xFu)+xv[2]*float((g0>>8)&0xFu)+xv[3]*float((g0>>12)&0xFu)
                     + xv[4]*float((g0>>16)&0xFu)+xv[5]*float((g0>>20)&0xFu)+xv[6]*float((g0>>24)&0xFu)+xv[7]*float((g0>>28)&0xFu)
                     + xv[8]*float(g1&0xFu)+xv[9]*float((g1>>4)&0xFu)+xv[10]*float((g1>>8)&0xFu)+xv[11]*float((g1>>12)&0xFu)
                     + xv[12]*float((g1>>16)&0xFu)+xv[13]*float((g1>>20)&0xFu)+xv[14]*float((g1>>24)&0xFu)+xv[15]*float((g1>>28)&0xFu);
            float du = xv[0]*float(u0&0xFu)+xv[1]*float((u0>>4)&0xFu)+xv[2]*float((u0>>8)&0xFu)+xv[3]*float((u0>>12)&0xFu)
                     + xv[4]*float((u0>>16)&0xFu)+xv[5]*float((u0>>20)&0xFu)+xv[6]*float((u0>>24)&0xFu)+xv[7]*float((u0>>28)&0xFu)
                     + xv[8]*float(u1&0xFu)+xv[9]*float((u1>>4)&0xFu)+xv[10]*float((u1>>8)&0xFu)+xv[11]*float((u1>>12)&0xFu)
                     + xv[12]*float((u1>>16)&0xFu)+xv[13]*float((u1>>20)&0xFu)+xv[14]*float((u1>>24)&0xFu)+xv[15]*float((u1>>28)&0xFu);
            int s_off = r_abs * K_G_GU + g_idx;
            ag = fma(dg, float(sg_e[s_off]), fma(xsum, float(bg_e[s_off]), ag));
            au = fma(du, float(su_e[s_off]), fma(xsum, float(bu_e[s_off]), au));
        }}
        float gv = simd_sum(ag);
        float uv = simd_sum(au);
        if (lane == 0) {{
            h_tg[r_abs] = ((gv / (1.0f + metal::exp(-gv))) * uv) * sc_e;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Zero-K-loop down_proj from h_tg[512] -> PART[slot, H]
    const device uint* wd_e = WD + size_t(eid) * STRIDE_W_DN;
    const device {T}* sd_e = SD + size_t(eid) * STRIDE_G_DN;
    const device {T}* bd_e = BD + size_t(eid) * STRIDE_G_DN;
    int dn_g = lane >> 2;
    const threadgroup float* hp = h_tg + lane * 16;
    float hsum = 0.0f;
    float hv[16];
    #pragma clang loop unroll(full)
    for (int i = 0; i < 16; ++i) {{ float a = hp[i]; hv[i] = a; hsum += a; }}

    int dn_r0 = sg * {rows_dn};
    for (int r = 0; r < {rows_dn}; ++r) {{
        int r_abs = dn_r0 + r;
        uint2 pk = *((const device uint2*)(wd_e + r_abs * K_W_DN + lane * 2));
        uint w0 = pk.x, w1 = pk.y;
        float d = hv[0]*float(w0&0xFu)+hv[1]*float((w0>>4)&0xFu)+hv[2]*float((w0>>8)&0xFu)+hv[3]*float((w0>>12)&0xFu)
                + hv[4]*float((w0>>16)&0xFu)+hv[5]*float((w0>>20)&0xFu)+hv[6]*float((w0>>24)&0xFu)+hv[7]*float((w0>>28)&0xFu)
                + hv[8]*float(w1&0xFu)+hv[9]*float((w1>>4)&0xFu)+hv[10]*float((w1>>8)&0xFu)+hv[11]*float((w1>>12)&0xFu)
                + hv[12]*float((w1>>16)&0xFu)+hv[13]*float((w1>>20)&0xFu)+hv[14]*float((w1>>24)&0xFu)+hv[15]*float((w1>>28)&0xFu);
        int s_off = r_abs * K_G_DN + dn_g;
        float acc = fma(d, float(sd_e[s_off]), hsum * float(bd_e[s_off]));
        float res = simd_sum(acc);
        if (lane == 0) {{
            PART[slot * H + r_abs] = res;
        }}
    }}
    """
    kern = _compile_cached(
        f"moe_mega_{H}_{D_MOE}_sg{NSG}",
        ["X", "IND", "SCORES", "WG", "SG", "BG", "WU", "SU", "BU", "WD", "SD", "BD"],
        ["PART"],
        source,
    )
    red_kern = _get_splitk_reduce_kernel(TOPK, H, T)
    grid = (TOPK * threads, 1, 1)
    tg = (threads, 1, 1)
    red_threads = H // 4
    red_tg = min(256, max(32, ((red_threads + 31) // 32) * 32))
    red_grid = (((red_threads + red_tg - 1) // red_tg) * red_tg, 1, 1)

    def runner(
        x: mx.array, indices: mx.array, scores: mx.array,
        wg: mx.array, sg: mx.array, bg: mx.array,
        wu: mx.array, su: mx.array, bu: mx.array,
        wd: mx.array, sd: mx.array, bd: mx.array,
        init_value: Optional[float] = None,
    ) -> mx.array:
        kw = {"init_value": init_value} if init_value is not None else {}
        part = kern(
            inputs=[x, indices, scores, wg, sg, bg, wu, su, bu, wd, sd, bd],
            grid=grid,
            threadgroup=tg,
            output_shapes=[(TOPK, H)],
            output_dtypes=[mx.float32],
            **kw,
        )[0]
        return red_kern(
            inputs=[part],
            grid=red_grid,
            threadgroup=(red_tg, 1, 1),
            output_shapes=[(1, H)],
            output_dtypes=[dtype],
            **kw,
        )[0]

    return runner, source


# ---------------------------------------------------------------------------
# Reference Functions for Fused SwiGLU & MoE Verification
# ---------------------------------------------------------------------------

def ref_swiglu_fp32(
    x: mx.array, wg: mx.array, sg: mx.array, bg: mx.array,
    wu: mx.array, su: mx.array, bu: mx.array,
) -> mx.array:
    xf = x.astype(mx.float32)
    g = xf @ _deq_fp32(wg, sg, bg, 64).T
    u = xf @ _deq_fp32(wu, su, bu, 64).T
    return nn.silu(g) * u


def stock_swiglu_mlx(
    x: mx.array, wg: mx.array, sg: mx.array, bg: mx.array,
    wu: mx.array, su: mx.array, bu: mx.array,
) -> mx.array:
    g = mx.quantized_matmul(x, wg, sg, bg, transpose=True, group_size=64, bits=4, mode="affine")
    u = mx.quantized_matmul(x, wu, su, bu, transpose=True, group_size=64, bits=4, mode="affine")
    return nn.silu(g) * u


def ref_moe_gate_up_fp32(
    x: mx.array, indices: mx.array,
    wg: mx.array, sg: mx.array, bg: mx.array,
    wu: mx.array, su: mx.array, bu: mx.array,
) -> mx.array:
    xf = x.astype(mx.float32).reshape(1, -1)
    idx_list = indices.tolist()
    outs = []
    for e in idx_list:
        g = xf @ _deq_fp32(wg[e], sg[e], bg[e], 64).T
        u = xf @ _deq_fp32(wu[e], su[e], bu[e], 64).T
        outs.append((nn.silu(g) * u).reshape(-1))
    return mx.stack(outs, axis=0)


def ref_moe_down_fp32(
    hm: mx.array, indices: mx.array, scores: mx.array,
    wd: mx.array, sd: mx.array, bd: mx.array,
) -> mx.array:
    idx_list = indices.tolist()
    sc_list = scores.astype(mx.float32).tolist()
    acc = mx.zeros((1, wd.shape[1]), dtype=mx.float32)
    for slot, e in enumerate(idx_list):
        h_e = hm[slot : slot + 1].astype(mx.float32)
        y_e = h_e @ _deq_fp32(wd[e], sd[e], bd[e], 64).T
        acc = acc + y_e * sc_list[slot]
    return acc


def ref_moe_full_fp32(
    x: mx.array, indices: mx.array, scores: mx.array,
    wg: mx.array, sg: mx.array, bg: mx.array,
    wu: mx.array, su: mx.array, bu: mx.array,
    wd: mx.array, sd: mx.array, bd: mx.array,
) -> mx.array:
    hm = ref_moe_gate_up_fp32(x, indices, wg, sg, bg, wu, su, bu)
    return ref_moe_down_fp32(hm, indices, scores, wd, sd, bd)


def stock_moe_full_mlx(
    x: mx.array, indices: mx.array, scores: mx.array,
    wg: mx.array, sg: mx.array, bg: mx.array,
    wu: mx.array, su: mx.array, bu: mx.array,
    wd: mx.array, sd: mx.array, bd: mx.array,
) -> mx.array:
    """Exact stock mlx_lm Qwen3NextSparseMoeBlock switch_mlp + weighted sum path."""
    x_exp = mx.expand_dims(x, (-2, -3))       # (1, 1, 1, H)
    idx_2d = indices.reshape(1, -1)           # (1, 8)
    x_up = mx.gather_qmm(x_exp, wu, su, bu, rhs_indices=idx_2d, transpose=True, group_size=64, bits=4, mode="affine")
    x_gate = mx.gather_qmm(x_exp, wg, sg, bg, rhs_indices=idx_2d, transpose=True, group_size=64, bits=4, mode="affine")
    h = nn.silu(x_gate) * x_up
    y = mx.gather_qmm(h, wd, sd, bd, rhs_indices=idx_2d, transpose=True, group_size=64, bits=4, mode="affine")
    y = y.squeeze(-2)                         # (1, 8, H)
    return (y * scores.reshape(1, -1, 1)).sum(axis=-2)


def stock_moe_gate_up_mlx(
    x: mx.array, indices: mx.array,
    wg: mx.array, sg: mx.array, bg: mx.array,
    wu: mx.array, su: mx.array, bu: mx.array,
) -> mx.array:
    x_exp = mx.expand_dims(x, (-2, -3))
    idx_2d = indices.reshape(1, -1)
    x_up = mx.gather_qmm(x_exp, wu, su, bu, rhs_indices=idx_2d, transpose=True, group_size=64, bits=4, mode="affine")
    x_gate = mx.gather_qmm(x_exp, wg, sg, bg, rhs_indices=idx_2d, transpose=True, group_size=64, bits=4, mode="affine")
    return (nn.silu(x_gate) * x_up).reshape(-1, wg.shape[1])


# ---------------------------------------------------------------------------
# Structured + Evolutionary Candidate Generator (60+ Distinct Schedules!)
# ---------------------------------------------------------------------------

def generate_dense_qmm_candidates(
    M: int, N: int, K: int, dtype: mx.Dtype = mx.bfloat16
) -> List[KernelCandidate]:
    """Generate shape-specialized ugly Metal 4 candidates for a given (M, N, K)."""
    shape_tag = f"M{M}_N{N}_K{K}"
    cands: List[KernelCandidate] = []
    K_G = K // 64

    def _add(family: str, genome: Dict[str, Any], builder: Callable) -> None:
        cid = f"{family}_{shape_tag}_{len(cands):02d}"
        c = KernelCandidate(cid=cid, arena="dense_qmm", family=family, shape_tag=shape_tag, genome=genome)
        try:
            fn, src = builder(M, N, K, genome, dtype)
            c.fn = fn
            c.source_lines = len(src.splitlines())
            c.compiled = True
        except Exception as e:
            c.compile_error = f"{type(e).__name__}: {e}"
        cands.append(c)

    if M <= 4:
        # Vectorized uint4/uint2/uint B=1..4 QMV schedules
        for nr, nsg, vb, sk, nm in [
            (2, 4, 128, 1, "shift"),
            (4, 4, 128, 1, "shift"),
            (4, 2, 128, 1, "mlx_div"),
            (8, 2, 128, 1, "shift"),
            (4, 4, 64,  1, "shift"),
            (4, 4, 64,  1, "mlx_div"),
            (2, 8, 64,  2, "shift"),
            (4, 4, 128, 2, "shift"),
            (2, 4, 128, 4, "shift"),
        ]:
            step_k = (vb // 32) * 8 * 32
            if K % (sk * step_k) == 0 and N % (nr * nsg) == 0:
                _add("vec_qmv_b1", {"NR": nr, "NSG": nsg, "vec_bits": vb, "split_k": sk, "nibble_mode": nm}, build_vec_qmv_b1)
        if M == 4:
            # Also race BM=8 dynamic-slice direct hardware uint4b_format NAX at M=4!
            for bn, nsg, sk, csb in [(32, 2, 4, False), (64, 4, 4, True), (64, 4, 8, False), (32, 2, 8, True)]:
                if K_G % sk == 0 and N % bn == 0:
                    _add("u4_nax_coop", {"BM": 8, "BN": bn, "NSG": nsg, "split_k": sk, "k_unroll": 2, "cache_sb": csb, "swizzle": 0, "stage_x": False, "relaxed": False}, build_u4_nax_coop)
        return cands

    # M >= 8: Full Metal 4 NAX zoo!
    bm_opts = [b for b in (8, 16, 32, 64) if b <= M]
    bm_primary = max(bm_opts)
    bm_alt = bm_opts[-2] if len(bm_opts) >= 2 else bm_primary

    # Family 1: Direct hardware uint4b_format cooperative NAX (diverse split_k, BN, NSG, unroll, swizzle, cache_sb)
    coop_specs = [
        (bm_primary, 64, 4, 1, 2, True,  0, False, False),
        (bm_primary, 64, 4, 2, 2, True,  0, False, False),
        (bm_primary, 64, 4, 4, 2, True,  0, False, False),
        (bm_primary, 64, 4, 8, 2, True,  0, False, False),
        (bm_primary, 64, 4, 4, 4, True,  0, False, False),
        (bm_primary, 64, 4, 4, 2, False, 0, False, False),
        (bm_primary, 64, 4, 4, 2, True,  2, False, False),
        (bm_primary, 64, 4, 4, 2, True,  0, True,  False),
        (bm_primary, 64, 4, 8, 2, False, 0, False, True),   # relaxed precision test!
        (bm_primary, 32, 2, 4, 2, True,  0, False, False),
        (bm_primary, 32, 4, 4, 2, True,  0, False, False),
        (bm_primary, 32, 2, 8, 2, False, 0, False, False),
        (bm_primary, 128, 4, 2, 2, True, 0, False, False),
        (bm_primary, 128, 4, 4, 2, True, 0, False, False),
        (bm_primary, 128, 4, 8, 2, True, 0, False, False),
        (bm_primary, 128, 4, 4, 4, True, 0, False, False),
        (bm_primary, 128, 8, 4, 2, True, 0, False, False),
        (bm_alt,     64, 4, 4, 2, True,  1, False, False),
        (bm_alt,     64, 4, 8, 2, True,  0, False, False),
        (bm_primary, 64, 4, 16, 2, True, 0, False, False),
    ]
    seen_coop = set()
    for spec in coop_specs:
        bm, bn, nsg, sk, unr, csb, sw, stx, rel = spec
        if spec in seen_coop:
            continue
        seen_coop.add(spec)
        if K_G % sk != 0 or (K_G // sk) % unr != 0 or N % bn != 0:
            continue
        _add("u4_nax_coop", {
            "BM": bm, "BN": bn, "NSG": nsg, "split_k": sk, "k_unroll": unr,
            "cache_sb": csb, "swizzle": sw, "stage_x": stx, "relaxed": rel,
        }, build_u4_nax_coop)

    # Family 2: Zero-barrier independent per-simdgroup direct uint4b_format NAX
    sg_specs = [
        (min(16, bm_primary), 32, 1, 2, 4, 2, 0),
        (min(16, bm_primary), 32, 1, 4, 4, 2, 0),
        (min(16, bm_primary), 32, 1, 2, 8, 2, 0),
        (min(16, bm_primary), 64, 1, 2, 4, 2, 0),
        (min(16, bm_primary), 32, 2, 2, 4, 2, 0),
        (min(32, bm_primary), 32, 1, 2, 4, 2, 0),
        (min(32, bm_primary), 32, 2, 2, 4, 1, 0),
        (8,                   32, 1, 2, 4, 2, 0),
    ]
    seen_sg = set()
    for spec in sg_specs:
        sm, sn, wm, wn, sk, unr, sw = spec
        if spec in seen_sg or sm * wm > M:
            continue
        seen_sg.add(spec)
        if K_G % sk != 0 or (K_G // sk) % unr != 0 or N % (sn * wn) != 0:
            continue
        _add("u4_nax_sg_nobar", {
            "SM": sm, "SN": sn, "WM": wm, "WN": wn, "split_k": sk, "k_unroll": unr, "swizzle": sw, "relaxed": False,
        }, build_u4_nax_sg_nobar)

    # Family 3: Split-K Threadgroup-Dequantized bfloat x bfloat NAX (multiply_accumulate)
    deq_specs = [
        (bm_primary, 64,  4, 1, 0, 1),
        (bm_primary, 64,  4, 2, 0, 1),
        (bm_primary, 64,  4, 4, 0, 1),
        (bm_primary, 64,  4, 8, 0, 1),
        (bm_primary, 64,  4, 4, 0, 2),  # BK=128 (half barriers!)
        (bm_primary, 64,  4, 2, 0, 2),
        (bm_primary, 64,  4, 1, 0, 2),
        (bm_primary, 64,  8, 4, 0, 2),
        (bm_primary, 128, 4, 4, 0, 1),
        (bm_primary, 128, 8, 2, 0, 1),
        (bm_primary, 32,  2, 4, 0, 1),
        (bm_primary, 32,  4, 4, 0, 2),
        (bm_alt,     64,  4, 4, 1, 1),
    ]
    seen_deq = set()
    for spec in deq_specs:
        bm, bn, nsg, sk, sw, bkg = spec
        if spec in seen_deq:
            continue
        seen_deq.add(spec)
        if K_G % (sk * bkg) != 0 or N % bn != 0:
            continue
        _add("tgp_deq_nax", {"BM": bm, "BN": bn, "NSG": nsg, "split_k": sk, "swizzle": sw, "bk_groups": bkg}, build_tgp_deq_nax)

    return cands


def generate_dense_swiglu_candidates(
    M: int, N: int, K: int, dtype: mx.Dtype = mx.bfloat16
) -> List[KernelCandidate]:
    """Generate fused gate+up+SiLU candidates for Qwen3.5-9B MLP (12288x4096)."""
    shape_tag = f"SwiGLU_M{M}_N{N}_K{K}"
    cands: List[KernelCandidate] = []
    K_G = K // 64

    def _add(family: str, genome: Dict[str, Any], builder: Callable) -> None:
        cid = f"{family}_{shape_tag}_{len(cands):02d}"
        c = KernelCandidate(cid=cid, arena="dense_swiglu", family=family, shape_tag=shape_tag, genome=genome)
        try:
            fn, src = builder(M, N, K, genome, dtype)
            c.fn = fn
            c.source_lines = len(src.splitlines())
            c.compiled = True
        except Exception as e:
            c.compile_error = f"{type(e).__name__}: {e}"
        cands.append(c)

    if M == 1:
        for nr, nsg, vb in [(2, 4, 128), (4, 4, 128), (4, 2, 128), (2, 4, 64), (4, 4, 64)]:
            if N % (nr * nsg) == 0:
                _add("fused_swiglu_b1_vec", {"NR": nr, "NSG": nsg, "vec_bits": vb}, build_fused_swiglu_b1_vec)
        return cands

    bm = min(64, max(b for b in (8, 16, 32, 64) if b <= M))
    # Family 5A: Direct U4 NAX fused SwiGLU
    for bn, nsg, sk, unr, csb, sw in [
        (64, 4, 1, 2, True,  0),
        (64, 4, 2, 2, True,  0),
        (64, 4, 4, 2, True,  0),
        (64, 4, 4, 2, False, 0),
        (32, 2, 2, 2, True,  0),
    ]:
        if K_G % sk == 0 and (K_G // sk) % unr == 0 and N % bn == 0:
            _add("fused_swiglu_u4_nax", {
                "BM": bm, "BN": bn, "NSG": nsg, "split_k": sk, "k_unroll": unr, "cache_sb": csb, "swizzle": sw,
            }, build_fused_swiglu_u4_nax)
    # Family 5B: Split-K Threadgroup-Dequant multiply_accumulate NAX fused SwiGLU (ZERO per-group FMA!)
    for bn, nsg, sk, sw, bkg in [
        (64, 4, 1, 0, 1),
        (64, 4, 2, 0, 1),
        (64, 4, 4, 0, 1),
        (64, 4, 1, 0, 2),
        (64, 4, 2, 0, 2),
        (64, 4, 4, 0, 2),
        (64, 8, 2, 0, 2),
        (128, 4, 2, 0, 1),
        (32, 2, 2, 0, 1),
    ]:
        if K_G % (sk * bkg) == 0 and N % bn == 0:
            _add("fused_swiglu_tgp_deq_nax", {
                "BM": bm, "BN": bn, "NSG": nsg, "split_k": sk, "swizzle": sw, "bk_groups": bkg,
            }, build_fused_swiglu_tgp_deq_nax)
    return cands


def generate_moe_candidates(
    H: int = 2048, D_MOE: int = 512, TOPK: int = 8, dtype: mx.Dtype = mx.bfloat16
) -> Tuple[List[KernelCandidate], List[KernelCandidate], List[KernelCandidate]]:
    """Generate 28+ specialized B=1 MoE kernels across M1 (gate+up+SwiGLU), M2 (down+sum), M3 (mega)."""
    gu_cands: List[KernelCandidate] = []
    dn_cands: List[KernelCandidate] = []
    mega_cands: List[KernelCandidate] = []

    # Family M1: 20 distinct fused 8-expert gate+up+SwiGLU schedules (incl. 0-barrier 128B-coalesced u32!)
    for nr, nsg, vb, xm, nm, go in [
        (4, 2, 32,  "direct_l1", "mlx_div", "expert_y"),  # 128B exact cache line, 8 regs, 0 barriers!
        (4, 4, 32,  "direct_l1", "mlx_div", "expert_y"),
        (2, 4, 32,  "direct_l1", "mlx_div", "expert_y"),
        (8, 2, 32,  "direct_l1", "mlx_div", "expert_y"),
        (4, 2, 32,  "direct_l1", "shift",   "expert_y"),
        (4, 4, 32,  "tgp_f32",   "mlx_div", "expert_y"),
        (4, 2, 64,  "direct_l1", "mlx_div", "expert_y"),
        (4, 4, 64,  "direct_l1", "mlx_div", "expert_y"),
        (2, 4, 64,  "direct_l1", "mlx_div", "expert_y"),
        (4, 4, 64,  "tgp_f32",   "mlx_div", "expert_y"),
        (2, 8, 64,  "tgp_f32",   "shift",   "expert_y"),
        (1, 8, 64,  "tgp_f32",   "shift",   "expert_y"),
        (2, 4, 64,  "direct_l1", "shift",   "expert_x"),
        (2, 4, 128, "direct_l1", "mlx_div", "expert_y"),
        (4, 2, 128, "tgp_f32",   "shift",   "expert_y"),
        (2, 4, 128, "tgp_f32",   "mlx_div", "expert_y"),
        (1, 8, 128, "direct_l1", "shift",   "expert_y"),
        (2, 8, 128, "tgp_f32",   "shift",   "expert_y"),
    ]:
        cid = f"moe_gu_{len(gu_cands):02d}"
        g = {"NR": nr, "NSG": nsg, "vec_bits": vb, "x_mode": xm, "nibble_mode": nm, "grid_order": go}
        c = KernelCandidate(cid=cid, arena="moe_gate_up", family="moe_fused_gate_up_swiglu", shape_tag=f"MoE_E{TOPK}_H{H}_D{D_MOE}", genome=g)
        try:
            fn, src = build_moe_fused_gate_up_swiglu(H, D_MOE, TOPK, g, dtype)
            c.fn, c.source_lines, c.compiled = fn, len(src.splitlines()), True
        except Exception as e:
            c.compile_error = f"{type(e).__name__}: {e}"
        gu_cands.append(c)

    # Family M2: 18 distinct fused 8-expert down+weighted-sum schedules (incl. 128B u32, 0-barrier L1, split_e=1/2/4/8!)
    for nr, nsg, hm, nm, se, vb in [
        (4, 2, "direct_l1",      "mlx_div", 1, 32),  # 128B coalesced, 0 barriers, 0 TGP SRAM!
        (4, 4, "direct_l1",      "mlx_div", 1, 32),
        (2, 4, "direct_l1",      "mlx_div", 1, 32),
        (4, 2, "direct_l1",      "mlx_div", 2, 32),
        (4, 2, "direct_l1",      "mlx_div", 4, 32),
        (4, 2, "direct_l1",      "mlx_div", 8, 32),  # 100% contiguous 1-expert/TG streaming!
        (4, 4, "direct_l1",      "mlx_div", 8, 32),
        (8, 2, "direct_l1",      "mlx_div", 8, 32),
        (4, 2, "direct_l1",      "mlx_div", 1, 64),  # zero K-loop, 0 barriers!
        (4, 2, "direct_l1",      "mlx_div", 4, 64),
        (4, 2, "direct_l1",      "mlx_div", 8, 64),
        (2, 4, "tgp_scaled_f32", "mlx_div", 1, 64),
        (4, 4, "tgp_scaled_f32", "shift",   1, 64),
        (2, 8, "tgp_scaled_f16", "shift",   1, 64),  # 8KB TGP SRAM
        (2, 4, "tgp_scaled_f16", "mlx_div", 2, 64),
        (2, 4, "tgp_scaled_f32", "shift",   2, 64),
        (4, 2, "tgp_scaled_f16", "mlx_div", 4, 32),
        (2, 4, "direct_l1",      "shift",   8, 32),
    ]:
        cid = f"moe_dn_{len(dn_cands):02d}"
        g = {"NR": nr, "NSG": nsg, "h_mode": hm, "nibble_mode": nm, "split_e": se, "vec_bits": vb}
        c = KernelCandidate(cid=cid, arena="moe_down", family="moe_fused_down_reduce", shape_tag=f"MoE_E{TOPK}_H{H}_D{D_MOE}", genome=g)
        try:
            fn, src = build_moe_fused_down_reduce(H, D_MOE, TOPK, g, dtype)
            c.fn, c.source_lines, c.compiled = fn, len(src.splitlines()), True
        except Exception as e:
            c.compile_error = f"{type(e).__name__}: {e}"
        dn_cands.append(c)

    # Family M3: True 1-dispatch mega-kernel (8 threadgroups, zero h DRAM round-trip!)
    for nsg in (8, 16):
        cid = f"moe_mega_{len(mega_cands):02d}"
        g = {"NSG": nsg}
        c = KernelCandidate(cid=cid, arena="moe_full", family="moe_mega_1dispatch", shape_tag=f"MoE_E{TOPK}_H{H}_D{D_MOE}", genome=g)
        try:
            fn, src = build_moe_mega_1dispatch(H, D_MOE, TOPK, g, dtype)
            c.fn, c.source_lines, c.compiled = fn, len(src.splitlines()), True
        except Exception as e:
            c.compile_error = f"{type(e).__name__}: {e}"
        mega_cands.append(c)

    return gu_cands, dn_cands, mega_cands


# ---------------------------------------------------------------------------
# Evolutionary Mutator (Spawns Ugly Mutated Children from Verified Parents!)
# ---------------------------------------------------------------------------

def mutate_dense_genome(parent: KernelCandidate, rng: random.Random, M: int, N: int, K: int) -> Optional[Tuple[str, Dict[str, Any], Callable]]:
    g = dict(parent.genome)
    K_G = K // 64
    if parent.family == "u4_nax_coop":
        gene = rng.choice(["split_k", "BN", "NSG", "k_unroll", "cache_sb", "swizzle", "stage_x"])
        if gene == "split_k":
            opts = [s for s in (1, 2, 4, 8, 16) if K_G % s == 0 and (K_G // s) % g["k_unroll"] == 0]
            g["split_k"] = rng.choice(opts)
        elif gene == "BN":
            opts = [b for b in (32, 64, 128) if N % b == 0]
            g["BN"] = rng.choice(opts)
        elif gene == "NSG":
            g["NSG"] = rng.choice([2, 4, 8])
        elif gene == "k_unroll":
            opts = [u for u in (1, 2, 4) if (K_G // g["split_k"]) % u == 0]
            g["k_unroll"] = rng.choice(opts)
        elif gene == "cache_sb":
            g["cache_sb"] = not g["cache_sb"]
        elif gene == "swizzle":
            g["swizzle"] = rng.choice([0, 1, 2])
        elif gene == "stage_x":
            g["stage_x"] = not g.get("stage_x", False)
        return ("u4_nax_coop", g, build_u4_nax_coop)
    if parent.family == "u4_nax_sg_nobar":
        gene = rng.choice(["split_k", "WN", "SN", "k_unroll", "swizzle"])
        if gene == "split_k":
            opts = [s for s in (1, 2, 4, 8) if K_G % s == 0 and (K_G // s) % g["k_unroll"] == 0]
            g["split_k"] = rng.choice(opts)
        elif gene == "WN":
            opts = [w for w in (1, 2, 4) if N % (g["SN"] * w) == 0]
            g["WN"] = rng.choice(opts)
        elif gene == "SN":
            opts = [s for s in (16, 32, 64) if N % (s * g["WN"]) == 0]
            g["SN"] = rng.choice(opts)
        elif gene == "k_unroll":
            opts = [u for u in (1, 2, 4) if (K_G // g["split_k"]) % u == 0]
            g["k_unroll"] = rng.choice(opts)
        elif gene == "swizzle":
            g["swizzle"] = rng.choice([0, 1])
        return ("u4_nax_sg_nobar", g, build_u4_nax_sg_nobar)
    if parent.family == "tgp_deq_nax":
        gene = rng.choice(["split_k", "BN", "NSG", "swizzle", "bk_groups"])
        bkg = g.get("bk_groups", 1)
        if gene == "split_k":
            opts = [s for s in (1, 2, 4, 8, 16) if K_G % (s * bkg) == 0]
            g["split_k"] = rng.choice(opts)
        elif gene == "BN":
            opts = [b for b in (32, 64, 128) if N % b == 0]
            g["BN"] = rng.choice(opts)
        elif gene == "NSG":
            g["NSG"] = rng.choice([2, 4, 8])
        elif gene == "swizzle":
            g["swizzle"] = rng.choice([0, 1, 2])
        elif gene == "bk_groups":
            opts = [b for b in (1, 2) if K_G % (g["split_k"] * b) == 0]
            g["bk_groups"] = rng.choice(opts)
        return ("tgp_deq_nax", g, build_tgp_deq_nax)
    return None


# ---------------------------------------------------------------------------
# Strict Verifier Gate Helpers
# ---------------------------------------------------------------------------

def verify_candidate_dense(c: KernelCandidate, M: int, N: int, K: int, dtype: mx.Dtype = mx.bfloat16) -> bool:
    if not c.compiled or c.fn is None:
        c.verified = False
        c.verify_info = {"fail": "compile_error"}
        return False
    try:
        ok, info = verify_qmm(c.fn, M, N, K, dtype=dtype, gs=64)
        c.verified = ok
        c.verify_info = info
        return ok
    except Exception as e:
        c.verified = False
        c.verify_info = {"fail": f"runtime_{type(e).__name__}: {e}"}
        return False


def make_swiglu_test_inputs(M: int, N: int, K: int, dtype: mx.Dtype = mx.bfloat16, seed: int = 101):
    mx.random.seed(seed)
    # Basis inputs: small integers + unit scales
    qg = mx.random.randint(0, 16, shape=(N, K), dtype=mx.uint8)
    qu = mx.random.randint(0, 16, shape=(N, K), dtype=mx.uint8)
    wg_b, wu_b = _pack_u4(qg), _pack_u4(qu)
    s1 = mx.ones((N, K // 64), dtype=dtype)
    b1 = mx.full((N, K // 64), 0.5, dtype=dtype)
    xb = mx.zeros((M, K), dtype=dtype)
    for m in range(M):
        xb[m, (m * 53) % K] = 0.25
        xb[m, (m * 53 + 11) % K] = -0.25
    basis = (xb, wg_b, s1, b1, wu_b, s1, b1)

    # Outlier random inputs
    scale = 1.0 / math.sqrt(K)
    wg, sg, bg = mx.quantize((mx.random.normal((N, K)) * scale).astype(dtype), group_size=64, bits=4, mode="affine")
    wu, su, bu = mx.quantize((mx.random.normal((N, K)) * scale).astype(dtype), group_size=64, bits=4, mode="affine")
    xr = _make_outliers((M, K), dtype, seed + 5)
    rand = (xr, wg, sg, bg, wu, su, bu)
    mx.eval(*basis, *rand)
    return basis, rand


def make_moe_test_inputs(E: int = 16, H: int = 2048, D_MOE: int = 512, TOPK: int = 8, dtype: mx.Dtype = mx.bfloat16, seed: int = 202):
    """Build basis + heavy-tailed outlier test fixtures for MoE verification (uses E=16 for speed)."""
    mx.random.seed(seed)
    indices = mx.array([1, 3, 5, 7, 9, 11, 13, 15][:TOPK], dtype=mx.uint32)
    scores_f = mx.softmax(mx.arange(TOPK, dtype=mx.float32) * 0.2).astype(dtype)

    # Basis inputs
    qg = mx.random.randint(0, 16, shape=(E, D_MOE, H), dtype=mx.uint8)
    qu = mx.random.randint(0, 16, shape=(E, D_MOE, H), dtype=mx.uint8)
    qd = mx.random.randint(0, 16, shape=(E, H, D_MOE), dtype=mx.uint8)
    wg_b, wu_b, wd_b = _pack_u4(qg), _pack_u4(qu), _pack_u4(qd)
    s_gu = mx.ones((E, D_MOE, H // 64), dtype=dtype) * 0.125
    b_gu = mx.full((E, D_MOE, H // 64), -0.5, dtype=dtype)
    s_dn = mx.ones((E, H, D_MOE // 64), dtype=dtype) * 0.125
    b_dn = mx.full((E, H, D_MOE // 64), -0.5, dtype=dtype)
    xb = mx.zeros((1, H), dtype=dtype)
    xb[0, 19] = 1.0
    xb[0, 517] = -0.5
    xb[0, 1500] = 0.5
    hb = mx.zeros((TOPK, D_MOE), dtype=dtype)
    for e in range(TOPK):
        hb[e, (e * 37) % D_MOE] = 1.0
        hb[e, (e * 37 + 13) % D_MOE] = -0.5

    # Outlier random inputs
    sc_gu = 1.0 / math.sqrt(H)
    sc_dn = 1.0 / math.sqrt(D_MOE)
    wg, sg, bg = mx.quantize((mx.random.normal((E, D_MOE, H)) * sc_gu).astype(dtype), group_size=64, bits=4, mode="affine")
    wu, su, bu = mx.quantize((mx.random.normal((E, D_MOE, H)) * sc_gu).astype(dtype), group_size=64, bits=4, mode="affine")
    wd, sd, bd = mx.quantize((mx.random.normal((E, H, D_MOE)) * sc_dn).astype(dtype), group_size=64, bits=4, mode="affine")
    xr = _make_outliers((1, H), dtype, seed + 7)
    hr = _make_outliers((TOPK, D_MOE), dtype, seed + 9)

    fixtures = {
        "gu_basis": (xb, indices, wg_b, s_gu, b_gu, wu_b, s_gu, b_gu),
        "gu_rand":  (xr, indices, wg, sg, bg, wu, su, bu),
        "dn_basis": (hb, indices, scores_f, wd_b, s_dn, b_dn),
        "dn_rand":  (hr, indices, scores_f, wd, sd, bd),
        "full_basis": (xb, indices, scores_f, wg_b, s_gu, b_gu, wu_b, s_gu, b_gu, wd_b, s_dn, b_dn),
        "full_rand":  (xr, indices, scores_f, wg, sg, bg, wu, su, bu, wd, sd, bd),
    }
    mx.eval(*[arr for tup in fixtures.values() for arr in tup])
    return fixtures


def verify_candidate_fused(c: KernelCandidate, ref_fn: Callable, basis_in: tuple, rand_in: tuple) -> bool:
    if not c.compiled or c.fn is None:
        c.verified = False
        c.verify_info = {"fail": "compile_error"}
        return False
    try:
        ok, info = verify_fused_op(c.fn, ref_fn, basis_in, rand_in)
        c.verified = ok
        c.verify_info = info
        return ok
    except Exception as e:
        c.verified = False
        c.verify_info = {"fail": f"runtime_{type(e).__name__}: {e}"}
        return False


# ---------------------------------------------------------------------------
# Fast Single-Dispatch Screening Timer (Used only for Evolutionary Ranking;
# Final Cold-SLC Interleaved Timings Live in run_bench.py!)
# ---------------------------------------------------------------------------

def quick_screen_us(fn: Callable, args: tuple, warmup: int = 2, iters: int = 6) -> float:
    for _ in range(warmup):
        y = fn(*args)
        mx.eval(y)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = fn(*args)
        mx.eval(y)
    mx.synchronize()
    return ((time.perf_counter() - t0) / iters) * 1e6


def evolve_dense_shape(
    M: int, N: int, K: int, dtype: mx.Dtype = mx.bfloat16,
    n_mutations: int = 8, screen: bool = True, seed: int = 7,
) -> Tuple[List[KernelCandidate], List[Tuple[KernelCandidate, float]]]:
    """Generate seed zoo + strictly verify + optionally evolve mutated children."""
    cands = generate_dense_qmm_candidates(M, N, K, dtype)
    for c in cands:
        verify_candidate_dense(c, M, N, K, dtype)

    verified = [c for c in cands if c.verified]
    ranked: List[Tuple[KernelCandidate, float]] = []
    if screen and verified:
        mx.random.seed(seed)
        x = mx.random.normal((M, K)).astype(dtype)
        w, s, b = mx.quantize((mx.random.normal((N, K)) * (1.0 / math.sqrt(K))).astype(dtype), group_size=64, bits=4, mode="affine")
        mx.eval(x, w, s, b)
        for c in verified:
            us = quick_screen_us(c.fn, (x, w, s, b))
            ranked.append((c, us))
        ranked.sort(key=lambda t: t[1])

        # Evolutionary generation 2: mutate top-4 verified parents!
        if n_mutations > 0 and M >= 8:
            rng = random.Random(seed + M * 31 + N)
            seen = {(c.family, tuple(sorted(c.genome.items()))) for c in cands}
            parents = [c for c, _ in ranked[:4]]
            for _ in range(n_mutations * 3):
                if sum(1 for c in cands if "_evo" in c.cid) >= n_mutations:
                    break
                p = rng.choice(parents)
                mut = mutate_dense_genome(p, rng, M, N, K)
                if mut is None:
                    continue
                fam, g, builder = mut
                sig = (fam, tuple(sorted(g.items())))
                if sig in seen:
                    continue
                seen.add(sig)
                cid = f"{fam}_M{M}_N{N}_K{K}_evo{len(cands):02d}"
                child = KernelCandidate(cid=cid, arena="dense_qmm", family=fam, shape_tag=f"M{M}_N{N}_K{K}", genome=g)
                try:
                    fn, src = builder(M, N, K, g, dtype)
                    child.fn, child.source_lines, child.compiled = fn, len(src.splitlines()), True
                except Exception as e:
                    child.compile_error = f"{type(e).__name__}: {e}"
                verify_candidate_dense(child, M, N, K, dtype)
                cands.append(child)
                if child.verified:
                    us = quick_screen_us(child.fn, (x, w, s, b))
                    ranked.append((child, us))
            ranked.sort(key=lambda t: t[1])

    return cands, ranked


def run_smoke_suite() -> Dict[str, Any]:
    """Brief compile + verifier + sanity check across all dense, SwiGLU, and MoE families (<10s)."""
    t0 = time.perf_counter()
    summary: Dict[str, Any] = {"arenas": {}, "total_generated": 0, "total_compiled": 0, "total_verified": 0}

    # 1. Dense QMM shapes (verify full zoo on M=32 4096x4096, plus M=1, M=16, M=64, 12288x4096, 4096x12288)
    dense_shapes = [
        (32, 4096, 4096),
        (64, 4096, 4096),
        (16, 12288, 4096),
        (32, 4096, 12288),
        (1,  4096, 4096),
    ]
    for M, N, K in dense_shapes:
        cands, ranked = evolve_dense_shape(M, N, K, dtype=mx.bfloat16, n_mutations=4 if M == 32 and N == 4096 and K == 4096 else 0, screen=True)
        n_gen = len(cands)
        n_cmp = sum(c.compiled for c in cands)
        n_ver = sum(c.verified for c in cands)
        summary["total_generated"] += n_gen
        summary["total_compiled"] += n_cmp
        summary["total_verified"] += n_ver
        top = ranked[0] if ranked else None
        fails = [c.to_record() for c in cands if not c.verified]
        summary["arenas"][f"dense_M{M}_N{N}_K{K}"] = {
            "generated": n_gen, "compiled": n_cmp, "verified": n_ver,
            "top_cid": top[0].cid if top else None,
            "top_screen_us": round(top[1], 1) if top else None,
            "failures": fails[:3],
        }
        print(f"[Dense M={M:2d} {N}x{K}] gen={n_gen} cmp={n_cmp} ver={n_ver}/{n_gen} | best={top[0].cid if top else 'NONE'} ({top[1]:.1f} us)")

    # 2. Fused Dense SwiGLU (M=1 and M=32 on 12288x4096)
    for M in (1, 32):
        N, K = 12288, 4096
        cands = generate_dense_swiglu_candidates(M, N, K, dtype=mx.bfloat16)
        basis_in, rand_in = make_swiglu_test_inputs(M, N, K, dtype=mx.bfloat16)
        for c in cands:
            verify_candidate_fused(c, ref_swiglu_fp32, basis_in, rand_in)
        n_gen, n_cmp, n_ver = len(cands), sum(c.compiled for c in cands), sum(c.verified for c in cands)
        summary["total_generated"] += n_gen
        summary["total_compiled"] += n_cmp
        summary["total_verified"] += n_ver
        ranked = sorted([(c, quick_screen_us(c.fn, rand_in)) for c in cands if c.verified], key=lambda t: t[1])
        stock_us = quick_screen_us(stock_swiglu_mlx, rand_in)
        top = ranked[0] if ranked else None
        summary["arenas"][f"swiglu_M{M}_N{N}_K{K}"] = {
            "generated": n_gen, "compiled": n_cmp, "verified": n_ver,
            "top_cid": top[0].cid if top else None,
            "top_screen_us": round(top[1], 1) if top else None,
            "stock_screen_us": round(stock_us, 1),
            "screen_speedup_x": round(stock_us / top[1], 2) if top else None,
            "failures": [c.to_record() for c in cands if not c.verified][:3],
        }
        print(f"[SwiGLU M={M:2d} {N}x{K}] gen={n_gen} cmp={n_cmp} ver={n_ver}/{n_gen} | best={top[0].cid if top else 'NONE'} ({top[1]:.1f} us vs stock {stock_us:.1f} us = {stock_us/top[1]:.2f}x)")

    # 3. MoE B=1 (Qwen3.6-35B-A3B exact active expert shapes: H=2048, D_MOE=512, TOPK=8)
    gu_cands, dn_cands, mega_cands = generate_moe_candidates(H=2048, D_MOE=512, TOPK=8, dtype=mx.bfloat16)
    fix = make_moe_test_inputs(E=16, H=2048, D_MOE=512, TOPK=8, dtype=mx.bfloat16)
    for c in gu_cands:
        verify_candidate_fused(c, ref_moe_gate_up_fp32, fix["gu_basis"], fix["gu_rand"])
    for c in dn_cands:
        verify_candidate_fused(c, ref_moe_down_fp32, fix["dn_basis"], fix["dn_rand"])
    for c in mega_cands:
        verify_candidate_fused(c, ref_moe_full_fp32, fix["full_basis"], fix["full_rand"])

    for label, clist, r_in in [
        ("moe_gate_up", gu_cands, fix["gu_rand"]),
        ("moe_down", dn_cands, fix["dn_rand"]),
        ("moe_mega_1dispatch", mega_cands, fix["full_rand"]),
    ]:
        n_gen, n_cmp, n_ver = len(clist), sum(c.compiled for c in clist), sum(c.verified for c in clist)
        summary["total_generated"] += n_gen
        summary["total_compiled"] += n_cmp
        summary["total_verified"] += n_ver
        ranked = sorted([(c, quick_screen_us(c.fn, r_in)) for c in clist if c.verified], key=lambda t: t[1])
        top = ranked[0] if ranked else None
        summary["arenas"][label] = {
            "generated": n_gen, "compiled": n_cmp, "verified": n_ver,
            "top_cid": top[0].cid if top else None,
            "top_screen_us": round(top[1], 1) if top else None,
            "failures": [c.to_record() for c in clist if not c.verified][:3],
        }
        print(f"[{label:18s}] gen={n_gen} cmp={n_cmp} ver={n_ver}/{n_gen} | best={top[0].cid if top else 'NONE'} ({top[1]:.1f} us)")

    summary["elapsed_s"] = round(time.perf_counter() - t0, 2)
    print(f"\nTOTAL SYNTHESIZED: {summary['total_generated']} | COMPILED: {summary['total_compiled']} | VERIFIED: {summary['total_verified']} ({summary['elapsed_s']}s)")
    return summary


if __name__ == "__main__":
    run_smoke_suite()
