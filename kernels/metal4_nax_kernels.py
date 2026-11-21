"""
Track A: Apple M5 Max Metal 4 Hardware NAX Kernels (`mpp::tensor_ops::matmul2d`).

Implements and caches correctness-verified Metal 4 Neural Accelerator kernels via
`mx.fast.metal_kernel` for MLX affine Q4 (group_size=64, bits=4, float16 & bfloat16)
and exact hardware integer NAX (`int8_t x int4b_format -> int32_t`,
`int8_t x int8_t -> int32_t`):

1. `u4_direct` (Direct hardware `T x uint4b_format -> float` grouped-Q4 Split-K NAX):
   Zero threadgroup weight dequant buffer, zero barriers, native MLX little-endian
   uint32 nibble layout, per-group K=64 hardware `matmul2d(..., mode::multiply)` with
   deduplicated register scale/bias/x_gsum accumulation.
2. `tg_deq` (128-bit/64-bit vectorized threadgroup-dequant Split-K Metal 4 NAX):
   `WM x WN` explicit `execution_simdgroup` `matmul2d(SM, SN, 64, ..., multiply_accumulate)`
   sharing one cooperatively dequantized `Ws` tile across `WM` M-simdgroups.
3. `int8_int4` / `int8_int8` hardware integer NAX:
   - Exact `int8_t x int4b_format -> int32_t` & `int8_t x int8_t -> int32_t` GEMM (0 error).
   - Dynamic per-group INT8 activation quant + hardware `int8_t x int4b_format -> int32_t`
     grouped-Q4 affine matmul.
4. `reg_ctb` (Register-dequant directly into right-operand `cooperative_tensor` ablation).
5. Zero-code stock MLX workarounds (`pad65`, `reshape_batch`, `stacked_n`).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import mlx.core as mx

HEADER_METAL4 = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""


def _dtype_str(dtype: mx.Dtype) -> str:
    if dtype == mx.float16:
        return "half"
    if dtype == mx.bfloat16:
        return "bfloat"
    raise ValueError(f"Unsupported activation dtype {dtype}; expected float16 or bfloat16")


def _vec4_str(dtype_str: str) -> str:
    return "half4" if dtype_str == "half" else "bfloat4"


# =============================================================================
# 1. Helper Kernels: Group Row-Sum (x_gsum) & Vectorized Split-K Reduction
# =============================================================================

@lru_cache(maxsize=8)
def get_x_gsum_kernel(dtype_str: str):
    """Compute x_gsum[M, K/64] = sum_{k=0..63} float(x[m, 64*g + k]) in float32."""
    vec4 = _vec4_str(dtype_str)
    src = f"""
    const int M = params[0];
    const int K = params[1];
    const int K_g = K >> 6;
    const int idx = int(thread_position_in_grid.x);
    if (idx < M * K_g) {{
        const int m = idx / K_g;
        const int g = idx % K_g;
        const device {vec4}* xp = (const device {vec4}*)(x + size_t(m) * K + (g << 6));
        float4 acc = float4(0.0f);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {{
            acc += float4(xp[i]);
        }}
        out[idx] = (acc.x + acc.y) + (acc.z + acc.w);
    }}
    """
    return mx.fast.metal_kernel(
        name=f"nax_xgsum_{dtype_str}",
        input_names=["x", "params"],
        output_names=["out"],
        header=HEADER_METAL4,
        source=src,
    )


def compute_x_gsum(x: mx.array) -> mx.array:
    """Compute per-K=64-group row sums of 2D matrix x (M, K) -> float32 (M, K/64)."""
    M, K = x.shape
    dstr = _dtype_str(x.dtype)
    k = get_x_gsum_kernel(dstr)
    n_tot = M * (K >> 6)
    params = mx.array([M, K], dtype=mx.int32)
    tgp = min(256, max(32, ((n_tot + 31) // 32) * 32))
    return k(
        inputs=[x, params],
        grid=(n_tot, 1, 1),
        threadgroup=(tgp, 1, 1),
        output_shapes=[(M, K >> 6)],
        output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=32)
def get_reduce_sk_kernel(dtype_str: str, split_k: int, partial_is_fp32: bool = False):
    """Vectorized 4-wide deterministic Split-K reduction: (split_k, M, N) -> (M, N)."""
    vec4_out = _vec4_str(dtype_str)
    if partial_is_fp32:
        p_t = "float"
        load_vec4 = "*(const device float4*)(partial + size_t(s) * total + idx)"
    else:
        p_t = dtype_str
        load_vec4 = f"float4(*(const device {vec4_out}*)(partial + size_t(s) * total + idx))"

    src = f"""
    const int idx = int(thread_position_in_grid.x) * 4;
    const int total = params[0];
    if (idx + 3 < total) {{
        float4 acc = float4(0.0f);
        #pragma unroll
        for (int s = 0; s < {split_k}; ++s) {{
            acc += {load_vec4};
        }}
        *(device {vec4_out}*)(out + idx) = {vec4_out}(acc);
    }} else if (idx < total) {{
        for (int j = idx; j < total; ++j) {{
            float a = 0.0f;
            #pragma unroll
            for (int s = 0; s < {split_k}; ++s) {{
                a += float(partial[size_t(s) * total + j]);
            }}
            out[j] = {dtype_str}(a);
        }}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"nax_red4_{dtype_str}_sk{split_k}_f{int(partial_is_fp32)}",
        input_names=["partial", "params"],
        output_names=["out"],
        header=HEADER_METAL4,
        source=src,
    )


def reduce_splitk(partial: mx.array, out_dtype: mx.Dtype) -> mx.array:
    """Reduce (split_k, M, N) -> (M, N) in out_dtype."""
    split_k, M, N = partial.shape
    total = M * N
    dstr = _dtype_str(out_dtype)
    rk = get_reduce_sk_kernel(dstr, split_k, partial_is_fp32=(partial.dtype == mx.float32))
    params = mx.array([total], dtype=mx.int32)
    n_thr = (total + 3) // 4
    return rk(
        inputs=[partial, params],
        grid=(n_thr, 1, 1),
        threadgroup=(min(256, max(32, n_thr)), 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[out_dtype],
    )[0]


# =============================================================================
# 2. Family 1: Direct Hardware `T x uint4b_format -> float` Grouped Q4 NAX
#    (ZERO threadgroup weight dequant buffer, ZERO barriers!)
# =============================================================================

@lru_cache(maxsize=128)
def get_u4_direct_kernel(
    dtype_str: str = "half",
    BM: int = 32,
    BN: int = 32,
    WM: int = 2,
    WN: int = 1,
    split_k: int = 4,
    atomic: bool = False,
    relaxed: bool = True,
    partial_fp32: bool = False,
):
    """
    Direct hardware `T x uint4b_format -> float` affine Q4 (gs=64) Split-K Metal 4 NAX.

    Exploits:
        y_g = x_g @ (s_g * u4_g + b_g)^T = s_g * (x_g @ u4_g^T) + b_g * sum(x_g)
    using `matmul2d(SM, SN, 64, false, true, relaxed, mode::multiply)` per K=64 group.
    For native SM=16, SN=32 tiles, deduplicates x_gsum (2 loads vs 16) and scales/biases
    (8 pairs vs 16) directly in registers using the verified cooperative_tensor layout!
    """
    SM = BM // WM
    SN = BN // WN
    assert BM % WM == 0 and BN % WN == 0
    rx = "true" if relaxed else "false"
    use_fast_16x32 = (SM == 16 and SN == 32)

    if use_fast_16x32:
        # Verified on M5 Max Metal 4: for SM=16, SN=32 execution_simdgroup,
        # acc capacity == 16 with elements [0..3]=(n0+0..3, m0), [4..7]=(n0+0..3, m1),
        # [8..11]=(n1+0..3, m0), [12..15]=(n1+0..3, m1).
        # We derive (n0, n1, m0, m1) dynamically from get_multidimensional_index so it
        # never hardcodes lane math and only reads 2 x_gsum + 8 scale/bias pairs per group!
        interior_loop = """
        const auto c0 = acc.get_multidimensional_index(uint16_t(0));
        const auto c4 = acc.get_multidimensional_index(uint16_t(4));
        const auto c8 = acc.get_multidimensional_index(uint16_t(8));
        const int nc0 = n_sg + int(c0[0]);
        const int nc1 = n_sg + int(c8[0]);
        const int mr0 = m_sg + int(c0[1]);
        const int mr1 = m_sg + int(c4[1]);
        const size_t xoff0 = size_t(mr0) * K_g;
        const size_t xoff1 = size_t(mr1) * K_g;

        for (int g = g_start; g < g_end; ++g) {
            auto sA = A.slice<64, SM>(g << 6, m_sg);
            auto sB = B.slice<64, SN>(g << 6, n_sg);
            op.run(sA, sB, tile_g);

            const float xm0 = x_gsum[xoff0 + g];
            const float xm1 = x_gsum[xoff1 + g];

            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const size_t soff = size_t(nc0 + q) * K_g + g;
                const float s = float(scales[soff]);
                const float b = float(biases[soff]);
                acc[q]     = fma(s, tile_g[q],     fma(b, xm0, acc[q]));
                acc[4 + q] = fma(s, tile_g[4 + q], fma(b, xm1, acc[4 + q]));
            }
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const size_t soff = size_t(nc1 + q) * K_g + g;
                const float s = float(scales[soff]);
                const float b = float(biases[soff]);
                acc[8 + q]  = fma(s, tile_g[8 + q],  fma(b, xm0, acc[8 + q]));
                acc[12 + q] = fma(s, tile_g[12 + q], fma(b, xm1, acc[12 + q]));
            }
        }
        """
    else:
        interior_loop = """
        for (int g = g_start; g < g_end; ++g) {
            auto sA = A.slice<64, SM>(g << 6, m_sg);
            auto sB = B.slice<64, SN>(g << 6, n_sg);
            op.run(sA, sB, tile_g);
            #pragma unroll
            for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
                if (acc.is_valid_element(i)) {
                    auto ids = acc.get_multidimensional_index(i);
                    const int ng = n_sg + int(ids[0]);
                    const int mg = m_sg + int(ids[1]);
                    const float s = float(scales[size_t(ng) * K_g + g]);
                    const float b = float(biases[size_t(ng) * K_g + g]);
                    acc[i] = fma(s, tile_g[i], fma(b, x_gsum[size_t(mg) * K_g + g], acc[i]));
                }
            }
        }
        """

    if atomic and split_k > 1:
        store_code = """
        device atomic_float* out_a = (device atomic_float*)out;
        #pragma unroll
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i)) {
                auto ids = acc.get_multidimensional_index(i);
                const int ng = n_sg + int(ids[0]);
                const int mg = m_sg + int(ids[1]);
                if (ng < N && mg < M) {
                    atomic_fetch_add_explicit(&out_a[size_t(mg) * N + ng], acc[i], memory_order_relaxed);
                }
            }
        }
        """
    elif split_k == 1:
        store_code = f"""
        #pragma unroll
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
            if (acc.is_valid_element(i)) {{
                auto ids = acc.get_multidimensional_index(i);
                const int ng = n_sg + int(ids[0]);
                const int mg = m_sg + int(ids[1]);
                if (ng < N && mg < M) {{
                    out[size_t(mg) * N + ng] = {dtype_str}(acc[i]);
                }}
            }}
        }}
        """
    else:
        p_t = "float" if partial_fp32 else dtype_str
        cast_e = "acc[i]" if partial_fp32 else f"{dtype_str}(acc[i])"
        store_code = f"""
        device {p_t}* out_sk = out + size_t(sk) * size_t(M) * size_t(N);
        #pragma unroll
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
            if (acc.is_valid_element(i)) {{
                auto ids = acc.get_multidimensional_index(i);
                const int ng = n_sg + int(ids[0]);
                const int mg = m_sg + int(ids[1]);
                if (ng < N && mg < M) {{
                    out_sk[size_t(mg) * N + ng] = {cast_e};
                }}
            }}
        }}
        """

    src = f"""
    constexpr int BM = {BM}, BN = {BN}, BK = 64;
    constexpr int WM = {WM}, WN = {WN}, SM = {SM}, SN = {SN}, SPLIT_K = {split_k};

    const int M = params[0];
    const int N = params[1];
    const int K = params[2];
    const int K_g = K >> 6;

    const int n0 = int(threadgroup_position_in_grid.x) * BN;
    const int m0 = int(threadgroup_position_in_grid.y) * BM;
    const int sk = int(threadgroup_position_in_grid.z);
    const int sgid = int(simdgroup_index_in_threadgroup);
    const int m_sg = m0 + (sgid / WN) * SM;
    const int n_sg = n0 + (sgid % WN) * SN;

    if (m_sg >= M || n_sg >= N) return;

    const int g_per_sk = (K_g + SPLIT_K - 1) / SPLIT_K;
    const int g_start = sk * g_per_sk;
    const int g_end = min(K_g, g_start + g_per_sk);
    if (g_start >= g_end) return;

    constexpr auto desc = matmul2d_descriptor(
        SM, SN, BK, false, true, {rx}, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;

    auto A = tensor<device {dtype_str}, dextents<int, 2>, tensor_inline>(
        const_cast<device {dtype_str}*>(x), dextents<int, 2>(K, M));
    auto B = tensor<device uint4b_format, dextents<int, 2>, tensor_inline>(
        const_cast<device uint8_t*>((const device uint8_t*)w), dextents<int, 2>(K, N));

    auto sA0 = A.slice(0, m_sg);
    auto sB0 = B.slice(0, n_sg);
    auto acc = op.get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), float>();
    auto tile_g = op.get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), float>();

    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    }}

    const bool interior = (m_sg + SM <= M) && (n_sg + SN <= N);
    if (interior) {{
        {interior_loop}
    }} else {{
        for (int g = g_start; g < g_end; ++g) {{
            auto sA = A.slice(g << 6, m_sg);
            auto sB = B.slice(g << 6, n_sg);
            op.run(sA, sB, tile_g);
            #pragma unroll
            for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
                if (acc.is_valid_element(i)) {{
                    auto ids = acc.get_multidimensional_index(i);
                    const int ng = n_sg + int(ids[0]);
                    const int mg = m_sg + int(ids[1]);
                    if (ng < N && mg < M) {{
                        const float s = float(scales[size_t(ng) * K_g + g]);
                        const float b = float(biases[size_t(ng) * K_g + g]);
                        acc[i] = fma(s, tile_g[i], fma(b, x_gsum[size_t(mg) * K_g + g], acc[i]));
                    }}
                }}
            }}
        }}
    }}

    {store_code}
    """
    return mx.fast.metal_kernel(
        name=f"nax_u4dir_{dtype_str}_{BM}_{BN}_{WM}x{WN}_sk{split_k}_a{int(atomic)}_r{int(relaxed)}_f{int(partial_fp32)}",
        input_names=["x", "w", "scales", "biases", "x_gsum", "params"],
        output_names=["out"],
        header=HEADER_METAL4,
        source=src,
    )


def qmm_u4_direct(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    BM: int = 32,
    BN: int = 32,
    WM: int = 2,
    WN: int = 1,
    split_k: int = 4,
    atomic: bool = False,
    relaxed: Optional[bool] = None,
    x_gsum: Optional[mx.array] = None,
    partial_fp32: Optional[bool] = None,
) -> mx.array:
    """Direct hardware `T x uint4b_format -> float` grouped affine Q4 (gs=64) matmul."""
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1])
    M, K = x2.shape
    N = w.shape[0]
    dstr = _dtype_str(x2.dtype)
    if partial_fp32 is None:
        partial_fp32 = (x2.dtype == mx.bfloat16)
    if relaxed is None:
        # Keep strict FP32 accumulation on bfloat16 for 100% greedy parity on real LLMs
        relaxed = (x2.dtype == mx.float16)
    if x_gsum is None:
        x_gsum = compute_x_gsum(x2)
    kern = get_u4_direct_kernel(dstr, BM, BN, WM, WN, split_k, atomic, relaxed, partial_fp32)
    tgp = WM * WN * 32
    grid = (((N + BN - 1) // BN) * tgp, (M + BM - 1) // BM, split_k)
    params = mx.array([M, N, K], dtype=mx.int32)

    if atomic and split_k > 1:
        y = kern(
            inputs=[x2, w, scales, biases, x_gsum, params],
            grid=grid,
            threadgroup=(tgp, 1, 1),
            output_shapes=[(M, N)],
            output_dtypes=[mx.float32],
            init_value=0.0,
        )[0].astype(x2.dtype)
    elif split_k == 1:
        y = kern(
            inputs=[x2, w, scales, biases, x_gsum, params],
            grid=grid,
            threadgroup=(tgp, 1, 1),
            output_shapes=[(M, N)],
            output_dtypes=[x2.dtype],
        )[0]
    else:
        p_dt = mx.float32 if partial_fp32 else x2.dtype
        partial = kern(
            inputs=[x2, w, scales, biases, x_gsum, params],
            grid=grid,
            threadgroup=(tgp, 1, 1),
            output_shapes=[(split_k, M, N)],
            output_dtypes=[p_dt],
        )[0]
        y = reduce_splitk(partial, x2.dtype)

    return y.reshape(*orig_shape[:-1], N)


# =============================================================================
# 3. Family 2: Vectorized Threadgroup-Dequant Split-K Metal 4 NAX
#    (`T x T -> float` `multiply_accumulate` with 128-bit `uint4` / 64-bit `uint2` loads)
# =============================================================================

@lru_cache(maxsize=128)
def get_tg_deq_nax_kernel(
    dtype_str: str = "half",
    BM: int = 32,
    BN: int = 32,
    BK: int = 64,
    WM: int = 2,
    WN: int = 1,
    split_k: int = 4,
    atomic: bool = False,
    relaxed: bool = True,
    partial_fp32: bool = False,
    double_buf: bool = False,
):
    """
    High-occupancy Split-K Metal 4 NAX kernel using `WM x WN` independent
    `execution_simdgroup` `matmul2d(SM, SN, BK, ..., multiply_accumulate)` ops sharing
    a cooperatively dequantized `Ws[BN * (BK+8)]` tile loaded via 128-bit `uint4` or 64-bit `uint2`.
    Supports BK in {64, 128}, optional double-buffering (1 barrier/step), and both `half` & `bfloat`.
    """
    NSG = WM * WN
    TGP = NSG * 32
    SM = BM // WM
    SN = BN // WN
    assert BM % WM == 0 and BN % WN == 0 and BK in (64, 128)
    BK_PAD = BK + 8
    GPS = BK // 64
    db_sz = 2 if double_buf else 1
    rx = "true" if relaxed else "false"
    vec4 = _vec4_str(dtype_str)
    bar = (
        "simdgroup_barrier(mem_flags::mem_threadgroup);"
        if NSG == 1
        else "threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    pre_bar = "" if double_buf else bar

    # Choose optimal vectorized load width so EVERY thread does equal work:
    # Each BK nibbles = BK/32 uint4s (16B) = BK/16 uint2s (8B).
    n_u4 = BN * (BK // 32)
    if TGP <= n_u4:
        U4_R = BK // 32
        if dtype_str == "half":
            unpack_body = """
            half s = scales[size_t(ng) * K_g + g_cur];
            half b = biases[size_t(ng) * K_g + g_cur];
            uint4 q = wu4[size_t(ng) * K_u4 + st * U4_R + h];
            uint32_t words[4] = {q.x, q.y, q.z, q.w};
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                uint32_t p = words[u];
                half4 v0 = half4(half(p & 0xFu), half((p >> 4) & 0xFu), half((p >> 8) & 0xFu), half((p >> 12) & 0xFu));
                half4 v1 = half4(half((p >> 16) & 0xFu), half((p >> 20) & 0xFu), half((p >> 24) & 0xFu), half((p >> 28) & 0xFu));
                dst[u * 2]     = fma(v0, s, b);
                dst[u * 2 + 1] = fma(v1, s, b);
            }
            """
        else:
            unpack_body = """
            float s = float(scales[size_t(ng) * K_g + g_cur]);
            float b = float(biases[size_t(ng) * K_g + g_cur]);
            uint4 q = wu4[size_t(ng) * K_u4 + st * U4_R + h];
            uint32_t words[4] = {q.x, q.y, q.z, q.w};
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                uint32_t p = words[u];
                float4 v0 = float4(float(p & 0xFu), float((p >> 4) & 0xFu), float((p >> 8) & 0xFu), float((p >> 12) & 0xFu));
                float4 v1 = float4(float((p >> 16) & 0xFu), float((p >> 20) & 0xFu), float((p >> 24) & 0xFu), float((p >> 28) & 0xFu));
                dst[u * 2]     = bfloat4(fma(v0, s, b));
                dst[u * 2 + 1] = bfloat4(fma(v1, s, b));
            }
            """
        deq_loop = f"""
        const device uint4* wu4 = (const device uint4*)w;
        const int K_u4 = K >> 5;
        constexpr int U4_R = {U4_R};
        for (int idx = tid; idx < BN * U4_R; idx += TGP) {{
            const int r = idx / U4_R;
            const int h = idx % U4_R;
            const int g_cur = st * GPS + (h >> 1);
            const int ng = n0 + r;
            threadgroup {vec4}* dst = (threadgroup {vec4}*)(Ws[buf] + r * BK_PAD + (h << 5));
            if (ng < N) {{
                {unpack_body}
            }} else {{
                #pragma unroll
                for (int u = 0; u < 8; ++u) dst[u] = {vec4}(0);
            }}
        }}
        """
    else:
        U2_R = BK // 16
        if dtype_str == "half":
            unpack_body = """
            half s = scales[size_t(ng) * K_g + g_cur];
            half b = biases[size_t(ng) * K_g + g_cur];
            uint2 q = wu2[size_t(ng) * K_u2 + st * U2_R + h];
            uint32_t words[2] = {q.x, q.y};
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                uint32_t p = words[u];
                half4 v0 = half4(half(p & 0xFu), half((p >> 4) & 0xFu), half((p >> 8) & 0xFu), half((p >> 12) & 0xFu));
                half4 v1 = half4(half((p >> 16) & 0xFu), half((p >> 20) & 0xFu), half((p >> 24) & 0xFu), half((p >> 28) & 0xFu));
                dst[u * 2]     = fma(v0, s, b);
                dst[u * 2 + 1] = fma(v1, s, b);
            }
            """
        else:
            unpack_body = """
            float s = float(scales[size_t(ng) * K_g + g_cur]);
            float b = float(biases[size_t(ng) * K_g + g_cur]);
            uint2 q = wu2[size_t(ng) * K_u2 + st * U2_R + h];
            uint32_t words[2] = {q.x, q.y};
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                uint32_t p = words[u];
                float4 v0 = float4(float(p & 0xFu), float((p >> 4) & 0xFu), float((p >> 8) & 0xFu), float((p >> 12) & 0xFu));
                float4 v1 = float4(float((p >> 16) & 0xFu), float((p >> 20) & 0xFu), float((p >> 24) & 0xFu), float((p >> 28) & 0xFu));
                dst[u * 2]     = bfloat4(fma(v0, s, b));
                dst[u * 2 + 1] = bfloat4(fma(v1, s, b));
            }
            """
        deq_loop = f"""
        const device uint2* wu2 = (const device uint2*)w;
        const int K_u2 = K >> 4;
        constexpr int U2_R = {U2_R};
        for (int idx = tid; idx < BN * U2_R; idx += TGP) {{
            const int r = idx / U2_R;
            const int h = idx % U2_R;
            const int g_cur = st * GPS + (h >> 2);
            const int ng = n0 + r;
            threadgroup {vec4}* dst = (threadgroup {vec4}*)(Ws[buf] + r * BK_PAD + (h << 4));
            if (ng < N) {{
                {unpack_body}
            }} else {{
                #pragma unroll
                for (int u = 0; u < 4; ++u) dst[u] = {vec4}(0);
            }}
        }}
        """

    if atomic and split_k > 1:
        store_code = """
        device atomic_float* out_a = (device atomic_float*)out;
        #pragma unroll
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i)) {
                auto ids = acc.get_multidimensional_index(i);
                const int ng = n0 + tn + int(ids[0]);
                const int mg = m0 + tm + int(ids[1]);
                if (ng < N && mg < M) {
                    atomic_fetch_add_explicit(&out_a[size_t(mg) * N + ng], acc[i], memory_order_relaxed);
                }
            }
        }
        """
    elif split_k == 1:
        store_code = f"""
        #pragma unroll
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
            if (acc.is_valid_element(i)) {{
                auto ids = acc.get_multidimensional_index(i);
                const int ng = n0 + tn + int(ids[0]);
                const int mg = m0 + tm + int(ids[1]);
                if (ng < N && mg < M) {{
                    out[size_t(mg) * N + ng] = {dtype_str}(acc[i]);
                }}
            }}
        }}
        """
    else:
        p_t = "float" if partial_fp32 else dtype_str
        cast_e = "acc[i]" if partial_fp32 else f"{dtype_str}(acc[i])"
        store_code = f"""
        device {p_t}* out_sk = out + size_t(sk) * size_t(M) * size_t(N);
        #pragma unroll
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
            if (acc.is_valid_element(i)) {{
                auto ids = acc.get_multidimensional_index(i);
                const int ng = n0 + tn + int(ids[0]);
                const int mg = m0 + tm + int(ids[1]);
                if (ng < N && mg < M) {{
                    out_sk[size_t(mg) * N + ng] = {cast_e};
                }}
            }}
        }}
        """

    buf_expr = "st & 1" if double_buf else "0"
    src = f"""
    constexpr int BM = {BM}, BN = {BN}, BK = {BK}, BK_PAD = {BK_PAD};
    constexpr int WM = {WM}, WN = {WN}, SM = {SM}, SN = {SN};
    constexpr int SPLIT_K = {split_k}, TGP = {TGP}, GPS = {GPS};

    const int M = params[0];
    const int N = params[1];
    const int K = params[2];
    const int K_g = K >> 6;
    const int n_steps = K / BK;

    const int n0 = int(threadgroup_position_in_grid.x) * BN;
    const int m0 = int(threadgroup_position_in_grid.y) * BM;
    const int sk = int(threadgroup_position_in_grid.z);
    const int tid = int(thread_index_in_threadgroup);
    const int sgid = int(simdgroup_index_in_threadgroup);
    const int tm = (sgid / WN) * SM;
    const int tn = (sgid % WN) * SN;

    const int s_per_sk = (n_steps + SPLIT_K - 1) / SPLIT_K;
    const int s_start = sk * s_per_sk;
    const int s_end = min(n_steps, s_start + s_per_sk);
    if (s_start >= s_end) return;

    threadgroup {dtype_str} Ws[{db_sz}][BN * BK_PAD];

    constexpr auto desc = matmul2d_descriptor(
        SM, SN, BK, false, true, {rx}, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;

    auto A = tensor<device {dtype_str}, dextents<int, 2>, tensor_inline>(
        const_cast<device {dtype_str}*>(x), dextents<int, 2>(K, M));
    auto B0 = tensor<threadgroup {dtype_str}, dextents<int, 2>, tensor_inline>(
        Ws[0], dextents<int, 2>(BK_PAD, BN));

    auto sA0 = A.slice(0, min(m0 + tm, max(0, M - 1)));
    auto sB0 = B0.slice<BK, SN>(0, tn);
    auto acc = op.get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), float>();
    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    }}

    const bool sg_active = (m0 + tm < M) && (n0 + tn < N);
    const bool interior_m = (m0 + tm + SM <= M);

    for (int st = s_start; st < s_end; ++st) {{
        const int buf = {buf_expr};
        {pre_bar}
        {deq_loop}
        {bar}
        if (sg_active) {{
            auto B_tg = tensor<threadgroup {dtype_str}, dextents<int, 2>, tensor_inline>(
                Ws[buf], dextents<int, 2>(BK_PAD, BN));
            auto sB = B_tg.slice<BK, SN>(0, tn);
            if (interior_m) {{
                auto sA = A.slice<BK, SM>(st * BK, m0 + tm);
                op.run(sA, sB, acc);
            }} else {{
                auto sA = A.slice(st * BK, m0 + tm);
                op.run(sA, sB, acc);
            }}
        }}
    }}

    if (sg_active) {{
        {store_code}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"nax_tgdeq_{dtype_str}_{BM}_{BN}_{BK}_{WM}x{WN}_sk{split_k}_a{int(atomic)}_r{int(relaxed)}_f{int(partial_fp32)}_d{int(double_buf)}",
        input_names=["x", "w", "scales", "biases", "params"],
        output_names=["out"],
        header=HEADER_METAL4,
        source=src,
    )


def qmm_tg_deq_nax(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    BM: int = 32,
    BN: int = 32,
    BK: int = 64,
    WM: int = 2,
    WN: int = 1,
    split_k: int = 4,
    atomic: bool = False,
    relaxed: bool = True,
    partial_fp32: Optional[bool] = None,
    double_buf: bool = False,
) -> mx.array:
    """Vectorized threadgroup-dequant Split-K Metal 4 NAX affine Q4 (gs=64) matmul."""
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1])
    M, K = x2.shape
    N = w.shape[0]
    dstr = _dtype_str(x2.dtype)
    if (K % BK) != 0:
        BK = 64
    if partial_fp32 is None:
        partial_fp32 = (x2.dtype == mx.bfloat16)
    kern = get_tg_deq_nax_kernel(
        dstr, BM, BN, BK, WM, WN, split_k, atomic, relaxed, partial_fp32, double_buf
    )
    tgp = WM * WN * 32
    grid = (((N + BN - 1) // BN) * tgp, (M + BM - 1) // BM, split_k)
    params = mx.array([M, N, K], dtype=mx.int32)

    if atomic and split_k > 1:
        y = kern(
            inputs=[x2, w, scales, biases, params],
            grid=grid,
            threadgroup=(tgp, 1, 1),
            output_shapes=[(M, N)],
            output_dtypes=[mx.float32],
            init_value=0.0,
        )[0].astype(x2.dtype)
    elif split_k == 1:
        y = kern(
            inputs=[x2, w, scales, biases, params],
            grid=grid,
            threadgroup=(tgp, 1, 1),
            output_shapes=[(M, N)],
            output_dtypes=[x2.dtype],
        )[0]
    else:
        p_dt = mx.float32 if partial_fp32 else x2.dtype
        partial = kern(
            inputs=[x2, w, scales, biases, params],
            grid=grid,
            threadgroup=(tgp, 1, 1),
            output_shapes=[(split_k, M, N)],
            output_dtypes=[p_dt],
        )[0]
        y = reduce_splitk(partial, x2.dtype)

    return y.reshape(*orig_shape[:-1], N)


# =============================================================================
# 4. Family 3: Register-Dequant Direct into `ct_b` Ablation (Zero Threadgroup Mem)
# =============================================================================

@lru_cache(maxsize=32)
def get_reg_deq_ctb_kernel(
    dtype_str: str = "half",
    BM: int = 32,
    BN: int = 32,
    WM: int = 2,
    WN: int = 1,
    split_k: int = 2,
):
    """Zero-threadgroup-memory register-dequant directly into right cooperative_tensor `ct_b`."""
    SM = BM // WM
    SN = BN // WN
    if dtype_str == "half":
        unpack_4 = """
        half s = scales[size_t(ng) * K_g + g];
        half b = biases[size_t(ng) * K_g + g];
        ct_b[i]     = fma(half(p & 0xFu), s, b);
        ct_b[i + 1] = fma(half((p >> 4) & 0xFu), s, b);
        ct_b[i + 2] = fma(half((p >> 8) & 0xFu), s, b);
        ct_b[i + 3] = fma(half((p >> 12) & 0xFu), s, b);
        """
    else:
        unpack_4 = """
        float s = float(scales[size_t(ng) * K_g + g]);
        float b = float(biases[size_t(ng) * K_g + g]);
        ct_b[i]     = bfloat(fma(float(p & 0xFu), s, b));
        ct_b[i + 1] = bfloat(fma(float((p >> 4) & 0xFu), s, b));
        ct_b[i + 2] = bfloat(fma(float((p >> 8) & 0xFu), s, b));
        ct_b[i + 3] = bfloat(fma(float((p >> 12) & 0xFu), s, b));
        """

    store_code = (
        f"""
        #pragma unroll
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
            if (acc.is_valid_element(i)) {{
                auto ids = acc.get_multidimensional_index(i);
                int ng = n0 + tn + int(ids[0]), mg = m0 + tm + int(ids[1]);
                if (ng < N && mg < M) out[size_t(mg) * N + ng] = {dtype_str}(acc[i]);
            }}
        }}
        """
        if split_k == 1
        else f"""
        device {dtype_str}* out_sk = out + size_t(sk) * size_t(M) * size_t(N);
        #pragma unroll
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
            if (acc.is_valid_element(i)) {{
                auto ids = acc.get_multidimensional_index(i);
                int ng = n0 + tn + int(ids[0]), mg = m0 + tm + int(ids[1]);
                if (ng < N && mg < M) out_sk[size_t(mg) * N + ng] = {dtype_str}(acc[i]);
            }}
        }}
        """
    )

    src = f"""
    constexpr int BM = {BM}, BN = {BN}, BK = 32;
    constexpr int WM = {WM}, WN = {WN}, SM = {SM}, SN = {SN}, SPLIT_K = {split_k};
    const int M = params[0], N = params[1], K = params[2];
    const int K_g = K >> 6, K_bytes = K >> 1;
    const int n0 = int(threadgroup_position_in_grid.x) * BN;
    const int m0 = int(threadgroup_position_in_grid.y) * BM;
    const int sk = int(threadgroup_position_in_grid.z);
    const int sgid = int(simdgroup_index_in_threadgroup);
    const int tm = (sgid / WN) * SM;
    const int tn = (sgid % WN) * SN;

    if (m0 + tm >= M || n0 + tn >= N) return;

    const int g_per_sk = (K_g + SPLIT_K - 1) / SPLIT_K;
    const int g_start = sk * g_per_sk;
    const int g_end = min(K_g, g_start + g_per_sk);

    constexpr auto desc = matmul2d_descriptor(
        SM, SN, BK, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;

    auto A = tensor<device {dtype_str}, dextents<int, 2>, tensor_inline>(
        const_cast<device {dtype_str}*>(x), dextents<int, 2>(K, M));
    auto sA0 = A.slice(0, m0 + tm);
    auto ct_b = op.get_right_input_cooperative_tensor<{dtype_str}, {dtype_str}, float>();
    auto acc = op.get_destination_cooperative_tensor<decltype(sA0), decltype(ct_b), float>();
    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    }}

    const device uint8_t* wb = (const device uint8_t*)w;
    const bool interior_m = (m0 + tm + SM <= M);

    for (int g = g_start; g < g_end; ++g) {{
        #pragma unroll
        for (int half_g = 0; half_g < 2; ++half_g) {{
            const int k_base = (g << 6) + (half_g << 5);
            #pragma unroll
            for (uint16_t i = 0; i < ct_b.get_capacity(); i += 4) {{
                auto c = ct_b.get_multidimensional_index(i);
                const int ng = n0 + tn + int(c[1]);
                const int kg = k_base + int(c[0]);
                if (ng < N) {{
                    uint16_t p = *(const device uint16_t*)(wb + size_t(ng) * K_bytes + (kg >> 1));
                    {unpack_4}
                }} else {{
                    ct_b[i] = {dtype_str}(0); ct_b[i + 1] = {dtype_str}(0);
                    ct_b[i + 2] = {dtype_str}(0); ct_b[i + 3] = {dtype_str}(0);
                }}
            }}
            if (interior_m) {{
                auto sA = A.slice<32, SM>(k_base, m0 + tm);
                op.run(sA, ct_b, acc);
            }} else {{
                auto sA = A.slice(k_base, m0 + tm);
                op.run(sA, ct_b, acc);
            }}
        }}
    }}
    {store_code}
    """
    return mx.fast.metal_kernel(
        name=f"nax_regctb_{dtype_str}_{BM}_{BN}_{WM}x{WN}_sk{split_k}",
        input_names=["x", "w", "scales", "biases", "params"],
        output_names=["out"],
        header=HEADER_METAL4,
        source=src,
    )


def qmm_reg_deq_ctb(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    BM: int = 32,
    BN: int = 32,
    WM: int = 2,
    WN: int = 1,
    split_k: int = 2,
) -> mx.array:
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1])
    M, K = x2.shape
    N = w.shape[0]
    dstr = _dtype_str(x2.dtype)
    kern = get_reg_deq_ctb_kernel(dstr, BM, BN, WM, WN, split_k)
    tgp = WM * WN * 32
    grid = (((N + BN - 1) // BN) * tgp, (M + BM - 1) // BM, split_k)
    params = mx.array([M, N, K], dtype=mx.int32)
    oshape = (M, N) if split_k == 1 else (split_k, M, N)
    y = kern(
        inputs=[x2, w, scales, biases, params],
        grid=grid,
        threadgroup=(tgp, 1, 1),
        output_shapes=[oshape],
        output_dtypes=[x2.dtype],
    )[0]
    if split_k > 1:
        y = reduce_splitk(y, x2.dtype)
    return y.reshape(*orig_shape[:-1], N)


# =============================================================================
# 5. Family 4: Hardware Integer NAX (`int8_t x int4b_format -> int32_t`,
#    `int8_t x int8_t -> int32_t`) + Grouped-Q4 Dynamic INT8xINT4 Kernel
# =============================================================================

@lru_cache(maxsize=16)
def get_int8_int4_exact_kernel(BM: int = 64, BN: int = 64, BK: int = 64, NSG: int = 4):
    """Exact hardware `int8_t x int4b_format -> int32_t` GEMM (`multiply_accumulate`, 0 error)."""
    scope = "execution_simdgroup" if NSG == 1 else f"execution_simdgroups<{NSG}>"
    src = f"""
    constexpr int BM = {BM}, BN = {BN}, BK = {BK};
    const int M = params[0], N = params[1], K = params[2];
    const int n0 = int(threadgroup_position_in_grid.x) * BN;
    const int m0 = int(threadgroup_position_in_grid.y) * BM;

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, {scope}> op;

    auto A = tensor<device int8_t, dextents<int, 2>, tensor_inline>(
        const_cast<device int8_t*>(a), dextents<int, 2>(K, M));
    auto B = tensor<device int4b_format, dextents<int, 2>, tensor_inline>(
        const_cast<device uint8_t*>((const device uint8_t*)b), dextents<int, 2>(K, N));
    auto C = tensor<device int32_t, dextents<int, 2>, tensor_inline>(
        out, dextents<int, 2>(N, M));

    auto sA0 = A.slice(0, m0);
    auto sB0 = B.slice(0, n0);
    auto acc = op.get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), int32_t>();
    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
        if (acc.is_valid_element(i)) acc[i] = 0;
    }}

    const bool interior = (m0 + BM <= M) && (n0 + BN <= N);
    if (interior) {{
        for (int k = 0; k < K; k += BK) {{
            auto sA = A.slice<BK, BM>(k, m0);
            auto sB = B.slice<BK, BN>(k, n0);
            op.run(sA, sB, acc);
        }}
    }} else {{
        for (int k = 0; k < K; k += BK) {{
            auto sA = A.slice(k, m0);
            auto sB = B.slice(k, n0);
            op.run(sA, sB, acc);
        }}
    }}
    auto sC = C.slice(n0, m0);
    acc.store(sC);
    """
    return mx.fast.metal_kernel(
        name=f"nax_i8i4_exact_{BM}_{BN}_{BK}_sg{NSG}",
        input_names=["a", "b", "params"],
        output_names=["out"],
        header=HEADER_METAL4,
        source=src,
    )


@lru_cache(maxsize=16)
def get_int8_int8_exact_kernel(BM: int = 64, BN: int = 64, BK: int = 64, NSG: int = 4):
    """Exact hardware `int8_t x int8_t -> int32_t` GEMM (`multiply_accumulate`, 0 error)."""
    scope = "execution_simdgroup" if NSG == 1 else f"execution_simdgroups<{NSG}>"
    src = f"""
    constexpr int BM = {BM}, BN = {BN}, BK = {BK};
    const int M = params[0], N = params[1], K = params[2];
    const int n0 = int(threadgroup_position_in_grid.x) * BN;
    const int m0 = int(threadgroup_position_in_grid.y) * BM;

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, {scope}> op;

    auto A = tensor<device int8_t, dextents<int, 2>, tensor_inline>(
        const_cast<device int8_t*>(a), dextents<int, 2>(K, M));
    auto B = tensor<device int8_t, dextents<int, 2>, tensor_inline>(
        const_cast<device int8_t*>(b), dextents<int, 2>(K, N));
    auto C = tensor<device int32_t, dextents<int, 2>, tensor_inline>(
        out, dextents<int, 2>(N, M));

    auto sA0 = A.slice(0, m0);
    auto sB0 = B.slice(0, n0);
    auto acc = op.get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), int32_t>();
    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
        if (acc.is_valid_element(i)) acc[i] = 0;
    }}

    const bool interior = (m0 + BM <= M) && (n0 + BN <= N);
    if (interior) {{
        for (int k = 0; k < K; k += BK) {{
            auto sA = A.slice<BK, BM>(k, m0);
            auto sB = B.slice<BK, BN>(k, n0);
            op.run(sA, sB, acc);
        }}
    }} else {{
        for (int k = 0; k < K; k += BK) {{
            auto sA = A.slice(k, m0);
            auto sB = B.slice(k, n0);
            op.run(sA, sB, acc);
        }}
    }}
    auto sC = C.slice(n0, m0);
    acc.store(sC);
    """
    return mx.fast.metal_kernel(
        name=f"nax_i8i8_exact_{BM}_{BN}_{BK}_sg{NSG}",
        input_names=["a", "b", "params"],
        output_names=["out"],
        header=HEADER_METAL4,
        source=src,
    )


def nax_int8_int4_exact(
    a_i8: mx.array,
    b_i4_u32: mx.array,
    BM: int = 64,
    BN: int = 64,
    BK: int = 64,
    NSG: int = 4,
) -> mx.array:
    """Exact hardware `int8_t (M, K) x int4b_format (N, K)^T -> int32_t (M, N)`."""
    M, K = a_i8.shape
    N = b_i4_u32.shape[0]
    kern = get_int8_int4_exact_kernel(BM, BN, BK, NSG)
    tgp = NSG * 32
    grid = (((N + BN - 1) // BN) * tgp, (M + BM - 1) // BM, 1)
    params = mx.array([M, N, K], dtype=mx.int32)
    return kern(
        inputs=[a_i8, b_i4_u32, params],
        grid=grid,
        threadgroup=(tgp, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.int32],
    )[0]


def nax_int8_int8_exact(
    a_i8: mx.array,
    b_i8: mx.array,
    BM: int = 64,
    BN: int = 64,
    BK: int = 64,
    NSG: int = 4,
) -> mx.array:
    """Exact hardware `int8_t (M, K) x int8_t (N, K)^T -> int32_t (M, N)`."""
    M, K = a_i8.shape
    N = b_i8.shape[0]
    kern = get_int8_int8_exact_kernel(BM, BN, BK, NSG)
    tgp = NSG * 32
    grid = (((N + BN - 1) // BN) * tgp, (M + BM - 1) // BM, 1)
    params = mx.array([M, N, K], dtype=mx.int32)
    return kern(
        inputs=[a_i8, b_i8, params],
        grid=grid,
        threadgroup=(tgp, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.int32],
    )[0]


def convert_mlx_u4_to_i4(
    w_u32: mx.array, scales: mx.array, biases: mx.array
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Convert MLX affine uint4 (u in [0..15]) to signed int4b_format (q = u - 8 in [-8..7]):
      - w_i4 = w_u32 ^ 0x88888888 (exact 4-bit two's complement shift by -8!)
      - b_shifted = biases + 8 * scales (so s*u + b == s*q + b_shifted identically!)
    """
    w_i4 = mx.bitwise_xor(w_u32, mx.array(0x88888888, dtype=mx.uint32))
    b_shifted = (biases.astype(mx.float32) + 8.0 * scales.astype(mx.float32)).astype(scales.dtype)
    return w_i4, scales, b_shifted


@lru_cache(maxsize=8)
def get_quantize_x_i8_group_kernel(dtype_str: str):
    """
    Fused dynamic per-group (K=64) symmetric INT8 activation quantizer + x_gsum:
    For each (m, g):
      s_x[m, g] = max(|x[m, 64g..64g+63]|) / 127.0f
      q_x[m, 64g+k] = int8_t(rint(x[m, 64g+k] / s_x[m, g]))
      x_gsum[m, g] = sum(x[m, 64g..64g+63])
    """
    vec4 = _vec4_str(dtype_str)
    src = f"""
    const int M = params[0];
    const int K = params[1];
    const int K_g = K >> 6;
    const int idx = int(thread_position_in_grid.x);
    if (idx < M * K_g) {{
        const int m = idx / K_g;
        const int g = idx % K_g;
        const device {vec4}* xp = (const device {vec4}*)(x + size_t(m) * K + (g << 6));
        float vals[64];
        float maxabs = 0.0f;
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {{
            float4 v = float4(xp[i]);
            vals[i * 4 + 0] = v.x;
            vals[i * 4 + 1] = v.y;
            vals[i * 4 + 2] = v.z;
            vals[i * 4 + 3] = v.w;
            float4 av = abs(v);
            maxabs = max(maxabs, max(max(av.x, av.y), max(av.z, av.w)));
            sum += (v.x + v.y) + (v.z + v.w);
        }}
        const float sx = (maxabs > 0.0f) ? (maxabs / 127.0f) : 0.0f;
        const float inv_sx = (maxabs > 0.0f) ? (127.0f / maxabs) : 0.0f;
        x_scale[idx] = sx;
        x_gsum[idx] = sum;
        device char4* qp = (device char4*)(q_x + size_t(m) * K + (g << 6));
        #pragma unroll
        for (int i = 0; i < 16; ++i) {{
            float4 v = float4(vals[i*4], vals[i*4+1], vals[i*4+2], vals[i*4+3]) * inv_sx;
            int4 q = int4(clamp(rint(v), -127.0f, 127.0f));
            qp[i] = char4(q);
        }}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"nax_qx_i8g_{dtype_str}",
        input_names=["x", "params"],
        output_names=["q_x", "x_scale", "x_gsum"],
        header=HEADER_METAL4,
        source=src,
    )


@lru_cache(maxsize=64)
def get_grouped_q4_i8i4_kernel(
    dtype_str: str = "half",
    BM: int = 32,
    BN: int = 32,
    WM: int = 2,
    WN: int = 1,
    split_k: int = 4,
):
    """
    Grouped-Q4 (gs=64) Split-K Metal 4 NAX using direct hardware
    `int8_t x int4b_format -> int32_t` (`matmul2d`, `mode::multiply`) + register rescaling!
    """
    SM = BM // WM
    SN = BN // WN
    assert BM % WM == 0 and BN % WN == 0
    use_fast_16x32 = (SM == 16 and SN == 32)

    if use_fast_16x32:
        interior_loop = """
        const auto c0 = acc.get_multidimensional_index(uint16_t(0));
        const auto c4 = acc.get_multidimensional_index(uint16_t(4));
        const auto c8 = acc.get_multidimensional_index(uint16_t(8));
        const int nc0 = n_sg + int(c0[0]);
        const int nc1 = n_sg + int(c8[0]);
        const int mr0 = m_sg + int(c0[1]);
        const int mr1 = m_sg + int(c4[1]);
        const size_t xoff0 = size_t(mr0) * K_g;
        const size_t xoff1 = size_t(mr1) * K_g;

        for (int g = g_start; g < g_end; ++g) {
            auto sA = A.slice<64, SM>(g << 6, m_sg);
            auto sB = B.slice<64, SN>(g << 6, n_sg);
            op.run(sA, sB, tile_g);

            const float sx0 = x_scale[xoff0 + g];
            const float sx1 = x_scale[xoff1 + g];
            const float xm0 = x_gsum[xoff0 + g];
            const float xm1 = x_gsum[xoff1 + g];

            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const size_t soff = size_t(nc0 + q) * K_g + g;
                const float sw = float(scales[soff]);
                const float bw = float(b_shifted[soff]);
                acc[q]     = fma(sw * sx0, float(tile_g[q]),     fma(bw, xm0, acc[q]));
                acc[4 + q] = fma(sw * sx1, float(tile_g[4 + q]), fma(bw, xm1, acc[4 + q]));
            }
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const size_t soff = size_t(nc1 + q) * K_g + g;
                const float sw = float(scales[soff]);
                const float bw = float(b_shifted[soff]);
                acc[8 + q]  = fma(sw * sx0, float(tile_g[8 + q]),  fma(bw, xm0, acc[8 + q]));
                acc[12 + q] = fma(sw * sx1, float(tile_g[12 + q]), fma(bw, xm1, acc[12 + q]));
            }
        }
        """
    else:
        interior_loop = """
        for (int g = g_start; g < g_end; ++g) {
            auto sA = A.slice<64, SM>(g << 6, m_sg);
            auto sB = B.slice<64, SN>(g << 6, n_sg);
            op.run(sA, sB, tile_g);
            #pragma unroll
            for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
                if (acc.is_valid_element(i)) {
                    auto ids = acc.get_multidimensional_index(i);
                    const int ng = n_sg + int(ids[0]);
                    const int mg = m_sg + int(ids[1]);
                    const size_t soff = size_t(ng) * K_g + g;
                    const size_t xoff = size_t(mg) * K_g + g;
                    acc[i] = fma(float(scales[soff]) * x_scale[xoff], float(tile_g[i]),
                                 fma(float(b_shifted[soff]), x_gsum[xoff], acc[i]));
                }
            }
        }
        """

    store_code = (
        f"""
        #pragma unroll
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
            if (acc.is_valid_element(i)) {{
                auto ids = acc.get_multidimensional_index(i);
                const int ng = n_sg + int(ids[0]);
                const int mg = m_sg + int(ids[1]);
                if (ng < N && mg < M) out[size_t(mg) * N + ng] = {dtype_str}(acc[i]);
            }}
        }}
        """
        if split_k == 1
        else f"""
        device {dtype_str}* out_sk = out + size_t(sk) * size_t(M) * size_t(N);
        #pragma unroll
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
            if (acc.is_valid_element(i)) {{
                auto ids = acc.get_multidimensional_index(i);
                const int ng = n_sg + int(ids[0]);
                const int mg = m_sg + int(ids[1]);
                if (ng < N && mg < M) out_sk[size_t(mg) * N + ng] = {dtype_str}(acc[i]);
            }}
        }}
        """
    )

    src = f"""
    constexpr int BM = {BM}, BN = {BN}, BK = 64;
    constexpr int WM = {WM}, WN = {WN}, SM = {SM}, SN = {SN}, SPLIT_K = {split_k};

    const int M = params[0], N = params[1], K = params[2];
    const int K_g = K >> 6;
    const int n0 = int(threadgroup_position_in_grid.x) * BN;
    const int m0 = int(threadgroup_position_in_grid.y) * BM;
    const int sk = int(threadgroup_position_in_grid.z);
    const int sgid = int(simdgroup_index_in_threadgroup);
    const int m_sg = m0 + (sgid / WN) * SM;
    const int n_sg = n0 + (sgid % WN) * SN;

    if (m_sg >= M || n_sg >= N) return;

    const int g_per_sk = (K_g + SPLIT_K - 1) / SPLIT_K;
    const int g_start = sk * g_per_sk;
    const int g_end = min(K_g, g_start + g_per_sk);
    if (g_start >= g_end) return;

    constexpr auto desc = matmul2d_descriptor(
        SM, SN, BK, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;

    auto A = tensor<device int8_t, dextents<int, 2>, tensor_inline>(
        const_cast<device int8_t*>(q_x), dextents<int, 2>(K, M));
    auto B = tensor<device int4b_format, dextents<int, 2>, tensor_inline>(
        const_cast<device uint8_t*>((const device uint8_t*)w_i4), dextents<int, 2>(K, N));

    auto sA0 = A.slice(0, m_sg);
    auto sB0 = B.slice(0, n_sg);
    auto tile_g = op.get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), int32_t>();
    auto acc = op.get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), float>();
    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    }}

    const bool interior = (m_sg + SM <= M) && (n_sg + SN <= N);
    if (interior) {{
        {interior_loop}
    }} else {{
        for (int g = g_start; g < g_end; ++g) {{
            auto sA = A.slice(g << 6, m_sg);
            auto sB = B.slice(g << 6, n_sg);
            op.run(sA, sB, tile_g);
            #pragma unroll
            for (uint16_t i = 0; i < acc.get_capacity(); ++i) {{
                if (acc.is_valid_element(i)) {{
                    auto ids = acc.get_multidimensional_index(i);
                    const int ng = n_sg + int(ids[0]);
                    const int mg = m_sg + int(ids[1]);
                    if (ng < N && mg < M) {{
                        const size_t soff = size_t(ng) * K_g + g;
                        const size_t xoff = size_t(mg) * K_g + g;
                        acc[i] = fma(float(scales[soff]) * x_scale[xoff], float(tile_g[i]),
                                     fma(float(b_shifted[soff]), x_gsum[xoff], acc[i]));
                    }}
                }}
            }}
        }}
    }}
    {store_code}
    """
    return mx.fast.metal_kernel(
        name=f"nax_gq4_i8i4_{dtype_str}_{BM}_{BN}_{WM}x{WN}_sk{split_k}",
        input_names=["q_x", "w_i4", "scales", "b_shifted", "x_scale", "x_gsum", "params"],
        output_names=["out"],
        header=HEADER_METAL4,
        source=src,
    )


def qmm_grouped_i8i4(
    x: mx.array,
    w_i4: mx.array,
    scales: mx.array,
    b_shifted: mx.array,
    BM: int = 32,
    BN: int = 32,
    WM: int = 2,
    WN: int = 1,
    split_k: int = 4,
) -> mx.array:
    """Dynamic per-group INT8 activation quant + hardware `int8_t x int4b_format -> int32_t` Q4 matmul."""
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1])
    M, K = x2.shape
    N = w_i4.shape[0]
    dstr = _dtype_str(x2.dtype)

    kq = get_quantize_x_i8_group_kernel(dstr)
    n_tot = M * (K >> 6)
    p_q = mx.array([M, K], dtype=mx.int32)
    q_x, x_scale, x_gsum = kq(
        inputs=[x2, p_q],
        grid=(n_tot, 1, 1),
        threadgroup=(min(256, max(32, n_tot)), 1, 1),
        output_shapes=[(M, K), (M, K >> 6), (M, K >> 6)],
        output_dtypes=[mx.int8, mx.float32, mx.float32],
    )

    kern = get_grouped_q4_i8i4_kernel(dstr, BM, BN, WM, WN, split_k)
    tgp = WM * WN * 32
    grid = (((N + BN - 1) // BN) * tgp, (M + BM - 1) // BM, split_k)
    params = mx.array([M, N, K], dtype=mx.int32)
    oshape = (M, N) if split_k == 1 else (split_k, M, N)
    y = kern(
        inputs=[q_x, w_i4, scales, b_shifted, x_scale, x_gsum, params],
        grid=grid,
        threadgroup=(tgp, 1, 1),
        output_shapes=[oshape],
        output_dtypes=[x2.dtype],
    )[0]
    if split_k > 1:
        y = reduce_splitk(y, x2.dtype)
    return y.reshape(*orig_shape[:-1], N)


# =============================================================================
# 6. Family 5: Zero-Code Stock MLX Workarounds
# =============================================================================

def qmm_pad65(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    """Zero-code MLX 0.31.2 workaround: pad M <= 64 to M=65 to trigger NAX `qmm_t_nax`."""
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1])
    M = x2.shape[0]
    if M <= 64:
        x_pad = mx.pad(x2, [(0, 65 - M), (0, 0)])
        y_pad = mx.quantized_matmul(
            x_pad, w, scales=scales, biases=biases,
            transpose=True, group_size=group_size, bits=bits, mode="affine",
        )
        y = y_pad[:M]
    else:
        y = mx.quantized_matmul(
            x2, w, scales=scales, biases=biases,
            transpose=True, group_size=group_size, bits=bits, mode="affine",
        )
    return y.reshape(*orig_shape[:-1], w.shape[0])


def qmm_reshape_batch(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    splits: int = 4,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    """Fable 5.2's batch-reshape diagnostic workaround: x[M, K] -> x[splits, M//splits, K]."""
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1])
    M, K = x2.shape
    if M % splits == 0 and M >= splits:
        xb = x2.reshape(splits, M // splits, K)
        yb = mx.quantized_matmul(
            xb, w, scales=scales, biases=biases,
            transpose=True, group_size=group_size, bits=bits, mode="affine",
        )
        y = yb.reshape(M, w.shape[0])
    else:
        y = mx.quantized_matmul(
            x2, w, scales=scales, biases=biases,
            transpose=True, group_size=group_size, bits=bits, mode="affine",
        )
    return y.reshape(*orig_shape[:-1], w.shape[0])


def qmm_stacked_gate_up(
    x: mx.array,
    w_cat: mx.array,
    scales_cat: mx.array,
    biases_cat: mx.array,
    split_n: int,
    backend: str = "stock",
) -> Tuple[mx.array, mx.array]:
    """Run concatenated [gate_proj; up_proj] (2*N, K) in one matmul and split along N."""
    if backend == "stock":
        y = mx.quantized_matmul(
            x, w_cat, scales=scales_cat, biases=biases_cat,
            transpose=True, group_size=64, bits=4, mode="affine",
        )
    elif backend == "pad65":
        y = qmm_pad65(x, w_cat, scales_cat, biases_cat)
    else:
        y = fast_qmm_nax(x, w_cat, scales_cat, biases_cat, backend=backend)
    return y[..., :split_n], y[..., split_n:]


# =============================================================================
# 7. Auto-Tuned Shape-Aware Dispatcher (M5 Max 40-Core NAX Schedule)
# =============================================================================

def pick_nax_schedule(M: int, N: int, K: int, backend: str = "auto"):
    """
    Empirically tuned shape-aware Metal 4 NAX schedule for Apple M5 Max (40 GPU cores).

    Measured silicon laws:
    1. Keep BM >= M (for M <= 64) so grid.y == 1 and weights are read ONCE from DRAM.
    2. Use native NAX per-simdgroup tile SM=16 (SM=8 when M<=8), SN=32.
    3. For N=4096, K=4096:
         - M<=16: BN=16/32, BK=64, sk=4 (1.30 ms)
         - M=32:  BN=32, BK=64, W=2x1, sk=4 (1.47 ms!)
         - M=64:  BN=32, BK=64, W=4x1, sk=2 (2.02 ms! vs 3.01 ms at sk=4)
    4. For wide-K down_proj (N=4096, K=12288):
         - M=32:  BN=64, BK=64, W=2x2, sk=2 (2.03 ms, 1.91x!)
         - M=64:  BN=64, BK=128, W=4x2, sk=1 (2.71 ms, 2.62x — zero reduce!)
    5. For wide-N gate/up/lm_head (N >= 12288, K=4096):
         - M<=32: BN=64, BK=128, W=WMx2, sk=2 (beats stock MLX by 1.17x–1.33x!)
         - M=64:  BN=64, BK=128, W=4x2, sk=1 (zero reduce; beats stock on 12288!)
         - Note: at M=64 on huge vocab N >= 65536 (lm_head), stock MLX already runs fast NAX,
           so `auto` / `tg_deq` delegates M=64 & N >= 65536 to stock MLX!
    """
    variant = "tg_deq" if backend == "auto" else backend

    if M <= 8:
        BM, WM = 8, 1
    elif M <= 16:
        BM, WM = 16, 1
    elif M <= 32:
        BM, WM = 32, 2
    else:
        BM, WM = 64, 4

    BK = 64
    double_buf = False

    if variant == "u4_direct":
        BN, WN = 32, 1
        if N >= 24576:
            sk = 1
        elif N >= 12288:
            sk = 2 if M <= 32 else 1
        elif K >= 12288:
            sk = 4 if M <= 32 else 2
        else:
            sk = 4 if M <= 32 else 2
        return variant, BM, BN, BK, WM, WN, sk, False, False

    # tg_deq optimal schedule
    if N >= 12288:
        BN, WN = 64, 2
        BK = 128 if (K % 128 == 0 and M >= 16) else 64
        sk = 2 if M <= 32 else 1
    elif K >= 12288:
        # Wide-K down_proj (e.g. 4096 x 12288): BN=64 W=WMx2 crushes!
        if M <= 16:
            BN, WN, BK, sk = 32, 1, 64, 4
        elif M <= 32:
            BN, WN, BK, sk = 64, 2, 64, 2
        else:
            BN, WN, BK, sk = 64, 2, 128, 1
    elif N >= 4096:
        if M <= 16:
            BN, WN, BK, sk = 32, 1, 64, 4
        elif M <= 32:
            BN, WN, BK, sk = 32, 1, 64, 4
        else:
            BN, WN, BK, sk = 32, 1, 64, 2
    else:
        # Small-N projections (e.g., k_proj/v_proj 1024x4096)
        BN, WN, BK = 32, 1, 64
        sk = 4 if K >= 4096 else 2

    return variant, BM, BN, BK, WM, WN, sk, False, double_buf


def fast_qmm_nax(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    backend: str = "auto",
    x_gsum: Optional[mx.array] = None,
) -> mx.array:
    """
    Drop-in replacement for `mx.quantized_matmul(..., transpose=True, group_size=64, bits=4, mode='affine')`.
    Automatically falls back to stock MLX for unsupported shapes/dtypes or M outside [8, 64] when backend=='auto'.
    """
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1])
    M, K = x2.shape
    N = w.shape[0]

    if (
        x2.dtype not in (mx.float16, mx.bfloat16)
        or (K & 63) != 0
        or N < 64
        or (backend == "auto" and (M < 8 or M > 64))
        or (backend in ("auto", "tg_deq", "u4_direct") and M >= 64 and N >= 65536)
    ):
        return mx.quantized_matmul(
            x, w, scales=scales, biases=biases,
            transpose=True, group_size=64, bits=4, mode="affine",
        )

    variant, BM, BN, BK, WM, WN, sk, atomic, double_buf = pick_nax_schedule(M, N, K, backend=backend)
    if variant == "u4_direct":
        y = qmm_u4_direct(x2, w, scales, biases, BM=BM, BN=BN, WM=WM, WN=WN, split_k=sk, atomic=atomic, x_gsum=x_gsum)
    elif variant == "tg_deq":
        y = qmm_tg_deq_nax(x2, w, scales, biases, BM=BM, BN=BN, BK=BK, WM=WM, WN=WN, split_k=sk, atomic=atomic, double_buf=double_buf)
    elif variant == "pad65":
        y = qmm_pad65(x2, w, scales, biases)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return y.reshape(*orig_shape[:-1], N)
