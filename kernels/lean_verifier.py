"""Tiny Trusted Verifier (<100 lines, Alpoge/Lean-style) for synthesized Metal Q4 kernels.

Audits any untrusted Metal kernel before benchmarking via three mandatory gates:
  (a) Fable 5.2 exact linear-basis + small-integer unit-scale probe (0.0 exact for linear ops),
  (b) Random + heavy-tailed outlier stress test vs FP32 dequantized reference (0 NaN/Inf),
  (c) Fresh-buffer anti-cheat check (poisoned init_value=1337.0 + non-zero coverage + non-aliasing).
"""
from __future__ import annotations
import math
from typing import Callable, Dict, Tuple
import mlx.core as mx

def _pack_u4(q_u8: mx.array) -> mx.array:
    """Pack [..., K] uint8 nibbles (0..15) into MLX uint32 [..., K//8] (low nibble first)."""
    q = q_u8.astype(mx.uint32).reshape(*q_u8.shape[:-1], -1, 8)
    shifts = mx.arange(0, 32, 4, dtype=mx.uint32)
    return mx.sum(q << shifts, axis=-1).astype(mx.uint32)

def _check_finite(y: mx.array) -> bool:
    return bool(mx.all(mx.isfinite(y)).item())

def _make_outliers(shape: Tuple[int, ...], dtype: mx.Dtype, seed: int) -> mx.array:
    mx.random.seed(seed)
    x = mx.random.normal(shape, dtype=mx.float32) * 0.5
    mask = mx.random.uniform(shape=shape) < 0.02
    spikes = mx.where(mx.random.uniform(shape=shape) < 0.5, 25.0, -25.0)
    return mx.where(mask, x + spikes, x).astype(dtype)

def _deq_fp32(w: mx.array, s: mx.array, b: mx.array, gs: int = 64) -> mx.array:
    return mx.dequantize(w, s, b, group_size=gs, bits=4, mode="affine").astype(mx.float32)

def verify_qmm(
    fn: Callable, M: int, N: int, K: int, dtype: mx.Dtype = mx.bfloat16,
    gs: int = 64, tol: float = 2e-2, seed: int = 42,
) -> Tuple[bool, Dict]:
    """Verify linear Q4 matmul candidate fn(x, w, s, b, init_value=...) -> y[M, N]."""
    mx.random.seed(seed)
    # (a) Exact linear-basis + small-integer unit-scale test (exact in bf16/fp16: sums <= 64)
    q_int = mx.random.randint(0, 16, shape=(N, K), dtype=mx.int32)
    w_u = _pack_u4(q_int.astype(mx.uint8))
    s_u = mx.ones((N, K // gs), dtype=dtype)
    b_0 = mx.zeros((N, K // gs), dtype=dtype)
    b_2 = mx.full((N, K // gs), 2.0, dtype=dtype)
    x_basis = mx.zeros((M, K), dtype=dtype)
    for m in range(M):
        x_basis[m, (m * 73) % K] = 1.0  # one-hot basis probe
        x_basis[m, (m * 73 + 17) % K] = 2.0  # small-integer linear combo (max sum 3*15=45)
    y_b0 = fn(x_basis, w_u, s_u, b_0, init_value=1337.0)
    y_b2 = fn(x_basis, w_u, s_u, b_2, init_value=1337.0)
    mx.eval(y_b0, y_b2)
    ref_b0 = (x_basis.astype(mx.float32) @ q_int.astype(mx.float32).T)
    ref_b2 = ref_b0 + (x_basis.astype(mx.float32).sum(axis=-1, keepdims=True) * 2.0)
    if not (_check_finite(y_b0) and _check_finite(y_b2)):
        return False, {"fail": "non_finite_basis"}
    err_b0 = float(mx.max(mx.abs(y_b0.astype(mx.float32) - ref_b0)).item())
    err_b2 = float(mx.max(mx.abs(y_b2.astype(mx.float32) - ref_b2)).item())
    if err_b0 != 0.0 or err_b2 != 0.0:
        return False, {"fail": "basis_exact", "err_b0": err_b0, "err_b2": err_b2}
    # (b) Random + heavy-tailed outlier test vs FP32 golden reference
    wf = mx.random.normal((N, K), dtype=mx.float32) * (1.0 / math.sqrt(K))
    w, s, b = mx.quantize(wf.astype(dtype), group_size=gs, bits=4, mode="affine")
    x = _make_outliers((M, K), dtype, seed + 1)
    y = fn(x, w, s, b, init_value=1337.0)
    y_ref = x.astype(mx.float32) @ _deq_fp32(w, s, b, gs).T
    mx.eval(y, y_ref)
    if not _check_finite(y) or tuple(y.shape) != (M, N):
        return False, {"fail": "shape_or_finite"}
    max_abs = float(mx.max(mx.abs(y.astype(mx.float32) - y_ref)).item())
    ref_max = float(mx.max(mx.abs(y_ref)).item()) + 1e-6
    rel = max_abs / ref_max
    # (c) Fresh-buffer anti-cheat: poisoned 1337.0 overwritten & non-aliasing across distinct inputs
    poison_hit = bool(mx.any(y == 1337.0).item()) or bool(mx.any(y_b0 == 1337.0).item())
    y_neg = fn(-x, w, s, b, init_value=1337.0)
    mx.eval(y_neg)
    err_neg = float(mx.max(mx.abs(y_neg.astype(mx.float32) + y_ref)).item())
    rel_neg = err_neg / ref_max
    ok = (not poison_hit) and (max_abs <= tol or rel <= 1.2e-2) and (err_neg <= tol or rel_neg <= 1.2e-2)
    return ok, {"basis_err": max(err_b0, err_b2), "max_abs_err": max_abs, "rel_err": rel, "poison_hit": poison_hit}

def verify_fused_op(
    fn: Callable, ref_fn: Callable, inputs_basis: tuple, inputs_rand: tuple,
    tol_abs: float = 5e-2, tol_rel: float = 2e-2,
) -> Tuple[bool, Dict]:
    """Verify nonlinear/MoE fused kernels (SwiGLU, SwitchGLU) on basis + outlier + poison gates."""
    yb = fn(*inputs_basis, init_value=1337.0)
    rb = ref_fn(*inputs_basis)
    yr = fn(*inputs_rand, init_value=1337.0)
    rr = ref_fn(*inputs_rand)
    mx.eval(yb, rb, yr, rr)
    if not (_check_finite(yb) and _check_finite(yr)) or tuple(yr.shape) != tuple(rr.shape):
        return False, {"fail": "shape_or_finite"}
    eb = float(mx.max(mx.abs(yb.astype(mx.float32) - rb.astype(mx.float32))).item())
    er = float(mx.max(mx.abs(yr.astype(mx.float32) - rr.astype(mx.float32))).item())
    rel_b = eb / (float(mx.max(mx.abs(rb)).item()) + 1e-6)
    rel_r = er / (float(mx.max(mx.abs(rr)).item()) + 1e-6)
    poison_hit = bool(mx.any(yr == 1337.0).item()) or bool(mx.any(yb == 1337.0).item())
    ok = (not poison_hit) and (eb <= tol_abs or rel_b <= tol_rel) and (er <= tol_abs or rel_r <= tol_rel)
    return ok, {"basis_err": eb, "max_abs_err": er, "rel_err": rel_r, "poison_hit": poison_hit}
