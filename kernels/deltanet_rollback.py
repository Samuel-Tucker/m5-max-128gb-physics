#!/usr/bin/env python3
"""
Exact Hybrid Gated DeltaNet + KVCache Checkpoint & Zero-Matmul Replay Engine.

Solves Track D's core architectural blocker on Apple M5 Max:
  - Qwen3.5-9B-4bit (24/32 layers) and Qwen3.6-35B-A3B-4bit (30/40 layers) are
    Gated DeltaNet hybrid models whose linear_attn layers use ArraysCache(size=2)
    holding (conv_state, S_state).
  - Stock mlx_lm speculative decoding fails because ArraysCache cannot be trimmed.

Key mechanism:
  1. Zero-copy checkpointing: MLX arrays are immutable, so saving references to
     (conv_state, S_state) before a V=(K+1)-token verify forward pass costs
     0.000 ms and 0 bytes copied. KVCache layers trim in O(1) via offset decrement.
  2. Factor capture during verify: Each GatedDeltaNet layer retains references to
     (conv_input, k, v, g, beta) already produced during verify (no extra matmuls
     or DRAM writes).
  3. Zero-matmul exact replay on rejection at depth a in [1..K]:
       - conv_state = mx.contiguous(conv_input[:, a : a + n_keep, :])
       - S_state    = gated_delta_replay_state_only(k, v, g, beta, prev_S, a)
         using a state-only Metal kernel (no q loads, no y output, no matmuls!).
       - Full acceptance (a == V) is a 0.000 ms no-op; a == 0 restores prev_S in
         0.000 ms.
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel, gated_delta_ops


def _make_gated_delta_replay_kernel(vectorized: bool = False):
    """State-only Metal kernel for exact Gated DeltaNet recurrent rollback.

    Replays ONLY t in [0, T_replay) to compute state_out from state_in, without
    loading q, computing attention output y, or allocating an output buffer.
    Accepts unsliced contiguous verify tensors via V_stride so zero slice copies
    are needed before launch.
    """
    if not mx.metal.is_available():
        return None

    if vectorized:
        g_setup = "auto g_ = g + (b_idx * V_stride * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_setup = "auto g_ = g + b_idx * V_stride * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // k: [B, V_stride, Hk, Dk], v: [B, V_stride, Hv, Dv]
        auto k_ = k + b_idx * V_stride * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * V_stride * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(i_state[s_idx]);
        }}

        {g_setup}
        auto beta_ = beta + b_idx * V_stride * Hv;

        for (int t = 0; t < T_replay; ++t) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] * {g_access};
                kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            for (int i = 0; i < n_per_t; ++i) {{
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + k_[s_idx] * delta;
            }}

            k_ += Hk * Dk;
            v_ += Hv * Dv;
            {g_advance}
            beta_ += Hv;
        }}

        for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """
    suffix = "_vec" if vectorized else ""
    return mx.fast.metal_kernel(
        name=f"gated_delta_replay_state_only{suffix}",
        input_names=["k", "v", "g", "beta", "state_in", "T_replay", "V_stride"],
        output_names=["state_out"],
        source=source,
    )


_replay_kernel = _make_gated_delta_replay_kernel(vectorized=False)
_replay_kernel_vec = _make_gated_delta_replay_kernel(vectorized=True)


def gated_delta_replay_state_only(
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state_in: mx.array,
    t_replay: int,
) -> mx.array:
    """Replay exactly t_replay steps of the Gated DeltaNet recurrence on state_in."""
    if t_replay <= 0:
        return state_in

    B, V_stride, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        _, state_out = gated_delta_ops(
            k[:, :t_replay],
            k[:, :t_replay],
            v[:, :t_replay],
            g[:, :t_replay],
            beta[:, :t_replay],
            state_in,
        )
        return state_out

    kernel = _replay_kernel_vec if g.ndim == 4 else _replay_kernel
    (state_out,) = kernel(
        inputs=[k, v, g, beta, state_in, t_replay, V_stride],
        template=[
            ("InT", k.dtype),
            ("StT", state_in.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[state_in.shape],
        output_dtypes=[state_in.dtype],
    )
    return state_out


@dataclass
class LayerVerifyCapture:
    prev_conv: Optional[mx.array]
    prev_state: Optional[mx.array]
    conv_input: mx.array
    k: mx.array
    v: mx.array
    g: mx.array
    beta: mx.array
    n_keep: int


@dataclass
class RollbackCheckpoint:
    linear_captures: Dict[int, LayerVerifyCapture]
    kv_offsets: Dict[int, int]
    total_verify_tokens: int


def _unwrap_text_model(model: nn.Module):
    """Return (text_model, backbone, args) for Qwen3_5Model or TextModel."""
    if hasattr(model, "language_model"):
        tm = model.language_model
    else:
        tm = model
    return tm, tm.model, tm.args


class DeltaNetRollbackEngine:
    """Exact hybrid Gated DeltaNet + KVCache verification and rollback engine."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.text_model, self.backbone, self.args = _unwrap_text_model(model)
        self.layers = self.backbone.layers
        self.linear_indices = [i for i, l in enumerate(self.layers) if l.is_linear]
        self.kv_indices = [i for i, l in enumerate(self.layers) if not l.is_linear]
        self._last_checkpoint: Optional[RollbackCheckpoint] = None

    def make_cache(self) -> List[Any]:
        return self.model.make_cache()

    def project_logits(self, hidden_states: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.backbone.embed_tokens.as_linear(hidden_states)
        return self.text_model.lm_head(hidden_states)

    def prefill(self, prompt_tokens: List[int], cache: List[Any]) -> mx.array:
        """Prefill prompt tokens and return ONLY the last-position logits [1, 1, V]."""
        x = mx.array([prompt_tokens], dtype=mx.int32)
        h = self.backbone(x, cache=cache)
        logits = self.project_logits(h[:, -1:, :])
        return logits

    def _run_linear_attn_capture(
        self,
        layer: nn.Module,
        x: mx.array,
        mask: Optional[mx.array],
        cache: ArraysCache,
    ) -> Tuple[mx.array, LayerVerifyCapture]:
        """Exact mirror of Qwen3.5 GatedDeltaNet.__call__ capturing replay factors."""
        la = layer.linear_attn
        B, S, _ = x.shape

        qkv = la.in_proj_qkv(x)
        z = la.in_proj_z(x).reshape(B, S, la.num_v_heads, la.head_v_dim)
        b = la.in_proj_b(x)
        a = la.in_proj_a(x)

        prev_conv = cache[0]
        if prev_conv is not None:
            conv_state = prev_conv
        else:
            conv_state = mx.zeros(
                (B, la.conv_kernel_size - 1, la.conv_dim),
                dtype=x.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)

        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        n_keep = la.conv_kernel_size - 1
        if cache.lengths is not None:
            ends = mx.clip(cache.lengths, 0, S)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])

        conv_out = nn.silu(la.conv1d(conv_input))
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [la.key_dim, 2 * la.key_dim], -1),
                [la.num_k_heads, la.num_k_heads, la.num_v_heads],
                [la.head_k_dim, la.head_k_dim, la.head_v_dim],
            )
        ]

        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        beta = mx.sigmoid(b)
        g = compute_g(la.A_log, a, la.dt_bias)

        prev_state = cache[1]
        state_in = prev_state
        if state_in is None:
            state_in = mx.zeros(
                (B, la.num_v_heads, la.head_v_dim, la.head_k_dim),
                dtype=mx.float32,
            )

        if mx.default_device() == mx.gpu and mx.metal.is_available():
            out, new_state = gated_delta_kernel(q, k, v, g, beta, state_in, mask)
        else:
            out, new_state = gated_delta_ops(q, k, v, g, beta, state_in, mask)

        cache[1] = new_state
        cache.advance(S)

        out = la.norm(out, z)
        out = la.out_proj(out.reshape(B, S, -1))

        capture = LayerVerifyCapture(
            prev_conv=prev_conv,
            prev_state=prev_state,
            conv_input=conv_input,
            k=k,
            v=v,
            g=g,
            beta=beta,
            n_keep=n_keep,
        )
        return out, capture

    def verify_forward(
        self,
        tokens: mx.array,
        cache: List[Any],
        record: bool = True,
    ) -> mx.array:
        """Run a parallel verify pass on tokens [1, V], optionally recording rollback factors.

        Uses every layer's exact submodules (including any Track A QuantizedLinear patches).
        """
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        V = int(tokens.shape[1])

        if not record or V <= 1:
            self._last_checkpoint = None
            return self.model(tokens, cache=cache)

        kv_offsets = {i: cache[i].offset for i in self.kv_indices}
        linear_captures: Dict[int, LayerVerifyCapture] = {}

        hidden_states = self.backbone.embed_tokens(tokens)
        fa_mask = create_attention_mask(hidden_states, cache[self.backbone.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.backbone.ssm_idx])

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            if layer.is_linear:
                r, cap = self._run_linear_attn_capture(
                    layer, layer.input_layernorm(hidden_states), ssm_mask, c
                )
                linear_captures[i] = cap
            else:
                r = layer.self_attn(layer.input_layernorm(hidden_states), fa_mask, c)
            h = hidden_states + r
            hidden_states = h + layer.mlp(layer.post_attention_layernorm(h))

        hidden_states = self.backbone.norm(hidden_states)
        logits = self.project_logits(hidden_states)

        self._last_checkpoint = RollbackCheckpoint(
            linear_captures=linear_captures,
            kv_offsets=kv_offsets,
            total_verify_tokens=V,
        )
        return logits

    def rollback(
        self,
        cache: List[Any],
        n_keep_tokens: int,
        async_eval: bool = False,
        sync_eval: bool = False,
    ) -> int:
        """Roll back hybrid cache to exact state after n_keep_tokens of last verify_forward.

        Args:
            cache: Model cache mutated by the preceding verify_forward call.
            n_keep_tokens: Number of verified prefix tokens to keep (0 <= n_keep_tokens <= V).
            async_eval: If True, immediately enqueue async GPU execution of replayed states
                so replay overlaps with CPU drafting.
            sync_eval: If True, evaluate and synchronize replayed states before returning.

        Returns:
            Number of rejected tokens rolled back (V - n_keep_tokens).
        """
        ckpt = self._last_checkpoint
        self._last_checkpoint = None
        if ckpt is None:
            return 0

        V = ckpt.total_verify_tokens
        if not (0 <= n_keep_tokens <= V):
            raise ValueError(f"n_keep_tokens={n_keep_tokens} out of range [0, {V}]")

        n_reject = V - n_keep_tokens
        if n_reject == 0:
            return 0

        # 1. O(1) KVCache trim for full-attention layers
        for i in self.kv_indices:
            cache[i].offset = ckpt.kv_offsets[i] + n_keep_tokens

        # 2. Exact zero-matmul Gated DeltaNet state replay for linear layers
        to_eval: List[mx.array] = []
        for i in self.linear_indices:
            cap = ckpt.linear_captures[i]
            c = cache[i]
            if n_keep_tokens == 0:
                c[0] = cap.prev_conv
                c[1] = cap.prev_state
                continue

            new_conv = mx.contiguous(
                cap.conv_input[:, n_keep_tokens : n_keep_tokens + cap.n_keep, :]
            )
            state_in = cap.prev_state
            if state_in is None:
                B = cap.k.shape[0]
                Hv, Dv = cap.v.shape[2:]
                Dk = cap.k.shape[3]
                state_in = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

            new_state = gated_delta_replay_state_only(
                cap.k, cap.v, cap.g, cap.beta, state_in, n_keep_tokens
            )
            c[0] = new_conv
            c[1] = new_state
            to_eval.append(new_conv)
            to_eval.append(new_state)

        if to_eval:
            if sync_eval:
                mx.eval(*to_eval)
                mx.synchronize()
            elif async_eval:
                mx.async_eval(*to_eval)

        return n_reject

    def clear_checkpoint(self) -> None:
        self._last_checkpoint = None

    def snapshot_state_only(self, cache: List[Any]) -> Tuple[Dict[int, Tuple[Any, Any]], Dict[int, int]]:
        """Zero-copy snapshot of current cache state (for draft model rollback to commit point)."""
        lin = {i: (cache[i][0], cache[i][1]) for i in self.linear_indices}
        kv = {i: cache[i].offset for i in self.kv_indices}
        return lin, kv

    def restore_state_only(
        self,
        cache: List[Any],
        snapshot: Tuple[Dict[int, Tuple[Any, Any]], Dict[int, int]],
    ) -> None:
        """Zero-copy restore of a snapshot taken by snapshot_state_only."""
        lin, kv = snapshot
        for i, (conv_s, rec_s) in lin.items():
            cache[i][0] = conv_s
            cache[i][1] = rec_s
        for i, off in kv.items():
            cache[i].offset = off

    def state_memory_bytes(self, cache: List[Any]) -> Dict[str, float]:
        """Compute exact MiB of Gated DeltaNet recurrent states & conv states."""
        conv_b = 0
        rec_b = 0
        kv_b = 0
        for i in self.linear_indices:
            if cache[i][0] is not None:
                conv_b += cache[i][0].nbytes
            if cache[i][1] is not None:
                rec_b += cache[i][1].nbytes
        for i in self.kv_indices:
            kv_b += cache[i].nbytes
        return {
            "linear_layers": len(self.linear_indices),
            "kv_layers": len(self.kv_indices),
            "conv_state_mib": conv_b / (1024**2),
            "recurrent_state_mib": rec_b / (1024**2),
            "total_deltanet_mib": (conv_b + rec_b) / (1024**2),
            "kv_cache_mib": kv_b / (1024**2),
        }


def _clone_cache_for_test(cache: List[Any]) -> List[Any]:
    """Deep-clone hybrid cache for isolated unit-test trials."""
    out = []
    for c in cache:
        if isinstance(c, ArraysCache):
            nc = ArraysCache(len(c.cache))
            nc.cache = [mx.array(x) if x is not None else None for x in c.cache]
            out.append(nc)
        elif isinstance(c, KVCache):
            nc = KVCache()
            if c.keys is not None:
                nc.keys = mx.array(c.keys)
                nc.values = mx.array(c.values)
                nc.offset = c.offset
            out.append(nc)
        else:
            raise TypeError(f"Unsupported cache type: {type(c)}")
    return out


def _max_state_diffs(cache_a: List[Any], cache_b: List[Any]) -> Dict[str, float]:
    max_conv = 0.0
    max_rec = 0.0
    mean_rec = 0.0
    max_kv = 0.0
    for ca, cb in zip(cache_a, cache_b):
        if isinstance(ca, ArraysCache):
            d0 = float(mx.max(mx.abs(ca[0].astype(mx.float32) - cb[0].astype(mx.float32))).item())
            diff_s = mx.abs(ca[1].astype(mx.float32) - cb[1].astype(mx.float32))
            d1 = float(mx.max(diff_s).item())
            m1 = float(mx.mean(diff_s).item())
            max_conv = max(max_conv, d0)
            max_rec = max(max_rec, d1)
            mean_rec = max(mean_rec, m1)
        elif isinstance(ca, KVCache):
            assert ca.offset == cb.offset, f"KVCache offset mismatch: {ca.offset} vs {cb.offset}"
            off = ca.offset
            dk = float(mx.max(mx.abs(ca.keys[..., :off, :].astype(mx.float32) - cb.keys[..., :off, :].astype(mx.float32))).item())
            dv = float(mx.max(mx.abs(ca.values[..., :off, :].astype(mx.float32) - cb.values[..., :off, :].astype(mx.float32))).item())
            max_kv = max(max_kv, dk, dv)
    return {
        "max_conv_diff": max_conv,
        "max_rec_diff": max_rec,
        "mean_rec_diff": mean_rec,
        "max_kv_diff": max_kv,
    }


def run_forced_rejection_unit_test(
    model_id: str,
    k_draft: int = 7,
    follow_steps: int = 12,
    tol: float = 1e-3,
    also_test_qmm_k: int = 15,
) -> Dict[str, Any]:
    """Strict forced-rejection parity test at EVERY depth d in 0..k_draft.

    Note on MLX 0.31.2 matmul dispatch precision:
      - For V = K + 1 <= 8 (e.g. K=7 -> V=8), MLX dispatches qmv (exact same
        accumulation order as serial B=1), yielding 0.000e+00 bit-exact state
        identity across ALL linear conv, recurrent S, and KVCache tensors!
      - For V = K + 1 >= 9 (e.g. K=15 -> V=16), MLX switches to qmm (tiled
        accumulation in bfloat16). We explicitly verify:
          (a) 0.000e+00 bit-identity of gated_delta_replay_state_only vs stock
              gated_delta_kernel,
          (b) 0.000e+00 causal zero-leakage from rejected suffix tokens,
          (c) mean_rec_diff < tol vs serial B=1, AND
          (d) 100% exact greedy token-for-token parity!
    """
    import mlx_lm

    print(f"\n=== Forced-Rejection Exact Parity Unit Test: {model_id} (K={k_draft}, V={k_draft+1}) ===")
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(model_id)
    print(f"Loaded in {time.perf_counter() - t0:.2f}s")

    engine = DeltaNetRollbackEngine(model)
    prompt = (
        "Explain the recurrence S_t = g_t * S_{t-1} + beta_t * (v_t - S_{t-1} k_t) k_t^T "
        "in Gated DeltaNet and how exact state rollback works in speculative decoding."
    )
    prompt_ids = tokenizer.encode(prompt)
    vocab_size = int(engine.args.vocab_size)

    # 0. Standalone state-only replay kernel bit-identity test vs stock gated_delta_kernel
    mx.random.seed(42)
    la0 = engine.layers[engine.linear_indices[0]].linear_attn
    B_t, V_t = 1, max(k_draft + 1, also_test_qmm_k + 1)
    q_t = mx.random.normal((B_t, V_t, la0.num_k_heads, la0.head_k_dim)).astype(mx.bfloat16)
    k_t = mx.random.normal((B_t, V_t, la0.num_k_heads, la0.head_k_dim)).astype(mx.bfloat16)
    v_t = mx.random.normal((B_t, V_t, la0.num_v_heads, la0.head_v_dim)).astype(mx.bfloat16)
    g_t = mx.sigmoid(mx.random.normal((B_t, V_t, la0.num_v_heads))).astype(mx.float32)
    b_t = mx.sigmoid(mx.random.normal((B_t, V_t, la0.num_v_heads))).astype(mx.bfloat16)
    s_t = mx.random.normal((B_t, la0.num_v_heads, la0.head_v_dim, la0.head_k_dim)).astype(mx.float32)
    max_kernel_err = 0.0
    for a_chk in [1, 2, 4, k_draft, V_t]:
        _, s_ref = gated_delta_kernel(q_t[:, :a_chk], k_t[:, :a_chk], v_t[:, :a_chk], g_t[:, :a_chk], b_t[:, :a_chk], s_t)
        s_rep = gated_delta_replay_state_only(k_t, v_t, g_t, b_t, s_t, a_chk)
        max_kernel_err = max(max_kernel_err, float(mx.max(mx.abs(s_ref - s_rep)).item()))
    assert max_kernel_err == 0.0, f"Replay kernel not bit-exact vs gated_delta_kernel: {max_kernel_err}"
    print(f"  [Replay Kernel Bit-Identity] max_abs_diff = {max_kernel_err:.3e} (0.0 exact) | PASS")

    # 1. Build golden serial B=1 reference trajectory and checkpoints at every step
    base_cache = engine.make_cache()
    logits0 = engine.prefill(prompt_ids, base_cache)
    mx.eval(logits0, *[x for c in base_cache for x in (c.state if isinstance(c, ArraysCache) else (c.keys, c.values)) if x is not None])
    mx.synchronize()

    mem_info = engine.state_memory_bytes(base_cache)
    print(
        f"Hybrid layers: {mem_info['linear_layers']} linear + {mem_info['kv_layers']} full-attn | "
        f"DeltaNet state: {mem_info['total_deltanet_mib']:.2f} MiB "
        f"(conv={mem_info['conv_state_mib']:.2f} MiB, S={mem_info['recurrent_state_mib']:.2f} MiB)"
    )

    y0 = int(mx.argmax(logits0[0, -1]).item())
    max_k_all = max(k_draft, also_test_qmm_k)
    total_ref_needed = max_k_all + 1 + follow_steps

    serial_cache = _clone_cache_for_test(base_cache)
    ref_tokens: List[int] = [y0]
    serial_caches_after: Dict[int, List[Any]] = {0: _clone_cache_for_test(serial_cache)}

    cur_tok = y0
    for step in range(1, total_ref_needed + 1):
        out_logits = model(mx.array([[cur_tok]], dtype=mx.int32), cache=serial_cache)
        nxt = int(mx.argmax(out_logits[0, -1]).item())
        mx.eval(out_logits)
        if step <= max_k_all + 1:
            serial_caches_after[step] = _clone_cache_for_test(serial_cache)
        ref_tokens.append(nxt)
        cur_tok = nxt
    mx.synchronize()

    # Warm up replay kernel once before measuring timings
    warm_cache = _clone_cache_for_test(base_cache)
    _ = engine.verify_forward(mx.array([ref_tokens[: k_draft + 1]], dtype=mx.int32), warm_cache, record=True)
    mx.synchronize()
    engine.rollback(warm_cache, n_keep_tokens=1, sync_eval=True)

    # 2. Test EVERY forced rejection depth d in 0..k_draft (plus explicit a=0 test!)
    depth_results = []
    all_passed = True

    c_zero = _clone_cache_for_test(base_cache)
    _ = engine.verify_forward(mx.array([ref_tokens[: k_draft + 1]], dtype=mx.int32), c_zero, record=True)
    mx.synchronize()
    t_rb0 = time.perf_counter()
    engine.rollback(c_zero, n_keep_tokens=0, sync_eval=True)
    rb0_ms = (time.perf_counter() - t_rb0) * 1e3
    diffs0 = _max_state_diffs(c_zero, serial_caches_after[0])
    assert max(diffs0.values()) == 0.0, f"a=0 rollback not bit-exact: {diffs0}"
    print(f"  [a=0 full restore] rollback={rb0_ms:.3f} ms | diffs=0.0 exact | PASS")

    is_moe = getattr(engine.args, "num_experts", 0) > 0

    # Pre-compute clean (uncorrupted) verify_forward(ref_tokens[:K+1]) captures
    # to assert 0.000e+00 causal zero-leakage at every rejection depth d < K.
    clean_ref_caches: Dict[int, List[Any]] = {}
    for a_clean in range(1, k_draft + 2):
        c_cl = _clone_cache_for_test(base_cache)
        _ = engine.verify_forward(mx.array([ref_tokens[: k_draft + 1]], dtype=mx.int32), c_cl, record=True)
        engine.rollback(c_cl, n_keep_tokens=a_clean, sync_eval=True)
        clean_ref_caches[a_clean] = c_cl

    for d in range(0, k_draft + 1):
        draft = []
        for j in range(k_draft):
            true_tok = ref_tokens[1 + j]
            if j < d:
                draft.append(true_tok)
            else:
                bad_tok = (true_tok + 137 + 17 * j) % vocab_size
                if bad_tok == true_tok:
                    bad_tok = (bad_tok + 1) % vocab_size
                draft.append(bad_tok)

        verify_in = [y0] + draft  # length K + 1
        spec_cache = _clone_cache_for_test(base_cache)

        t_ver = time.perf_counter()
        v_logits = engine.verify_forward(mx.array([verify_in], dtype=mx.int32), spec_cache, record=True)
        v_argmax = mx.argmax(v_logits[0], axis=-1).tolist()
        mx.synchronize()
        ver_ms = (time.perf_counter() - t_ver) * 1e3

        m = 0
        while m < k_draft and v_argmax[m] == draft[m]:
            m += 1
        bonus_tok = int(v_argmax[m])
        emitted = draft[:m] + [bonus_tok]
        expected_emitted = ref_tokens[1 : 1 + d + 1]

        a = m + 1
        t_rb = time.perf_counter()
        n_rej = engine.rollback(spec_cache, n_keep_tokens=a, sync_eval=True)
        rb_ms = (time.perf_counter() - t_rb) * 1e3

        diffs = _max_state_diffs(spec_cache, serial_caches_after[a])
        leak_diffs = _max_state_diffs(spec_cache, clean_ref_caches[a])
        zero_leak = max(
            leak_diffs["max_conv_diff"],
            leak_diffs["max_rec_diff"],
            leak_diffs["max_kv_diff"],
        ) == 0.0

        post_tokens = []
        tok_cur = bonus_tok
        for _ in range(follow_steps):
            lg = model(mx.array([[tok_cur]], dtype=mx.int32), cache=spec_cache)
            nxt = int(mx.argmax(lg[0, -1]).item())
            post_tokens.append(nxt)
            tok_cur = nxt
        mx.synchronize()

        expected_post = ref_tokens[1 + d + 1 : 1 + d + 1 + follow_steps]
        exact_m = m == d
        exact_emit = emitted == expected_emitted
        exact_post = post_tokens == expected_post
        if ((not is_moe) and (k_draft + 1 <= 8)) or (is_moe and (k_draft + 1 <= 4)):
            # Exact qmv regime (Dense V<=8, MoE V<=4): strict max_abs_diff < tol (0.0 bit-exact!)
            within_tol = (
                diffs["max_conv_diff"] < tol
                and diffs["max_rec_diff"] < tol
                and diffs["max_kv_diff"] < tol
            )
        else:
            # MoE V>=5 (8-bit gate qmm) or dense V>=9 (qmm): 0.0 causal leakage + mean_rec_diff < tol
            within_tol = zero_leak and (diffs["mean_rec_diff"] < tol)
        passed = exact_m and exact_emit and exact_post and zero_leak and within_tol
        all_passed = all_passed and passed

        depth_results.append(
            {
                "forced_reject_depth": d,
                "detected_m": m,
                "kept_a": a,
                "rejected": n_rej,
                "verify_ms": round(ver_ms, 3),
                "rollback_ms": round(rb_ms, 3),
                "max_conv_diff": diffs["max_conv_diff"],
                "max_rec_diff": diffs["max_rec_diff"],
                "mean_rec_diff": diffs["mean_rec_diff"],
                "max_kv_diff": diffs["max_kv_diff"],
                "causal_leak_max": max(
                    leak_diffs["max_conv_diff"],
                    leak_diffs["max_rec_diff"],
                    leak_diffs["max_kv_diff"],
                ),
                "exact_emit_parity": exact_emit,
                "exact_post_parity": exact_post,
                "passed": passed,
            }
        )
        status = "PASS" if passed else "FAIL"
        print(
            f"  [d={d:2d} -> a={a:2d}, rej={n_rej:2d}] ver={ver_ms:6.2f} ms | "
            f"rb={rb_ms:6.3f} ms | leak={depth_results[-1]['causal_leak_max']:.1e} | "
            f"S_mean={diffs['mean_rec_diff']:.2e} S_max={diffs['max_rec_diff']:.2e} | "
            f"emit={exact_emit} post({follow_steps})={exact_post} | {status}"
        )

    # 3. Also test wide qmm regime (K=15 -> V=16): causal zero-leakage + exact greedy parity
    qmm_check = {}
    if also_test_qmm_k and also_test_qmm_k > k_draft:
        a_test = 5
        seq_good = ref_tokens[: also_test_qmm_k + 1]
        seq_bad = ref_tokens[:a_test] + [
            (ref_tokens[j] + 211 + 19 * j) % vocab_size
            for j in range(a_test, also_test_qmm_k + 1)
        ]
        c_g = _clone_cache_for_test(base_cache)
        _ = engine.verify_forward(mx.array([seq_good], dtype=mx.int32), c_g, record=True)
        engine.rollback(c_g, n_keep_tokens=a_test, sync_eval=True)

        c_b = _clone_cache_for_test(base_cache)
        lg_b = engine.verify_forward(mx.array([seq_bad], dtype=mx.int32), c_b, record=True)
        v_arg_b = mx.argmax(lg_b[0], axis=-1).tolist()
        mx.synchronize()
        t_r = time.perf_counter()
        engine.rollback(c_b, n_keep_tokens=a_test, sync_eval=True)
        rb_qmm_ms = (time.perf_counter() - t_r) * 1e3

        leak = _max_state_diffs(c_g, c_b)
        vs_ser = _max_state_diffs(c_b, serial_caches_after[a_test])
        # Verify post-rollback greedy parity
        post_b = []
        tok_c = int(v_arg_b[a_test - 1])
        for _ in range(follow_steps):
            lg = model(mx.array([[tok_c]], dtype=mx.int32), cache=c_b)
            tok_c = int(mx.argmax(lg[0, -1]).item())
            post_b.append(tok_c)
        mx.synchronize()
        qmm_exact = (
            v_arg_b[:a_test] == ref_tokens[1 : a_test + 1]
            and post_b == ref_tokens[a_test + 1 : a_test + 1 + follow_steps]
        )
        zero_leak = max(leak["max_conv_diff"], leak["max_rec_diff"], leak["max_kv_diff"]) == 0.0
        assert zero_leak and qmm_exact and vs_ser["mean_rec_diff"] < tol, (
            f"Wide K={also_test_qmm_k} check failed: zero_leak={zero_leak}, qmm_exact={qmm_exact}, vs_ser={vs_ser}"
        )
        qmm_check = {
            "k": also_test_qmm_k,
            "a": a_test,
            "rollback_ms": round(rb_qmm_ms, 3),
            "causal_leak_max": 0.0,
            "mean_rec_diff_vs_serial": vs_ser["mean_rec_diff"],
            "exact_greedy_parity": qmm_exact,
        }
        print(
            f"  [Wide qmm K={also_test_qmm_k}, a={a_test}] rb={rb_qmm_ms:.3f} ms | "
            f"causal_leak=0.0 exact | S_mean_vs_B1={vs_ser['mean_rec_diff']:.2e} | "
            f"greedy_parity={qmm_exact} | PASS"
        )

    # 4. Microbenchmark pure rollback timing across a in {1, 2, 4, 7} (median of 5 reps)
    replay_bench = {}
    for a_bench in [1, 2, 4, k_draft]:
        if a_bench >= k_draft + 1:
            continue
        c_tmp = _clone_cache_for_test(base_cache)
        _ = engine.verify_forward(mx.array([ref_tokens[: k_draft + 1]], dtype=mx.int32), c_tmp, record=True)
        mx.synchronize()
        ckpt = engine._last_checkpoint
        times = []
        for _ in range(5):
            engine._last_checkpoint = ckpt
            t_s = time.perf_counter()
            engine.rollback(c_tmp, n_keep_tokens=a_bench, sync_eval=True)
            times.append((time.perf_counter() - t_s) * 1e3)
        replay_bench[f"a={a_bench}"] = round(sorted(times)[len(times) // 2], 3)
    print(f"  Median pure state-only replay ms (5 reps): {replay_bench}")

    del model, base_cache, serial_cache, serial_caches_after
    gc.collect()
    mx.clear_cache()

    if not all_passed:
        raise AssertionError(f"Forced-rejection parity test FAILED for {model_id}!")
    print(f"=== ALL {k_draft + 1} FORCED-REJECTION DEPTHS PASSED 100% EXACT PARITY ({model_id}) ===")
    return {
        "model_id": model_id,
        "k_draft": k_draft,
        "memory": mem_info,
        "kernel_bit_identity_max_err": max_kernel_err,
        "a0_rollback_ms": round(rb0_ms, 3),
        "depth_results": depth_results,
        "qmm_wide_check": qmm_check,
        "replay_median_ms": replay_bench,
        "all_passed": all_passed,
    }


def main():
    parser = argparse.ArgumentParser(description="Gated DeltaNet exact rollback unit test")
    parser.add_argument(
        "--models",
        type=str,
        default="mlx-community/Qwen3.5-9B-4bit,mlx-community/Qwen3.6-35B-A3B-4bit",
        help="Comma-separated model IDs to test",
    )
    parser.add_argument("--k", type=int, default=0, help="Draft length K for forced-rejection sweep (0 = auto bit-exact qmv max: 7 for dense 9B, 3 for MoE 35B)")
    parser.add_argument("--wide-k", type=int, default=15, help="Wide qmm K check (0 to skip)")
    parser.add_argument("--follow", type=int, default=12, help="Post-rollback greedy verification tokens")
    parser.add_argument("--tol", type=float, default=1e-3, help="Max abs state diff tolerance")
    args = parser.parse_args()

    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        k_use = args.k if args.k > 0 else (3 if ("35B" in mid or "moe" in mid.lower()) else 7)
        run_forced_rejection_unit_test(
            mid,
            k_draft=k_use,
            follow_steps=args.follow,
            tol=args.tol,
            also_test_qmm_k=args.wide_k,
        )


if __name__ == "__main__":
    main()
