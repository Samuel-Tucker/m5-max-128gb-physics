#!/usr/bin/env python3
"""
Native Qwen3.5 / Qwen3.6 Multi-Token Prediction (MTP) Head in MLX.

Exact match to official Qwen3.5 / vLLM `Qwen3_5MultiTokenPredictor`:
  1. e = pre_fc_norm_embedding(embed_tokens(next_token_ids))
  2. h = pre_fc_norm_hidden(prev_hidden_states)
  3. x = fc(concat([e, h], axis=-1))
  4. x = mtp_decoder_layer(x, mask, cache)   # full-attention + dense MLP (9B) or SparseMoE (35B)
  5. out = norm(x)                           # fed to lm_head AND chained into next MTP step

Supports:
  - Dense Qwen3.5-9B (BF16 or Q4)
  - MoE Qwen3.6-35B-A3B (Q4 + 8-bit router gates)
  - Zero-copy / O(1) KVCache offset rollback (`cache.offset = saved_offset`)
  - Exact batched MTP KV synchronization on accepted prefix tokens
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5 import TextModelArgs
from mlx_lm.models.qwen3_next import (
    Qwen3NextAttention as Attention,
    Qwen3NextMLP as MLP,
    Qwen3NextSparseMoeBlock as SparseMoeBlock,
)


class Qwen35MTPDecoderLayer(nn.Module):
    """Single full-attention Qwen3.5 / Qwen3.6 decoder layer used inside `mtp.layers.0`."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        if args.num_experts > 0:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen35MTPHead(nn.Module):
    """Native Qwen3.5 / Qwen3.6 Multi-Token Prediction module."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        self.layers = [Qwen35MTPDecoderLayer(args)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def make_cache(self) -> KVCache:
        return KVCache()

    def forward_states(
        self,
        token_ids: mx.array,
        hidden_states: mx.array,
        embed_tokens: nn.Module,
        cache: Optional[KVCache] = None,
        concat_order: str = "eh",
    ) -> Tuple[mx.array, mx.array]:
        """Run MTP block and return `(post_norm_hidden, pre_norm_hidden)`.

        Args:
            token_ids: `[B, S]` next-token ids aligned with `hidden_states` (`t_{p+1}` for `h_p`).
            hidden_states: `[B, S, H]` target backbone hidden states (or previous MTP hidden states).
            embed_tokens: Shared target `embed_tokens` module.
            cache: Single `KVCache` for `mtp.layers.0`.
            concat_order: `"eh"` for `[emb, hidden]` (official) or `"he"` (diagnostic ablation).
        """
        if token_ids.ndim == 1:
            token_ids = token_ids[None, :]
        if hidden_states.ndim == 2:
            hidden_states = hidden_states[None, :, :]

        emb = embed_tokens(token_ids)
        e_norm = self.pre_fc_norm_embedding(emb)
        h_norm = self.pre_fc_norm_hidden(hidden_states)
        if concat_order == "eh":
            fused = mx.concatenate([e_norm, h_norm], axis=-1)
        else:
            fused = mx.concatenate([h_norm, e_norm], axis=-1)

        x = self.fc(fused)
        mask = create_attention_mask(x, cache)
        pre_norm = self.layers[0](x, mask=mask, cache=cache)
        post_norm = self.norm(pre_norm)
        return post_norm, pre_norm


def load_mtp_head(
    target_model: nn.Module,
    weights_path: str | Path,
    quantized: bool = True,
    bits: int = 4,
    group_size: int = 64,
) -> Qwen35MTPHead:
    """Instantiate and strictly load a Qwen35MTPHead matching `target_model`."""
    tm = target_model.language_model if hasattr(target_model, "language_model") else target_model
    args: TextModelArgs = tm.args
    mtp = Qwen35MTPHead(args)

    raw_weights: Dict[str, mx.array] = mx.load(str(weights_path))
    stripped: Dict[str, mx.array] = {}
    for k, v in raw_weights.items():
        nk = k[len("mtp.") :] if k.startswith("mtp.") else k
        stripped[nk] = v

    if quantized:
        def _pred(path: str, m: nn.Module):
            if not hasattr(m, "to_quantized"):
                return False
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": group_size, "bits": 8}
            return {"group_size": group_size, "bits": bits}

        nn.quantize(mtp, group_size=group_size, bits=bits, class_predicate=_pred)

    mtp.load_weights(list(stripped.items()), strict=True)
    mx.eval(mtp.parameters())
    return mtp


def patch_mtp_with_track_a_nax(mtp: Qwen35MTPHead, backend: str = "tg_deq", min_m: int = 8) -> int:
    """Optionally patch dense QuantizedLinear modules in `mtp` with Track A Metal 4 NAX."""
    import sys
    kernels_dir = Path(__file__).resolve().parent
    if str(kernels_dir) not in sys.path:
        sys.path.insert(0, str(kernels_dir))
    from patch_qwen9b_nax import NAXQuantizedLinear

    count = 0
    for attr in ("fc",):
        mod = getattr(mtp, attr, None)
        if isinstance(mod, nn.QuantizedLinear):
            setattr(mtp, attr, NAXQuantizedLinear(mod, backend=backend, min_m=min_m, max_m=64))
            count += 1
    layer = mtp.layers[0]
    for parent in (layer.self_attn, layer.mlp):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            mod = getattr(parent, name, None)
            if isinstance(mod, nn.QuantizedLinear):
                setattr(parent, name, NAXQuantizedLinear(mod, backend=backend, min_m=min_m, max_m=64))
                count += 1
    return count
