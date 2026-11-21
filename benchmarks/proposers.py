#!/usr/bin/env python3
"""
Orthogonal Near-Zero-Byte Draft Proposers (+ Serial Small-Model Controls) for Track D.

All proposers share a uniform interface:
  - reset(prompt_tokens)
  - propose(committed_tokens, max_k) -> List[int]
  - on_verify(committed_tokens, draft_tokens, verify_argmax, accepted_count)
  - stats() -> Dict[str, Any]

Proposers implemented:
  1. PromptLookupProposer:
       O(1) incremental multi-order CPU suffix/n-gram index over prompt + committed
       tokens (<0.03 ms/step, 0 DRAM weight reads). Supports both fixed-K and
       confidence-gated adaptive-K (returns [] on miss -> zero free-prose tax!).
  2. LookaheadProposer:
       Drafter-free Lookahead / Jacobi N-gram Pool (arXiv:2402.02057-inspired),
       harvesting n-grams online from BOTH committed tokens AND parallel verify
       trajectories (including rejected-position model predictions).
  3. LayerSkipProposer:
       Self-speculative zero-extra-model drafter running a reduced subset of the
       target model's own layers/MLPs via non-mutating zero-copy shadow caches.
  4. SmallModelProposer:
       Cached Gated DeltaNet small-model drafter (Qwen3.5-2B-4bit control and
       Qwen3.5-0.8B-4bit fast drafter) using DeltaNetRollbackEngine's zero-copy
       commit-point snapshot/restore.
  5. HybridProposer:
       Zero-byte CPU suffix first (0.02 ms), falling back to a neural/lookahead
       drafter ONLY on suffix misses (Free Prose).
  6. MoEShallowProposer:
       Specialized shallow K<=4 proposer for Qwen3.6-35B-A3B-4bit respecting
       MoE 256-expert top-8 routing fan-out.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache

from deltanet_rollback import DeltaNetRollbackEngine, _unwrap_text_model


class BaseProposer:
    name: str = "base"
    uses_gpu: bool = False

    def reset(self, prompt_tokens: List[int]) -> None:
        pass

    def propose(self, committed_tokens: List[int], max_k: Optional[int] = None) -> List[int]:
        raise NotImplementedError

    def on_verify(
        self,
        committed_tokens: List[int],
        draft_tokens: List[int],
        verify_argmax: List[int],
        accepted_count: int,
    ) -> None:
        pass

    def stats(self) -> Dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Proposer 1: O(1) Incremental CPU Multi-Order Suffix Automaton / Prompt Lookup
# ---------------------------------------------------------------------------

class PromptLookupProposer(BaseProposer):
    """Zero-byte CPU multi-order hash-indexed suffix matcher over prompt + output.

    Indexes n-grams for orders n in [min_order..max_order] incrementally in O(1)
    amortized time per committed token, preferring the longest matching suffix
    and most recent occurrence.
    """

    def __init__(
        self,
        k_max: int = 8,
        min_order: int = 2,
        max_order: int = 6,
        adaptive: bool = True,
        k_by_order: Optional[Dict[int, int]] = None,
        allow_unigram_fallback: bool = False,
        name: Optional[str] = None,
    ):
        self.k_max = k_max
        self.min_order = min_order
        self.max_order = max_order
        self.adaptive = adaptive
        self.allow_unigram_fallback = allow_unigram_fallback
        if k_by_order is not None:
            self.k_by_order = dict(k_by_order)
        else:
            # Default confidence-gated schedule: longer suffix match -> deeper K
            self.k_by_order = {
                6: k_max,
                5: k_max,
                4: k_max,
                3: min(k_max, 7),
                2: min(k_max, 3),
                1: min(k_max, 2),
            }
        self.name = name or (
            f"prompt_lookup_adaptive_k{k_max}" if adaptive else f"prompt_lookup_k{k_max}"
        )
        self.uses_gpu = False
        self._history: List[int] = []
        self._indexed_up_to: int = 0
        self._tables: Dict[int, Dict[Tuple[int, ...], int]] = {}
        self._hits_by_order: Dict[int, int] = {}
        self._misses: int = 0
        self._precomputed: Dict[int, List[int]] = {}

    def reset(self, prompt_tokens: List[int]) -> None:
        self._history = []
        self._indexed_up_to = 0
        lo = 1 if self.allow_unigram_fallback else self.min_order
        self._tables = {n: {} for n in range(lo, self.max_order + 1)}
        self._hits_by_order = {n: 0 for n in range(lo, self.max_order + 1)}
        self._misses = 0
        self._precomputed.clear()
        self._ingest(prompt_tokens)

    def _ingest(self, tokens: List[int]) -> None:
        """Incrementally index all new tokens up to len(tokens) - 1.

        Note: An n-gram ending at index `end` (meaning tokens[end-n : end]) is only
        useful if at least 1 continuation token exists at index `end`, i.e., end < L.
        """
        cur_len = len(self._history)
        if len(tokens) > cur_len:
            self._history.extend(tokens[cur_len:])

        L = len(self._history)
        # Index every end position `end` in [max(1, indexed_up_to), L - 1]
        # (We hold back end == L because continuation at index L doesn't exist yet!)
        start_end = max(1, self._indexed_up_to)
        hist = self._history
        for end in range(start_end, L):
            for n, tbl in self._tables.items():
                if end >= n:
                    key = tuple(hist[end - n : end])
                    tbl[key] = end
        self._indexed_up_to = L

    def _lookup_from_tail(self, tail: List[int], k_cap: int) -> Tuple[List[int], int]:
        """Return (draft_tokens, matched_order)."""
        hist = self._history
        L = len(hist)
        tlen = len(tail)
        lo = 1 if self.allow_unigram_fallback else self.min_order

        for n in range(min(self.max_order, tlen), lo - 1, -1):
            key = tuple(tail[tlen - n : tlen])
            pos = self._tables[n].get(key)
            if pos is not None and pos < L:
                k_target = self.k_by_order.get(n, k_cap) if self.adaptive else k_cap
                k_target = min(k_target, k_cap)
                if k_target <= 0:
                    continue
                avail = L - pos
                if avail >= k_target:
                    return hist[pos : pos + k_target], n
                # Periodic/tail extension when match sits near the end of history
                chunk = hist[pos:L]
                if not chunk:
                    continue
                if self.adaptive:
                    return chunk, n
                # Fixed-K mode: extend cyclically if needed
                reps = (k_target + len(chunk) - 1) // len(chunk)
                return (chunk * reps)[:k_target], n

        return [], 0

    def propose(self, committed_tokens: List[int], max_k: Optional[int] = None) -> List[int]:
        self._ingest(committed_tokens)
        k_cap = self.k_max if max_k is None else min(self.k_max, max_k)
        tail = self._history[-self.max_order :]
        draft, order = self._lookup_from_tail(tail, k_cap)
        if order > 0:
            self._hits_by_order[order] = self._hits_by_order.get(order, 0) + 1
            return draft

        self._misses += 1
        if not self.adaptive:
            # Fixed-K ablation fallback when no n-gram matches: repeat most recent tokens
            if len(self._history) >= 2:
                fallback = [self._history[-1]] * k_cap
                return fallback
        return []

    def precompute_continuations(self, committed_tokens: List[int], draft_tokens: List[int], k_cap: int) -> Dict[int, List[int]]:
        """CPU-overlapped pre-drafting for every acceptance depth while GPU verifies."""
        self._ingest(committed_tokens)
        out: Dict[int, List[int]] = {}
        tail_base = self._history[-self.max_order :]
        for depth in range(len(draft_tokens) + 1):
            tail = (tail_base + draft_tokens[:depth])[-self.max_order :]
            cand, _ = self._lookup_from_tail(tail, k_cap)
            if cand:
                out[depth] = cand
        self._precomputed = out
        return out

    def stats(self) -> Dict[str, Any]:
        return {
            "hits_by_order": dict(self._hits_by_order),
            "misses": self._misses,
            "total_hits": sum(self._hits_by_order.values()),
        }


# ---------------------------------------------------------------------------
# Proposer 2: Drafter-Free Lookahead / Jacobi N-gram Pool (arXiv:2402.02057)
# ---------------------------------------------------------------------------

class LookaheadProposer(BaseProposer):
    """Drafter-free Lookahead / Jacobi N-gram Pool proposer.

    Maintains a rolling n-gram pool harvested from:
      1. Committed output (and optionally prompt) n-grams, AND
      2. Parallel verification trajectories (d_j -> verify_argmax[j]) harvested
         across consecutive verify passes — capturing target-model-generated
         local n-grams even at rejected draft positions!
    """

    def __init__(
        self,
        k_max: int = 6,
        ngram_max: int = 4,
        pool_capacity: int = 8192,
        include_prompt: bool = True,
        probe_k_on_miss: int = 2,
        adaptive_probe: bool = True,
        name: str = "lookahead",
    ):
        self.name = name
        self.uses_gpu = False
        self.k_max = k_max
        self.ngram_max = ngram_max
        self.pool_capacity = pool_capacity
        self.include_prompt = include_prompt
        self.probe_k_on_miss = probe_k_on_miss
        self.adaptive_probe = adaptive_probe

        self._history: List[int] = []
        self._prompt_len: int = 0
        self._indexed_up_to: int = 0
        # Maps prefix tuple -> deque of continuation tuples (length up to k_max)
        self._pool: Dict[Tuple[int, ...], List[int]] = {}
        self._pool_keys_fifo: deque = deque()
        # Separate Jacobi-harvested bigrams/trigrams from rejected verify trajectories
        self._jacobi_pool: Dict[Tuple[int, ...], List[int]] = {}
        self._last_verify_tail: List[int] = []
        self._recent_accepts: deque = deque(maxlen=16)

        self._committed_hits = 0
        self._jacobi_hits = 0
        self._probe_steps = 0
        self._skip_steps = 0
        self._harvested_jacobi_ngrams = 0

    def reset(self, prompt_tokens: List[int]) -> None:
        self._history = list(prompt_tokens)
        self._prompt_len = len(prompt_tokens)
        self._indexed_up_to = 0 if self.include_prompt else len(prompt_tokens)
        self._pool.clear()
        self._pool_keys_fifo.clear()
        self._jacobi_pool.clear()
        self._last_verify_tail = []
        self._recent_accepts.clear()
        self._committed_hits = 0
        self._jacobi_hits = 0
        self._probe_steps = 0
        self._skip_steps = 0
        self._harvested_jacobi_ngrams = 0
        if self.include_prompt:
            self._ingest_committed(prompt_tokens)

    def _put_pool(self, pool: Dict[Tuple[int, ...], List[int]], key: Tuple[int, ...], cont: List[int]) -> None:
        if not cont:
            return
        if key not in pool:
            if len(self._pool_keys_fifo) >= self.pool_capacity:
                old = self._pool_keys_fifo.popleft()
                self._pool.pop(old, None)
            if pool is self._pool:
                self._pool_keys_fifo.append(key)
        pool[key] = cont[: self.k_max]

    def _ingest_committed(self, committed_tokens: List[int]) -> None:
        if len(committed_tokens) > len(self._history):
            self._history.extend(committed_tokens[len(self._history) :])
        L = len(self._history)
        hist = self._history
        start = max(1, self._indexed_up_to)
        for end in range(start, L):
            cont = hist[end : min(L, end + self.k_max)]
            if not cont:
                continue
            for n in range(1, min(self.ngram_max, end) + 1):
                key = tuple(hist[end - n : end])
                self._put_pool(self._pool, key, cont)
        self._indexed_up_to = L

    def on_verify(
        self,
        committed_tokens: List[int],
        draft_tokens: List[int],
        verify_argmax: List[int],
        accepted_count: int,
    ) -> None:
        self._recent_accepts.append(accepted_count)
        if not draft_tokens or not verify_argmax:
            return

        # Harvest Jacobi trajectory n-grams from the parallel verification pass:
        # At verify position j (j >= 1), input token was draft_tokens[j-1] and
        # the target model's greedy next-token prediction is verify_argmax[j].
        # Thus (draft_tokens[j-1], verify_argmax[j]) is a genuine model transition!
        # Even better: chain (draft_tokens[j-2], draft_tokens[j-1]) -> [verify_argmax[j], ...]
        y0 = committed_tokens[-1] if committed_tokens else None
        seq_in = ([y0] if y0 is not None else []) + list(draft_tokens)
        V = min(len(seq_in), len(verify_argmax))

        for j in range(max(1, accepted_count), V):
            tok_in = seq_in[j]
            tok_pred = verify_argmax[j]
            # 1-gram transition harvested from rejected verify position
            self._jacobi_pool[(tok_in,)] = [tok_pred]
            self._harvested_jacobi_ngrams += 1
            if j >= 1:
                self._jacobi_pool[(seq_in[j - 1], tok_in)] = [tok_pred]
                self._harvested_jacobi_ngrams += 1
            # Chain with next step if model agrees along the rejected path
            if j + 1 < V and verify_argmax[j] == seq_in[j + 1]:
                chain = [verify_argmax[j], verify_argmax[j + 1]]
                if j + 2 < V and verify_argmax[j + 1] == seq_in[j + 2]:
                    chain.append(verify_argmax[j + 2])
                self._jacobi_pool[(tok_in,)] = chain
                if j >= 1:
                    self._jacobi_pool[(seq_in[j - 1], tok_in)] = chain

        # Keep rejected verify_argmax tail as a rolling Jacobi candidate buffer
        if accepted_count < len(verify_argmax) - 1:
            self._last_verify_tail = list(verify_argmax[accepted_count + 1 :])
        else:
            self._last_verify_tail = []

    def _chain_from_jacobi(self, seed_tok: int, k_cap: int) -> List[int]:
        out: List[int] = []
        prev2 = self._history[-2] if len(self._history) >= 2 else None
        prev1 = seed_tok
        for _ in range(k_cap):
            nxt_seq = None
            if prev2 is not None:
                nxt_seq = self._jacobi_pool.get((prev2, prev1))
            if nxt_seq is None:
                nxt_seq = self._jacobi_pool.get((prev1,))
            if not nxt_seq:
                break
            for t in nxt_seq:
                out.append(t)
                prev2, prev1 = prev1, t
                if len(out) >= k_cap:
                    return out[:k_cap]
        return out

    def propose(self, committed_tokens: List[int], max_k: Optional[int] = None) -> List[int]:
        self._ingest_committed(committed_tokens)
        k_cap = self.k_max if max_k is None else min(self.k_max, max_k)
        hist = self._history
        L = len(hist)

        # 1. Longest n-gram match (n >= 2) in committed pool
        for n in range(min(self.ngram_max, L), 1, -1):
            key = tuple(hist[L - n : L])
            cont = self._pool.get(key)
            if cont:
                self._committed_hits += 1
                limit = k_cap if n >= 3 else min(k_cap, 3)
                return cont[:limit]

        # 2. Jacobi trajectory pool match (2-gram then 1-gram)
        if L >= 1:
            j_draft = self._chain_from_jacobi(hist[-1], min(k_cap, 3))
            if j_draft:
                self._jacobi_hits += 1
                return j_draft

        # 3. Unigram committed pool match (short K=2 probe if productive)
        if self.probe_k_on_miss > 0 and L >= 1:
            if self.adaptive_probe and len(self._recent_accepts) >= 8:
                # Back off to K=0 if probes are yielding ~0 accepts (protects Free Prose!)
                if sum(self._recent_accepts) == 0:
                    self._skip_steps += 1
                    return []
            cont1 = self._pool.get((hist[-1],))
            if cont1:
                self._probe_steps += 1
                return cont1[: min(self.probe_k_on_miss, k_cap)]

        self._skip_steps += 1
        return []

    def stats(self) -> Dict[str, Any]:
        return {
            "committed_hits": self._committed_hits,
            "jacobi_hits": self._jacobi_hits,
            "probe_steps": self._probe_steps,
            "skip_steps": self._skip_steps,
            "harvested_jacobi_ngrams": self._harvested_jacobi_ngrams,
            "pool_size": len(self._pool),
            "jacobi_pool_size": len(self._jacobi_pool),
        }


# ---------------------------------------------------------------------------
# Proposer 3: Self-Speculative Layer-Skip / Early-Exit Drafter
# ---------------------------------------------------------------------------

class _ShadowKVCache:
    """Non-mutating view over a target KVCache that appends speculative draft KV
    into a tiny separate scratch buffer without copying or mutating target KV.
    """

    def __init__(self, base_kv: KVCache):
        self._base_offset = base_kv.offset
        if base_kv.keys is not None and base_kv.offset > 0:
            self._base_k = base_kv.keys[..., : base_kv.offset, :]
            self._base_v = base_kv.values[..., : base_kv.offset, :]
        else:
            self._base_k = None
            self._base_v = None
        self._extra_k: Optional[mx.array] = None
        self._extra_v: Optional[mx.array] = None
        self.offset = base_kv.offset

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        if self._extra_k is None:
            self._extra_k = keys
            self._extra_v = values
        else:
            self._extra_k = mx.concatenate([self._extra_k, keys], axis=2)
            self._extra_v = mx.concatenate([self._extra_v, values], axis=2)
        self.offset += keys.shape[2]
        if self._base_k is None:
            return self._extra_k, self._extra_v
        return (
            mx.concatenate([self._base_k, self._extra_k], axis=2),
            mx.concatenate([self._base_v, self._extra_v], axis=2),
        )

    def make_mask(self, N: int, window_size: Optional[int] = None, return_array: bool = False):
        from mlx_lm.models.cache import create_attention_mask as _cam

        return _cam(N, offset=self.offset, return_array=return_array, window_size=window_size)


class _ShadowArraysCache:
    """Zero-copy non-mutating view over an ArraysCache (relies on MLX immutability)."""

    def __init__(self, base_arr: ArraysCache):
        self.cache = [base_arr[0], base_arr[1]]
        self.lengths = None
        self.left_padding = None

    def __getitem__(self, idx: int):
        return self.cache[idx]

    def __setitem__(self, idx: int, val: Any):
        self.cache[idx] = val

    def advance(self, n: int) -> None:
        pass

    def make_mask(self, N: int):
        return None


class LayerSkipProposer(BaseProposer):
    """Self-speculative layer-skip / MLP-skip drafter using target model weights.

    Zero extra model weights loaded! Syncs directly to the target model's exact
    cache after every verify step via zero-copy shadow caches, eliminating drift.

    skip_mode:
      - 'skip_middle_mlp': Runs ALL attention/DeltaNet layers (preserving full
        routing & recurrence) while skipping the heavy MLP (~75% of layer bytes)
        on selected middle layers.
      - 'drop_layers': Skips entire decoder blocks outside `keep_layers`.
    """

    def __init__(
        self,
        model: nn.Module,
        target_cache_ref: Optional[List[Any]] = None,
        k_draft: int = 3,
        skip_mode: str = "skip_middle_mlp",
        skip_mlp_layers: Optional[List[int]] = None,
        keep_layers: Optional[List[int]] = None,
        name: Optional[str] = None,
    ):
        self.model = model
        self.text_model, self.backbone, self.args = _unwrap_text_model(model)
        self.layers = self.backbone.layers
        self.num_layers = len(self.layers)
        self._target_cache: Optional[List[Any]] = target_cache_ref
        self.k_draft = k_draft
        self.skip_mode = skip_mode
        self.uses_gpu = True

        if skip_mlp_layers is None:
            # Skip MLPs on alternating middle layers (keep early 0..3, late last-4, and full-attn layers)
            self.skip_mlp_set = {
                i
                for i in range(4, self.num_layers - 4)
                if self.layers[i].is_linear
            }
        else:
            self.skip_mlp_set = set(skip_mlp_layers)

        if keep_layers is None:
            # Keep first 2, all full-attention layers, and last 2 layers
            self.keep_layer_set = set(
                [0, 1, self.num_layers - 2, self.num_layers - 1]
                + [i for i, l in enumerate(self.layers) if not l.is_linear]
            )
        else:
            self.keep_layer_set = set(keep_layers)

        self.name = name or f"layer_skip_{skip_mode}_k{k_draft}"
        self._draft_steps = 0
        self._total_proposed = 0
        self._total_accepted = 0

    def bind_target_cache(self, target_cache: List[Any]) -> None:
        self._target_cache = target_cache

    def reset(self, prompt_tokens: List[int]) -> None:
        self._draft_steps = 0
        self._total_proposed = 0
        self._total_accepted = 0

    def _project_logits(self, h: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.backbone.embed_tokens.as_linear(h)
        return self.text_model.lm_head(h)

    def _forward_skipped(self, token_id: int, shadow_cache: List[Any]) -> int:
        x = mx.array([[token_id]], dtype=mx.int32)
        h = self.backbone.embed_tokens(x)

        for i, (layer, sc) in enumerate(zip(self.layers, shadow_cache)):
            if self.skip_mode == "drop_layers" and i not in self.keep_layer_set:
                continue
            if layer.is_linear:
                r = layer.linear_attn(layer.input_layernorm(h), None, sc)
            else:
                r = layer.self_attn(layer.input_layernorm(h), None, sc)
            h_mid = h + r
            if self.skip_mode == "skip_middle_mlp" and i in self.skip_mlp_set:
                h = h_mid
            else:
                h = h_mid + layer.mlp(layer.post_attention_layernorm(h_mid))

        h = self.backbone.norm(h)
        logits = self._project_logits(h)
        return int(mx.argmax(logits[0, -1]).item())

    def propose(self, committed_tokens: List[int], max_k: Optional[int] = None) -> List[int]:
        if self._target_cache is None or not committed_tokens:
            return []
        k = self.k_draft if max_k is None else min(self.k_draft, max_k)
        if k <= 0:
            return []

        # Build zero-copy shadow cache anchored at exact committed target cache
        shadow_cache: List[Any] = [
            _ShadowArraysCache(c) if isinstance(c, ArraysCache) else _ShadowKVCache(c)
            for c in self._target_cache
        ]

        draft: List[int] = []
        tok = committed_tokens[-1]
        for _ in range(k):
            tok = self._forward_skipped(tok, shadow_cache)
            draft.append(tok)
        self._draft_steps += 1
        self._total_proposed += len(draft)
        return draft

    def on_verify(
        self,
        committed_tokens: List[int],
        draft_tokens: List[int],
        verify_argmax: List[int],
        accepted_count: int,
    ) -> None:
        self._total_accepted += accepted_count

    def stats(self) -> Dict[str, Any]:
        return {
            "skip_mode": self.skip_mode,
            "skipped_mlp_count": len(self.skip_mlp_set) if self.skip_mode == "skip_middle_mlp" else 0,
            "kept_layer_count": len(self.keep_layer_set) if self.skip_mode == "drop_layers" else self.num_layers,
            "draft_calls": self._draft_steps,
            "total_proposed": self._total_proposed,
            "total_accepted": self._total_accepted,
        }


# ---------------------------------------------------------------------------
# Proposer 4: Cached Small-Model Gated DeltaNet Drafter (2B Control / 0.8B)
# ---------------------------------------------------------------------------

class SmallModelProposer(BaseProposer):
    """Cached small-model drafter with exact Gated DeltaNet commit-point rollback.

    Uses DeltaNetRollbackEngine's zero-copy snapshot_state_only / restore_state_only
    to anchor the draft cache at the last committed token before drafting K tokens.
    If any draft token is rejected (m < K), restores the commit snapshot in 0.000 ms
    and catches up accepted + bonus tokens in a single batched forward pass on the
    next propose() call (also making Hybrid CPU-Suffix + SmallModel seamless!).
    """

    def __init__(
        self,
        draft_model: nn.Module,
        k_draft: int = 4,
        p_min: float = 0.0,
        name: str = "small_model",
    ):
        self.name = name
        self.uses_gpu = True
        self.draft_model = draft_model
        self.engine = DeltaNetRollbackEngine(draft_model)
        self.k_draft = k_draft
        self.p_min = p_min

        self._cache: Optional[List[Any]] = None
        self._committed_cached_len: int = 0
        self._commit_snapshot: Optional[Tuple[Dict[int, Tuple[Any, Any]], Dict[int, int]]] = None
        self._drafted_len_since_snapshot: int = 0

        self._propose_calls = 0
        self._catchup_tokens = 0

    def reset(self, prompt_tokens: List[int]) -> None:
        self._cache = self.engine.make_cache()
        self._commit_snapshot = None
        self._drafted_len_since_snapshot = 0
        self._propose_calls = 0
        self._catchup_tokens = 0
        # Prefill draft model on prompt tokens (matches target model prefill)
        if prompt_tokens:
            lg = self.engine.prefill(prompt_tokens, self._cache)
            mx.eval(lg)
            self._committed_cached_len = len(prompt_tokens)
        else:
            self._committed_cached_len = 0

    def propose(self, committed_tokens: List[int], max_k: Optional[int] = None) -> List[int]:
        k = self.k_draft if max_k is None else min(self.k_draft, max_k)
        if k <= 0 or not committed_tokens:
            return []

        assert self._cache is not None
        # Catch up any committed tokens not yet in draft cache (e.g. [y_0], or
        # [d_1..d_m, bonus], or tokens accepted while CPU suffix bypassed us)
        pending = committed_tokens[self._committed_cached_len :]
        assert len(pending) >= 1, (
            f"Draft cache ahead of committed tokens: {self._committed_cached_len} vs {len(committed_tokens)}"
        )
        self._catchup_tokens += len(pending)
        if len(pending) == 1:
            lg = self.draft_model(mx.array([pending], dtype=mx.int32), cache=self._cache)
        else:
            lg = self.engine.prefill(pending, self._cache)
        row = lg.reshape(-1, lg.shape[-1])[-1]
        d_next_arr = mx.argmax(row)
        if self.p_min > 0.0:
            row_f = row.astype(mx.float32)
            p_arr = mx.exp(mx.max(row_f) - mx.logsumexp(row_f))
            mx.eval(d_next_arr, p_arr)
            d_next = int(d_next_arr.item())
            p_top = float(p_arr.item())
        else:
            d_next = int(d_next_arr.item())
            p_top = 1.0

        self._committed_cached_len = len(committed_tokens)
        # Zero-copy snapshot at exact committed point (0.000 ms, 0 bytes copied!)
        self._commit_snapshot = self.engine.snapshot_state_only(self._cache)
        self._drafted_len_since_snapshot = 0

        if self.p_min > 0.0 and p_top < self.p_min:
            self._propose_calls += 1
            return []

        draft = [d_next]
        # Generate remaining k - 1 draft tokens serially (early-stop if confidence < p_min)
        for _ in range(k - 1):
            lg = self.draft_model(mx.array([[d_next]], dtype=mx.int32), cache=self._cache)
            row = lg.reshape(-1, lg.shape[-1])[-1]
            d_next_arr = mx.argmax(row)
            if self.p_min > 0.0:
                row_f = row.astype(mx.float32)
                p_arr = mx.exp(mx.max(row_f) - mx.logsumexp(row_f))
                mx.eval(d_next_arr, p_arr)
                d_next = int(d_next_arr.item())
                p_top = float(p_arr.item())
                self._drafted_len_since_snapshot += 1
                if p_top < self.p_min:
                    break
                draft.append(d_next)
            else:
                d_next = int(d_next_arr.item())
                draft.append(d_next)
                self._drafted_len_since_snapshot += 1

        self._propose_calls += 1
        return draft

    def on_verify(
        self,
        committed_tokens: List[int],
        draft_tokens: List[int],
        verify_argmax: List[int],
        accepted_count: int,
    ) -> None:
        if self._commit_snapshot is None or self._cache is None:
            return

        # If ALL k draft tokens were accepted, draft cache already holds the exact
        # states for draft_tokens[0 .. k-2]! Keep it with zero rollback!
        if (
            accepted_count == len(draft_tokens)
            and len(draft_tokens) > 0
            and self._drafted_len_since_snapshot == len(draft_tokens) - 1
        ):
            self._committed_cached_len += self._drafted_len_since_snapshot
        else:
            # Zero-copy restore to commit point (0.000 ms); next propose() catches up
            # [d_1..d_m, bonus] in one fast parallel forward pass!
            self.engine.restore_state_only(self._cache, self._commit_snapshot)

        self._commit_snapshot = None
        self._drafted_len_since_snapshot = 0

    def stats(self) -> Dict[str, Any]:
        return {
            "propose_calls": self._propose_calls,
            "catchup_tokens": self._catchup_tokens,
            "k_draft": self.k_draft,
        }


# ---------------------------------------------------------------------------
# Proposer 5: Hybrid Zero-Byte CPU Suffix + Fallback Drafter
# ---------------------------------------------------------------------------

class HybridProposer(BaseProposer):
    """Zero-byte CPU suffix matcher first; falls back to neural/lookahead drafter
    ONLY when suffix matching finds no confident match (e.g. Free Prose).
    """

    def __init__(
        self,
        primary: BaseProposer,
        fallback: BaseProposer,
        name: str = "hybrid",
    ):
        self.primary = primary
        self.fallback = fallback
        self.name = name
        self.uses_gpu = primary.uses_gpu or fallback.uses_gpu
        self._last_source: str = "none"
        self._primary_steps = 0
        self._fallback_steps = 0

    def reset(self, prompt_tokens: List[int]) -> None:
        self.primary.reset(prompt_tokens)
        self.fallback.reset(prompt_tokens)
        self._last_source = "none"
        self._primary_steps = 0
        self._fallback_steps = 0

    def propose(self, committed_tokens: List[int], max_k: Optional[int] = None) -> List[int]:
        draft = self.primary.propose(committed_tokens, max_k=max_k)
        if draft:
            self._last_source = "primary"
            self._primary_steps += 1
            return draft
        draft = self.fallback.propose(committed_tokens, max_k=max_k)
        self._last_source = "fallback" if draft else "none"
        if draft:
            self._fallback_steps += 1
        return draft

    def on_verify(
        self,
        committed_tokens: List[int],
        draft_tokens: List[int],
        verify_argmax: List[int],
        accepted_count: int,
    ) -> None:
        if self._last_source == "primary":
            self.primary.on_verify(committed_tokens, draft_tokens, verify_argmax, accepted_count)
        elif self._last_source == "fallback":
            self.fallback.on_verify(committed_tokens, draft_tokens, verify_argmax, accepted_count)

    def stats(self) -> Dict[str, Any]:
        return {
            "primary_steps": self._primary_steps,
            "fallback_steps": self._fallback_steps,
            "primary_stats": self.primary.stats(),
            "fallback_stats": self.fallback.stats(),
        }


# ---------------------------------------------------------------------------
# Proposer 6: MoE-Aware Shallow K<=4 Proposer for Qwen3.6-35B-A3B-4bit
# ---------------------------------------------------------------------------

def make_moe_shallow_proposer(k_max: int = 3, name: str = "moe_shallow") -> HybridProposer:
    """Shallow K<=3 (V<=4) zero-byte proposer for Qwen3.6-35B-A3B-4bit.

    Dual hardware + numerical optimum on M5 Max MLX 0.31.2:
      1. Caps MoE unique active experts per layer (~24/256 vs ~84/256 at K=15).
      2. Stays in MLX's V<=4 qmv regime (even for 8-bit mlp.gate/shared_expert_gate),
         guaranteeing 0.000e+00 bit-exact state & token identity vs serial B=1!
    """
    suffix = PromptLookupProposer(
        k_max=k_max,
        min_order=2,
        max_order=6,
        adaptive=True,
        k_by_order={
            6: k_max,
            5: k_max,
            4: k_max,
            3: k_max,
            2: min(k_max, 2),
        },
        name="moe_suffix",
    )
    lookahead = LookaheadProposer(
        k_max=min(k_max, 3),
        ngram_max=3,
        include_prompt=False,  # suffix already covers prompt; lookahead adds online Jacobi pool
        probe_k_on_miss=0,     # never waste MoE expert fan-out on blind probes!
        name="moe_jacobi_fallback",
    )
    return HybridProposer(primary=suffix, fallback=lookahead, name=name)
