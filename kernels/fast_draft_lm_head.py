#!/usr/bin/env python3
"""
Fast Draft-Only LM Heads for MTP Speculative Decoding on Apple M5 Max.

Why this exists:
  - On Qwen3.5-9B-4bit: `mtp.layers.0` Q4 = 121 MB (~0.28 ms), while `lm_head` Q4
    (248,320 x 4096) = 572 MB (~1.24 ms) -> 82% of every MTP draft step is `lm_head`!
  - On Qwen3.6-35B-A3B-4bit: active MTP layer Q4 = 32 MB (~0.14 ms), while `lm_head` Q4
    (248,320 x 2048) = 286 MB (~0.65 ms) -> 84% of every MTP draft step is `lm_head`!
  - Because the TARGET model always verifies every draft token with its EXACT Q4 `lm_head`,
    a draft-only `lm_head` does NOT need to be lossless — it only needs high greedy top-1
    agreement at a fraction of the DRAM bytes!

Four pluggable draft-only heads benchmarked head-to-head:
  1. `ExactDraftLMHead`: target's exact Q4 `lm_head` (572 MB / 286 MB).
  2. `QuantizedDraftLMHead(bits=2|3)`: re-quantized full-vocab Q2 (297/149 MB) or Q3 copy.
  3. `ActiveVocabSubHead(vocab_ids)`: contiguous Q4 sub-matrix of ASCII/Latin/code/English
     tokens (e.g. 49,152 or 65,536 rows -> 113 MB on 9B, 56 MB on 35B!). Zero gather overhead!
  4. `PCAGatherDraftLMHead(rank=64, top_k=4096)`: SLC-resident `r=64` Q4 randomized-SVD
     coarse screen (~8.4 MB in 48 MB SLC!) + Track C's exact gathered Metal Q4 matvec kernel
     over ONLY the top-`top_k` surviving rows!
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class BaseDraftLMHead:
    name: str = "base"
    weight_bytes: int = 0

    def sample_greedy_and_prob(self, h_last: mx.array) -> Tuple[mx.array, mx.array]:
        """Given `h_last` shape `[1, 1, H]` (or `[1, H]`), return `(tok_id_scalar, max_prob_scalar)`."""
        raise NotImplementedError

    def argmax_only(self, h_last: mx.array) -> mx.array:
        """Fast GPU-only greedy argmax (skips FP32 softmax/logsumexp for zero-sync pipelining)."""
        tok, _ = self.sample_greedy_and_prob(h_last)
        return tok

    def logits_and_ids(self, h_last: mx.array) -> Tuple[mx.array, Optional[mx.array]]:
        raise NotImplementedError


def _get_target_lm_head(target_model: nn.Module) -> nn.Module:
    tm = target_model.language_model if hasattr(target_model, "language_model") else target_model
    if tm.args.tie_word_embeddings:
        return tm.model.embed_tokens
    return tm.lm_head


class ExactDraftLMHead(BaseDraftLMHead):
    """Exact target Q4 `lm_head` (zero extra memory)."""

    def __init__(self, target_model: nn.Module):
        self.name = "exact_q4"
        tm = target_model.language_model if hasattr(target_model, "language_model") else target_model
        self._tied = tm.args.tie_word_embeddings
        self._mod = _get_target_lm_head(target_model)
        self.weight_bytes = int(
            self._mod.weight.nbytes
            + getattr(self._mod, "scales", mx.array([], dtype=mx.uint8)).nbytes
            + getattr(self._mod, "biases", mx.array([], dtype=mx.uint8)).nbytes
        )

    def _project(self, h_last: mx.array) -> mx.array:
        if h_last.ndim == 3:
            h = h_last[0, -1:, :]
        elif h_last.ndim == 2:
            h = h_last[-1:, :]
        else:
            h = h_last[None, :]
        if self._tied:
            return self._mod.as_linear(h)[0]
        return self._mod(h)[0]

    def argmax_only(self, h_last: mx.array) -> mx.array:
        return mx.argmax(self._project(h_last)).astype(mx.int32)

    def sample_greedy_and_prob(self, h_last: mx.array) -> Tuple[mx.array, mx.array]:
        row = self._project(h_last)
        tok = mx.argmax(row).astype(mx.int32)
        row_f = row.astype(mx.float32)
        p_max = mx.exp(mx.max(row_f) - mx.logsumexp(row_f))
        return tok, p_max


class QuantizedDraftLMHead(BaseDraftLMHead):
    """Full-vocab re-quantized draft `lm_head` (bits=2 or bits=3, group_size=64)."""

    def __init__(self, target_model: nn.Module, bits: int = 2, group_size: int = 64):
        self.name = f"draft_q{bits}"
        self.bits = bits
        self.group_size = group_size
        mod = _get_target_lm_head(target_model)

        # Dequantize in row chunks to keep peak memory modest
        V = mod.weight.shape[0]
        H = mod.weight.shape[1] * (32 // mod.bits) if hasattr(mod, "bits") else mod.weight.shape[1]
        chunk = 32768
        w_parts, s_parts, b_parts = [], [], []
        for s0 in range(0, V, chunk):
            s1 = min(V, s0 + chunk)
            if hasattr(mod, "scales"):
                wf = mx.dequantize(
                    mod.weight[s0:s1],
                    mod.scales[s0:s1],
                    mod.biases[s0:s1],
                    group_size=mod.group_size,
                    bits=mod.bits,
                )
            else:
                wf = mod.weight[s0:s1]
            qw, qs, qb = mx.quantize(wf, group_size=group_size, bits=bits)
            mx.eval(qw, qs, qb)
            w_parts.append(qw)
            s_parts.append(qs)
            b_parts.append(qb)

        self.weight = mx.concatenate(w_parts, axis=0)
        self.scales = mx.concatenate(s_parts, axis=0)
        self.biases = mx.concatenate(b_parts, axis=0)
        mx.eval(self.weight, self.scales, self.biases)
        self.weight_bytes = int(self.weight.nbytes + self.scales.nbytes + self.biases.nbytes)

    def sample_greedy_and_prob(self, h_last: mx.array) -> Tuple[mx.array, mx.array]:
        h = h_last.reshape(-1, h_last.shape[-1])[-1:, :]
        row = mx.quantized_matmul(
            h,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )[0]
        tok = mx.argmax(row)
        row_f = row.astype(mx.float32)
        p_max = mx.exp(mx.max(row_f) - mx.logsumexp(row_f))
        return tok, p_max


def build_true_ascii_vocab_indices(
    tokenizer: Any,
    vocab_size: int,
    max_rows: int = 65536,
    include_latin1: bool = False,
    extra_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """Select up to `max_rows` TRUE UTF-8 decoded printable ASCII/code/special tokens in BPE rank order.

    Properly reverses GPT-2/Qwen byte-level BPE unicode mapping before checking UTF-8 ASCII,
    filtering out ~120K CJK/Arabic/Cyrillic/multilingual tokens cleanly.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    u2b = {chr(c): b for b, c in zip(bs, cs)}

    keep: List[int] = []
    seen = set()
    if extra_ids:
        for tid in extra_ids:
            if 0 <= tid < vocab_size and tid not in seen:
                keep.append(tid)
                seen.add(tid)

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for tid in sorted(special_ids):
        if 0 <= tid < vocab_size and tid not in seen:
            keep.append(tid)
            seen.add(tid)

    all_toks = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    extra_punct = {"—", "–", "’", "‘", "“", "”", "•", "…", "£", "€", "×", "→", "≤", "≥", "≈"}
    for tid, tstr in enumerate(all_toks):
        if tid in seen or tstr is None:
            continue
        try:
            s = bytes(u2b[ch] for ch in tstr).decode("utf-8")
        except Exception:
            continue
        if include_latin1:
            ok = all((32 <= ord(c) <= 255) or c in ("\n", "\r", "\t") or c in extra_punct for c in s)
        else:
            ok = all((32 <= ord(c) <= 126) or c in ("\n", "\r", "\t") for c in s)
        if ok:
            keep.append(tid)
            seen.add(tid)
            if len(keep) >= max_rows:
                break

    return np.array(sorted(keep), dtype=np.int32)


def build_latin_code_vocab_indices(
    tokenizer: Any,
    vocab_size: int,
    max_rows: int = 65536,
    extra_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """Select up to `max_rows` high-value ASCII/Latin/code/special token IDs.

    In Qwen3.5's BPE vocabulary (248,320 tokens), lower merge-rank IDs correspond to
    more frequent subwords, while ~140K+ higher-ID tokens are CJK/Arabic/Cyrillic/rare
    multilingual scripts. Filtering by printable ASCII + Latin-1 + special tokens in
    BPE merge-rank order yields a tiny, ultra-dense contiguous Q4 `lm_head` sub-matrix!
    """
    keep: List[int] = []
    seen = set()

    # Always include special tokens and extra prompt tokens first
    if extra_ids:
        for tid in extra_ids:
            if 0 <= tid < vocab_size and tid not in seen:
                keep.append(tid)
                seen.add(tid)

    # First 256 byte tokens + common special tokens
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for tid in sorted(special_ids):
        if 0 <= tid < vocab_size and tid not in seen:
            keep.append(tid)
            seen.add(tid)

    # Scan vocabulary in BPE merge-rank order (ascending token ID = higher frequency!)
    # Use tokenizer.convert_ids_to_tokens in batch for speed
    all_ids = list(range(vocab_size))
    toks = tokenizer.convert_ids_to_tokens(all_ids)
    # GPT-2/Qwen byte-level BPE printable ASCII/Latin chars include 'Ġ' (\u0120) and 'Ċ' (\u010a)
    for tid, tok in enumerate(toks):
        if tid in seen or tok is None:
            continue
        # Decode byte-level markers lightly: check if every char has ord <= 0x024F (Latin + BPE byte table)
        is_latin_bpe = True
        for ch in tok:
            oc = ord(ch)
            if oc > 0x024F and ch not in ("–", "—", "‘", "’", "“", "”", "•", "…", "£", "€", "×", "→", "≤", "≥", "≈"):
                is_latin_bpe = False
                break
        if is_latin_bpe:
            keep.append(tid)
            seen.add(tid)
            if len(keep) >= max_rows:
                break

    # Pad to multiple of 64 if needed
    arr = np.array(sorted(keep), dtype=np.int32)
    return arr


class ActiveVocabSubHead(BaseDraftLMHead):
    """Contiguous Q4 (or Q2) sub-matrix over an active vocab subset (e.g. 32K–65K rows).

    Zero gather kernel overhead — uses MLX's native contiguous `quantized_matmul` on
    only `V_sub / 248320` of the rows (4x–8x smaller than full `lm_head`!).
    """

    def __init__(
        self,
        target_model: nn.Module,
        vocab_ids: np.ndarray,
        bits: int = 4,
        group_size: int = 64,
        label: Optional[str] = None,
    ):
        self.vocab_ids_np = np.asarray(vocab_ids, dtype=np.int32)
        self.vocab_ids = mx.array(self.vocab_ids_np, dtype=mx.int32)
        self.v_sub = int(self.vocab_ids_np.shape[0])
        self.bits = bits
        self.group_size = group_size
        self.name = label or f"subvocab_{self.v_sub//1000}k_q{bits}"

        mod = _get_target_lm_head(target_model)
        if hasattr(mod, "scales") and mod.bits == bits and mod.group_size == group_size:
            self.weight = mx.contiguous(mod.weight[self.vocab_ids])
            self.scales = mx.contiguous(mod.scales[self.vocab_ids])
            self.biases = mx.contiguous(mod.biases[self.vocab_ids])
        else:
            # Dequantize selected rows and quantize to requested bits
            if hasattr(mod, "scales"):
                wf = mx.dequantize(
                    mod.weight[self.vocab_ids],
                    mod.scales[self.vocab_ids],
                    mod.biases[self.vocab_ids],
                    group_size=mod.group_size,
                    bits=mod.bits,
                )
            else:
                wf = mod.weight[self.vocab_ids]
            self.weight, self.scales, self.biases = mx.quantize(
                wf, group_size=group_size, bits=bits
            )
        mx.eval(self.weight, self.scales, self.biases, self.vocab_ids)
        self.weight_bytes = int(self.weight.nbytes + self.scales.nbytes + self.biases.nbytes)

    def argmax_only(self, h_last: mx.array) -> mx.array:
        h = h_last.reshape(-1, h_last.shape[-1])[-1:, :]
        row = mx.quantized_matmul(
            h,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )[0]
        return self.vocab_ids[mx.argmax(row)].astype(mx.int32)

    def sample_greedy_and_prob(self, h_last: mx.array) -> Tuple[mx.array, mx.array]:
        h = h_last.reshape(-1, h_last.shape[-1])[-1:, :]
        row = mx.quantized_matmul(
            h,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )[0]
        local_idx = mx.argmax(row)
        tok = self.vocab_ids[local_idx].astype(mx.int32)
        row_f = row.astype(mx.float32)
        p_max = mx.exp(mx.max(row_f) - mx.logsumexp(row_f))
        return tok, p_max


# Custom Metal gathered Q4 matvec kernel (from Track C — hits 2.52x at 25% rows, >5x at 2% rows!)
_GATHERED_Q4_MATVEC_SRC = r"""
uint gid = thread_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint tg_id = threadgroup_position_in_grid.x;

uint m_idx = tg_id;
if (m_idx >= M_out) return;

uint row = (uint)idx[m_idx];
const device uint32_t* w_row = W + (size_t)row * (size_t)K_u32;
const device T* s_row = S + (size_t)row * (size_t)K_grp;
const device T* b_row = B + (size_t)row * (size_t)K_grp;

float acc = 0.0f;
for (uint g = tid; g < K_grp; g += 32) {
    float sc = (float)s_row[g];
    float bi = (float)b_row[g];
    uint base_u32 = g * 8;
    uint base_k = g * 64;
    float g_dot = 0.0f;
    float x_sum = 0.0f;
    #pragma unroll
    for (uint j = 0; j < 8; ++j) {
        uint32_t pack = w_row[base_u32 + j];
        uint k0 = base_k + j * 8;
        #pragma unroll
        for (uint n = 0; n < 8; ++n) {
            float xv = (float)X[k0 + n];
            float qv = (float)((pack >> (n * 4)) & 0xFu);
            g_dot += xv * qv;
            x_sum += xv;
        }
    }
    acc += sc * g_dot + bi * x_sum;
}
acc = simd_sum(acc);
if (tid == 0) {
    OUT[m_idx] = (T)acc;
}
"""

_gathered_q4_kernel = None


def _get_gathered_q4_kernel():
    global _gathered_q4_kernel
    if _gathered_q4_kernel is None:
        _gathered_q4_kernel = mx.fast.metal_kernel(
            name="mtp_gathered_q4_matvec",
            input_names=["X", "W", "S", "B", "idx"],
            output_names=["OUT"],
            source=_GATHERED_Q4_MATVEC_SRC,
        )
    return _gathered_q4_kernel


class PCAGatherDraftLMHead(BaseDraftLMHead):
    """2-stage SLC-resident randomized-SVD coarse screen + exact Metal gathered Q4 `lm_head`.

    Stage 1: `z = h @ V_r` (`[1, r]`), `coarse = quantized_matmul(z, U_q4)` (`~8.4 MB` in SLC!).
    Stage 2: `top_idx = argpartition(coarse, -top_k)[-top_k:]` -> custom Metal `q4_gather_matvec`
             reading ONLY `top_k` rows of target's EXACT Q4 `lm_head` directly from DRAM!
    """

    def __init__(
        self,
        target_model: nn.Module,
        rank: int = 64,
        top_k: int = 4096,
        oversample: int = 16,
    ):
        self.name = f"pca_r{rank}_top{top_k}"
        self.rank = rank
        self.top_k = top_k
        mod = _get_target_lm_head(target_model)
        self.W = mod.weight
        self.S = mod.scales
        self.B = mod.biases
        self.group_size = mod.group_size
        self.bits = mod.bits
        self.V = int(self.W.shape[0])
        self.K_u32 = int(self.W.shape[1])
        self.K_grp = int(self.S.shape[1])
        self.H = self.K_u32 * (32 // self.bits)

        # Data-free randomized SVD on target lm_head (1 power iteration -> sharp spectrum!)
        rng = np.random.default_rng(42)
        p_dim = rank + oversample
        omega = mx.array(rng.standard_normal((self.H, p_dim)).astype(np.float32))
        # Y = W @ omega  [V, p_dim] via quantized_matmul (fast on GPU!)
        Y = mx.quantized_matmul(
            omega.T.astype(self.S.dtype),
            self.W,
            self.S,
            self.B,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        ).T.astype(mx.float32)  # [V, p_dim]
        mx.eval(Y)
        # Orthonormalize columns via small Gram matrix Y^T Y [p_dim, p_dim] on CPU/GPU
        # Better: compute B_mat = Y^T @ W via chunked dequantize? No, one power step via
        # Q_orth from Y: QR on CPU for [V, p_dim] is only ~0.4s!
        Y_np = np.array(Y, dtype=np.float32)
        Q_np, _ = np.linalg.qr(Y_np, mode="reduced")  # [V, p_dim]
        # Compute B_mat = Q^T @ W [p_dim, H] in chunks
        B_acc = np.zeros((p_dim, self.H), dtype=np.float32)
        chunk = 32768
        for s0 in range(0, self.V, chunk):
            s1 = min(self.V, s0 + chunk)
            wf = mx.dequantize(
                self.W[s0:s1], self.S[s0:s1], self.B[s0:s1],
                group_size=self.group_size, bits=self.bits,
            ).astype(mx.float32)
            qt = mx.array(Q_np[s0:s1].T)
            part = qt @ wf
            mx.eval(part)
            B_acc += np.array(part, dtype=np.float32)
        Ub, sb, Vt = np.linalg.svd(B_acc, full_matrices=False)
        # Keep top `rank`: W ≈ (Q @ Ub[:, :r] * sb[:r]) @ Vt[:r, :] = U_scaled @ V_r^T
        U_scaled = (Q_np @ Ub[:, :rank]) * sb[:rank][None, :]  # [V, rank]
        V_r = Vt[:rank, :].T  # [H, rank]

        self.V_r = mx.array(V_r, dtype=self.S.dtype)  # [H, rank]
        # Store U_scaled in Q4 (group_size=64) if rank divisible by 64, else FP16
        U_mx = mx.array(U_scaled, dtype=self.S.dtype)
        if rank % 64 == 0:
            self.U_w, self.U_s, self.U_b = mx.quantize(U_mx, group_size=64, bits=4)
            self._u_quant = True
            mx.eval(self.V_r, self.U_w, self.U_s, self.U_b)
            self.weight_bytes = int(self.V_r.nbytes + self.U_w.nbytes + self.U_s.nbytes + self.U_b.nbytes)
        else:
            self.U_fp = U_mx
            self._u_quant = False
            mx.eval(self.V_r, self.U_fp)
            self.weight_bytes = int(self.V_r.nbytes + self.U_fp.nbytes)

    def sample_greedy_and_prob(self, h_last: mx.array) -> Tuple[mx.array, mx.array]:
        h = h_last.reshape(-1, h_last.shape[-1])[-1:, :]
        z = h @ self.V_r  # [1, rank]
        if self._u_quant:
            coarse = mx.quantized_matmul(
                z, self.U_w, self.U_s, self.U_b, transpose=True, group_size=64, bits=4
            )[0]
        else:
            coarse = (z @ self.U_fp.T)[0]
        top_idx = mx.argpartition(coarse, kth=-self.top_k)[-self.top_k :].astype(mx.int32)
        kern = _get_gathered_q4_kernel()
        (sub_logits,) = kern(
            inputs=[h[0], self.W, self.S, self.B, top_idx],
            template=[
                ("T", h.dtype),
                ("M_out", self.top_k),
                ("K_u32", self.K_u32),
                ("K_grp", self.K_grp),
            ],
            grid=(self.top_k * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(self.top_k,)],
            output_dtypes=[h.dtype],
        )
        local_best = mx.argmax(sub_logits)
        tok = top_idx[local_best]
        row_f = sub_logits.astype(mx.float32)
        p_max = mx.exp(mx.max(row_f) - mx.logsumexp(row_f))
        return tok, p_max
