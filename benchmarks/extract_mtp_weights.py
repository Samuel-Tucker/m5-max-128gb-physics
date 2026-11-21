#!/usr/bin/env python3
"""
Extract ONLY the native `mtp.*` Multi-Token Prediction tensors from official
HuggingFace `Qwen/Qwen3.5-9B` (486.6 MB) and `Qwen/Qwen3.6-35B-A3B` (1,689.3 MB)
repositories via HTTP Range requests — zero full-model download!

Saves clean MLX safetensors files:
  - `weights/qwen35_9b_mtp_bf16.safetensors`
  - `weights/qwen35_9b_mtp_q4.safetensors`
  - `weights/qwen36_35b_mtp_q4.safetensors`
  (and prints exact norm weight distributions so we can verify zero-centered +1.0 shift.)
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np


OUT_DIR = Path(__file__).resolve().parent.parent / "weights"


def _http_get_bytes(url: str, start: int | None = None, end: int | None = None, retries: int = 4) -> bytes:
    headers = {"User-Agent": "python-urllib-mtp-extractor"}
    if start is not None and end is not None:
        headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                chunks = []
                while True:
                    chunk = resp.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                return b"".join(chunks)
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"  [retry {attempt+1}] {exc}", flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def fetch_mtp_tensors_bf16(repo: str) -> Dict[str, mx.array]:
    print(f"\n=== Fetching MTP tensors from {repo} ===", flush=True)
    idx_url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json"
    idx = json.loads(_http_get_bytes(idx_url).decode("utf-8"))
    wm: Dict[str, str] = idx["weight_map"]
    mtp_keys = sorted([k for k in wm if k.startswith("mtp.")])
    if not mtp_keys:
        raise RuntimeError(f"No mtp.* keys found in {repo}")

    by_shard: Dict[str, List[str]] = {}
    for k in mtp_keys:
        by_shard.setdefault(wm[k], []).append(k)

    tensors: Dict[str, mx.array] = {}
    total_mb = 0.0
    t0 = time.perf_counter()

    for shard, keys in sorted(by_shard.items()):
        shard_url = f"https://huggingface.co/{repo}/resolve/main/{shard}"
        hdr_len = struct.unpack("<Q", _http_get_bytes(shard_url, 0, 7))[0]
        hdr = json.loads(_http_get_bytes(shard_url, 8, 8 + hdr_len - 1).decode("utf-8"))
        data_base = 8 + hdr_len

        # Sort keys by byte offset and coalesce adjacent/nearby ranges (gap <= 1 MB)
        spans: List[Tuple[int, int, str, dict]] = []
        for k in keys:
            meta = hdr[k]
            s, e = meta["data_offsets"]
            spans.append((s, e, k, meta))
        spans.sort(key=lambda x: x[0])

        groups: List[List[Tuple[int, int, str, dict]]] = []
        for sp in spans:
            if not groups or sp[0] - groups[-1][-1][1] > 1 * 1024 * 1024:
                groups.append([sp])
            else:
                groups[-1].append(sp)

        for grp in groups:
            g_start = grp[0][0]
            g_end = grp[-1][1]
            nbytes = g_end - g_start
            print(
                f"  Range GET {shard} [{g_start}:{g_end}] ({nbytes/1e6:.2f} MB, {len(grp)} tensors)...",
                flush=True,
            )
            raw = _http_get_bytes(shard_url, data_base + g_start, data_base + g_end - 1)
            assert len(raw) == nbytes, f"Expected {nbytes} bytes, got {len(raw)}"
            total_mb += nbytes / 1e6

            for s, e, k, meta in grp:
                rel_s = s - g_start
                rel_e = e - g_start
                dtype_str = meta["dtype"]
                shape = tuple(meta["shape"])
                buf = raw[rel_s:rel_e]
                if dtype_str == "BF16":
                    u16 = np.frombuffer(buf, dtype=np.uint16).reshape(shape).copy()
                    arr = mx.array(u16).view(mx.bfloat16)
                elif dtype_str == "F16":
                    f16 = np.frombuffer(buf, dtype=np.float16).reshape(shape).copy()
                    arr = mx.array(f16)
                elif dtype_str == "F32":
                    f32 = np.frombuffer(buf, dtype=np.float32).reshape(shape).copy()
                    arr = mx.array(f32)
                else:
                    raise ValueError(f"Unsupported dtype {dtype_str} for {k}")
                tensors[k] = arr

    dt = time.perf_counter() - t0
    print(f"Downloaded {len(tensors)} MTP tensors ({total_mb:.1f} MB) in {dt:.1f}s ({total_mb/max(dt,1e-6):.1f} MB/s)")
    print("RMSNorm weight stats (RAW from HF before +1.0 shift):")
    for k in sorted(tensors):
        if "norm" in k:
            v = tensors[k].astype(mx.float32)
            mx.eval(v)
            print(
                f"  {k:48s} shape={list(v.shape)} mean={float(mx.mean(v)):+.4f} "
                f"std={float(mx.std(v)):.4f} min={float(mx.min(v)):+.4f} max={float(mx.max(v)):+.4f}"
            )
    return tensors


def quantize_mtp_dict(
    raw_tensors: Dict[str, mx.array],
    is_moe: bool,
    bits: int = 4,
    group_size: int = 64,
    shift_norms: bool = True,
) -> Dict[str, mx.array]:
    """Sanitize (split MoE gate_up, +1.0 shift norms) and quantize linear/expert weights."""
    out: Dict[str, mx.array] = {}
    work = dict(raw_tensors)

    if is_moe:
        gu_key = "mtp.layers.0.mlp.experts.gate_up_proj"
        dn_key = "mtp.layers.0.mlp.experts.down_proj"
        if gu_key in work:
            gu = work.pop(gu_key)
            mid = gu.shape[-2] // 2
            work["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"] = gu[..., :mid, :]
            work["mtp.layers.0.mlp.switch_mlp.up_proj.weight"] = gu[..., mid:, :]
        if dn_key in work:
            work["mtp.layers.0.mlp.switch_mlp.down_proj.weight"] = work.pop(dn_key)

    for k, v in sorted(work.items()):
        if "norm" in k and v.ndim == 1:
            out[k] = (v.astype(mx.float32) + 1.0).astype(mx.bfloat16) if shift_norms else v
            continue

        # Router gates stay 8-bit (matching mlx_lm Qwen3.5 quant_predicate) or BF16 if small
        if k.endswith("mlp.gate.weight") or k.endswith("shared_expert_gate.weight"):
            if v.shape[-1] % group_size == 0 and v.ndim >= 2:
                qw, qs, qb = mx.quantize(v, group_size=group_size, bits=8)
                base = k[:-len(".weight")]
                out[f"{base}.weight"] = qw
                out[f"{base}.scales"] = qs
                out[f"{base}.biases"] = qb
            else:
                out[k] = v
            continue

        # All other 2D/3D weights -> Q4 (group_size=64)
        if v.ndim >= 2 and v.shape[-1] % group_size == 0:
            qw, qs, qb = mx.quantize(v, group_size=group_size, bits=bits)
            base = k[:-len(".weight")] if k.endswith(".weight") else k
            out[f"{base}.weight"] = qw
            out[f"{base}.scales"] = qs
            out[f"{base}.biases"] = qb
        else:
            out[k] = v

    mx.eval(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["9b", "35b"], choices=["9b", "35b"])
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if "9b" in args.models:
        bf16_path = OUT_DIR / "qwen35_9b_mtp_bf16.safetensors"
        q4_path = OUT_DIR / "qwen35_9b_mtp_q4.safetensors"
        raw = fetch_mtp_tensors_bf16("Qwen/Qwen3.5-9B")
        # Save BF16 (with +1.0 shifted norms too so BF16 vs Q4 uses identical sanitized weights)
        bf16_san = {}
        for k, v in raw.items():
            bf16_san[k] = (v.astype(mx.float32) + 1.0).astype(mx.bfloat16) if ("norm" in k and v.ndim == 1) else v
        mx.eval(bf16_san)
        mx.save_safetensors(str(bf16_path), bf16_san)
        print(f"Saved {bf16_path} ({bf16_path.stat().st_size / 1e6:.1f} MB)")

        q4 = quantize_mtp_dict(raw, is_moe=False, bits=4, group_size=64, shift_norms=True)
        mx.save_safetensors(str(q4_path), q4)
        print(f"Saved {q4_path} ({q4_path.stat().st_size / 1e6:.1f} MB)")

    if "35b" in args.models:
        q4_path = OUT_DIR / "qwen36_35b_mtp_q4.safetensors"
        raw = fetch_mtp_tensors_bf16("Qwen/Qwen3.6-35B-A3B")
        q4 = quantize_mtp_dict(raw, is_moe=True, bits=4, group_size=64, shift_norms=True)
        mx.save_safetensors(str(q4_path), q4)
        print(f"Saved {q4_path} ({q4_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
