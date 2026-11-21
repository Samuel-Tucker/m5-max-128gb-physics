#!/usr/bin/env python3
"""Exact safetensors byte audit & M5 Max roofline calculator for cached Qwen3.5/3.6 models.

Reads ONLY safetensors JSON headers + config.json (zero weight tensor allocation).
Reports exact bytes for:
  - total safetensors tensor bytes (and on-disk file bytes)
  - vision tower bytes (never touched in text decode)
  - embedding table bytes (total vs 1-row read per decode token)
  - lm_head bytes (handles both untied lm_head and tied embed_tokens)
  - routed MoE expert bytes (total vs exact top-k / num_experts active fraction)
  - shared expert / router / GatedDeltaNet / attention / norm / dense MLP breakdown
  - whether ANY mtp.* weights exist in any cached checkpoint
  - exact achieved GB/s and % of M5 Max 503.2 GB/s copy & 614.0 GB/s peak roofline
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

TARGET_MODELS = [
    "mlx-community/Qwen3.5-2B-4bit",
    "mlx-community/Qwen3.5-4B-4bit",
    "mlx-community/Qwen3.5-9B-4bit",
    "mlx-community/Qwen3.6-35B-A3B-4bit",
]

# Measured B=1 decode baselines on this Apple M5 Max (2026-11-20, mlx_lm prompt=146 tok)
# From docs/council/20261120-rh-m5max/brief.md
MEASURED_DECODE_BASELINES: Dict[str, Dict[str, float]] = {
    "mlx-community/Qwen3.5-2B-4bit": {"decode_ms": 3.85, "decode_tps": 236.0},
    "mlx-community/Qwen3.5-9B-4bit": {"decode_ms": 10.66, "decode_tps": 91.3},
    "mlx-community/Qwen3.6-35B-A3B-4bit": {"decode_ms": 9.15, "decode_tps": 113.9},
}

ROOFLINE_COPY_GBS = 503.2
ROOFLINE_PEAK_GBS = 614.0


def find_snapshot_dir(repo_id: str, cache_dir: str = HF_CACHE_DIR) -> Path:
    folder = "models--" + repo_id.replace("/", "--")
    snaps = sorted(glob.glob(os.path.join(cache_dir, folder, "snapshots", "*")))
    if not snaps:
        raise FileNotFoundError(f"No cached HF snapshot found for {repo_id} under {cache_dir}")
    # Prefer snapshot containing config.json
    for s in reversed(snaps):
        if os.path.exists(os.path.join(s, "config.json")):
            return Path(s)
    return Path(snaps[-1])


def read_safetensors_header(sf_path: Path) -> Dict[str, Any]:
    with open(sf_path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr_bytes = f.read(hdr_len)
    return json.loads(hdr_bytes.decode("utf-8"))


def classify_tensor(key: str) -> str:
    kl = key.lower()
    if "mtp" in kl:
        return "mtp"
    if kl.startswith("vision_tower.") or kl.startswith("model.visual.") or ".visual." in kl:
        return "vision_tower"
    if "embed_tokens" in kl:
        return "embed_tokens"
    if "lm_head" in kl:
        return "lm_head"
    if "switch_mlp" in kl or ".mlp.experts." in kl:
        return "moe_routed_experts"
    if "shared_expert_gate" in kl or kl.endswith(".mlp.gate.weight") or kl.endswith(".mlp.gate.scales") or kl.endswith(".mlp.gate.biases"):
        return "moe_router_and_shared_gate"
    if "shared_expert." in kl:
        return "moe_shared_expert"
    if ".mlp." in kl:
        return "dense_mlp"
    if ".linear_attn." in kl:
        return "linear_attn_deltanet"
    if ".self_attn." in kl:
        return "self_attn"
    if "norm" in kl:
        return "norms"
    return "other_text"


def infer_effective_bits(tensor_entries: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Summarize quantized vs unquantized weight tensors."""
    by_prefix: Dict[str, Dict[str, Any]] = {}
    for k, meta in tensor_entries:
        parts = k.rsplit(".", 1)
        if len(parts) == 2 and parts[1] in ("weight", "scales", "biases"):
            by_prefix.setdefault(parts[0], {})[parts[1]] = meta
    bit_counts: Dict[str, int] = {}
    for prefix, grp in by_prefix.items():
        if "weight" in grp and "scales" in grp:
            w_shape = grp["weight"]["shape"]
            s_shape = grp["scales"]["shape"]
            w_dtype = grp["weight"]["dtype"]
            # MLX packs U32 along last dim; group_size=64 (or inferred from config)
            # bits = (w_last * 32) / (s_last * group_size)
            if w_dtype == "U32" and len(w_shape) == len(s_shape):
                ratio = (w_shape[-1] * 32) / max(1, s_shape[-1])
                # Assuming group_size=64 -> bits = ratio / 64
                bits_64 = round(ratio / 64)
                label = f"Q{bits_64}_gs64"
            else:
                label = f"quant_{w_dtype}"
            b = sum(m["data_offsets"][1] - m["data_offsets"][0] for m in grp.values())
            bit_counts[label] = bit_counts.get(label, 0) + b
        elif "weight" in grp:
            m = grp["weight"]
            label = f"dense_{m['dtype']}"
            b = m["data_offsets"][1] - m["data_offsets"][0]
            bit_counts[label] = bit_counts.get(label, 0) + b
    return bit_counts


def audit_model(repo_id: str, cache_dir: str = HF_CACHE_DIR) -> Dict[str, Any]:
    snap = find_snapshot_dir(repo_id, cache_dir)
    cfg = json.loads((snap / "config.json").read_text())
    tc = cfg.get("text_config", cfg)

    vocab_size = int(tc.get("vocab_size", cfg.get("vocab_size", 0)))
    hidden_size = int(tc.get("hidden_size", 0))
    num_layers = int(tc.get("num_hidden_layers", 0))
    num_experts = int(tc.get("num_experts", 0) or 0)
    num_experts_per_tok = int(tc.get("num_experts_per_tok", 0) or 0)
    tie_word_embeddings = bool(tc.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)))
    quant_cfg = cfg.get("quantization", {})
    default_bits = quant_cfg.get("bits", None)
    default_gs = quant_cfg.get("group_size", None)

    sf_files = sorted(snap.glob("*.safetensors"))
    if not sf_files:
        raise FileNotFoundError(f"No .safetensors files in {snap}")

    total_file_bytes = sum(p.stat().st_size for p in sf_files)
    cat_bytes: Dict[str, int] = {}
    cat_tensors: Dict[str, int] = {}
    mtp_keys: List[str] = []
    all_entries: List[Tuple[str, Dict[str, Any]]] = []
    text_entries: List[Tuple[str, Dict[str, Any]]] = []
    sample_mlp_shapes: Dict[str, Dict[str, Any]] = {}

    for sf in sf_files:
        hdr = read_safetensors_header(sf)
        for k, meta in hdr.items():
            if k == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            nbytes = int(end - start)
            cat = classify_tensor(k)
            cat_bytes[cat] = cat_bytes.get(cat, 0) + nbytes
            cat_tensors[cat] = cat_tensors.get(cat, 0) + 1
            all_entries.append((k, meta))
            if cat != "vision_tower":
                text_entries.append((k, meta))
            if cat == "mtp":
                mtp_keys.append(k)
            if "layers.0.mlp." in k:
                sample_mlp_shapes[k] = {
                    "dtype": meta["dtype"],
                    "shape": meta["shape"],
                    "bytes": nbytes,
                }

    total_tensor_bytes = sum(cat_bytes.values())
    vision_bytes = cat_bytes.get("vision_tower", 0)
    embed_bytes = cat_bytes.get("embed_tokens", 0)
    lm_head_separate_bytes = cat_bytes.get("lm_head", 0)
    routed_total_bytes = cat_bytes.get("moe_routed_experts", 0)
    mtp_bytes = cat_bytes.get("mtp", 0)

    # Active bytes per B=1 decode token:
    # 1. Embed lookup reads exactly 1 row out of vocab_size rows
    embed_active_row_bytes = (embed_bytes / vocab_size) if vocab_size > 0 else 0.0
    # 2. lm_head reads full lm_head; if tied and no separate lm_head exists, reads full embed_tokens
    if lm_head_separate_bytes > 0:
        lm_head_active_bytes = lm_head_separate_bytes
        lm_head_source = "separate_lm_head"
    elif tie_word_embeddings:
        lm_head_active_bytes = embed_bytes
        lm_head_source = "tied_embed_tokens"
    else:
        lm_head_active_bytes = 0
        lm_head_source = "missing"

    # 3. Routed MoE experts read exactly num_experts_per_tok / num_experts fraction at B=1
    if num_experts > 0 and num_experts_per_tok > 0:
        routed_active_bytes = routed_total_bytes * (num_experts_per_tok / num_experts)
    else:
        routed_active_bytes = routed_total_bytes

    # 4. All other text weights (always active every decode token)
    always_active_cats = [
        "dense_mlp",
        "moe_shared_expert",
        "moe_router_and_shared_gate",
        "linear_attn_deltanet",
        "self_attn",
        "norms",
        "other_text",
    ]
    always_active_text_bytes = sum(cat_bytes.get(c, 0) for c in always_active_cats)

    active_decode_bytes = (
        always_active_text_bytes
        + routed_active_bytes
        + lm_head_active_bytes
        + embed_active_row_bytes
    )

    # Check config for any MTP fields
    cfg_mtp_fields = {
        k: v
        for d in (cfg, tc)
        for k, v in d.items()
        if "mtp" in k.lower() or "nextn" in k.lower()
    }

    # Quantization breakdown of text weights
    text_quant_breakdown = infer_effective_bits(text_entries)
    per_layer_overrides = {
        k: v for k, v in quant_cfg.items() if isinstance(v, dict)
    }

    # Roofline analysis
    active_gb_si = active_decode_bytes / 1e9
    active_gib = active_decode_bytes / (1024**3)
    roofline: Dict[str, Any] = {
        "min_ms_at_503gbs_copy": (active_gb_si / ROOFLINE_COPY_GBS) * 1000.0,
        "min_ms_at_614gbs_peak": (active_gb_si / ROOFLINE_PEAK_GBS) * 1000.0,
        "max_tps_at_503gbs_copy": ROOFLINE_COPY_GBS / active_gb_si if active_gb_si > 0 else 0.0,
        "max_tps_at_614gbs_peak": ROOFLINE_PEAK_GBS / active_gb_si if active_gb_si > 0 else 0.0,
    }
    if repo_id in MEASURED_DECODE_BASELINES:
        bm = MEASURED_DECODE_BASELINES[repo_id]
        ms = bm["decode_ms"]
        tps = bm["decode_tps"]
        bw_from_ms = active_gb_si / (ms / 1000.0)
        bw_from_tps = active_gb_si * tps
        roofline.update({
            "measured_decode_ms": ms,
            "measured_decode_tps": tps,
            "achieved_gbs_from_ms": bw_from_ms,
            "achieved_gbs_from_tps": bw_from_tps,
            "pct_of_503gbs_copy_from_ms": 100.0 * bw_from_ms / ROOFLINE_COPY_GBS,
            "pct_of_614gbs_peak_from_ms": 100.0 * bw_from_ms / ROOFLINE_PEAK_GBS,
            "pct_of_503gbs_copy_from_tps": 100.0 * bw_from_tps / ROOFLINE_COPY_GBS,
            "pct_of_614gbs_peak_from_tps": 100.0 * bw_from_tps / ROOFLINE_PEAK_GBS,
            "b1_headroom_to_503gbs_x": ROOFLINE_COPY_GBS / bw_from_ms,
            "b1_headroom_to_614gbs_x": ROOFLINE_PEAK_GBS / bw_from_ms,
        })

    return {
        "repo_id": repo_id,
        "snapshot": str(snap),
        "num_safetensors_files": len(sf_files),
        "total_file_bytes": total_file_bytes,
        "total_tensor_bytes": total_tensor_bytes,
        "architecture": {
            "model_type": cfg.get("model_type"),
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "intermediate_size": tc.get("intermediate_size"),
            "moe_intermediate_size": tc.get("moe_intermediate_size"),
            "shared_expert_intermediate_size": tc.get("shared_expert_intermediate_size"),
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "vocab_size": vocab_size,
            "tie_word_embeddings": tie_word_embeddings,
            "default_bits": default_bits,
            "default_group_size": default_gs,
            "num_quant_overrides": len(per_layer_overrides),
        },
        "bytes_breakdown": {
            "vision_tower_bytes": vision_bytes,
            "embed_tokens_total_bytes": embed_bytes,
            "embed_tokens_1row_bytes": embed_active_row_bytes,
            "lm_head_separate_bytes": lm_head_separate_bytes,
            "lm_head_active_bytes": lm_head_active_bytes,
            "lm_head_source": lm_head_source,
            "moe_routed_total_bytes": routed_total_bytes,
            "moe_routed_active_bytes": routed_active_bytes,
            "always_active_text_bytes": always_active_text_bytes,
            "mtp_bytes": mtp_bytes,
            "active_decode_bytes_per_token": active_decode_bytes,
            "active_decode_gb_si": active_gb_si,
            "active_decode_gib": active_gib,
            "by_category": cat_bytes,
            "tensor_counts_by_category": cat_tensors,
            "text_quant_breakdown": text_quant_breakdown,
        },
        "mtp_audit": {
            "has_mtp_weights": len(mtp_keys) > 0,
            "mtp_tensor_count": len(mtp_keys),
            "mtp_keys_sample": mtp_keys[:20],
            "config_mtp_fields": cfg_mtp_fields,
        },
        "sample_layer0_mlp_shapes": sample_mlp_shapes,
        "roofline": roofline,
    }


def run_byte_audit(models: Optional[List[str]] = None) -> Dict[str, Any]:
    models = models or TARGET_MODELS
    results = {}
    for repo in models:
        results[repo] = audit_model(repo)
    any_mtp = any(r["mtp_audit"]["has_mtp_weights"] for r in results.values())
    return {
        "roofline_constants": {
            "copy_gbs": ROOFLINE_COPY_GBS,
            "peak_gbs": ROOFLINE_PEAK_GBS,
        },
        "any_mtp_weights_in_any_model": any_mtp,
        "models": results,
    }


def _fmt_gb(nbytes: float) -> str:
    return f"{nbytes / 1e9:.4f} GB ({nbytes / (1024**3):.4f} GiB)"


def print_report(report: Dict[str, Any]) -> None:
    print("=" * 96)
    print("EXACT SAFETENSORS BYTE AUDIT & APPLE M5 MAX ROOFLINE REPORT")
    print(f"Roofline refs: {ROOFLINE_COPY_GBS:.1f} GB/s measured R+W copy | {ROOFLINE_PEAK_GBS:.1f} GB/s theoretical peak")
    print("=" * 96)
    for repo, r in report["models"].items():
        b = r["bytes_breakdown"]
        a = r["architecture"]
        rf = r["roofline"]
        mtp = r["mtp_audit"]
        print(f"\n[{repo}] ({a['model_type']}, L={a['num_layers']}, H={a['hidden_size']}, V={a['vocab_size']}, tie={a['tie_word_embeddings']})")
        print(f"  Total safetensors tensor bytes : {_fmt_gb(r['total_tensor_bytes'])}  (disk files: {r['total_file_bytes']/1e9:.4f} GB)")
        print(f"  Vision tower (untouched text)  : {_fmt_gb(b['vision_tower_bytes'])}")
        print(f"  embed_tokens (total / 1-row)   : {_fmt_gb(b['embed_tokens_total_bytes'])} / {b['embed_tokens_1row_bytes']:.1f} B")
        print(f"  lm_head (active, {b['lm_head_source']:17s}): {_fmt_gb(b['lm_head_active_bytes'])}")
        if a["num_experts"] > 0:
            print(
                f"  MoE routed experts ({a['num_experts_per_tok']}/{a['num_experts']} active) : "
                f"total {_fmt_gb(b['moe_routed_total_bytes'])} -> active {_fmt_gb(b['moe_routed_active_bytes'])}"
            )
        print(f"  Always-active text weights     : {_fmt_gb(b['always_active_text_bytes'])}")
        for k in ("dense_mlp", "moe_shared_expert", "moe_router_and_shared_gate", "linear_attn_deltanet", "self_attn", "norms", "other_text"):
            if k in b["by_category"]:
                print(f"    - {k:28s}: {b['by_category'][k]/1e6:9.2f} MB  ({b['tensor_counts_by_category'][k]} tensors)")
        print(f"  Quantization mix (text)        : " + ", ".join(f"{k}={v/1e6:.1f}MB" for k, v in sorted(b["text_quant_breakdown"].items())))
        print(f"  >>> ACTIVE TEXT DECODE / TOK   : {_fmt_gb(b['active_decode_bytes_per_token'])}")
        print(f"  >>> MTP WEIGHTS PRESENT?       : {mtp['has_mtp_weights']} (count={mtp['mtp_tensor_count']}, cfg={mtp['config_mtp_fields']})")
        if "measured_decode_ms" in rf:
            print(
                f"  >>> ROOFLINE (from {rf['measured_decode_ms']:.2f} ms/tok): "
                f"{rf['achieved_gbs_from_ms']:.1f} GB/s = "
                f"{rf['pct_of_503gbs_copy_from_ms']:.1f}% of 503.2 copy | "
                f"{rf['pct_of_614gbs_peak_from_ms']:.1f}% of 614 peak "
                f"(B=1 headroom: {rf['b1_headroom_to_503gbs_x']:.2f}x to copy / {rf['b1_headroom_to_614gbs_x']:.2f}x to peak)"
            )
            print(
                f"      (from {rf['measured_decode_tps']:.1f} tok/s end-to-end): "
                f"{rf['achieved_gbs_from_tps']:.1f} GB/s = "
                f"{rf['pct_of_503gbs_copy_from_tps']:.1f}% of copy | "
                f"{rf['pct_of_614gbs_peak_from_tps']:.1f}% of peak"
            )
        else:
            print(
                f"  >>> ROOFLINE FLOOR             : "
                f"{rf['min_ms_at_503gbs_copy']:.2f} ms ({rf['max_tps_at_503gbs_copy']:.1f} tok/s) @ 503.2 GB/s | "
                f"{rf['min_ms_at_614gbs_peak']:.2f} ms ({rf['max_tps_at_614gbs_peak']:.1f} tok/s) @ 614 GB/s"
            )
    print("\n" + "=" * 96)
    print(f"ANY mtp.* WEIGHTS IN ANY CACHED CHECKPOINT: {report['any_mtp_weights_in_any_model']}")
    print("=" * 96)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact safetensors byte audit for cached Qwen3.5/3.6 models")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()
    report = run_byte_audit()
    print_report(report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Saved JSON audit to {out_path}")


if __name__ == "__main__":
    main()
