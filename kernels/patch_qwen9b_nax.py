"""
Track A: Drop-in Metal 4 Split-K NAX Patch for `mlx-community/Qwen3.5-9B-4bit`.

Replaces qualifying `nn.QuantizedLinear` modules (affine Q4, group_size=64) with
`NAXQuantizedLinear`, which routes `M in [min_m, max_m]` (default 8..64) to our
Metal 4 Split-K NAX kernel (`tg_deq`, `u4_direct`, or `pad65`) while leaving B=1
decode and large-M prefill on stock MLX. Optionally stacks `gate_proj + up_proj`
into a single concatenated `24576 x 4096` projection (`StackedNAXMLP`).

Designed to be directly importable by Track D (speculative verification) AND
runnable standalone to verify greedy token parity and benchmark Verify K=1..64.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import metal4_nax_kernels as nk


class NAXQuantizedLinear(nn.Module):
    """Drop-in replacement for `nn.QuantizedLinear` routing M in [min_m, max_m] to Metal 4 NAX."""

    def __init__(
        self,
        orig: nn.QuantizedLinear,
        backend: str = "tg_deq",
        min_m: int = 8,
        max_m: int = 64,
    ):
        super().__init__()
        self.weight = orig.weight
        self.scales = orig.scales
        self.biases = orig.biases
        if "bias" in orig:
            self.bias = orig.bias
        self.group_size = orig.group_size
        self.bits = orig.bits
        self.mode = orig.mode
        self._nax_backend = backend
        self._min_m = min_m
        self._max_m = max_m
        self._out_dims = orig.weight.shape[0]
        self._in_dims = (orig.weight.shape[1] * 32) // orig.bits
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        M = x.size // x.shape[-1]
        if (
            self._nax_backend != "stock"
            and self._min_m <= M <= self._max_m
            and self.bits == 4
            and self.group_size == 64
            and self.mode == "affine"
            and self._out_dims >= 64
            and (self._in_dims & 63) == 0
            and x.dtype in (mx.float16, mx.bfloat16)
        ):
            y = nk.fast_qmm_nax(
                x, self.weight, self.scales, self.biases, backend=self._nax_backend
            )
        else:
            y = mx.quantized_matmul(
                x,
                self.weight,
                scales=self.scales,
                biases=self.biases,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
        if "bias" in self:
            y = y + self.bias
        return y


class StackedNAXMLP(nn.Module):
    """
    Fused-N MLP replacing separate `gate_proj` (12288x4096) + `up_proj` (12288x4096)
    with one concatenated `24576x4096` quantized matmul, plus NAX `down_proj`.
    """

    def __init__(
        self,
        orig_mlp: nn.Module,
        backend: str = "tg_deq",
        min_m: int = 8,
        max_m: int = 64,
    ):
        super().__init__()
        gp = orig_mlp.gate_proj
        up = orig_mlp.up_proj
        self._split_n = gp.weight.shape[0]
        self._in_dims = (gp.weight.shape[1] * 32) // gp.bits
        self.group_size = gp.group_size
        self.bits = gp.bits
        self.mode = gp.mode
        self._nax_backend = backend
        self._min_m = min_m
        self._max_m = max_m

        self.gu_weight = mx.concatenate([gp.weight, up.weight], axis=0)
        self.gu_scales = mx.concatenate([gp.scales, up.scales], axis=0)
        self.gu_biases = mx.concatenate([gp.biases, up.biases], axis=0)
        mx.eval(self.gu_weight, self.gu_scales, self.gu_biases)

        self.down_proj = NAXQuantizedLinear(
            orig_mlp.down_proj, backend=backend, min_m=min_m, max_m=max_m
        )
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        M = x.size // x.shape[-1]
        if (
            self._nax_backend != "stock"
            and self._min_m <= M <= self._max_m
            and x.dtype in (mx.float16, mx.bfloat16)
        ):
            gu = nk.fast_qmm_nax(
                x, self.gu_weight, self.gu_scales, self.gu_biases, backend=self._nax_backend
            )
        else:
            gu = mx.quantized_matmul(
                x,
                self.gu_weight,
                scales=self.gu_scales,
                biases=self.gu_biases,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
        gate = gu[..., : self._split_n]
        up = gu[..., self._split_n :]
        return self.down_proj(swiglu(gate, up))


def _is_eligible_qlinear(mod: Any) -> bool:
    return (
        isinstance(mod, nn.QuantizedLinear)
        and mod.bits == 4
        and mod.group_size == 64
        and mod.mode == "affine"
        and mod.weight.shape[0] >= 64
        and (((mod.weight.shape[1] * 32) // mod.bits) & 63) == 0
    )


def patch_qwen9b_model(
    model: nn.Module,
    backend: str = "tg_deq",
    min_m: int = 8,
    max_m: int = 64,
    stack_gate_up: bool = False,
    patch_lm_head: bool = True,
) -> Dict[str, Any]:
    """
    Patch `Qwen3.5-9B-4bit` in-place (reversibly by storing originals in returned dict).
    Returns `{"patched_count", "stacked_mlp_count", "restore": callable}`.
    """
    lm = getattr(model, "language_model", model)
    originals: List[Tuple[Any, str, Any]] = []
    patched_count = 0
    stacked_count = 0

    def _replace(parent: Any, attr: str, new_mod: Any):
        originals.append((parent, attr, getattr(parent, attr)))
        setattr(parent, attr, new_mod)

    for layer in lm.model.layers:
        # Attention / GatedDeltaNet projections
        attn = layer.linear_attn if layer.is_linear else layer.self_attn
        for name in ("in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"):
            sub = getattr(attn, name, None)
            if sub is not None and _is_eligible_qlinear(sub):
                _replace(attn, name, NAXQuantizedLinear(sub, backend=backend, min_m=min_m, max_m=max_m))
                patched_count += 1

        # MLP
        mlp = layer.mlp
        if (
            stack_gate_up
            and hasattr(mlp, "gate_proj")
            and hasattr(mlp, "up_proj")
            and hasattr(mlp, "down_proj")
            and _is_eligible_qlinear(mlp.gate_proj)
            and _is_eligible_qlinear(mlp.up_proj)
            and _is_eligible_qlinear(mlp.down_proj)
        ):
            _replace(layer, "mlp", StackedNAXMLP(mlp, backend=backend, min_m=min_m, max_m=max_m))
            stacked_count += 1
            patched_count += 3
        else:
            for name in ("gate_proj", "up_proj", "down_proj"):
                sub = getattr(mlp, name, None)
                if sub is not None and _is_eligible_qlinear(sub):
                    _replace(mlp, name, NAXQuantizedLinear(sub, backend=backend, min_m=min_m, max_m=max_m))
                    patched_count += 1

    # lm_head (248320 x 4096 = 572 MB — huge impact on verify K=8..64!)
    if patch_lm_head and hasattr(lm, "lm_head") and _is_eligible_qlinear(lm.lm_head):
        _replace(lm, "lm_head", NAXQuantizedLinear(lm.lm_head, backend=backend, min_m=min_m, max_m=max_m))
        patched_count += 1

    def restore():
        for parent, attr, orig in reversed(originals):
            setattr(parent, attr, orig)

    return {
        "patched_count": patched_count,
        "stacked_mlp_count": stacked_count,
        "patch_lm_head": patch_lm_head,
        "backend": backend,
        "min_m": min_m,
        "max_m": max_m,
        "restore": restore,
    }


def _clone_cache(cache: List[Any]) -> List[Any]:
    """Deep-copy hybrid KVCache + ArraysCache list so every verify pass sees identical prefix state."""
    from mlx_lm.models.cache import ArraysCache, KVCache

    out = []
    for c in cache:
        if isinstance(c, KVCache):
            nc = KVCache()
            if c.keys is not None:
                nc.keys = mx.array(c.keys)
                nc.values = mx.array(c.values)
            nc.offset = c.offset
            out.append(nc)
        elif isinstance(c, ArraysCache):
            nc = ArraysCache(size=len(c.cache))
            nc.cache = [mx.array(a) if a is not None else None for a in c.cache]
            nc.left_padding = mx.array(c.left_padding) if c.left_padding is not None else None
            nc.lengths = mx.array(c.lengths) if c.lengths is not None else None
            out.append(nc)
        else:
            raise TypeError(f"Unsupported cache type: {type(c)}")
    mx.eval([a for c in out for a in (c.cache if hasattr(c, "cache") else [c.keys, c.values]) if a is not None])
    return out


def prepare_verify_fixture(model: nn.Module, tokenizer: Any, max_k: int = 64):
    """
    Prefill a realistic prompt and generate `max_k` greedy reference tokens from stock model,
    returning `(base_cache, draft_tokens_mx)` where `draft_tokens_mx` has shape `(1, max_k)`.
    """
    prompt = (
        "Explain the roofline model for quantized LLM inference on Apple Silicon unified memory, "
        "comparing decode batch size 1 against speculative verification of 32 tokens."
    )
    prompt_ids = tokenizer.encode(prompt)
    cache = model.make_cache()
    logits = model(mx.array([prompt_ids]), cache=cache)
    mx.eval(logits)
    first_tok = int(mx.argmax(logits[0, -1]).item())
    base_cache = _clone_cache(cache)

    # Generate max_k greedy tokens for realistic verify block
    toks = [first_tok]
    cur = first_tok
    gen_cache = _clone_cache(base_cache)
    for _ in range(max_k - 1):
        out = model(mx.array([[cur]]), cache=gen_cache)
        cur = int(mx.argmax(out[0, -1]).item())
        toks.append(cur)
    mx.eval(mx.array(toks))
    return base_cache, mx.array([toks], dtype=mx.int32)


def benchmark_qwen9b_verify(
    model_id: str = "mlx-community/Qwen3.5-9B-4bit",
    k_values: Tuple[int, ...] = (1, 4, 8, 16, 32, 64),
    variants: Optional[List[Dict[str, Any]]] = None,
    iters: int = 5,
    warmup: int = 2,
    flush_slc: bool = True,
) -> Dict[str, Any]:
    """
    Load Qwen3.5-9B-4bit ONCE, verify exact greedy parity across K=1..64 for every variant,
    and run synchronized, SLC-flushed interleaved A/B verify timings.
    """
    from mlx_lm import load

    model, tokenizer = load(model_id)
    mx.synchronize()

    base_cache, all_tokens = prepare_verify_fixture(model, tokenizer, max_k=max(k_values))
    flush_buf = mx.zeros((128 * 1024 * 1024,), dtype=mx.float32) if flush_slc else None
    if flush_buf is not None:
        mx.eval(flush_buf)

    if variants is None:
        variants = [
            {"name": "tg_deq_nax", "backend": "tg_deq", "min_m": 8, "max_m": 64, "stack_gate_up": False, "patch_lm_head": True},
            {"name": "tg_deq_nax_stacked", "backend": "tg_deq", "min_m": 8, "max_m": 64, "stack_gate_up": True, "patch_lm_head": True},
            {"name": "u4_direct_nax", "backend": "u4_direct", "min_m": 8, "max_m": 64, "stack_gate_up": False, "patch_lm_head": True},
            {"name": "pad65_workaround", "backend": "pad65", "min_m": 8, "max_m": 64, "stack_gate_up": False, "patch_lm_head": True},
        ]

    # 1. Capture stock reference logits for every K
    stock_ref_logits: Dict[int, mx.array] = {}
    for K in k_values:
        c = _clone_cache(base_cache)
        lg = model(all_tokens[:, :K], cache=c)
        mx.eval(lg)
        stock_ref_logits[K] = lg.astype(mx.float32)
    mx.eval(*stock_ref_logits.values())
    mx.synchronize()

    def _do_flush():
        if flush_buf is not None:
            f = flush_buf + 1.0
            mx.eval(f)
            mx.synchronize()

    def _time_single_verify(K: int) -> float:
        c = _clone_cache(base_cache)
        _do_flush()
        t0 = time.perf_counter()
        out = model(all_tokens[:, :K], cache=c)
        mx.eval(out)
        mx.synchronize()
        return (time.perf_counter() - t0) * 1e3

    # 2. For each variant: verify parity + run interleaved A/B timings against stock!
    results: Dict[str, Any] = {
        "model_id": model_id,
        "k_values": list(k_values),
        "iters": iters,
        "flush_slc": flush_slc,
        "variants": {},
    }

    for v in variants:
        vname = v["name"]
        patch_info = patch_qwen9b_model(
            model,
            backend=v["backend"],
            min_m=v.get("min_m", 8),
            max_m=v.get("max_m", 64),
            stack_gate_up=v.get("stack_gate_up", False),
            patch_lm_head=v.get("patch_lm_head", True),
        )
        v_res: Dict[str, Any] = {
            "config": {k: val for k, val in v.items()},
            "patched_count": patch_info["patched_count"],
            "stacked_mlp_count": patch_info["stacked_mlp_count"],
            "per_k": {},
        }

        # Parity check across all K
        for K in k_values:
            c = _clone_cache(base_cache)
            lg = model(all_tokens[:, :K], cache=c)
            mx.eval(lg)
            lg32 = lg.astype(mx.float32)
            ref32 = stock_ref_logits[K]
            diff = mx.abs(lg32 - ref32)
            max_diff = float(mx.max(diff).item())
            mean_diff = float(mx.mean(diff).item())
            ref_toks = mx.argmax(ref32[0], axis=-1)
            got_toks = mx.argmax(lg32[0], axis=-1)
            matches = int(mx.sum(ref_toks == got_toks).item())
            v_res["per_k"][str(K)] = {
                "max_logit_diff": max_diff,
                "mean_logit_diff": mean_diff,
                "greedy_matches": matches,
                "greedy_total": K,
                "greedy_exact": (matches == K),
            }

        # Warmup patched model on all K
        for _ in range(warmup):
            for K in k_values:
                c = _clone_cache(base_cache)
                out = model(all_tokens[:, :K], cache=c)
                mx.eval(out)
        mx.synchronize()

        # Interleaved A/B timing per K: restore -> time stock -> re-patch -> time variant!
        # To keep patch/restore cheap and zero-copy, only do restore/re-patch if not stacking,
        # or reuse the exact StackedNAXMLP objects (no re-concat!).
        patch_info["restore"]()
        # Capture patched layer objects once so re-apply is O(1) pointer swap!
        patched_objs = patch_qwen9b_model(
            model,
            backend=v["backend"],
            min_m=v.get("min_m", 8),
            max_m=v.get("max_m", 64),
            stack_gate_up=v.get("stack_gate_up", False),
            patch_lm_head=v.get("patch_lm_head", True),
        )
        # Collect forward/restore swaps
        swaps: List[Tuple[Any, str, Any, Any]] = []
        # Instead of re-allocating stacked weights every iter, simply time patched and stock
        # interleaved by toggling `_nax_backend` on every patched module!
        patched_modules: List[Any] = []
        lm = getattr(model, "language_model", model)
        for layer in lm.model.layers:
            attn = layer.linear_attn if layer.is_linear else layer.self_attn
            for attr in ("in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"):
                sub = getattr(attn, attr, None)
                if isinstance(sub, NAXQuantizedLinear):
                    patched_modules.append(sub)
            if isinstance(layer.mlp, StackedNAXMLP):
                patched_modules.append(layer.mlp)
                patched_modules.append(layer.mlp.down_proj)
            else:
                for attr in ("gate_proj", "up_proj", "down_proj"):
                    sub = getattr(layer.mlp, attr, None)
                    if isinstance(sub, NAXQuantizedLinear):
                        patched_modules.append(sub)
        if hasattr(lm, "lm_head") and isinstance(lm.lm_head, NAXQuantizedLinear):
            patched_modules.append(lm.lm_head)

        def _set_backend(bname: str):
            for m in patched_modules:
                m._nax_backend = bname

        # If stacked MLP is enabled, stock control MUST use unstacked originals for honest A/B!
        # We can swap layer.mlp cleanly without reallocating tensors.
        mlp_pairs = []
        if v.get("stack_gate_up", False):
            # Pull original MLPs by temporarily restoring? Simpler: save patched MLPs, restore to get origs!
            stacked_mlps = [layer.mlp for layer in lm.model.layers]
            patched_objs["restore"]()
            orig_mlps = [layer.mlp for layer in lm.model.layers]
            mlp_pairs = list(zip(lm.model.layers, orig_mlps, stacked_mlps))
            # Re-apply non-MLP NAX patches
            patched_objs = patch_qwen9b_model(
                model,
                backend=v["backend"],
                min_m=v.get("min_m", 8),
                max_m=v.get("max_m", 64),
                stack_gate_up=False,
                patch_lm_head=v.get("patch_lm_head", True),
            )
            # Rebuild patched_modules list for non-MLP + stacked MLPs
            patched_modules = []
            for layer in lm.model.layers:
                attn = layer.linear_attn if layer.is_linear else layer.self_attn
                for attr in ("in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"):
                    sub = getattr(attn, attr, None)
                    if isinstance(sub, NAXQuantizedLinear):
                        patched_modules.append(sub)
                for attr in ("gate_proj", "up_proj", "down_proj"):
                    sub = getattr(layer.mlp, attr, None)
                    if isinstance(sub, NAXQuantizedLinear):
                        patched_modules.append(sub)
            for _, _, sm in mlp_pairs:
                patched_modules.append(sm)
                patched_modules.append(sm.down_proj)
            if hasattr(lm, "lm_head") and isinstance(lm.lm_head, NAXQuantizedLinear):
                patched_modules.append(lm.lm_head)

        for K in k_values:
            stock_ts: List[float] = []
            var_ts: List[float] = []
            # Warmup stock once for K
            if mlp_pairs:
                for layer, om, _ in mlp_pairs:
                    layer.mlp = om
            _set_backend("stock")
            _ = _time_single_verify(K)

            for it in range(iters):
                # Interleave order across iterations (AB, BA, AB, ...)
                order = ("stock", "var") if (it % 2 == 0) else ("var", "stock")
                for arm in order:
                    if arm == "stock":
                        if mlp_pairs:
                            for layer, om, _ in mlp_pairs:
                                layer.mlp = om
                        _set_backend("stock")
                        stock_ts.append(_time_single_verify(K))
                    else:
                        if mlp_pairs:
                            for layer, _, sm in mlp_pairs:
                                layer.mlp = sm
                        _set_backend(v["backend"])
                        var_ts.append(_time_single_verify(K))

            s_med = sorted(stock_ts)[len(stock_ts) // 2]
            v_med = sorted(var_ts)[len(var_ts) // 2]
            s_min = min(stock_ts)
            v_min = min(var_ts)
            v_res["per_k"][str(K)].update({
                "stock_ms_median": round(s_med, 3),
                "stock_ms_min": round(s_min, 3),
                "variant_ms_median": round(v_med, 3),
                "variant_ms_min": round(v_min, 3),
                "speedup_median": round(s_med / v_med, 3),
                "speedup_min": round(s_min / v_min, 3),
                "stock_trials_ms": [round(t, 3) for t in stock_ts],
                "variant_trials_ms": [round(t, 3) for t in var_ts],
            })

        # Restore completely clean stock model before next variant
        if mlp_pairs:
            for layer, om, _ in mlp_pairs:
                layer.mlp = om
        patched_objs["restore"]()
        results["variants"][vname] = v_res

    return results


def main():
    parser = argparse.ArgumentParser(description="Patch & benchmark Qwen3.5-9B-4bit Verify K=1..64")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-9B-4bit")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--backend", default="all", choices=["all", "tg_deq", "u4_direct", "pad65"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.backend == "all":
        variants = None
    else:
        variants = [
            {"name": f"{args.backend}_nax", "backend": args.backend, "min_m": 8, "max_m": 64, "stack_gate_up": False, "patch_lm_head": True},
        ]

    res = benchmark_qwen9b_verify(
        model_id=args.model,
        k_values=tuple(args.k_values),
        variants=variants,
        iters=args.iters,
    )
    print("\n=== Qwen3.5-9B-4bit Verify Summary ===")
    for vname, vr in res["variants"].items():
        print(f"\nVariant: {vname} (patched={vr['patched_count']}, stacked_mlp={vr['stacked_mlp_count']})")
        for K in res["k_values"]:
            r = vr["per_k"][str(K)]
            print(
                f"  K={K:2d}: stock={r['stock_ms_median']:6.2f} ms | {vname}={r['variant_ms_median']:6.2f} ms "
                f"({r['speedup_median']:.2f}x) | max_logit_diff={r['max_logit_diff']:.4f} | "
                f"greedy={r['greedy_matches']}/{r['greedy_total']}"
            )
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
