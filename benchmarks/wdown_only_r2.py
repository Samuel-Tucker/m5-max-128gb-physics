#!/usr/bin/env python3
"""Round-2 Check: W_down-Only Dynamic Magnitude Sparsity on Qwen3.5-9B-4bit.

Inspired by Sam's RTX 3070 Round 3 finding (gate+up exact, prune ONLY down_proj columns
with smallest |SiLU(gate) * up| magnitude). Tests zero-shot PPL/KL/greedy parity across
30%, 40%, 50%, 60% W_down pruning vs cold-DRAM W_down speedup on Apple M5 Max.
"""
import json, math, time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm.models.activations import swiglu
import numpy as np

EVAL_TEXT = (
    "In modern unified-memory architectures such as the Apple M5 Max, autoregressive language "
    "model decoding operates far below the arithmetic ridge point because every generated token "
    "requires streaming gigabytes of quantized weights from LPDDR5X DRAM for a single vector-matrix "
    "product. Meanwhile, per-core GPU Neural Accelerators deliver tens of teraflops that sit idle "
    "until batch or verification width exceeds roughly thirty tokens. Understanding how SwiGLU "
    "feed-forward activations distribute their energy across hidden channels reveals why naive "
    "gate-sign pruning destroys perplexity whereas post-activation magnitude pruning on the down "
    "projection can remove substantial weight traffic with minimal distortion."
)

def eval_ppl_and_logits(model, tokens):
    x = mx.array([tokens[:-1]], dtype=mx.int32)
    y = mx.array([tokens[1:]], dtype=mx.int32)
    logits = model(x).astype(mx.float32)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(logp, y[..., None], axis=-1).squeeze(-1)
    ppl = float(mx.exp(mx.mean(nll)).item())
    preds = mx.argmax(logits[0], axis=-1)
    mx.eval(preds, logp)
    return ppl, np.array(preds), np.array(logp[0])

def main():
    model, tok = mlx_lm.load("mlx-community/Qwen3.5-9B-4bit")
    tokens = tok.encode(EVAL_TEXT)
    base_ppl, base_preds, base_logp = eval_ppl_and_logits(model, tokens)
    base_p = np.exp(base_logp)
    print(f"Baseline Qwen3.5-9B-4bit ({len(tokens)} tok): PPL={base_ppl:.4f}")

    lm = model.language_model.model
    orig_calls = [type(l.mlp).__call__ for l in lm.layers]

    results = {"model_id": "mlx-community/Qwen3.5-9B-4bit", "tokens": len(tokens), "baseline_ppl": round(base_ppl, 4), "arms": []}

    for skip_frac in [0.30, 0.40, 0.50, 0.60, 0.70]:
        keep_k = int(round(12288 * (1.0 - skip_frac)))
        # Patch all 32 MLP blocks with exact gate+up, top-k magnitude mask on h_mid before down_proj
        orig_mlps = [l.mlp for l in lm.layers]
        class SparseDownMLP(nn.Module):
            def __init__(self, m, k_keep):
                super().__init__()
                self.m = m; self.k_keep = k_keep
            def __call__(self, x):
                h = swiglu(self.m.gate_proj(x), self.m.up_proj(x))
                mag = mx.abs(h)
                # threshold at (12288 - k_keep)-th smallest per token
                kth = 12288 - self.k_keep
                thresh = mx.partition(mag, kth=kth, axis=-1)[..., kth : kth + 1]
                h_sp = mx.where(mag >= thresh, h, mx.zeros_like(h))
                return self.m.down_proj(h_sp)

        for i, l in enumerate(lm.layers):
            l.mlp = SparseDownMLP(orig_mlps[i], keep_k)

        ppl, preds, logp = eval_ppl_and_logits(model, tokens)
        kl = float(np.mean(np.sum(base_p * (base_logp - logp), axis=-1)))
        agree = float(np.mean(preds == base_preds))
        dppl = ppl - base_ppl
        dppl_pct = (dppl / base_ppl) * 100.0
        # Restore
        for i, l in enumerate(lm.layers):
            l.mlp = orig_mlps[i]
        rec = {
            "wdown_skip_pct": int(skip_frac * 100),
            "keep_k": keep_k,
            "ppl": round(ppl, 4),
            "delta_ppl": round(dppl, 4),
            "delta_ppl_pct": round(dppl_pct, 2),
            "kl_div": round(kl, 5),
            "greedy_agree_pct": round(agree * 100, 2),
        }
        results["arms"].append(rec)
        print(f"  W_down skip {int(skip_frac*100)}% (keep {keep_k}/12288): PPL={ppl:.4f} (dPPL={dppl:+.4f}, {dppl_pct:+.2f}%) | KL={kl:.5f} | agree={agree*100:.1f}%")

    out_path = Path("results/track_c_wdown_r2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
