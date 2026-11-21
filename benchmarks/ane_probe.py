#!/usr/bin/env python3
"""
Track E.4 — Apple Neural Engine (ANE) / CoreML Real Hardware Probe (M5 Max)

Zero dependency on coremltools!
Generates valid CoreML NeuralNetwork .mlmodel protobuf files directly in pure Python
(1x1 Conv2D in NCHW FP16 weights — Apple ANE's native fast path), compiles them via
`xcrun coremlcompiler compile`, verifies exact ANE placement via Apple's official
`MLComputePlan` API in Swift (`ane_runner`), and measures:
  1. ANE invocation latency floor (small 4096->1024->4096 draft MLP, S=1 and S=16).
  2. ANE compute & DRAM throughput (4096x4096 1x1 Conv at S=1, S=64, S=256).
  3. ANE vs CPU vs GPU placement & latency comparison.
  4. Concurrent ANE + GPU 2 GB pure-read DRAM contention test (both small draft MLP
     and 134 MB >SLC 4-layer stack running concurrently with GPU pure-read!).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

_MLX_PY = "python3"
try:
    import mlx.core as mx  # noqa: F401
except ImportError:
    if os.path.exists(_MLX_PY) and os.path.realpath(sys.executable) != os.path.realpath(_MLX_PY):
        os.execv(_MLX_PY, [_MLX_PY] + sys.argv)
    raise

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS_DIR = os.path.join(os.path.dirname(HERE), "kernels")
RUNNER_SWIFT = os.path.join(KERNELS_DIR, "ane_runner.swift")
RUNNER_BIN = os.path.join(KERNELS_DIR, "ane_runner")
PURE_READ_PY = os.path.join(HERE, "pure_read_bw.py")


# ---------------------------------------------------------------------------
# Pure-Python CoreML Protobuf Builder (zero coremltools needed!)
# ---------------------------------------------------------------------------

def _varint(v: int) -> bytes:
    out = bytearray()
    while v > 0x7F:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)
    return bytes(out)


def _f_varint(num: int, v: int) -> bytes:
    return _varint((num << 3) | 0) + _varint(v)


def _f_bytes(num: int, b: bytes) -> bytes:
    return _varint((num << 3) | 2) + _varint(len(b)) + b


def _f_str(num: int, s: str) -> bytes:
    return _f_bytes(num, s.encode("utf-8"))


def _make_array_feature(name: str, shape: list[int], dtype: int = 65568) -> bytes:
    # dtype: 65568 = FLOAT32, 65552 = FLOAT16 (spec >= 7)
    shape_bytes = b"".join(_varint(s) for s in shape)
    aft = _f_bytes(1, shape_bytes) + _f_varint(2, dtype)
    ft = _f_bytes(5, aft)
    return _f_str(1, name) + _f_bytes(3, ft)


def _make_conv1x1_layer(name: str, inp: str, out: str, c_in: int, c_out: int, seed_val: int = 0x2666) -> bytes:
    """
    Build CoreML ConvolutionLayerParams (1x1 conv with FP16 weights in WeightParams.float16Value=2).
    Official CoreML NeuralNetwork.proto field tags:
      outputChannels=1, kernelChannels=2, nGroups=10, kernelSize=20, stride=30,
      dilationFactor=40, valid=50, isDeconvolution=60, hasBias=70, weights=90.
    """
    # Balanced small FP16 pattern (~ +/-0.004 with tiny positive net mean ~+0.00015)
    # so chained 4096-wide layers stay finite & non-zero without FP16 overflow.
    pat = struct.pack("<8H", 0x1c00, 0x9c00, 0x1a00, 0x9a00, 0x1d00, 0x9c00, 0x1800, 0x9800)
    total_elems = c_out * c_in
    assert total_elems % 8 == 0
    w_bytes = pat * (total_elems // 8)
    wp = _f_bytes(2, w_bytes)  # WeightParams.float16Value = field 2
    conv = (
        _f_varint(1, c_out)
        + _f_varint(2, c_in)
        + _f_varint(10, 1)  # nGroups
        + _f_bytes(20, _varint(1) + _varint(1))  # kernelSize [1, 1]
        + _f_bytes(30, _varint(1) + _varint(1))  # stride [1, 1]
        + _f_bytes(40, _varint(1) + _varint(1))  # dilationFactor [1, 1]
        + _f_bytes(50, b"")  # ValidPadding
        + _f_bytes(90, wp)   # weights
    )
    return _f_str(1, name) + _f_str(2, inp) + _f_str(3, out) + _f_bytes(100, conv)


def _make_relu_layer(name: str, inp: str, out: str) -> bytes:
    # ActivationParams = field 130; ActivationReLU = field 10 (empty message)
    act = _f_bytes(10, b"")
    return _f_str(1, name) + _f_str(2, inp) + _f_str(3, out) + _f_bytes(130, act)


def build_mlmodel_conv_stack(
    path: str,
    dims: list[tuple[int, int]],
    seq_len: int,
    *,
    with_relu: bool = False,
    fp16_io: bool = True,
) -> dict:
    """Write a CoreML NeuralNetwork .mlmodel containing chained 1x1 FP16 Conv2D layers."""
    layers_bytes = bytearray()
    cur = "x"
    total_params = 0
    for idx, (cin, cout) in enumerate(dims):
        is_last = idx == len(dims) - 1
        out_name = "y" if is_last else f"h{idx}"
        layers_bytes += _f_bytes(1, _make_conv1x1_layer(f"conv{idx}", cur, out_name, cin, cout))
        total_params += cin * cout
        cur = out_name
        if with_relu and not is_last:
            act_out = f"a{idx}"
            layers_bytes += _f_bytes(1, _make_relu_layer(f"relu{idx}", cur, act_out))
            cur = act_out

    # arrayInputShapeMapping = 3 (EXACT_ARRAY_MAPPING)
    nn = bytes(layers_bytes) + _f_varint(10, 3)
    spec_ver = 7 if fp16_io else 4
    io_dtype = 65552 if fp16_io else 65568
    c_first = dims[0][0]
    c_last = dims[-1][1]
    # CoreML NeuralNetwork image-like MLMultiArray uses 3D [C, H, W] (implicit batch=1)
    desc = (
        _f_bytes(1, _make_array_feature("x", [c_first, 1, seq_len], io_dtype))
        + _f_bytes(10, _make_array_feature("y", [c_last, 1, seq_len], io_dtype))
    )
    model_bytes = _f_varint(1, spec_ver) + _f_bytes(2, desc) + _f_bytes(500, nn)
    with open(path, "wb") as f:
        f.write(model_bytes)
    return {
        "total_params": total_params,
        "weight_fp16_mb": round(total_params * 2 / 1e6, 2),
        "flops_per_call": 2 * total_params * seq_len,
    }


def ensure_runner_compiled() -> None:
    if (
        not os.path.exists(RUNNER_BIN)
        or os.path.getmtime(RUNNER_SWIFT) > os.path.getmtime(RUNNER_BIN)
    ):
        subprocess.run(
            ["swiftc", "-O", "-framework", "CoreML", "-framework", "Foundation", RUNNER_SWIFT, "-o", RUNNER_BIN],
            check=True,
        )


def compile_mlmodel(mlmodel_path: str, out_dir: str) -> str:
    r = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", mlmodel_path, out_dir],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"coremlcompiler failed: {r.stderr}\n{r.stdout}")
    base = os.path.splitext(os.path.basename(mlmodel_path))[0] + ".mlmodelc"
    return os.path.join(out_dir, base)


def run_swift_json(cmd: list[str]) -> dict:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": r.stderr.strip() or r.stdout.strip(), "returncode": r.returncode}
    return json.loads(r.stdout)


def bench_model(
    work_dir: str,
    name: str,
    dims: list[tuple[int, int]],
    seq_len: int,
    *,
    with_relu: bool = False,
    units_list: list[str] = ("ane",),
    iters: int = 30,
    warmup: int = 5,
    fp16_io: bool = True,
) -> tuple[dict, str]:
    mlmodel_path = os.path.join(work_dir, f"{name}.mlmodel")
    meta = build_mlmodel_conv_stack(mlmodel_path, dims, seq_len, with_relu=with_relu, fp16_io=fp16_io)
    mlmodelc = compile_mlmodel(mlmodel_path, work_dir)
    os.remove(mlmodel_path)

    unit_results = {}
    for units in units_list:
        cmd = [
            RUNNER_BIN,
            "--bench", mlmodelc,
            "--units", units,
            "--c", str(dims[0][0]),
            "--s", str(seq_len),
            "--iters", str(iters),
            "--warmup", str(warmup),
        ]
        if fp16_io:
            cmd.append("--fp16")
        res = run_swift_json(cmd)
        if "median_ms" in res and res["median_ms"] > 0:
            med_s = res["median_ms"] / 1000.0
            res["effective_tflops"] = round((meta["flops_per_call"] / 1e12) / med_s, 3)
            res["effective_weight_gb_s"] = round((meta["total_params"] * 2 / 1e9) / med_s, 2)
        unit_results[units] = res

    return {
        "name": name,
        "dims": dims,
        "seq_len": seq_len,
        "fp16_io": fp16_io,
        **meta,
        "runs": unit_results,
    }, mlmodelc


def run_concurrent_ane_gpu(mlmodelc: str, c_in: int, seq_len: int, seconds: float, fp16_io: bool) -> dict:
    """Measure solo vs concurrent ANE + GPU 2 GB pure-read DRAM streaming."""
    ane_cmd = [
        RUNNER_BIN, "--loop", mlmodelc,
        "--units", "ane", "--c", str(c_in), "--s", str(seq_len),
        "--seconds", str(seconds),
    ]
    if fp16_io:
        ane_cmd.append("--fp16")
    gpu_cmd = [sys.executable, PURE_READ_PY, "--stream-seconds", str(seconds)]

    # Solo baselines
    ane_solo = run_swift_json(ane_cmd)
    gpu_solo = run_swift_json(gpu_cmd)

    # Concurrent execution (launch both simultaneously)
    p_ane = subprocess.Popen(ane_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    p_gpu = subprocess.Popen(gpu_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ane_out, ane_err = p_ane.communicate()
    gpu_out, gpu_err = p_gpu.communicate()
    ane_conc = json.loads(ane_out) if p_ane.returncode == 0 else {"error": ane_err}
    gpu_conc = json.loads(gpu_out) if p_gpu.returncode == 0 else {"error": gpu_err}

    gpu_retention_pct = None
    ane_slowdown = None
    if "gpu_pure_read_gb_s" in gpu_solo and "gpu_pure_read_gb_s" in gpu_conc:
        gpu_retention_pct = round(100.0 * gpu_conc["gpu_pure_read_gb_s"] / gpu_solo["gpu_pure_read_gb_s"], 2)
    if "mean_ms" in ane_solo and "mean_ms" in ane_conc:
        ane_slowdown = round(ane_conc["mean_ms"] / ane_solo["mean_ms"], 3)

    return {
        "ane_solo": ane_solo,
        "gpu_solo": gpu_solo,
        "ane_concurrent": ane_conc,
        "gpu_concurrent": gpu_conc,
        "gpu_bw_retention_pct": gpu_retention_pct,
        "ane_latency_ratio_concurrent_vs_solo": ane_slowdown,
    }


def run_all(*, quick: bool = False) -> dict:
    ensure_runner_compiled()
    info = run_swift_json([RUNNER_BIN, "--info"])
    iters = 15 if quick else 50
    conc_sec = 2.0 if quick else 4.0

    with tempfile.TemporaryDirectory(prefix="track_e_ane_") as tmp:
        models = {}

        # 0. Invocation latency floor: tiny 512x512 (0.5 MB)
        m_tiny, _ = bench_model(
            tmp, "tiny_512x512_s1",
            [(512, 512)], seq_len=1,
            units_list=["ane", "cpu"], iters=iters, warmup=5, fp16_io=True,
        )
        models["tiny_512_s1"] = m_tiny

        # 1. Tiny L2/SRAM-scale draft MLP: 4096 -> 1024 -> 4096 (16.8 MB FP16)
        m_draft_s1, draft_mlc = bench_model(
            tmp, "draft_mlp_4096_1024_4096_s1",
            [(4096, 1024), (1024, 4096)], seq_len=1, with_relu=True,
            units_list=["ane", "gpu", "cpu"], iters=iters, warmup=5, fp16_io=True,
        )
        models["draft_mlp_s1"] = m_draft_s1

        m_draft_s16, _ = bench_model(
            tmp, "draft_mlp_4096_1024_4096_s16",
            [(4096, 1024), (1024, 4096)], seq_len=16, with_relu=True,
            units_list=["ane", "gpu"], iters=iters, warmup=5, fp16_io=True,
        )
        models["draft_mlp_s16"] = m_draft_s16

        # 2. Single 4096x4096 GEMM (33.6 MB FP16) + SRAM-friendly compute peak (4x2048 S=256)
        m_gemm_s1, _ = bench_model(
            tmp, "gemm_4096x4096_s1",
            [(4096, 4096)], seq_len=1,
            units_list=["ane", "gpu"], iters=iters, warmup=5, fp16_io=True,
        )
        models["gemm_4096_s1"] = m_gemm_s1

        m_gemm_s64, _ = bench_model(
            tmp, "gemm_4096x4096_s64",
            [(4096, 4096)], seq_len=64,
            units_list=["ane", "gpu"], iters=iters, warmup=5, fp16_io=True,
        )
        models["gemm_4096_s64"] = m_gemm_s64

        m_peak_4x2048, _ = bench_model(
            tmp, "peak_4x2048x2048_s256",
            [(2048, 2048)] * 4, seq_len=256,
            units_list=["ane"], iters=iters, warmup=5, fp16_io=True,
        )
        models["peak_4x2048_s256"] = m_peak_4x2048

        # 3. 4-layer 4096x4096 stack (134.2 MB > 48 MB SLC!) -> true ANE DRAM streaming & GPU contention
        m_dram4, dram4_mlc = bench_model(
            tmp, "dram_4x4096x4096_s1",
            [(4096, 4096)] * 4, seq_len=1,
            units_list=["ane"], iters=max(10, iters // 2), warmup=3, fp16_io=True,
        )
        models["dram_4x4096_s1"] = m_dram4

        # 4. Concurrent ANE + GPU 2 GB pure-read DRAM contention tests
        conc_draft = run_concurrent_ane_gpu(draft_mlc, 4096, 1, seconds=conc_sec, fp16_io=True)
        conc_dram4 = run_concurrent_ane_gpu(dram4_mlc, 4096, 1, seconds=conc_sec, fp16_io=True)

    return {
        "coremltools_installed": False,
        "method": "Zero-dependency pure-Python CoreML NeuralNetwork v7 FP16 protobuf + xcrun coremlcompiler + Swift MLModel/MLComputePlan",
        "device_info": info,
        "models": models,
        "concurrent_gpu_contention": {
            "draft_mlp_16p8MB_vs_gpu_2GB_read": conc_draft,
            "dram_4x4096_134MB_vs_gpu_2GB_read": conc_dram4,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Track E.4: Apple M5 Max CoreML/ANE Probe")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    res = run_all(quick=args.quick)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)

    print("=== ANE Device Info ===")
    print(json.dumps(res["device_info"], indent=2))
    print("\n=== ANE Model Benchmarks ===")
    for k, m in res["models"].items():
        ane = m["runs"].get("ane", {})
        plan = ane.get("compute_plan", {})
        print(
            f"  {k:20s} ({m['weight_fp16_mb']:5.1f} MB): "
            f"ANE med={ane.get('median_ms', -1):6.3f} ms | "
            f"{ane.get('effective_tflops', 0):6.3f} TFLOP/s | "
            f"{ane.get('effective_weight_gb_s', 0):6.1f} GB/s | "
            f"plan(ANE={plan.get('ane_layers')},GPU={plan.get('gpu_layers')},CPU={plan.get('cpu_layers')})"
        )
    print("\n=== Concurrent ANE + GPU 2GB Pure-Read Contention ===")
    for k, c in res["concurrent_gpu_contention"].items():
        print(
            f"  {k}: GPU solo={c['gpu_solo'].get('gpu_pure_read_gb_s')} -> conc={c['gpu_concurrent'].get('gpu_pure_read_gb_s')} GB/s "
            f"({c['gpu_bw_retention_pct']}% retained) | ANE solo={c['ane_solo'].get('mean_ms', 0):.3f} ms -> conc={c['ane_concurrent'].get('mean_ms', 0):.3f} ms "
            f"({c['ane_latency_ratio_concurrent_vs_solo']}x)"
        )


if __name__ == "__main__":
    main()
