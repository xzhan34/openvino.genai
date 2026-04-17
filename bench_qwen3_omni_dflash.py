#!/usr/bin/env python3
"""bench_qwen3_omni_dflash.py — Benchmark Qwen3-Omni DFlash speculative decoding.


Cross-platform (Linux / Windows) benchmark script.
Usage:
    python bench_qwen3_omni_dflash.py [--device GPU.1] [--max-tokens 128] [--block-size 16]
                                      [--target-dir PATH] [--draft-dir PATH]
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Defaults (edit for your environment) ─────────────────────────────────────
DEFAULT_MODEL_BASE = r"C:\work\models" if sys.platform == "win32" else "/home/xzhan34/work/models"
DEFAULT_TARGET_DIR = os.path.join(DEFAULT_MODEL_BASE, "Qwen3-Omni-4B-Instruct-multilingual")
DEFAULT_DRAFT_DIR  = os.path.join(DEFAULT_MODEL_BASE, "Qwen3-4B-DFlash-b16")
DEFAULT_IMAGE_PATH = str(Path(__file__).resolve().parent / "testdata" / "get_started_with_cpp.jpg")
DEFAULT_DEVICE     = "GPU"
DEFAULT_MAX_TOKENS = 128
DEFAULT_BLOCK_SIZE = 16
DEFAULT_PRECISION  = "mixed"
# "mixed" uses GPU defaults (inference_precision=f16, kv_cache=f16) — matches DFlash target config.
# Other options: default/fp32/inf_fp32_kv_int8/inf_fp32_kv_int4/inf_fp16_kv_int8/inf_fp16_kv_int4
# inf_fp32_kv_fp32_w_int8/inf_fp32_kv_fp32_w_int4_asym/inf_fp32_kv_int8_w_int4_asym/inf_fp16_kv_int8_w_int4_asym

def find_genai_dir() -> Path:
    """Locate the openvino.genai repo root (directory containing this script)."""
    return Path(__file__).resolve().parent


def find_exe(genai_dir: Path, name: str) -> Path:
    """Find an executable in well-known build output directories."""
    candidates = [
        genai_dir / "build" / "src" / "cpp" / "src" / "modeling" / "samples" / name,
        genai_dir / "build" / "bin" / "RelWithDebInfo" / name,
        genai_dir / "build" / "bin" / "Release" / name,
        genai_dir / "build" / "bin" / name,
        genai_dir / "build-master" / "src" / "cpp" / "src" / "modeling" / "samples" / "RelWithDebInfo" / name,
        genai_dir / "build-master" / "src" / "cpp" / "src" / "modeling" / "samples" / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fallback: search recursively (slow)
    for p in genai_dir.rglob(name):
        return p
    raise FileNotFoundError(f"Cannot find executable '{name}' under {genai_dir}")


def setup_env(genai_dir: Path) -> dict:
    """Prepare environment variables for running the benchmarks."""
    env = os.environ.copy()

    if sys.platform == "win32":
        ov_dir = genai_dir.parent / "openvino"
        extra_paths = [
            str(ov_dir / "bin" / "intel64" / "RelWithDebInfo"),
            str(ov_dir / "temp" / "Windows_AMD64" / "tbb" / "bin"),
            str(genai_dir / "build" / "openvino_genai"),
        ]
        env["PATH"] = ";".join(extra_paths) + ";" + env.get("PATH", "")

        tokenizers_dll = genai_dir / "build" / "openvino_genai" / "openvino_tokenizers.dll"
        if tokenizers_dll.exists():
            env["OPENVINO_TOKENIZERS_PATH_GENAI"] = str(tokenizers_dll)
    else:
        # Linux: adjust LD_LIBRARY_PATH
        ov_dir = genai_dir.parent / "openvino"
        lib_dirs = [
            str(ov_dir / "bin" / "intel64" / "RelWithDebInfo"),
            str(genai_dir / "build" / "openvino_genai"),
        ]
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")
        env["OV_TOKENIZERS_LIB_PATH"] = str(genai_dir / "build" / "openvino_genai")

    # Use split draft model (context_kv + step) with f16 precision
    # (fc matmul overflow fixed: dflash_draft.cpp scales input by 1/128, RMSNorm is scale-invariant)
    env["OV_GENAI_SPLIT_DRAFT"] = "1"
    env["OV_GENAI_DRAFT_PRECISION"] = "f16"
    env["OV_GENAI_DISABLE_THINKING"] = "1"

    return env


# ── Metric parsing ───────────────────────────────────────────────────────────

@dataclass
class Metrics:
    """Performance metrics parsed from program output.

    AccRate (Acceptance Rate) = accepted_draft_tokens / generated_tokens
      - accepted_draft_tokens: total tokens accepted by target verify across all draft steps
      - generated_tokens: total generated tokens (output_len - prompt_len)
      - range [0, 1], higher means draft model matches target better

    AvgAccept = accepted_draft_tokens / draft_steps
      - draft_steps: total draft inference steps
      - with block_size=16, ideal value ≈ 15
    """
    prompt_tokens: str = "—"
    output_tokens: str = "—"
    ttft_ms: str = "—"
    tpot_ms: str = "—"
    throughput: str = "—"
    acc_rate: str = "—"
    avg_accept: str = "—"


# Regex patterns matching "Key: value [unit]" output lines
_METRIC_PATTERNS = {
    "prompt_tokens": re.compile(r"Prompt token size:\s*(\d+)"),
    "output_tokens": re.compile(r"Output token size:\s*(\d+)"),
    "ttft_ms":       re.compile(r"TTFT:\s*([\d.]+)"),
    "tpot_ms":       re.compile(r"TPOT:\s*([\d.]+)"),
    "throughput":    re.compile(r"Throughput:\s*([\d.]+)"),
    "acc_rate":      re.compile(r"Acceptance rate:\s*([\d.]+)"),
    "avg_accept":    re.compile(r"Avg accepted per step:\s*([\d.]+)"),
}


def parse_metrics(output: str) -> Metrics:
    m = Metrics()
    for attr, pat in _METRIC_PATTERNS.items():
        match = pat.search(output)
        if match:
            setattr(m, attr, match.group(1))
    return m


# ── Test runner ──────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    metrics: Metrics = field(default_factory=Metrics)
    output: str = ""
    returncode: int = 0


def run_test(cmd: list, name: str, env: dict, log_dir: Path, test_num: int) -> TestResult:
    """Run a single benchmark test and capture output."""
    print(f"\n=== [Test {test_num}] {name} ===")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    output = result.stdout + result.stderr

    # Save log
    log_file = log_dir / f"test{test_num}.log"
    log_file.write_text(output, encoding="utf-8")

    # Print output
    print(output)

    if result.returncode != 0:
        print(f"  *** Test {test_num} FAILED (exit code {result.returncode}) ***")

    # Cool down GPU between tests
    print(f"  Cooling down GPU for 15s...")
    time.sleep(15)

    return TestResult(
        name=name,
        metrics=parse_metrics(output),
        output=output,
        returncode=result.returncode,
    )


# ── Summary table ────────────────────────────────────────────────────────────

def print_summary(results: list, device: str):
    """Print a formatted summary table of all test results."""
    sep  = "+-----+------------------------+--------+--------+-----------+------------+---------+----------+-----------+"
    hdr  = "| #   | Test                   | Prompt | Output | TTFT (ms) | TPOT (ms)  | Tok/s   | AccRate  | AvgAccept |"

    print()
    print("=" * 60)
    print(f"  BENCHMARK SUMMARY  —  Device: {device}")
    print("=" * 60)
    print()
    print("  AccRate  = accepted_draft_tokens / generated_tokens")
    print("             (how well the draft model matches the target; higher is better)")
    print("  AvgAccept = accepted_draft_tokens / draft_steps")
    print("             (average tokens accepted per draft step; ideal ≈ block_size - 1)")
    print()
    print(sep)
    print(hdr)
    print(sep)

    for i, r in enumerate(results, 1):
        m = r.metrics
        row = (
            f"| {i:<3} "
            f"| {r.name:<22} "
            f"| {m.prompt_tokens:>6} "
            f"| {m.output_tokens:>6} "
            f"| {m.ttft_ms:>9} "
            f"| {m.tpot_ms:>10} "
            f"| {m.throughput:>7} "
            f"| {m.acc_rate:>8} "
            f"| {m.avg_accept:>9} |"
        )
        print(row)

    print(sep)
    print()

    # Pairwise speedup
    pairs = [
        (0, 1, "Text FP16"),
        (2, 3, "Text INT4+INT4"),
        (4, 5, "Codegen"),
        (6, 7, "VL mode"),
    ]
    print("  Speedup (DFlash vs Baseline):")
    print("  " + "-" * 50)
    for base_idx, dflash_idx, label in pairs:
        if base_idx >= len(results) or dflash_idx >= len(results):
            continue
        b = results[base_idx].metrics.throughput
        d = results[dflash_idx].metrics.throughput
        try:
            bf, df = float(b), float(d)
            speedup = df / bf if bf > 0 else 0
            print(f"    {label:<14} {df:>7.2f} tok/s vs {bf:>7.2f} tok/s  =>  {speedup:.2f}x")
        except (ValueError, ZeroDivisionError):
            print(f"    {label:<14} {d} tok/s vs {b} tok/s")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-Omni DFlash speculative decoding")
    parser.add_argument("--model-dir",   default=DEFAULT_MODEL_BASE,  help="Base model directory")
    parser.add_argument("--target-dir",  default=None,                help="Target model directory (default: <model-dir>/Qwen3-Omni-4B-Instruct-multilingual)")
    parser.add_argument("--draft-dir",   default=None,                help="DFlash draft model directory (default: <model-dir>/Qwen3-4B-DFlash-b16)")
    parser.add_argument("--image",       default=DEFAULT_IMAGE_PATH,  help="Image path for VL tests")
    parser.add_argument("--device",      default=DEFAULT_DEVICE,      help="Device (GPU / GPU.1 / CPU)")
    parser.add_argument("--max-tokens",  default=DEFAULT_MAX_TOKENS,  type=int, help="Max new tokens")
    parser.add_argument("--block-size",  default=DEFAULT_BLOCK_SIZE,  type=int, help="DFlash block size")
    parser.add_argument("--precision",   default=DEFAULT_PRECISION,   help="Baseline precision mode (default: mixed = f16 inference + f16 kv cache)")
    args = parser.parse_args()
    if args.target_dir is None:
        args.target_dir = os.path.join(args.model_dir, "Qwen3-Omni-4B-Instruct-multilingual")
    if args.draft_dir is None:
        args.draft_dir = os.path.join(args.model_dir, "Qwen3-4B-DFlash-b16")

    genai_dir = find_genai_dir()
    ext = ".exe" if sys.platform == "win32" else ""
    try:
        dflash_exe = find_exe(genai_dir, f"modeling_qwen3_omni_dflash{ext}")
    except FileNotFoundError:
        dflash_exe = None
    baseline_exe = find_exe(genai_dir, f"modeling_qwen3_omni{ext}")

    print(f"DFlash exe:   {dflash_exe or '(not built — DFlash tests will be skipped)'}")
    print(f"Baseline exe: {baseline_exe}")

    env = setup_env(genai_dir)
    log_dir = Path(tempfile.mkdtemp(prefix="qwen3_omni_dflash_bench_"))
    print(f"Log directory: {log_dir}")

    results: list[TestResult] = []
    test_num = 0

    # Helper to build DFlash command (same positional args as modeling_qwen3_5_dflash)
    #   <TARGET_MODEL_DIR> <DRAFT_MODEL_DIR> [PROMPT] [DEVICE] [MAX_NEW_TOKENS]
    #   [BLOCK_SIZE] [TARGET_QUANT] [DRAFT_QUANT] [IMAGE_PATH]
    def dflash_cmd(prompt, max_tok=None, tgt_quant="FP16", draft_quant="FP16", image=None):
        cmd = [
            str(dflash_exe),
            args.target_dir,
            args.draft_dir,
            prompt,
            args.device,
            str(max_tok or args.max_tokens),
            str(args.block_size),
            tgt_quant,
            draft_quant,
        ]
        if image:
            cmd.append(image)
        return cmd

    # Helper to build baseline command (modeling_qwen3_omni uses named args)
    #   --model-dir PATH --image PATH [--prompt TEXT] [--device NAME]
    #   [--output-tokens N] [--precision MODE]
    def baseline_cmd(prompt, max_tok=None, image=None, precision=None):
        cmd = [
            str(baseline_exe),
            "--model-dir", args.target_dir,
            "--image", image or args.image,
            "--prompt", prompt,
            "--device", args.device,
            "--output-tokens", str(max_tok or args.max_tokens),
            "--precision", precision or args.precision,
        ]
        return cmd

    # ── Test definitions ──────────────────────────────────────────────────
    _PROMPT_CASES_1_4 = "Suggest five award-winning documentary films with brief background descriptions for aspiring filmmakers to study"

    tests = [
        # (name, cmd_builder)
        ("Baseline FP16 text",
         lambda: baseline_cmd(_PROMPT_CASES_1_4, max_tok=128),
         False),

        ("DFlash FP16 text",
         lambda: dflash_cmd(_PROMPT_CASES_1_4, max_tok=128),
         True),

        ("Baseline INT4 text",
         lambda: baseline_cmd(_PROMPT_CASES_1_4, max_tok=128, precision="inf_fp16_kv_int8_w_int4_asym"),
         False),

        ("DFlash INT4+INT4 text",
         lambda: dflash_cmd(_PROMPT_CASES_1_4, max_tok=128, tgt_quant="INT4_ASYM", draft_quant="INT4_ASYM"),
         True),

        ("Baseline FP16 codegen",
         lambda: baseline_cmd("Write a Python function to sort a list using quicksort"),
         False),

        ("DFlash FP16 codegen",
         lambda: dflash_cmd("Write a Python function to sort a list using quicksort"),
         True),

        ("Baseline FP16 VL",
         lambda: baseline_cmd("Describe this image in detail", image=args.image),
         False),

        ("DFlash FP16 VL",
         lambda: dflash_cmd("Describe this image in detail", image=args.image),
         True),
    ]

    # ── Run all tests ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Qwen3-Omni DFlash Benchmark")
    print(f"  Target: {args.target_dir}")
    print(f"  Draft:  {args.draft_dir}")
    print(f"  Device: {args.device}  Max tokens: {args.max_tokens}  Block size: {args.block_size}")
    print(f"  Baseline precision: {args.precision}")
    print("=" * 60)

    for name, cmd_fn, needs_dflash in tests:
        test_num += 1
        if needs_dflash and dflash_exe is None:
            print(f"\n=== [Test {test_num}] {name} === SKIPPED (DFlash exe not built)")
            continue
        cmd = cmd_fn()
        r = run_test(cmd, name, env, log_dir, test_num)
        results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(results, args.device)
    print(f"  Logs saved in: {log_dir}")
    print("=" * 60)
    print("  Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
