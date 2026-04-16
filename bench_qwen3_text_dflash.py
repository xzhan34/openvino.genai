#!/usr/bin/env python3
"""bench_qwen3_text_dflash.py — Benchmark Qwen3 text-only DFlash speculative decoding.

Cross-platform (Linux / Windows) benchmark script using the modeling_dflash executable.
The executable runs DFlash speculative decoding followed by a baseline comparison internally.

Usage:
    python bench_qwen3_text_dflash.py [--device GPU] [--max-tokens 128] [--block-size 16]
                                      [--target-dir PATH] [--draft-dir PATH]
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Defaults (edit for your environment) ─────────────────────────────────────
DEFAULT_MODEL_BASE = r"C:\work\models" if sys.platform == "win32" else "/home/xzhan34/work/models"
DEFAULT_TARGET_DIR = os.path.join(DEFAULT_MODEL_BASE, "Qwen3-4B")
DEFAULT_DRAFT_DIR  = os.path.join(DEFAULT_MODEL_BASE, "Qwen3-4B-DFlash-b16")
DEFAULT_DEVICE     = "GPU"
DEFAULT_MAX_TOKENS = 128
DEFAULT_BLOCK_SIZE = 16


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
        # Linux: derive OV dir from genai sibling directory
        ov_dir = genai_dir.parent / "openvino"
        lib_dirs = [
            str(ov_dir / "bin" / "intel64" / "RelWithDebInfo"),
            str(genai_dir / "build" / "openvino_genai"),
        ]
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")
        env["OV_TOKENIZERS_LIB_PATH"] = str(genai_dir / "build" / "openvino_genai")

    return env


# ── Metric parsing ───────────────────────────────────────────────────────────

@dataclass
class Metrics:
    """Performance metrics parsed from modeling_dflash output.

    The executable prints two sections:
      1. DFlash results: [Tokens] and [Latency] lines
      2. Baseline results: [Target-only] line

    AccRate (Acceptance Rate) = accepted_draft_tokens / generated_tokens
    AvgAccept = accepted_draft_tokens / draft_steps
    """
    prompt_tokens: str = "—"
    output_tokens: str = "—"
    ttft_ms: str = "—"
    tpot_ms: str = "—"
    throughput: str = "—"
    acc_rate: str = "—"
    avg_accept: str = "—"


def parse_dflash_metrics(output: str) -> Metrics:
    """Parse DFlash section metrics from modeling_dflash output."""
    m = Metrics()

    # [Tokens] prompt=XX, generated=XX, draft_accepted=XX, target_only=XX, avg_accept_per_block=XX
    tok_match = re.search(r"\[Tokens\]\s*prompt=(\d+),\s*generated=(\d+),\s*draft_accepted=(\d+).*avg_accept_per_block=([\d.]+)", output)
    if tok_match:
        m.prompt_tokens = tok_match.group(1)
        m.output_tokens = tok_match.group(2)
        accepted = int(tok_match.group(3))
        generated = int(tok_match.group(2))
        m.avg_accept = tok_match.group(4)
        if generated > 0:
            m.acc_rate = f"{accepted / generated:.4f}"

    # [Latency] TTFT=XX ms, TPOT=XX ms/token, total_generate=XX ms, throughput=XX tokens/s
    lat_match = re.search(r"\[Latency\]\s*TTFT=([\d.]+)\s*ms,\s*TPOT=([\d.]+)\s*ms/token.*throughput=([\d.]+)\s*tokens/s", output)
    if lat_match:
        m.ttft_ms = lat_match.group(1)
        m.tpot_ms = lat_match.group(2)
        m.throughput = lat_match.group(3)

    return m


def parse_baseline_metrics(output: str) -> Metrics:
    """Parse Baseline section metrics from modeling_dflash output."""
    m = Metrics()

    # [Target-only] generated=XX, TTFT=XX ms, TPOT=XX ms/token, total_generate=XX ms, throughput=XX tokens/s
    baseline_match = re.search(
        r"\[Target-only\]\s*generated=(\d+),\s*TTFT=([\d.]+)\s*ms,\s*TPOT=([\d.]+)\s*ms/token.*throughput=([\d.]+)\s*tokens/s",
        output)
    if baseline_match:
        m.output_tokens = baseline_match.group(1)
        m.ttft_ms = baseline_match.group(2)
        m.tpot_ms = baseline_match.group(3)
        m.throughput = baseline_match.group(4)

    # Get prompt tokens from [Tokens] line (same run)
    tok_match = re.search(r"\[Tokens\]\s*prompt=(\d+)", output)
    if tok_match:
        m.prompt_tokens = tok_match.group(1)

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

    return TestResult(
        name=name,
        output=output,
        returncode=result.returncode,
    )


def run_dflash_test(cmd: list, name: str, env: dict, log_dir: Path, test_num: int):
    """Run modeling_dflash and return (dflash_result, baseline_result) pair."""
    r = run_test(cmd, name, env, log_dir, test_num)

    dflash_result = TestResult(
        name=f"DFlash {name}",
        metrics=parse_dflash_metrics(r.output),
        output=r.output,
        returncode=r.returncode,
    )
    baseline_result = TestResult(
        name=f"Baseline {name}",
        metrics=parse_baseline_metrics(r.output),
        output=r.output,
        returncode=r.returncode,
    )
    return dflash_result, baseline_result


# ── Summary table ────────────────────────────────────────────────────────────

def print_summary(results: list, device: str):
    """Print a formatted summary table of all test results."""
    sep  = "+-----+----------------------------+--------+--------+-----------+------------+---------+----------+-----------+"
    hdr  = "| #   | Test                       | Prompt | Output | TTFT (ms) | TPOT (ms)  | Tok/s   | AccRate  | AvgAccept |"

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
            f"| {r.name:<26} "
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

    # Pairwise speedup: each test produces (baseline, dflash) pair
    print("  Speedup (DFlash vs Baseline):")
    print("  " + "-" * 50)
    for i in range(0, len(results), 2):
        if i + 1 >= len(results):
            break
        baseline = results[i]
        dflash = results[i + 1]
        b = baseline.metrics.throughput
        d = dflash.metrics.throughput
        label = dflash.name.replace("DFlash ", "")
        try:
            bf, df = float(b), float(d)
            speedup = df / bf if bf > 0 else 0
            print(f"    {label:<20} {df:>7.2f} tok/s vs {bf:>7.2f} tok/s  =>  {speedup:.2f}x")
        except (ValueError, ZeroDivisionError):
            print(f"    {label:<20} {d} tok/s vs {b} tok/s")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3 text-only DFlash speculative decoding")
    parser.add_argument("--model-dir",   default=DEFAULT_MODEL_BASE,  help="Base model directory")
    parser.add_argument("--target-dir",  default=None,                help="Target model directory (default: <model-dir>/Qwen3-4B)")
    parser.add_argument("--draft-dir",   default=None,                help="DFlash draft model directory (default: <model-dir>/Qwen3-4B-DFlash-b16)")
    parser.add_argument("--device",      default=DEFAULT_DEVICE,      help="Device (GPU / CPU)")
    parser.add_argument("--max-tokens",  default=DEFAULT_MAX_TOKENS,  type=int, help="Max new tokens")
    parser.add_argument("--block-size",  default=DEFAULT_BLOCK_SIZE,  type=int, help="DFlash block size")
    args = parser.parse_args()
    if args.target_dir is None:
        args.target_dir = os.path.join(args.model_dir, "Qwen3-4B")
    if args.draft_dir is None:
        args.draft_dir = os.path.join(args.model_dir, "Qwen3-4B-DFlash-b16")

    genai_dir = find_genai_dir()
    ext = ".exe" if sys.platform == "win32" else ""
    dflash_exe = find_exe(genai_dir, f"modeling_dflash{ext}")

    print(f"DFlash exe: {dflash_exe}")

    env = setup_env(genai_dir)
    log_dir = Path(tempfile.mkdtemp(prefix="dflash_qwen3_bench_"))
    print(f"Log directory: {log_dir}")

    results: list[TestResult] = []
    test_num = 0

    # Helper to build DFlash command
    # modeling_dflash <TARGET_DIR> <DRAFT_DIR> [PROMPT] [DEVICE] [MAX_NEW_TOKENS] [BLOCK_SIZE]
    def dflash_cmd(prompt, max_tok=None):
        return [
            str(dflash_exe),
            args.target_dir,
            args.draft_dir,
            prompt,
            args.device,
            str(max_tok or args.max_tokens),
            str(args.block_size),
        ]

    # ── Test definitions ──────────────────────────────────────────────────
    # Each test runs modeling_dflash which produces both DFlash and baseline results.
    tests = [
        ("FP16 text",
         lambda: dflash_cmd("What is the capital of France?")),

        ("FP16 long(256)",
         lambda: dflash_cmd("Write a detailed explanation of how neural networks work", max_tok=256)),

        ("FP16 codegen",
         lambda: dflash_cmd("Write a Python function to sort a list using quicksort")),

        ("FP16 reasoning",
         lambda: dflash_cmd("Explain the difference between TCP and UDP protocols")),
    ]

    # ── Run all tests ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Qwen3 Text DFlash Benchmark")
    print(f"  Target: {args.target_dir}")
    print(f"  Draft:  {args.draft_dir}")
    print(f"  Device: {args.device}  Max tokens: {args.max_tokens}  Block size: {args.block_size}")
    print("=" * 60)

    for name, cmd_fn in tests:
        test_num += 1
        cmd = cmd_fn()
        baseline_r, dflash_r = run_dflash_test(cmd, name, env, log_dir, test_num)
        results.append(baseline_r)
        results.append(dflash_r)

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(results, args.device)
    print(f"  Logs saved in: {log_dir}")
    print("=" * 60)
    print("  Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
