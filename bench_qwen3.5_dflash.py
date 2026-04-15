#!/usr/bin/env python3
"""bench_qwen3.5_dflash.py — Benchmark Qwen3.5 DFlash speculative decoding.

Cross-platform (Linux / Windows) benchmark script.
Usage:
    python bench_qwen3.5_dflash.py [--device GPU] [--max-tokens 128] [--block-size 16]
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
DEFAULT_TARGET_DIR = r"C:\work\models\Qwen3.5-4B" if sys.platform == "win32" else "/work/models/Qwen3.5-4B"
DEFAULT_DRAFT_DIR  = r"C:\work\models\Qwen3.5-4B-DFlash" if sys.platform == "win32" else "/work/models/Qwen3.5-4B-DFlash"
DEFAULT_IMAGE_PATH = (
    r"C:\work\openvino_ws\openvino.liangali\docs\articles_en\assets\images\get_started_with_cpp.jpg"
    if sys.platform == "win32"
    else "/work/openvino_ws/openvino.liangali/docs/articles_en/assets/images/get_started_with_cpp.jpg"
)
DEFAULT_DEVICE     = "GPU"
DEFAULT_MAX_TOKENS = 128
DEFAULT_BLOCK_SIZE = 16


def find_genai_dir() -> Path:
    """Locate the openvino.genai repo root (directory containing this script)."""
    return Path(__file__).resolve().parent


def find_exe(genai_dir: Path, name: str) -> Path:
    """Find an executable in well-known build output directories."""
    candidates = [
        genai_dir / "build" / "bin" / "RelWithDebInfo" / name,
        genai_dir / "build" / "bin" / "Release" / name,
        genai_dir / "build" / "bin" / name,
        genai_dir / "build-master" / "src" / "cpp" / "src" / "modeling" / "samples" / "RelWithDebInfo" / name,
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
        ov_dir = Path(r"C:\work\openvino_ws\openvino.liangali")
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
        # Linux: adjust LD_LIBRARY_PATH if needed
        ov_dir = Path(os.environ.get("OV_DIR", "/work/openvino_ws/openvino.liangali/build"))
        lib_dirs = [
            str(ov_dir / "lib"),
            str(genai_dir / "build" / "openvino_genai"),
        ]
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")

    return env


# ── Metric parsing ───────────────────────────────────────────────────────────

@dataclass
class Metrics:
    """Performance metrics parsed from program output.

    AccRate (Acceptance Rate) = accepted_draft_tokens / generated_tokens
      - accepted_draft_tokens: 所有 draft step 中被 target verify 通过的 token 总数
      - generated_tokens: 总生成 token 数 (output_len - prompt_len)
      - 值域 [0, 1]，越高表示 draft 模型与 target 越一致，加速效果越好

    AvgAccept = accepted_draft_tokens / draft_steps
      - draft_steps: 总 draft 推理步数
      - block_size=16 时，理想值接近 15（第一个位置是 last_accepted，不计入 draft）
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
        (0, 2, "Text INT4"),
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
    parser = argparse.ArgumentParser(description="Benchmark Qwen3.5 DFlash speculative decoding")
    parser.add_argument("--target-dir",  default=DEFAULT_TARGET_DIR,  help="Target model directory")
    parser.add_argument("--draft-dir",   default=DEFAULT_DRAFT_DIR,   help="DFlash draft model directory")
    parser.add_argument("--image",       default=DEFAULT_IMAGE_PATH,  help="Image path for VL tests")
    parser.add_argument("--device",      default=DEFAULT_DEVICE,      help="Device (GPU / CPU)")
    parser.add_argument("--max-tokens",  default=DEFAULT_MAX_TOKENS,  type=int, help="Max new tokens")
    parser.add_argument("--block-size",  default=DEFAULT_BLOCK_SIZE,  type=int, help="DFlash block size")
    args = parser.parse_args()

    genai_dir = find_genai_dir()
    ext = ".exe" if sys.platform == "win32" else ""
    dflash_exe  = find_exe(genai_dir, f"modeling_qwen3_5_dflash{ext}")
    baseline_exe = find_exe(genai_dir, f"modeling_qwen3_5{ext}")

    print(f"DFlash exe:   {dflash_exe}")
    print(f"Baseline exe: {baseline_exe}")

    env = setup_env(genai_dir)
    log_dir = Path(tempfile.mkdtemp(prefix="dflash_bench_"))
    print(f"Log directory: {log_dir}")

    results: list[TestResult] = []
    test_num = 0

    # Helper to build DFlash command
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

    # Helper to build baseline command
    def baseline_cmd(prompt, mode="text", max_tok=None, image=None):
        cmd = [
            str(baseline_exe),
            "--model", args.target_dir,
            "--mode", mode,
            "--prompt", prompt,
            "--device", args.device,
            "--output-tokens", str(max_tok or args.max_tokens),
        ]
        if image:
            cmd.extend(["--image", image])
        return cmd

    # ── Test definitions ──────────────────────────────────────────────────
    tests = [
        # (name, cmd_builder)
        ("Baseline FP16 text",
         lambda: baseline_cmd("What is the capital of France?")),

        ("DFlash FP16 text",
         lambda: dflash_cmd("What is the capital of France?")),

        ("DFlash INT4+FP16 text",
         lambda: dflash_cmd("What is the capital of France?", tgt_quant="INT4_ASYM")),

        ("DFlash FP16 long(256)",
         lambda: dflash_cmd("Write a detailed explanation of how neural networks work", max_tok=256)),

        ("Baseline FP16 codegen",
         lambda: baseline_cmd("Write a Python function to sort a list using quicksort")),

        ("DFlash FP16 codegen",
         lambda: dflash_cmd("Write a Python function to sort a list using quicksort")),

        ("Baseline FP16 VL",
         lambda: baseline_cmd("Describe this image in detail", mode="vl", image=args.image)),

        ("DFlash FP16 VL",
         lambda: dflash_cmd("Describe this image in detail", image=args.image)),
    ]

    # ── Run all tests ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Qwen3.5 DFlash Benchmark")
    print(f"  Target: {args.target_dir}")
    print(f"  Draft:  {args.draft_dir}")
    print(f"  Device: {args.device}  Max tokens: {args.max_tokens}  Block size: {args.block_size}")
    print("=" * 60)

    for name, cmd_fn in tests:
        test_num += 1
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
