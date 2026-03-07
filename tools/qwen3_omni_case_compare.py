#!/usr/bin/env python3

import argparse
import glob
import json
import math
import os
import signal
import subprocess
import sys
import threading
import wave
from datetime import datetime
from pathlib import Path


def ensure_test_wav(path: Path, sample_rate: int = 16000, seconds: float = 1.0, freq_hz: float = 440.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(sample_rate * seconds)
    amplitude = 0.25
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for i in range(num_samples):
            sample = amplitude * math.sin(2.0 * math.pi * freq_hz * (i / sample_rate))
            value = int(max(-1.0, min(1.0, sample)) * 32767)
            frames.extend(value.to_bytes(2, byteorder="little", signed=True))
        wav.writeframes(bytes(frames))


def parse_cpp_text(stdout: str) -> str:
    """Extract model text output from C++ stdout.

    The C++ binary prints stats lines (containing ':') before the generated text,
    and '[SafetensorsWeightFinalizer]' lines after it.  We capture everything
    between the last stats line (Throughput) and the first '[' bracket line that
    follows the generated text.
    """
    lines = stdout.splitlines()
    # Find the start: line after the last "Throughput:" stats line
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Throughput:"):
            start = i + 1
    # Find the end: first '[' line after start
    end = len(lines)
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("["):
            end = i
            break
    text = "\n".join(lines[start:end]).strip()
    return text


def run_cpp_case(cpp_bin: Path, model_dir: Path, image_path: Path, prompt: str, max_new_tokens: int, precision: str = "fp32") -> dict:
    cmd = [
        str(cpp_bin),
        str(model_dir),
        str(image_path),
        prompt,
        "CPU",
        str(max_new_tokens),
        "",
        "",
        precision,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    result = {
        "supported": True,
        "return_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "text_output": parse_cpp_text(proc.stdout) if proc.returncode == 0 else "",
    }
    return result


def parse_kv_stdout(stdout: str) -> dict:
    parsed = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def run_cpp_tts_case(
    cpp_tts_bin: Path,
    model_dir: Path,
    case_id: int,
    prompt: str,
    wav_out: Path,
    max_new_tokens: int,
    image_path: Path = None,
    audio_path: Path = None,
) -> dict:
    cmd = [
        str(cpp_tts_bin),
        str(model_dir),
        str(case_id),
        prompt,
        str(wav_out),
        str(image_path) if image_path else "none",
        str(audio_path) if audio_path else "none",
        "CPU",
        str(max_new_tokens),
    ]
    # Set env vars for the C++ binary to find the bridge script and Python executable
    env = dict(os.environ)
    # This script lives at <repo>/tools/qwen3_omni_case_compare.py
    # Bridge script is at  <repo>/src/cpp/src/modeling/models/qwen3_omni/
    bridge_dir = Path(__file__).resolve().parent.parent / "src" / "cpp" / "src" / "modeling" / "models" / "qwen3_omni"
    if bridge_dir.exists():
        env["QWEN3_OMNI_BRIDGE_DIR"] = str(bridge_dir)
    env.setdefault("PYTHON_EXECUTABLE", sys.executable)
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    parsed = parse_kv_stdout(proc.stdout) if proc.returncode == 0 else {}
    return {
        "supported": True,
        "return_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "text_output": parsed.get("TEXT_OUTPUT", "").replace("\\n", "\n") if proc.returncode == 0 else "",
        "wav_output": parsed.get("WAV_OUTPUT", "") if proc.returncode == 0 else "",
        "audio_samples": int(parsed.get("AUDIO_SAMPLES", "0")) if proc.returncode == 0 else 0,
        "audio_sample_rate": int(parsed.get("AUDIO_SAMPLE_RATE", "0")) if proc.returncode == 0 else 0,
        "tts_backend": parsed.get("TTS_BACKEND", "") if proc.returncode == 0 else "",
        "tts_note": parsed.get("TTS_NOTE", "") if proc.returncode == 0 else "",
    }


def decode_python_generation(result, tokenizer, input_length: int):
    audio_keys = ["audio", "audios", "waveform", "waveforms", "tts", "speech"]
    text = ""
    has_audio = False

    if hasattr(result, "sequences"):
        sequences = result.sequences
        decoded = tokenizer.batch_decode(sequences[:, input_length:], skip_special_tokens=True)
        text = decoded[0] if decoded else ""
        for key in audio_keys:
            if hasattr(result, key):
                value = getattr(result, key)
                has_audio = value is not None
                break
        return text, has_audio

    if isinstance(result, dict):
        sequences = result.get("sequences")
        if sequences is not None:
            decoded = tokenizer.batch_decode(sequences[:, input_length:], skip_special_tokens=True)
            text = decoded[0] if decoded else ""
        for key in audio_keys:
            if key in result and result[key] is not None:
                has_audio = True
                break
        return text, has_audio

    if hasattr(result, "shape"):
        decoded = tokenizer.batch_decode(result[:, input_length:], skip_special_tokens=True)
        text = decoded[0] if decoded else ""
        return text, False

    return str(result), False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--cpp-bin", required=True)
    parser.add_argument("--cpp-tts-bin", required=False, default="")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--cpp-precision", type=str, default="fp32",
                        help="Precision mode for C++ binary (fp32/inf_fp32_kv_fp32_w_int4_asym/etc.)")
    parser.add_argument("--test-audio", type=str, default="",
                        help="Path to a real speech WAV file for case 3 (default: auto-generated 440Hz test tone)")
    parser.add_argument("--py-max-new-tokens", type=int, default=0,
                        help="Max new tokens for Python inference (default: same as --max-new-tokens, "
                             "set lower to avoid slow CPU inference)")
    parser.add_argument("--py-timeout", type=int, default=600,
                        help="Timeout in seconds for each Python case (default: 600). 0=no timeout.")
    parser.add_argument("--cpp-only", action="store_true",
                        help="Skip Python model loading and inference, only run C++ cases")
    parser.add_argument("--cases", type=str, default="",
                        help="Comma-separated list of case IDs to run (e.g. '4' or '3,4'). Empty = all.")
    args = parser.parse_args()
    py_max_new_tokens = args.py_max_new_tokens if args.py_max_new_tokens > 0 else args.max_new_tokens

    model_dir = Path(args.model_dir)
    image_path = Path(args.image)
    cpp_bin = Path(args.cpp_bin)
    cpp_tts_bin = Path(args.cpp_tts_bin) if args.cpp_tts_bin else None
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if args.test_audio:
        test_audio = Path(args.test_audio)
        if not test_audio.exists():
            raise FileNotFoundError(f"Test audio not found: {test_audio}")
    else:
        test_audio = out_json.parent / "qwen3_omni_test_tone.wav"
        ensure_test_wav(test_audio)

    case_filter = set(int(x) for x in args.cases.split(",") if x.strip()) if args.cases else set()

    model = None
    processor = None
    process_mm_info = None
    if not args.cpp_only:
        base = Path(__file__).resolve().parents[2] / "qwen3_omni_4B_final_release"
        internal = glob.glob(str(base / "transformers-internal-q3o-dense-*" / "src"))
        if not internal:
            raise RuntimeError("Cannot locate internal transformers source")
        sys.path.insert(0, internal[0])
        sys.path.insert(0, str(base))

        from transformers import Qwen3OmniForConditionalGeneration, Qwen3OmniProcessor
        from qwen_omni_utils import process_mm_info as _process_mm_info
        process_mm_info = _process_mm_info

        model = Qwen3OmniForConditionalGeneration.from_pretrained(
            str(model_dir),
            dtype="auto",
            device_map="cpu",
            trust_remote_code=True,
        )
        model.eval()
        processor = Qwen3OmniProcessor.from_pretrained(str(model_dir), trust_remote_code=True, use_fast=False)

    # Prompt from Python demo (qwen3_omni_4b_demo/run_qwen3_omni_dense.py)
    demo_prompt = "What can you see in the image and hear in the audio wav? Answer in one short sentence."

    cases = [
        {
            "id": 1,
            "name": "image+text -> text",
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": "Describe this image in detail."},
                    ],
                }
            ],
            "cpp_supported": True,
            "expect_tts": False,
        },
        {
            "id": 2,
            "name": "image+text -> text+tts",
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": "Describe this image and also output speech audio."},
                    ],
                }
            ],
            "cpp_supported": cpp_tts_bin is not None,
            "cpp_mode": "tts_min",
            "cpp_prompt": "Describe this image and provide a speech response.",
            "expect_tts": True,
        },
        {
            "id": 3,
            "name": "audio+text -> text+tts",
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": str(test_audio)},
                        {"type": "text", "text": "What sound do you hear in the audio? Answer in one short sentence."},
                    ],
                }
            ],
            "cpp_supported": cpp_tts_bin is not None,
            "cpp_mode": "tts_min",
            "cpp_prompt": "What sound do you hear in the audio? Answer in one short sentence.",
            "expect_tts": True,
        },
        {
            "id": 4,
            "name": "image+audio+text -> text+tts (demo)",
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "audio", "audio": str(test_audio)},
                        {"type": "text", "text": demo_prompt},
                    ],
                }
            ],
            "cpp_supported": cpp_tts_bin is not None,
            "cpp_mode": "tts_min",
            "cpp_prompt": demo_prompt,
            "expect_tts": True,
        },
    ]

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "image": str(image_path),
        "audio": str(test_audio),
        "compare_target": f"C++ {args.cpp_precision} vs Python default(dtype=auto)",
        "cases": [],
    }

    for case in cases:
        if case_filter and case["id"] not in case_filter:
            continue

        # Extract the user text prompt from conversation
        user_text_prompt = ""
        for msg in case["conversation"]:
            if isinstance(msg.get("content"), list):
                for c in msg["content"]:
                    if c.get("type") == "text":
                        user_text_prompt = c.get("text", "")
        cpp_prompt = case.get("cpp_prompt", user_text_prompt)

        case_result = {
            "id": case["id"],
            "name": case["name"],
            "expect_tts": case["expect_tts"],
            "python_prompt": user_text_prompt,
            "cpp_prompt": cpp_prompt,
            "cpp": {"supported": case["cpp_supported"]},
            "python": {},
        }

        if not args.cpp_only and processor is not None:
            text = processor.apply_chat_template(case["conversation"], add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(case["conversation"], use_audio_in_video=True)

            py_result_container = {}
            def _run_py_case():
                try:
                    inputs = processor(
                        text=text,
                        audio=audios,
                        images=images,
                        videos=videos,
                        return_tensors="pt",
                        padding=True,
                        use_audio_in_video=True,
                    )
                    inputs = inputs.to(model.device).to(model.dtype)
                    print(f"  [Python] Starting generate (max_new_tokens={py_max_new_tokens}, "
                          f"return_audio={case['expect_tts']})...", flush=True)
                    generation = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=py_max_new_tokens,
                        speaker="f245",
                        return_audio=case["expect_tts"],
                        use_audio_in_video=True,
                        thinker_return_dict_in_generate=True,
                    )

                    py_has_audio = False
                    generation_for_decode = generation
                    if isinstance(generation, tuple):
                        generation_for_decode = generation[0]
                        py_has_audio = generation[1] is not None

                    py_text, py_has_audio_from_decode = decode_python_generation(
                        generation_for_decode,
                        processor.tokenizer,
                        int(inputs["input_ids"].shape[1]),
                    )
                    py_has_audio = py_has_audio or py_has_audio_from_decode
                    py_result_container["result"] = {
                        "ok": True,
                        "text_output": py_text,
                        "has_tts_audio": py_has_audio,
                    }
                except Exception as error:
                    py_result_container["result"] = {
                        "ok": False,
                        "error": str(error),
                    }

            timeout = args.py_timeout if args.py_timeout > 0 else None
            py_thread = threading.Thread(target=_run_py_case, daemon=True)
            py_thread.start()
            py_thread.join(timeout=timeout)
            if py_thread.is_alive():
                print(f"  [Python] TIMEOUT after {timeout}s - skipping", flush=True)
                case_result["python"] = {
                    "ok": False,
                    "error": f"timeout ({timeout}s)",
                }
            elif "result" in py_result_container:
                case_result["python"] = py_result_container["result"]
            else:
                case_result["python"] = {
                    "ok": False,
                    "error": "unknown error (no result)",
                }
        else:
            case_result["python"] = {"ok": False, "error": "skipped (--cpp-only)"}

        if case["cpp_supported"]:
            if case.get("cpp_mode") == "tts_min":
                wav_out = out_json.parent / f"case{case['id']}_cpp_tts.wav"
                # Only pass image_path / audio_path when the case conversation actually uses them
                case_has_image = any(
                    c.get("type") == "image"
                    for msg in case["conversation"]
                    for c in (msg.get("content") if isinstance(msg.get("content"), list) else [])
                )
                case_has_audio = any(
                    c.get("type") == "audio"
                    for msg in case["conversation"]
                    for c in (msg.get("content") if isinstance(msg.get("content"), list) else [])
                )
                case_result["cpp"] = run_cpp_tts_case(
                    cpp_tts_bin,
                    model_dir,
                    case["id"],
                    case.get("cpp_prompt", ""),
                    wav_out,
                    args.max_new_tokens,
                    image_path=image_path if case_has_image else None,
                    audio_path=test_audio if case_has_audio else None,
                )
            else:
                cpp_prompt = "Describe this image in detail."
                case_result["cpp"] = run_cpp_case(cpp_bin, model_dir, image_path, cpp_prompt, args.max_new_tokens, args.cpp_precision)
        else:
            case_result["cpp"] = {
                "supported": False,
                "reason": "C++ tts sample binary is not provided. Pass --cpp-tts-bin to enable case2/3/4.",
            }

        report["cases"].append(case_result)

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"=== Qwen3 Omni Case Compare (C++ {args.cpp_precision} vs Python default) ===")
    for case in report["cases"]:
        py = case["python"]
        cpp = case["cpp"]
        py_status = "OK" if py.get("ok") else f"FAIL: {py.get('error', '')}"
        if cpp.get("supported"):
            cpp_status = "OK" if cpp.get("return_code") == 0 else f"FAIL(rc={cpp.get('return_code')})"
        else:
            cpp_status = f"UNSUPPORTED ({cpp.get('reason')})"
        print(f"[{case['id']}] {case['name']}")
        # Show input modalities
        modalities = []
        if case.get('python_prompt'):
            modalities.append(f"text=\"{case['python_prompt']}\"")
        if case.get('cpp_prompt') and case.get('cpp_prompt') != case.get('python_prompt'):
            modalities.append(f"cpp_prompt=\"{case['cpp_prompt']}\"")
        print(f"  Prompt: {modalities[0] if modalities else '(none)'}")
        if len(modalities) > 1:
            print(f"  C++ Prompt: {modalities[1]}")
        print(f"  Python: {py_status}")
        if py.get("ok"):
            py_text = py.get('text_output', '')
            py_text_display = (py_text[:200] + '...') if len(py_text) > 200 else py_text
            print(f"  Python text: {py_text_display}")
            print(f"  Python has_tts_audio: {py.get('has_tts_audio')}")
        print(f"  C++: {cpp_status}")
        if cpp.get("supported") and cpp.get("return_code") == 0:
            cpp_text = cpp.get('text_output', '')
            cpp_text_display = (cpp_text[:200] + '...') if len(cpp_text) > 200 else cpp_text
            print(f"  C++ text: {cpp_text_display}")
            if cpp.get("wav_output"):
                print(f"  C++ wav: {cpp.get('wav_output')}")
                print(f"  C++ audio_samples: {cpp.get('audio_samples')}")
                if cpp.get("tts_backend"):
                    print(f"  C++ tts_backend: {cpp.get('tts_backend')}")

    print(f"\nSaved report: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
