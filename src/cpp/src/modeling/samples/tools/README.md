# Dev Tools

Dev-only Python utilities for model preparation. **Not included in release builds.**

## Setup

```bash
pip install -r requirements.txt
```

## Tools

### hf_tokenizer_to_ov.py

Convert a HuggingFace tokenizer to OpenVINO IR (`openvino_tokenizer.xml` + `openvino_detokenizer.xml`).

```bash
# Basic usage — output defaults to model_id directory
python hf_tokenizer_to_ov.py <model_id_or_path>

# Specify output directory
python hf_tokenizer_to_ov.py Qwen/Qwen3-0.6B -o ./qwen3_tokenizer_ir
```

Options:

| Flag | Description |
|------|-------------|
| `-o, --output-dir` | Output directory (default: model_id directory) |
