---
name: ov-modeling-new-multimodal-model
description: |
  Add new multimodal model support to the OpenVINO GenAI Modeling framework. This skill guides the full-stack process of implementing explicit model expressions in C++ for complex multimodal GenAI models (VLM, omni-modal, speech-enabled). Use this skill when: (1) User provides a HuggingFace transformers PyTorch multimodal model and asks to implement it in the modeling framework, (2) User wants to add support for a new multimodal model architecture like Qwen3-Omni, (3) The model involves multiple encoders (vision, audio), LLM backbone, TTS pipeline, or other complex multi-component architectures. This skill is an enhanced version of ov-modeling-new-model that incorporates battle-tested lessons from the Qwen3-Omni implementation — a model with 5 sub-models, 3 weight mapping strategies, multimodal preprocessing, and a full text+audio generation pipeline.
---

# OpenVINO Modeling — Add New Multimodal Model Support

This skill provides a comprehensive, battle-tested guide for implementing complex multimodal model architectures in the OpenVINO GenAI Modeling framework. It is based on the successful implementation of Qwen3-Omni (image + audio + video + text → text + audio), incorporating all lessons learned from the initial conversion and subsequent bug fixes.

## CRITICAL PRINCIPLES

Before starting implementation, internalize these hard-won lessons:

### Weight Loading Principles
1. **Weight names MUST match exactly** — The OpenVINO Module hierarchy generates parameter paths that must exactly match HuggingFace safetensors weight names (modulo `layers.N` ↔ `layers[N]` auto-conversion). A single wrong name causes "Unknown parameter" errors.

2. **ALWAYS read the PyTorch source first** — Never guess weight names, layer names, or forward pass logic. Different models use different conventions.

3. **Weight tying breaks during load_model()** — If `tie_word_embeddings=true`, `load_model()` iterates ALL safetensors keys and may `bind()` the lm_head weight with a separate tensor from safetensors, breaking the tie set in the constructor. **You MUST re-tie after load_model():**
   ```cpp
   if (cfg.tie_word_embeddings) {
       model.get_parameter("lm_head.weight").tie_to(
           model.get_parameter("language_model.embed_tokens.weight"));
   }
   ```
   This was the root cause of garbage output (e.g., "蟊Prev mg ostensibly pathogens..." instead of coherent text).

4. **Multimodal models have multiple weight prefixes** — A single safetensors file contains weights for multiple sub-models (e.g., `thinker.`, `talker.`, `code2wav.`). Use `PrefixMappedWeightSource` to scope each sub-model's weight loading.

5. **Complex models need custom WeightSource adapters** — When C++ module hierarchy names differ significantly from safetensors names (e.g., speech decoder), implement a custom `WeightSource` adapter that handles: key name translation, weight sharing (codebook weights mapped to 16 layers), and synthetic weight generation (identity/zero tensors for projection layers not in safetensors).

### Architecture Principles
6. **Use the direct factory function pattern** — Models are registered via public `create_xxx_model()` functions called from `safetensors_modeling.cpp`, NOT via ModelBuilder self-registration.

7. **Follow the `ops::` function signatures exactly** — Check actual header files. For example, `ops::slice(data, start, stop, step, axis)` takes 5 args, not 4.

8. **Dynamic slicing requires OV ops** — For slicing with dynamic dimensions (e.g., taking last N elements), use `ov::op::v8::Slice` directly instead of `ops::slice` (which only handles static int64_t indices).

9. **Build and test incrementally** — Compile after each sub-model. Don't write the entire multimodal pipeline before compiling.

### Multimodal Pipeline Principles
10. **TTS/audio-output models are precision-sensitive** — Small TTS models (talker, code predictor, speech decoder) degrade with reduced precision (int8/fp16). Always compile them with `inference_precision=f32`, independent of precision mode for the main LLM.

11. **Speech decoder (BigVGAN) must run on CPU** — GPU plugins downcast to fp16 despite fp32 hints, causing ~10x amplitude loss in SnakeBeta activations across the 480x upsampling chain. Force CPU for speech decoder.

12. **Nested config.json requires sub-object fallback** — Some models store sub-component configs in nested objects (e.g., `talker_config.text_config.hidden_size=1024`), not at the top level. Your config parser must check both the top level and the sub-object:
    ```cpp
    auto val = raw.value("hidden_size", tc.value("hidden_size", default_val));
    ```

13. **Config field names may differ between components** — e.g., `codec_bos_id` in one component vs `codec_bos_token_id` in another. Handle both:
    ```cpp
    cfg.codec_bos_token_id = raw.value("codec_bos_token_id",
                                        raw.value("codec_bos_id", default_val));
    ```

## Prerequisites

User should provide:
1. **HuggingFace PyTorch implementation** — The transformers source code for reference
2. **SafeTensors model folder** — Downloaded model weights with `config.json`
3. **Preprocessing utilities** — Any custom vision/audio preprocessing code
4. **Demo/inference script** — Working end-to-end inference script

## Workflow Overview

```
1. Analyze PyTorch Implementation (ALL sub-models)
         |
         v
2. Map Complete Weight Name Hierarchy
         |
         v
3. Analyze Config.json Nesting Structure
         |
         v
4. Design Sub-Model Decomposition Strategy
         |
         v
5. Plan Weight Loading Strategy Per Sub-Model
         |
         v
6. Implement Config Parsing + Sub-Config Conversion
         |
         v
7. Implement Core LLM (Text Model) + Weight Mapping
         |
         v
8. Implement Vision Encoder + Weight Mapping
         |
         v
9. Implement Audio Encoder + Weight Mapping
         |
         v
10. Implement Multimodal Fusion (Embedding Injection, DeepStack)
         |
         v
11. Implement TTS Pipeline (Talker + Code Predictor + Speech Decoder)
         |
         v
12. Implement Preprocessing (Vision + Audio, C++ native)
         |
         v
13. Implement Pipeline Orchestration (Multi-Model Assembly)
         |
         v
14. Implement Sample/Demo Program
         |
         v
15. Register in safetensors_modeling.cpp
         |
         v
16. Build, Fix Compilation Errors, Run Tests
         |
         v
17. E2E Validation: Text-Only → Image → Audio → Video → Combined
```

---

## Step 1: Analyze PyTorch Implementation

This is THE MOST IMPORTANT step for multimodal models. You must understand **every sub-model**, not just the LLM backbone.

### What to Extract for Each Sub-Model

**1. The Complete Class Hierarchy** — Every `nn.Module` class, its parent class, its constructor parameters, and its submodules:
```python
class Qwen3OmniForConditionalGeneration(PreTrainedModel):
    def __init__(self, config):
        self.thinker = Qwen3OmniThinkerForConditionalGeneration(config.thinker_config)
        self.talker = Qwen3OmniTalkerForConditionalGeneration(config.talker_config)
        self.code2wav = Qwen3OmniCode2Wav(config.code2wav_config)
```

**2. Weight Parameter Names** — The EXACT names as they appear in safetensors. Pay special attention to:
- Models that use `out_proj` vs `o_proj`
- Models that use `fc1`/`fc2` vs `gate_proj`/`up_proj`/`down_proj`
- Models that use `ln_q` vs `norm` for LayerNorm
- Models that use `mlp.0`/`mlp.2` (Sequential) vs named linear layers
- Models with fused QKV (`qkv.weight`) vs separate Q/K/V

**3. Config Structure** — Nested config hierarchy:
```
TopConfig
├── sub_model_1_config
│   ├── text_config          ← nested sub-object!
│   ├── vision_config
│   └── audio_config
├── sub_model_2_config
│   └── code_predictor_config  ← double-nested!
└── sub_model_3_config
```

**4. Forward Pass Flow** — For each sub-model AND for the overall pipeline:
- Inputs: what tensors, what shapes, what dtypes
- Outputs: what tensors, what shapes
- Fusion points: where different modalities merge

**5. Special Patterns**:
- **3D MRoPE** (temporal, height, width position IDs)
- **DeepStack** (injecting intermediate ViT features into early LLM layers)
- **QK-Norm** (RMSNorm on Q and K per head)
- **Windowed attention** (chunked/sliding window)
- **Multi-codebook generation** (autoregressive across codebook layers)
- **Codebook weight sharing** (16 codebook layers sharing one embedding)

### Key Files to Read

```
# Model implementation
transformers/models/<model>/modeling_<model>.py       # ALL classes
transformers/models/<model>/configuration_<model>.py  # ALL config classes
transformers/models/<model>/processing_<model>.py     # Processor (token replacement)
transformers/models/<model>/modular_<model>.py        # If exists, canonical source

# Preprocessing
<utils>/audio_process.py      # Audio loading, format conversion
<utils>/vision_process.py     # Image/video loading, smart resize

# The actual model
<model_dir>/config.json        # Runtime configuration (check nesting!)

# Demo script
<demo>/run_<model>.py          # E2E inference flow
```

### Checklist for Each Sub-Model

For every sub-model, answer these questions before writing any C++ code:

- [ ] What is the module name prefix in safetensors? (e.g., `thinker.`, `talker.`, `code2wav.`)
- [ ] What are ALL weight parameter paths? (list every single one)
- [ ] Are there any name differences between PyTorch module names and safetensors keys?
- [ ] Does the model share weights with another model? (e.g., `tie_word_embeddings`)
- [ ] Does the model have weights NOT in safetensors? (synthetic/computed weights)
- [ ] What is the input/output tensor contract?
- [ ] What are the attention patterns? (causal, windowed, full, bidirectional)
- [ ] Does it use KV cache? (stateful prefill+decode vs one-shot)
- [ ] What position embedding scheme? (standard RoPE, 3D MRoPE, sinusoidal, learned)

---

## Step 2: Map Complete Weight Name Hierarchy

Create a comprehensive table of ALL weight paths in safetensors. This is your ground truth.

### Method: Inspect Safetensors Index

Read `model.safetensors.index.json` to get ALL weight keys:
```python
import json
with open("model.safetensors.index.json") as f:
    index = json.load(f)
for key in sorted(index["weight_map"].keys()):
    print(key)
```

### Categorize by Sub-Model Prefix

Group all weights by their top-level prefix:
```
thinker.audio_tower.*          → Audio Encoder (X weights)
thinker.visual.*               → Vision Encoder (Y weights)
thinker.model.*                → LLM Text Model (Z weights)
thinker.lm_head.*              → LM Head (1 weight)
talker.text_projection.*       → Talker Text Projection (4 weights)
talker.hidden_projection.*     → Talker Hidden Projection (4 weights)
talker.model.*                 → Talker Transformer (W weights)
talker.codec_head.*            → Talker Codec Head (1 weight)
talker.code_predictor.*        → Code Predictor (V weights)
code2wav.*                     → Speech Decoder (U weights)
```

### Identify Weight Name Mismatches Early

Cross-reference PyTorch `nn.Module` names with safetensors keys. Common discrepancies in multimodal models:

| PyTorch Module | Safetensors Convention | C++ Adaptation Needed |
|---|---|---|
| `nn.Sequential([Linear, GELU, Linear])` | `mlp.0.weight`, `mlp.2.weight` | Map to `linear_fc1`, `linear_fc2` |
| `LayerNorm("ln_q")` | `ln_q.weight`, `ln_q.bias` | Map to `norm.weight`, `norm.bias` |
| `ModuleList` with indexed access | `merger_list.0.xxx` | May need to rename (e.g., `deepstack_merger_list.0.xxx`) |
| Shared codebook embedding | `code_embedding.weight` (1 tensor) | Must map to N module references |
| Buffers (not parameters) | `positional_embedding`, `inv_freq`, `code_offset` | May or may not be in safetensors |

---

## Step 3: Analyze Config.json Nesting Structure

Multimodal models have deeply nested configurations. Understanding this prevents critical config misreads.

### Example: Qwen3-Omni Config Nesting

```json
{
  "model_type": "qwen3_omni",
  "tts_pad_token_id": 151671,        // Top-level special tokens
  "tts_eos_token_id": 151673,
  "thinker_config": {
    "audio_config": { ... },          // Audio encoder params
    "vision_config": { ... },         // Vision encoder params
    "text_config": {                  // LLM params
      "hidden_size": 4096,
      "num_hidden_layers": 32,
      ...
    }
  },
  "talker_config": {
    "thinker_hidden_size": 2048,      // Top-level talker params
    "text_config": {                  // NESTED talker transformer params
      "hidden_size": 1024,            // Different from thinker_hidden_size!
      "num_hidden_layers": 28,
      ...
    },
    "code_predictor_config": {
      "code_predictor_config": { ... }  // DOUBLE-nested!
    }
  },
  "code2wav_config": { ... }
}
```

### Config Parsing Pitfalls

**Pitfall 1: Reading wrong `hidden_size`**
The talker's `hidden_size=1024` is under `text_config`, but the top-level talker config has `thinker_hidden_size=2048`. Reading from the wrong level gives wrong dimensions.

**Solution**: Always try the sub-object first, then fall back:
```cpp
const auto& tc = raw.contains("text_config") ? raw.at("text_config") : empty_obj;
cfg.hidden_size = raw.value("hidden_size", tc.value("hidden_size", default));
```

**Pitfall 2: Field name variations**
```cpp
// talker uses "codec_bos_id", not "codec_bos_token_id"
cfg.codec_bos_token_id = raw.value("codec_bos_token_id",
                                    raw.value("codec_bos_id", default));
```

**Pitfall 3: Arrays in nested objects**
```cpp
// mrope_section might be in text_config, not at talker top level
if (raw.contains("mrope_section") && raw.at("mrope_section").is_array()) {
    // parse from top level
} else if (tc.contains("mrope_section") && tc.at("mrope_section").is_array()) {
    // parse from text_config sub-object
}
```

---

## Step 4: Design Sub-Model Decomposition Strategy

For multimodal models, you must decide how to decompose the single PyTorch model into multiple C++ OV models. Each OV model is compiled and run independently.

### Decomposition Criteria

1. **Each model gets its own `ov::Model` graph** — created by a `create_xxx_model()` factory function
2. **Models that share weights** should load from the same `WeightSource`
3. **Models with different inference patterns** (prefill vs decode, with/without KV cache) become separate OV models
4. **Models with different precision requirements** must be separate (e.g., TTS at fp32, LLM at int8)

### Example Decomposition: Qwen3-Omni

| C++ OV Model | Purpose | Weight Prefix | KV Cache | Precision |
|---|---|---|---|---|
| Text Model | Thinker LLM | `thinker.` | Yes (stateful) | User-specified |
| Vision Model | ViT encoder | `thinker.` | No | User-specified |
| Audio Encoder | Whisper encoder | `thinker.audio_tower.` | No | User-specified |
| Talker Embedding | Text+codec embed | `talker.` | No | fp32 |
| Talker Codec Embedding | Codec-only embed | `talker.` | No | fp32 |
| Talker (no cache) | Forward without KV | `talker.` | No | fp32 |
| Talker Prefill | Forward with KV init | `talker.` | Yes (explicit) | fp32 |
| Talker Decode | Forward with KV step | `talker.` | Yes (explicit) | fp32 |
| Code Predictor AR [×N] | Per-step codebook pred | `talker.code_predictor.` | No | fp32 |
| Code Predictor Codec Embed | Sum of codec embeds | `talker.code_predictor.` | No | fp32 |
| Speech Decoder | Codes → waveform | `code2wav.` | No | fp32, CPU only |

**Key insight**: The talker and code predictor need multiple model variants (with/without cache, per-step) because the C++ framework builds static OV graphs. Different control flows require separate graphs.

---

## Step 5: Plan Weight Loading Strategy Per Sub-Model

For each OV model, determine the weight loading strategy:

### Strategy 1: Direct Mapping (with prefix stripping)
Use when C++ module names match safetensors names after stripping a prefix.
```cpp
PrefixMappedWeightSource scoped_source(source, "prefix.");
load_model(model, scoped_source, finalizer);
```
**Example**: Audio encoder — `thinker.audio_tower.layers.0.self_attn.q_proj.weight` → strip `thinker.audio_tower.` → `layers.0.self_attn.q_proj.weight` → matches C++ param directly.

### Strategy 2: Prefix + packed_mapping Rules
Use when C++ module names partially differ from safetensors names.
```cpp
PrefixMappedWeightSource scoped_source(source, "prefix.");
model.packed_mapping().rules.push_back({"source_prefix", "cpp_prefix"});
load_model(model, scoped_source, finalizer);
```
**Example**: Vision merger — `visual.merger.ln_q.weight` mapped to `visual.merger.norm.weight` via packed_mapping rule `{"visual.merger.ln_q.", "visual.merger.norm."}`.

### Strategy 3: Custom WeightSource Adapter
Use when:
- C++ and safetensors naming differ significantly
- Multiple C++ parameters share one safetensors weight (codebook sharing)
- C++ parameters have no corresponding safetensors weight (synthetic weights)

```cpp
class CustomMappedWeightSource : public WeightSource {
    // to_source_name(): C++ param name → safetensors key
    // to_model_name(): safetensors key → C++ param name
    // Synthetic weights: identity/zero tensors generated in constructor
    // Shared weights: multiple C++ params → one safetensors tensor
};
```
**Example**: Speech decoder requires all three strategies simultaneously.

### Strategy 4: No Mapping (Direct Source Keys)
Use when safetensors keys already include the full module path.
```cpp
load_model(model, source, finalizer);  // No prefix stripping
```
**Example**: Talker models — safetensors key `talker.model.layers.0...` matches C++ module path `talker.model.layers[0]...` (only bracket conversion needed).

### Weight Loading Options

| Option | Usage |
|---|---|
| `allow_unmatched = true` | When source has keys not belonging to this model (shared source) |
| `allow_missing = true` | When model has params not in source (synthetic weights) |
| `report_missing = true` | Debug: log missing params during loading |
| `report_unmatched = true` | Debug: log unmatched source keys |

**Best practice**: Use `allow_unmatched = true` for all multimodal sub-models (since they share a source). Use `allow_missing = true` only for models with synthetic weights.

---

## Step 6: Implement Config Parsing

### Create All Config Structs

For each sub-model, create a config struct. The top-level config aggregates them:

```cpp
struct MyModelTextConfig {
    int32_t hidden_size = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t head_dim = 0;
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t vocab_size = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    std::string hidden_act = "silu";
    bool attention_bias = false;
    bool tie_word_embeddings = false;
    std::vector<int32_t> mrope_section;  // For 3D MRoPE
};

struct MyModelVisionConfig {
    int32_t depth = 0;
    int32_t hidden_size = 0;
    int32_t num_heads = 0;
    int32_t intermediate_size = 0;
    int32_t in_channels = 3;
    int32_t patch_size = 16;
    int32_t temporal_patch_size = 2;
    int32_t spatial_merge_size = 2;
    int32_t out_hidden_size = 0;
    std::string hidden_act = "gelu_pytorch_tanh";
    std::vector<int32_t> deepstack_visual_indexes;
};

struct MyModelAudioConfig {
    int32_t num_mel_bins = 128;
    int32_t d_model = 0;
    int32_t encoder_layers = 0;
    int32_t encoder_attention_heads = 0;
    int32_t encoder_ffn_dim = 0;
    int32_t output_dim = 0;
    int32_t n_window = 100;
    int32_t n_window_infer = 400;
    std::string activation_function = "gelu";
};

struct MyModelConfig {
    MyModelTextConfig text;
    MyModelVisionConfig vision;
    MyModelAudioConfig audio;

    // Special token IDs
    int32_t image_token_id = 0;
    int32_t video_token_id = 0;
    int32_t audio_token_id = 0;
    int32_t tts_pad_token_id = 0;
    int32_t tts_eos_token_id = 0;

    // Raw JSON for sub-model configs (parsed by their own converters)
    nlohmann::json talker_config_raw;
    nlohmann::json code2wav_config_raw;

    static MyModelConfig from_json(const nlohmann::json& data);
};
```

### Implement Nested Config Parsing

```cpp
MyModelConfig MyModelConfig::from_json(const nlohmann::json& data) {
    MyModelConfig cfg;

    // Top-level fields
    read_json_param(data, "tts_pad_token_id", cfg.tts_pad_token_id);
    read_json_param(data, "tts_eos_token_id", cfg.tts_eos_token_id);
    read_json_param(data, "tie_word_embeddings", cfg.text.tie_word_embeddings);

    // Navigate to thinker_config
    const nlohmann::json* thinker = nullptr;
    if (data.contains("thinker_config") && data["thinker_config"].is_object()) {
        thinker = &data["thinker_config"];

        // Read special token IDs from thinker level
        read_json_param(*thinker, "image_token_id", cfg.image_token_id);
        read_json_param(*thinker, "audio_token_id", cfg.audio_token_id);

        // Parse text_config from thinker_config
        if (thinker->contains("text_config")) {
            parse_text_config(thinker->at("text_config"), cfg.text);
        }
        // Parse vision_config from thinker_config
        if (thinker->contains("vision_config")) {
            parse_vision_config(thinker->at("vision_config"), cfg.vision);
        }
        // Parse audio_config from thinker_config
        if (thinker->contains("audio_config")) {
            parse_audio_config(thinker->at("audio_config"), cfg.audio);
        }
    }

    // Store raw JSON for sub-models that are parsed by their own converters
    if (data.contains("talker_config")) {
        cfg.talker_config_raw = data["talker_config"];
    }
    if (data.contains("code2wav_config")) {
        cfg.code2wav_config_raw = data["code2wav_config"];
    }

    return cfg;
}
```

### Implement Sub-Config Conversion with Fallback

```cpp
TalkerConfig to_talker_config(const MyModelConfig& cfg) {
    TalkerConfig talker_cfg;
    const auto& raw = cfg.talker_config_raw;
    if (raw.is_null() || !raw.is_object()) return talker_cfg;

    // CRITICAL: Check text_config sub-object for transformer parameters
    const nlohmann::json empty_obj = nlohmann::json::object();
    const auto& tc = (raw.contains("text_config") && raw["text_config"].is_object())
                         ? raw["text_config"] : empty_obj;

    auto val_i32 = [&](const char* key, int32_t def) -> int32_t {
        return raw.value(key, tc.value(key, def));
    };
    auto val_f64 = [&](const char* key, double def) -> double {
        return raw.value(key, tc.value(key, def));
    };

    talker_cfg.hidden_size = val_i32("hidden_size", 1024);
    talker_cfg.num_attention_heads = val_i32("num_attention_heads", 16);
    // IMPORTANT: "thinker_hidden_size" at top level, not in text_config
    talker_cfg.text_hidden_size = raw.value("thinker_hidden_size",
                                    raw.value("text_hidden_size", 2048));
    // Handle field name variations
    talker_cfg.codec_bos_token_id = raw.value("codec_bos_token_id",
                                    raw.value("codec_bos_id", 4197));
    talker_cfg.codec_pad_token_id = raw.value("codec_pad_token_id",
                                    raw.value("codec_pad_id", 4196));

    // Arrays may be in text_config
    if (raw.contains("mrope_section") && raw["mrope_section"].is_array()) {
        for (const auto& s : raw["mrope_section"])
            talker_cfg.mrope_section.push_back(s.get<int32_t>());
    } else if (tc.contains("mrope_section") && tc["mrope_section"].is_array()) {
        for (const auto& s : tc["mrope_section"])
            talker_cfg.mrope_section.push_back(s.get<int32_t>());
    }

    return talker_cfg;
}
```

---

## Step 7: Implement Core LLM (Text Model)

### Headers and Structure

```cpp
// modeling_mymodel_text.cpp
#include "modeling_mymodel.hpp"            // Config + declarations
#include "modeling_mymodel_internal.hpp"    // PrefixMappedWeightSource
#include "modeling/module.hpp"
#include "modeling/ops/llm.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/kv_cache.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/weights/weight_loader.hpp"
```

### Attention with QK-Norm and 3D MRoPE

Many recent models apply RMSNorm to Q and K after projection:
```cpp
class MyModelTextAttention : public Module {
    Parameter* q_proj_param_;
    Parameter* k_proj_param_;
    Parameter* v_proj_param_;
    Parameter* o_proj_param_;
    RMSNorm q_norm_;   // Per-head QK normalization
    RMSNorm k_norm_;
    int32_t num_heads_, num_kv_heads_, head_dim_;
    float scaling_;

    Tensor forward(const Tensor& hidden_states, const Tensor& beam_idx,
                   const Tensor& rope_cos, const Tensor& rope_sin,
                   const Tensor& attention_mask) const {
        auto q = ops::linear(hidden_states, q_proj_weight());
        auto k = ops::linear(hidden_states, k_proj_weight());
        auto v = ops::linear(hidden_states, v_proj_weight());

        // Reshape to heads
        auto q_heads = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
        auto k_heads = k.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});
        auto v_heads = v.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});

        // QK-Norm: normalize per head_dim (NOT hidden_size!)
        q_heads = q_norm_.forward(q_heads);
        k_heads = k_norm_.forward(k_heads);

        // Apply RoPE (standard or 3D MRoPE — handled by rope_cos/sin computation)
        auto* policy = &ctx().op_policy();
        auto q_rot = ops::llm::apply_rope(q_heads, rope_cos, rope_sin, head_dim_, policy);
        auto k_rot = ops::llm::apply_rope(k_heads, rope_cos, rope_sin, head_dim_, policy);

        // KV cache, GQA expansion, SDPA
        auto cached = ops::append_kv_cache(k_rot, v_heads, beam_idx,
                                           num_kv_heads_, head_dim_, full_path(), ctx());
        auto k_expanded = ops::llm::repeat_kv(cached.first, num_heads_, num_kv_heads_, head_dim_);
        auto v_expanded = ops::llm::repeat_kv(cached.second, num_heads_, num_kv_heads_, head_dim_);
        auto mask = ops::llm::build_kv_causal_mask_with_attention(q_rot, k_expanded, attention_mask);
        auto context = ops::llm::sdpa(q_rot, k_expanded, v_expanded, scaling_, 3, &mask, false, policy);

        const int64_t attn_out_dim = static_cast<int64_t>(num_heads_) * head_dim_;
        auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});
        return ops::linear(merged, o_proj_weight());
    }
};
```

### Multimodal Injection Modules

These modules handle placing vision/audio embeddings into the LLM hidden states:

```cpp
// Embedding Injector: replaces token embeddings at multimodal positions
class EmbeddingInjector : public Module {
    // No weights — pure computation
    Tensor forward(const Tensor& embeddings, const Tensor& multimodal_embeds,
                   const Tensor& pos_mask) const {
        // masked_scatter: at positions where pos_mask=true,
        // replace embedding values with multimodal_embeds
        // Implementation uses ov ops for gather/scatter
    }
};

// DeepStack Injector: ADDS intermediate vision features to hidden states
class DeepStackInjector : public Module {
    // No weights — masked addition
    Tensor forward(const Tensor& hidden_states, const Tensor& pos_mask,
                   const Tensor& deepstack_embeds) const {
        // At positions where pos_mask=true:
        // hidden_states[pos] += deepstack_embeds[pos]
    }
};
```

### Text Model with Multimodal Support

```cpp
class TextModel : public Module {
    VocabEmbedding embed_tokens_;
    EmbeddingInjector embedding_injector_;
    DeepStackInjector deepstack_injector_;
    std::vector<TextDecoderLayer> layers_;
    RMSNorm norm_;

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,       // [3, B, seq] for 3D MRoPE
                   const Tensor& beam_idx,
                   const Tensor& attention_mask,
                   const Tensor& visual_embeds,       // Multimodal inputs
                   const Tensor& visual_pos_mask,
                   const Tensor& audio_features,
                   const Tensor& audio_pos_mask,
                   const std::vector<Tensor>& deepstack_embeds) {

        auto hidden = embed_tokens_.forward(input_ids);

        // Inject vision embeddings at image/video token positions
        hidden = embedding_injector_.forward(hidden, visual_embeds, visual_pos_mask);

        // Inject audio features at audio token positions
        hidden = embedding_injector_.forward(hidden, audio_features, audio_pos_mask);

        // Compute 3D MRoPE cos/sin from position_ids
        auto cos_sin = compute_mrope(position_ids);

        for (size_t i = 0; i < layers_.size(); ++i) {
            hidden = layers_[i].forward(hidden, beam_idx,
                                        cos_sin.first, cos_sin.second,
                                        attention_mask);

            // DeepStack: inject intermediate vision features into early layers
            if (i < deepstack_embeds.size()) {
                hidden = deepstack_injector_.forward(hidden, visual_pos_mask,
                                                     deepstack_embeds[i]);
            }
        }

        return norm_.forward(hidden);
    }
};
```

### Factory Function with Weight Tie Fix

```cpp
std::shared_ptr<ov::Model> create_text_model(
    const MyModelConfig& cfg,
    WeightSource& source,
    WeightFinalizer& finalizer,
    bool enable_multimodal_inputs) {

    // 1. Scope weights to "thinker." prefix
    PrefixMappedWeightSource thinker_source(source, "thinker.");

    // 2. Build module hierarchy
    BuilderContext ctx;
    TextForCausalLM model(ctx, cfg.text);

    // 3. Add packed_mapping rules for name differences
    model.packed_mapping().rules.push_back({"model.embed_tokens.", "language_model.embed_tokens."});
    model.packed_mapping().rules.push_back({"model.norm.", "language_model.norm."});
    for (int i = 0; i < cfg.text.num_hidden_layers; ++i) {
        auto si = std::to_string(i);
        model.packed_mapping().rules.push_back(
            {"model.layers[" + si + "].", "language_model.layers." + si + "."});
        model.packed_mapping().rules.push_back(
            {"model.layers." + si + ".", "language_model.layers." + si + "."});
    }

    // 4. Load weights
    LoadOptions options;
    options.allow_unmatched = true;
    options.report_unmatched = true;
    load_model(model, thinker_source, finalizer, options);

    // 5. CRITICAL: Re-tie lm_head after load_model()
    // load_model() iterates ALL safetensors keys including "thinker.lm_head.weight",
    // which may call bind() and break the tie set in the constructor.
    if (cfg.text.tie_word_embeddings) {
        model.get_parameter("lm_head.weight").tie_to(
            model.get_parameter("language_model.embed_tokens.weight"));
    }

    // 6. Create input parameters and forward pass
    auto input_ids = ctx.parameter("input_ids", ov::element::i64, {-1, -1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, {-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, {3, -1, -1});  // 3D MRoPE
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, {-1});

    // Multimodal inputs (optional)
    Tensor visual_embeds, visual_pos_mask, audio_features, audio_pos_mask;
    std::vector<Tensor> deepstack_embeds;
    if (enable_multimodal_inputs) {
        visual_embeds = ctx.parameter("visual_embeds_padded", ov::element::f16, {-1, -1, cfg.vision.out_hidden_size});
        visual_pos_mask = ctx.parameter("visual_pos_mask", ov::element::boolean, {-1, -1});
        audio_features = ctx.parameter("audio_features_padded", ov::element::f16, {-1, -1, cfg.audio.output_dim});
        audio_pos_mask = ctx.parameter("audio_pos_mask", ov::element::boolean, {-1, -1});
        for (size_t i = 0; i < cfg.vision.deepstack_visual_indexes.size(); ++i) {
            deepstack_embeds.push_back(
                ctx.parameter("deepstack_" + std::to_string(i), ov::element::f16,
                              {-1, -1, cfg.vision.out_hidden_size}));
        }
    }

    auto logits = model.forward(input_ids, position_ids, beam_idx, attention_mask,
                                visual_embeds, visual_pos_mask,
                                audio_features, audio_pos_mask, deepstack_embeds);

    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    result->output(0).set_names({"logits"});
    return ctx.build_model({result->output(0)});
}
```

---

## Step 8: Implement Vision Encoder

### Key Vision Architecture Patterns

**Conv3D Patch Embedding** (for video+image):
```cpp
class VisionPatchEmbed : public Module {
    Parameter* proj_weight_param_;  // Conv3D: [embed_dim, in_ch, temporal_ps, ps, ps]
    Parameter* proj_bias_param_;

    Tensor forward(const Tensor& pixel_values, const Tensor& grid_thw) const {
        // pixel_values: [N, in_ch*temporal_ps*ps*ps] (flattened patches)
        // Or [N, C, T, H, W] for 5D input → Conv3D → reshape
        auto out = conv3d(pixel_values, proj_weight(), proj_bias());
        return out.reshape({0, embed_dim_});
    }
};
```

**Fused QKV Attention**:
```cpp
class VisionAttention : public Module {
    Parameter* qkv_param_;   // Fused [3*hidden_size, hidden_size]
    Parameter* proj_param_;  // [hidden_size, hidden_size]

    Tensor forward(const Tensor& hidden_states, const Tensor& cu_seqlens,
                   const Tensor& rotary_pos_emb) const {
        auto qkv = ops::linear(hidden_states, qkv_weight());
        // Split into Q, K, V
        auto q = ops::slice(qkv, 0, hidden_size_, 1, /*axis=*/2);
        auto k = ops::slice(qkv, hidden_size_, 2*hidden_size_, 1, 2);
        auto v = ops::slice(qkv, 2*hidden_size_, 3*hidden_size_, 1, 2);
        // Apply rotary, attention, proj...
    }
};
```

**PatchMerger with Name Mapping**:
```cpp
class VisionPatchMerger : public Module {
    // C++ uses "norm" but PyTorch uses "ln_q"
    LayerNorm norm_;       // → packed_mapping: "ln_q." → "norm."
    // C++ uses "linear_fc1"/"linear_fc2" but PyTorch uses "mlp.0"/"mlp.2"
    Parameter* linear_fc1_weight_;  // → packed_mapping: "mlp.0." → "linear_fc1."
    Parameter* linear_fc2_weight_;  // → packed_mapping: "mlp.2." → "linear_fc2."
};
```

**DeepStack Mergers**:
```cpp
// Multiple merger modules for intermediate ViT layer features
// PyTorch: model.visual.merger_list.{0,1,2} (at layers 8, 16, 24)
// C++: visual.deepstack_merger_list.{0,1,2}
// Requires packed_mapping rules for each index:
model.packed_mapping().rules.push_back(
    {"visual.merger_list.0.ln_q.", "visual.deepstack_merger_list.0.norm."});
model.packed_mapping().rules.push_back(
    {"visual.merger_list.0.mlp.0.", "visual.deepstack_merger_list.0.linear_fc1."});
model.packed_mapping().rules.push_back(
    {"visual.merger_list.0.mlp.2.", "visual.deepstack_merger_list.0.linear_fc2."});
// Repeat for indices 1, 2...
```

---

## Step 9: Implement Audio Encoder

### Audio Encoder Architecture (Whisper-like)

```cpp
class AudioEncoder : public Module {
    // CNN front-end
    Parameter* conv2d1_weight_, *conv2d1_bias_;  // Conv2d(1, 480, 3, stride=2, pad=1)
    Parameter* conv2d2_weight_, *conv2d2_bias_;
    Parameter* conv2d3_weight_, *conv2d3_bias_;
    Parameter* conv_out_weight_, *conv_out_bias_; // Linear(480*freq_bins, d_model)

    // Transformer layers
    std::vector<AudioEncoderLayer> layers_;

    // Post-processing
    LayerNorm ln_post_;
    Parameter* proj1_weight_, *proj1_bias_;      // Linear(d_model, d_model)
    Parameter* proj2_weight_, *proj2_bias_;      // Linear(d_model, output_dim)

    Tensor forward(const Tensor& input_features,
                   const Tensor& feature_attention_mask) const {
        // 1. CNN: 3× Conv2d(stride=2) with GELU → 8× time downsampling
        auto x = input_features;  // [B, 1, n_mels, T]
        x = ops::gelu(conv2d(x, conv2d1_weight(), conv2d1_bias()));  // stride 2
        x = ops::gelu(conv2d(x, conv2d2_weight(), conv2d2_bias()));
        x = ops::gelu(conv2d(x, conv2d3_weight(), conv2d3_bias()));

        // 2. Flatten freq dimension: [B, T', channels*freq_bins]
        // 3. Linear projection: → [B, T', d_model]
        x = ops::linear(x, conv_out_weight());

        // 4. Add sinusoidal positional embeddings
        x = x + positional_embedding_;

        // 5. Build windowed attention mask (chunk-based)
        auto mask = build_windowed_mask(x, n_window_infer_);

        // 6. Transformer encoder layers
        for (auto& layer : layers_) {
            x = layer.forward(x, mask);
        }

        // 7. Post-processing: LN → proj1 → GELU → proj2
        x = ln_post_.forward(x);
        x = ops::gelu(ops::linear(x, proj1_weight()));
        x = ops::linear(x, proj2_weight());  // → output_dim
        return x;
    }
};
```

**Key differences from text attention**:
- Uses `out_proj` not `o_proj`
- Has biases on all projections
- Uses `self_attn_layer_norm` / `final_layer_norm` not `input_layernorm` / `post_attention_layernorm`
- Uses `fc1` / `fc2` not `gate_proj` / `up_proj` / `down_proj` (no gating)
- Uses GELU activation (not SwiGLU)
- No KV cache (full encoder, not autoregressive)

---

## Step 10: Implement Multimodal Fusion

### Embedding Injection (masked_scatter)

The core pattern for injecting multimodal features into the LLM:

```cpp
// Replace embedding values at positions where pos_mask=True
// hidden_states: [B, seq_len, hidden_size]
// multimodal_embeds: [total_tokens, hidden_size] (packed, no padding)
// pos_mask: [B, seq_len] boolean
Tensor inject_embeddings(const Tensor& hidden_states,
                         const Tensor& multimodal_embeds,
                         const Tensor& pos_mask) {
    // Use ov::op to perform masked_scatter
    // At True positions in pos_mask, replace hidden_states values
    // with sequential values from multimodal_embeds
}
```

### DeepStack Injection (masked_add)

```cpp
// ADD intermediate vision features at visual token positions
// hidden_states: [B, seq_len, hidden_size]
// deepstack_embeds: [B, seq_len, hidden_size] (padded to match)
// pos_mask: [B, seq_len]
Tensor inject_deepstack(const Tensor& hidden_states,
                         const Tensor& pos_mask,
                         const Tensor& deepstack_embeds) {
    // hidden_states[pos_mask] += deepstack_embeds[pos_mask]
}
```

---

## Step 11: Implement TTS Pipeline

### TTS Sub-Models Overview

The TTS pipeline consists of:
1. **Talker** — Autoregressive model that generates first codec token per timestep
2. **Code Predictor** — Predicts remaining N-1 codec layers given the first
3. **Speech Decoder** — Neural vocoder converting codec codes to waveform

### Talker Architecture

```cpp
class TalkerForConditionalGeneration : public Module {
    // Input projection
    TextProjection text_projection_;     // Thinker embeds → talker dim
    TextProjection hidden_projection_;   // Thinker hidden[layer_N] → talker dim

    // Core transformer
    TalkerModel model_;                  // 28 layers, 3D MRoPE

    // Output head
    LMHead codec_head_;                  // → first codebook logits

    // Forward: inputs_embeds → codec_logits + hidden_states
};
```

**TextProjection** (MLP):
```
PyTorch: talker.text_projection.linear_fc1.{weight,bias}
         talker.text_projection.linear_fc2.{weight,bias}
```

### Code Predictor Architecture

```cpp
class CodePredictorForConditionalGeneration : public Module {
    CodePredictorModel model_;
    // 31 separate LM heads (one per remaining codebook layer)
    std::vector<LMHead> lm_heads_;       // lm_head[0..30]
    // 31 separate codec embeddings
    // model_.codec_embedding[0..30]

    // Forward per step: input → logits for one codebook layer
    Tensor forward(const Tensor& inputs_embeds,
                   const Tensor& position_ids,
                   int generation_step) const {
        auto hidden = model_.forward_no_cache(inputs_embeds, position_ids);
        return lm_heads_[generation_step].forward(hidden);
    }
};
```

### Speech Decoder with Custom WeightSource

This is the most complex weight mapping in the model:

```cpp
class OmniCode2WavMappedWeightSource : public WeightSource {
    WeightSource& source_;
    std::unordered_map<std::string, ov::Tensor> synthetic_;

    OmniCode2WavMappedWeightSource(WeightSource& source, const DecoderConfig& cfg) {
        const size_t dim = cfg.latent_dim;
        constexpr float mean_scale = 1.0f / 16.0f;  // 1/num_quantizers

        // Synthetic identity weights (not in safetensors)
        synthetic_["decoder.quantizer.rvq_first.output_proj.weight"] =
            make_conv1d_identity_weight(dim, dim, 1, mean_scale);
        synthetic_["decoder.quantizer.rvq_rest.output_proj.weight"] =
            make_conv1d_identity_weight(dim, dim, 1, mean_scale);
        synthetic_["decoder.pre_conv.conv.weight"] =
            make_conv1d_identity_weight(dim, dim, 5);  // kernel=5, identity at center
        synthetic_["decoder.pre_conv.conv.bias"] = make_zero_tensor({dim});
        synthetic_["decoder.pre_transformer.input_proj.weight"] =
            make_identity_weight(dim, dim);
        synthetic_["decoder.pre_transformer.input_proj.bias"] = make_zero_tensor({dim});
        synthetic_["decoder.pre_transformer.output_proj.weight"] =
            make_identity_weight(dim, dim);
        synthetic_["decoder.pre_transformer.output_proj.bias"] = make_zero_tensor({dim});
    }

    // Name conversion: C++ param → safetensors key
    static std::string to_source_name(const std::string& model_name) {
        // Strip "decoder." prefix
        // Convert layers[N] → layers.N
    }

    // Name conversion: safetensors key → C++ param
    static std::string to_model_name(const std::string& src_name) {
        // Prepend "decoder."
        // Convert layers.N → layers[N]
    }

    // All quantizer codebook layers share one tensor
    bool is_quantizer_codebook(const std::string& name) const {
        return name.find("rvq_first.vq.layers") != std::string::npos ||
               name.find("rvq_rest.vq.layers") != std::string::npos;
    }

    const ov::Tensor& get_tensor(const std::string& name) const override {
        // 1. Check synthetic weights first
        auto it = synthetic_.find(name);
        if (it != synthetic_.end()) return it->second;

        // 2. Codebook sharing: all 16 layers → one "code_embedding.weight"
        if (is_quantizer_codebook(name) && source_.has("code_embedding.weight"))
            return source_.get_tensor("code_embedding.weight");

        // 3. Normal mapping: strip prefix + convert brackets
        return source_.get_tensor(to_source_name(name));
    }
};
```

---

## Step 12: Implement Preprocessing (C++ Native)

### Audio Preprocessing: Mel Spectrogram

Implement a pure C++ Whisper mel spectrogram extractor (no Python dependency):

```cpp
class WhisperMelSpectrogram {
public:
    struct MelResult {
        ov::Tensor features;           // [1, n_mels, T_frames]
        ov::Tensor attention_mask;     // [1, T_frames]
        size_t num_frames;
    };

    // Core pipeline:
    // 1. Read WAV file → float32 mono 16kHz
    // 2. Reflect-pad waveform by n_fft/2
    // 3. Hann-windowed STFT (n_fft=400, hop=160)
    // 4. Power spectrum → Slaney mel filterbank (128 bins)
    // 5. Log10 → clamp(max-8) → normalize (x+4)/4

    static std::vector<float> read_wav_to_float32(const std::string& path);
    static MelResult extract_mel_features(const std::vector<float>& audio,
                                           int n_mels = 128,
                                           int n_fft = 400,
                                           int hop_length = 160);

    // Audio encoder chunking:
    // Split mel into chunks of n_window*2 frames
    // Compute CNN output lengths per chunk
    static void chunk_mel_for_audio_encoder(const MelResult& mel,
                                             int n_window,
                                             /* out */ ov::Tensor& chunked_features,
                                             /* out */ std::vector<int>& feature_lengths);
};
```

### Vision Preprocessing

```cpp
class VisionPreprocessor {
public:
    struct VisionResult {
        ov::Tensor pixel_values;       // Patchified [N, C*Tp*Ps*Ps]
        std::vector<int64_t> grid_thw; // [T, H, W] per image/video
        ov::Tensor pos_embed;          // Position embeddings
        ov::Tensor rotary_cos, rotary_sin;  // Rotary embeddings
    };

    // Smart resize: round to factor, respect min/max pixel budgets
    static std::pair<int, int> smart_resize(int h, int w, int factor = 28,
                                             int min_pixels = 3136,
                                             int max_pixels = 12845056);

    // Smart frame count for video
    static int smart_nframes(int total_frames, float fps,
                             float target_fps = 2.0,
                             int min_frames = 4, int max_frames = 768);

    VisionResult preprocess(const std::vector<uint8_t>& image_data,
                             const ov::Tensor& pos_embed_weight);
    VisionResult preprocess_video(const std::vector<ov::Tensor>& frames,
                                   const ov::Tensor& pos_embed_weight);
};
```

---

## Step 13: Implement Pipeline Orchestration

### Pipeline Model Assembly

```cpp
struct PipelineModels {
    std::shared_ptr<ov::Model> text;
    std::shared_ptr<ov::Model> vision;
    std::shared_ptr<ov::Model> audio_encoder;
    std::shared_ptr<ov::Model> talker_embedding;
    std::shared_ptr<ov::Model> talker_codec_embedding;
    std::shared_ptr<ov::Model> talker;
    std::shared_ptr<ov::Model> talker_prefill;
    std::shared_ptr<ov::Model> talker_decode;
    std::vector<std::shared_ptr<ov::Model>> code_predictor_ar;
    std::shared_ptr<ov::Model> code_predictor_codec_embedding;
    std::vector<std::shared_ptr<ov::Model>> code_predictor_single_codec_embedding;
    std::shared_ptr<ov::Model> speech_decoder;
};

PipelineModels create_pipeline_models(
    const MyModelConfig& cfg,
    WeightSource& source,
    WeightFinalizer& finalizer) {

    PipelineModels models;

    models.text = create_text_model(cfg, source, finalizer,
                                     /*enable_multimodal_inputs=*/true);
    models.vision = create_vision_model(cfg, source, finalizer);
    models.audio_encoder = create_audio_encoder_model(cfg, source, finalizer);

    // TTS models (all share the same source, loaded independently)
    models.talker_embedding = create_talker_embedding_model(cfg, source, finalizer);
    models.talker_codec_embedding = create_talker_codec_embedding_model(cfg, source, finalizer);
    models.talker = create_talker_model(cfg, source, finalizer);
    models.talker_prefill = create_talker_prefill_model(cfg, source, finalizer);
    models.talker_decode = create_talker_decode_model(cfg, source, finalizer);

    // Code predictor: one AR model per generation step
    int num_code_groups = cfg.talker_config_raw.value("num_code_groups", 32);
    for (int step = 0; step < num_code_groups - 1; ++step) {
        models.code_predictor_ar.push_back(
            create_code_predictor_ar_model(cfg, step, source, finalizer));
        models.code_predictor_single_codec_embedding.push_back(
            create_code_predictor_single_codec_embed_model(cfg, step, source, finalizer));
    }
    models.code_predictor_codec_embedding =
        create_code_predictor_codec_embed_model(cfg, source, finalizer);

    models.speech_decoder = create_speech_decoder_model(cfg, source, finalizer);

    return models;
}
```

---

## Step 14: Implement Sample/Demo Program

### Pipeline Runtime Flow

```cpp
int main(int argc, char* argv[]) {
    // 1. Load config and weights
    auto cfg = MyModelConfig::from_json(load_config(model_dir));
    SafetensorsWeightSource source(model_dir);
    WeightFinalizer finalizer;

    // 2. Build all models
    auto models = create_pipeline_models(cfg, source, finalizer);

    // 3. Compile with appropriate precision
    ov::Core core;
    auto text_compiled = core.compile_model(models.text, device, text_props);
    auto vision_compiled = core.compile_model(models.vision, device, vision_props);
    auto audio_compiled = core.compile_model(models.audio_encoder, device, audio_props);

    // TTS: ALWAYS fp32, speech decoder ALWAYS CPU
    ov::AnyMap tts_props = {{"INFERENCE_PRECISION_HINT", "f32"}};
    auto talker_compiled = core.compile_model(models.talker_prefill, device, tts_props);
    auto speech_decoder_compiled = core.compile_model(models.speech_decoder, "CPU", tts_props);

    // 4. Preprocess inputs
    auto vision_result = preprocessor.preprocess(image_data, pos_embed_weight);
    auto mel_result = WhisperMelSpectrogram::extract_mel_features(audio_data);

    // 5. Vision inference
    vision_request.set_tensor("pixel_values", vision_result.pixel_values);
    vision_request.infer();
    auto visual_embeds = vision_request.get_tensor("visual_embeds");
    auto deepstack = extract_deepstack(vision_request);

    // 6. Audio inference
    audio_request.set_tensor("input_features", mel_result.features);
    audio_request.infer();
    auto audio_features = audio_request.get_tensor("audio_features");

    // 7. Text prefill with multimodal inputs
    text_request.set_tensor("input_ids", tokenized_prompt);
    text_request.set_tensor("visual_embeds_padded", scatter_visual_embeds(visual_embeds));
    text_request.set_tensor("visual_pos_mask", visual_pos_mask);
    text_request.set_tensor("audio_features_padded", scatter_audio_features(audio_features));
    text_request.set_tensor("audio_pos_mask", audio_pos_mask);
    for (size_t i = 0; i < deepstack.size(); ++i)
        text_request.set_tensor("deepstack_" + std::to_string(i), deepstack[i]);
    text_request.infer();

    // 8. Text decode loop
    while (!eos) {
        auto next_token = argmax(text_request.get_tensor("logits"));
        // Set zero-size multimodal inputs for decode steps
        text_request.set_tensor("visual_embeds_padded", zeros(1,1,dim));
        // ... (single token decode)
    }

    // 9. TTS pipeline (if audio output enabled)
    // 9a. Talker prefill with projected thinker embeddings
    // 9b. Talker AR decode: generate codec tokens
    // 9c. Code predictor: expand to full codebook
    // 9d. Speech decoder: codec codes → waveform

    // 10. Output
    save_wav("output.wav", waveform, 24000);
}
```

---

## Step 15: Register in safetensors_modeling.cpp

For multimodal models, registration follows the same 3-location pattern as simple LLMs:

### Location 1: Include header
```cpp
#include "modeling/models/mymodel/modeling_mymodel.hpp"
```

### Location 2: Force modeling flag
```cpp
const bool force_modeling_api = (model_type == "mymodel" || ...);
```

### Location 3: Config mapping block
```cpp
} else if (hf_config.model_type == "mymodel") {
    // Parse the full multimodal config
    auto json_data = load_json(config_path);
    auto cfg = MyModelConfig::from_json(json_data);
    // Build just the text model for the standard CausalLM path
    ov_model = create_text_model(cfg, source, finalizer, /*multimodal=*/false);
```

**Note**: The `safetensors_modeling.cpp` path typically only creates the text model. The full multimodal pipeline (vision, audio, TTS) is assembled by the sample program or a dedicated pipeline class.

---

## Step 16: Build and Debug

### Build Command
```bash
cmake --build build --target test_modeling_api -j8
# For samples:
cmake --build build --target modeling_qwen3_omni -j8
```

### Common Multimodal-Specific Errors

**Error: Garbage output ("蟊Prev mg ostensibly...")**
- Cause: `lm_head.weight` tie broken by `load_model()`
- Fix: Re-tie after `load_model()` when `tie_word_embeddings` is true

**Error: Wrong talker dimensions (e.g., 2048 instead of 1024)**
- Cause: Reading `hidden_size` from talker top-level config instead of `text_config` sub-object
- Fix: Config parser must check nested `text_config` sub-object with fallback

**Error: TTS audio quality degradation (quiet, wrong length)**
- Cause: TTS models compiled with reduced precision (int8/fp16)
- Fix: Always compile TTS models with `inference_precision=f32`

**Error: Speech decoder produces noise on GPU**
- Cause: GPU plugin downcasts to fp16 despite fp32 hint, accumulates rounding errors in SnakeBeta activation across 480x upsampling
- Fix: Force speech decoder to `device="CPU"`

**Error: Vision model crashes with dynamic-size images**
- Cause: Static `ops::slice` indices exceed actual sequence length
- Fix: Use `ov::op::v8::Slice` with dynamic safe indices: `min(desired_index, seq_len - 1)`

**Error: "Missing mapped code2wav tensor"**
- Cause: Speech decoder weight source doesn't handle the name translation between safetensors and C++ module paths
- Fix: Implement `to_source_name()`/`to_model_name()` in custom WeightSource, handle codebook sharing and synthetic weights

**Error: Audio encoder produces wrong feature counts**
- Cause: CNN output length formula doesn't match Python's `_get_feat_extract_output_lengths`
- Fix: Implement exact formula: 3× stride-2 convolution: `ceil(ceil(ceil(L/2)/2)/2)`, plus windowed chunking logic

---

## Step 17: E2E Validation Strategy

Validate incrementally, adding one modality at a time:

### Test Matrix

| Case | Input | Output | What It Validates |
|---|---|---|---|
| 1 | Text only | Text | LLM backbone, tokenization, position IDs |
| 2 | Text + Image | Text | Vision encoder, embedding injection, deepstack, 3D MRoPE |
| 3 | Text + Audio | Text | Audio encoder, mel extraction, audio injection |
| 4 | Text + Image + Audio | Text | Combined multimodal fusion |
| 5 | Text + Image | Text + Audio | Full TTS pipeline: talker, code predictor, speech decoder |
| 6 | Text + Audio | Text + Audio | Audio-in + audio-out |
| 7 | Text + Video | Text | Video preprocessing, temporal patches |
| 8 | Text + Video + Audio | Text + Audio | Full multimodal with video |

### Validation Approach

1. **Binary comparison with Python reference**: Run PyTorch inference, dump all intermediate tensors, compare with C++ outputs at every stage
2. **Token-by-token comparison**: The first N generated tokens should match exactly between Python and C++ (fp32 mode)
3. **Audio quality check**: Compare waveform amplitude (mean_abs), duration (sample count), and perceptual quality

---

## Available Operations Reference

### Core Ops (`ops::`)
| Function | Signature | Notes |
|----------|-----------|-------|
| `linear` | `(x, weight)` | Matrix multiply: x @ weight^T |
| `matmul` | `(a, b, ta, tb)` | General matmul with transpose flags |
| `slice` | `(data, start, stop, step, axis)` | **5 args, all int64_t** |
| `gather` | `(data, indices, axis)` | Gather along axis |
| `concat` | `(tensors_vec, axis)` | Concatenate tensors |
| `silu` | `(x)` | SiLU activation |
| `rms` | `(x, weight, eps)` | RMS normalization |
| `const_scalar` | `(ctx, value)` | Create scalar constant |
| `const_vec` | `(ctx, values)` | Create vector constant |
| `constant` | `(tensor, ctx)` | Wrap ov::Tensor as constant |

### LLM Ops (`ops::llm::`)
| Function | Signature | Notes |
|----------|-----------|-------|
| `rope_cos_sin` | `(positions, head_dim, theta, policy)` | Standard RoPE |
| `apply_rope` | `(x, cos, sin, head_dim, policy)` | Apply RoPE to tensor |
| `sdpa` | `(q, k, v, scale, softmax_axis, mask, causal, policy)` | Attention |
| `repeat_kv` | `(x, num_heads, num_kv_heads, head_dim)` | GQA head expansion |
| `build_kv_causal_mask_with_attention` | `(q, k, attention_mask)` | Causal mask |
| `append_kv_cache` | `(k, v, beam_idx, num_kv_heads, head_dim, prefix, ctx)` | KV cache |

### NN Ops (`ops::nn::`)
| Function | Signature | Notes |
|----------|-----------|-------|
| `gelu` | `(x, approximate)` | GELU activation |
| `layer_norm` | `(x, weight, bias, eps, axis)` | Layer normalization |
| `group_norm` | `(x, weight, bias, groups, eps)` | Group normalization |

### Shape Ops (`shape::`)
| Function | Signature | Notes |
|----------|-----------|-------|
| `dim` | `(tensor, axis)` | Get dimension size (1D tensor) |
| `of` | `(tensor)` | Get full shape |
| `make` | `({dim1, dim2, ...})` | Construct shape |
| `broadcast_to` | `(tensor, shape)` | Broadcast tensor |

### Tensor Methods
| Method | Notes |
|--------|-------|
| `.reshape({dims})` | Reshape; 0 = keep dim, -1 = infer |
| `.permute({order})` | Transpose dimensions |
| `.unsqueeze(axis)` | Add dimension |
| `.squeeze(axis)` | Remove dimension |
| `.to(dtype)` | Type conversion |
| `.softmax(axis)` | Softmax |
| `.mean(axis, keepdim)` | Mean reduction |
| `+`, `-`, `*`, `/` | Element-wise arithmetic |

---

## Weight Name Mapping Patterns Summary

### Pattern 1: Prefix Stripping
```
Safetensors: thinker.model.layers.0.self_attn.q_proj.weight
                ↓ PrefixMappedWeightSource("thinker.")
Scoped:     model.layers.0.self_attn.q_proj.weight
```

### Pattern 2: packed_mapping Rules
```
Scoped:     model.layers.0.self_attn.q_proj.weight
                ↓ packed_mapping rule: {"model.layers.0.", "language_model.layers.0."}
C++ Param:  language_model.layers.0.self_attn.q_proj.weight
```

### Pattern 3: Automatic Layer Index Conversion
```
Scoped:     layers.0.self_attn.q_proj.weight
                ↓ WeightNameMapper (automatic)
C++ Param:  layers[0].self_attn.q_proj.weight
```

### Pattern 4: Custom WeightSource Adapter
```
Safetensors: code2wav.backbone.transformer.layers.0.self_attn.q_proj.weight
                ↓ PrefixMappedWeightSource("code2wav.")
Scoped:     backbone.transformer.layers.0.self_attn.q_proj.weight
                ↓ OmniCode2WavMappedWeightSource.to_model_name()
C++ Param:  decoder.backbone.transformer.layers[0].self_attn.q_proj.weight
```

### Pattern 5: Shared Weights
```
Safetensors: code2wav.code_embedding.weight (1 tensor)
                ↓ OmniCode2WavMappedWeightSource
C++ Param:  decoder.quantizer.rvq_first.vq.layers[0]._codebook.embed  (same tensor)
            decoder.quantizer.rvq_rest.vq.layers[0]._codebook.embed   (same tensor)
            decoder.quantizer.rvq_rest.vq.layers[1]._codebook.embed   (same tensor)
            ... (all 16 layers share one tensor)
```

### Pattern 6: Synthetic Weights (Not In Safetensors)
```
(no safetensors key)
    ↓ Generated in OmniCode2WavMappedWeightSource constructor
C++ Param:  decoder.quantizer.rvq_first.output_proj.weight  →  identity × 1/16
            decoder.pre_conv.conv.weight                     →  identity at center
            decoder.pre_transformer.input_proj.weight        →  identity matrix
```

### Pattern 7: Post-Load Re-Tying
```
Constructor: lm_head.tie_to(embed_tokens.weight)
    ↓ load_model() loads separate lm_head.weight → BREAKS TIE
    ↓ Post-load fix:
model.get_parameter("lm_head.weight").tie_to(
    model.get_parameter("language_model.embed_tokens.weight"));
```

---

## File Checklist

### New Files to Create

| File | Purpose |
|------|---------|
| `modeling/models/<model>/modeling_<model>.hpp` | Config structs + all declarations + factory function prototypes |
| `modeling/models/<model>/modeling_<model>_internal.hpp` | PrefixMappedWeightSource, utility helpers |
| `modeling/models/<model>/modeling_<model>_text.cpp` | LLM text model + multimodal injection |
| `modeling/models/<model>/modeling_<model>_vision.cpp` | Vision encoder (ViT) |
| `modeling/models/<model>/modeling_<model>_audio.cpp` | Audio encoder (Whisper-like) |
| `modeling/models/<model>/modeling_<model>_tts.cpp` | TTS: talker + code predictor + speech decoder weight mapping |
| `modeling/models/<model>/modeling_<model>_config.cpp` | Config parsing from JSON |
| `modeling/models/<model>/<model>_pipeline.hpp` | Pipeline model struct + multimodal info types |
| `modeling/models/<model>/<model>_pipeline.cpp` | Pipeline assembly (create all models) |
| `modeling/models/<model>/processing_<model>_audio.hpp/.cpp` | Audio preprocessing |
| `modeling/models/<model>/processing_<model>_vision.hpp/.cpp` | Vision preprocessing |
| `modeling/models/<model>/processing_<model>_vl.hpp/.cpp` | VL preprocessing (delegates to base) |
| `modeling/models/<model>/whisper_mel_spectrogram.hpp/.cpp` | Pure C++ mel spectrogram extraction |
| `modeling/samples/modeling_<model>.cpp` | Sample/demo program |
| `modeling/tests/<model>_spec_test.cpp` | Unit tests |

### Existing Files to Modify

| File | Change |
|------|--------|
| `safetensors_utils/safetensors_modeling.cpp` | 3 locations: include, force flag, config mapping |
| `safetensors_utils/hf_config.hpp/.cpp` | Add model-specific config fields (if needed) |
| `modeling/samples/CMakeLists.txt` | Add new sample targets |

### Files That Do NOT Need Changes

| File | Reason |
|------|--------|
| `modeling/CMakeLists.txt` | Sources are glob-matched automatically |
| `modeling/tests/CMakeLists.txt` | Tests are glob-matched automatically |

---

## Key Lessons from Qwen3-Omni Implementation

### Lesson 1: Weight Tying Must Survive load_model()
`load_model()` iterates ALL safetensors keys via the weight source. For tied weights (lm_head ↔ embed_tokens), if the safetensors file has a separate `lm_head.weight` key, `load_model()` will call `bind()` on it, breaking the tie. Always re-tie after loading.

### Lesson 2: Nested Config Is The Norm For Multimodal Models
Real multimodal configs have 2-3 levels of nesting. The same field name (e.g., `hidden_size`) appears at multiple levels with different values. Your parser must navigate correctly.

### Lesson 3: Different Sub-Models Need Different Precision
LLM backbone can use int8/int4 for efficiency. TTS pipeline (talker, code predictor, speech decoder) MUST use fp32 — they're sensitive to quantization. Speech decoder specifically must run on CPU because GPU plugins silently downcast to fp16.

### Lesson 4: Weight Name Mapping Scales With Complexity
Simple models: just prefix stripping + auto bracket conversion.
Moderate models: prefix + packed_mapping rules.
Complex models (speech decoder): need full custom WeightSource with name translation, codebook sharing, and synthetic weight generation.

### Lesson 5: Vision / Audio Preprocessing Is Half The Work
For multimodal models, the preprocessing pipeline (smart resize, mel spectrogram, patch embedding, position computation) is as complex as the model itself. Implement C++ native preprocessing early, with Python bridge as fallback.

### Lesson 6: Multiple OV Models Per Pipeline
Unlike simple LLMs (1 model), multimodal models decompose into 10+ separate OV models. Each has different: weight loading, precision, device, input/output contract. Plan the decomposition before writing any model code.

### Lesson 7: Debug By Binary Comparison
For complex multimodal models, the only reliable debug method is dumping intermediate tensors from both Python and C++ inference, then comparing them stage by stage. Build dump infrastructure from the start.

### Lesson 8: Dynamic Slicing For Debug Code
Using `ops::slice` with static indices in debug code can crash when actual tensor dimensions are smaller (e.g., small test images). Always use `ov::op::v8::Slice` with clamped dynamic indices for debug outputs.

---

## Reference: Existing Model Implementations

| Model | Directory | Pattern | Special Features |
|-------|-----------|---------|------------------|
| Qwen3 Dense | `models/qwen3/` | Standard LLM | Fused norm+residual, attention bias |
| Qwen3 MoE | `models/qwen3_moe/` | MoE | Expert routing, top-k selection |
| Qwen3 VL | `models/qwen3_vl/` | VLM | Vision + text, base for Omni |
| Qwen3 Omni | `models/qwen3_omni/` | Full multimodal | Vision + audio + text + TTS, DeepStack, 3D MRoPE |
| Qwen3 TTS | `models/qwen3_tts/` | TTS | Speech decoder, code predictor (shared with Omni) |
| SmolLM3 | `models/smollm3/` | Standard LLM | MLP bias support |
| LFM2 | `models/lfm2/` | Hybrid | Attention/conv alternating, per-head norms |
