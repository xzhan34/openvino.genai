// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3-Omni DFlash Speculative Decoding sample.
//
// Combines a target Qwen3-Omni-4B thinker model with a Qwen3-4B-DFlash-b16
// draft model for speculative decoding.  Text-only and VL (vision+language)
// modes are supported.
//
// CLI (same positional convention as modeling_qwen3_5_dflash):
//   <TARGET_MODEL_DIR> <DRAFT_MODEL_DIR> [PROMPT] [DEVICE] [MAX_NEW_TOKENS]
//   [BLOCK_SIZE] [TARGET_QUANT] [DRAFT_QUANT] [IMAGE_PATH]

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <openvino/openvino.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>

#include "load_image.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "loaders/model_config.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"
#include "modeling/weights/quantization_config.hpp"
#include "utils.hpp"

#include "modeling/models/dflash_draft/dflash_draft.hpp"
#include "modeling/models/qwen3_omni/modeling_qwen3_omni.hpp"
#include "modeling/models/qwen3_omni/modeling_qwen3_omni_internal.hpp"
#include "modeling/models/qwen3_omni/processing_qwen3_omni_vl.hpp"
#include "modeling/models/qwen3_omni/processing_qwen3_omni_vision.hpp"

#ifdef _WIN32
#  define NOMINMAX
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <shellapi.h>
#  pragma comment(lib, "shell32.lib")
#endif

namespace {

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// Performance tracking
// ---------------------------------------------------------------------------

struct StageStats {
    double total_ms = 0.0;
    double max_ms = 0.0;
    size_t count = 0;

    void add(double ms) {
        total_ms += ms;
        max_ms = std::max(max_ms, ms);
        ++count;
    }

    double avg() const {
        return count ? total_ms / static_cast<double>(count) : 0.0;
    }
};

struct PerfStats {
    StageStats prefill_wall;
    StageStats draft_wall;
    StageStats verify_wall;
    StageStats other_wall;
    StageStats postproc_wall;

    size_t draft_steps = 0;
    size_t accepted_tokens = 0;
    size_t generated_tokens = 0;
    double ttft_ms = 0.0;
    double total_generate_ms = 0.0;
    std::vector<size_t> accepted_per_step;
};

double duration_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_stage_stats(const std::string& name, const StageStats& wall) {
    std::cout << "  " << name << ": wall " << wall.avg() << " ms (max " << wall.max_ms
              << ", " << wall.count << " runs)" << std::endl;
}

// ---------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------

std::vector<int64_t> tensor_to_ids(const ov::Tensor& ids_tensor) {
    const auto shape = ids_tensor.get_shape();
    if (shape.size() != 2 || shape[0] != 1)
        throw std::runtime_error("input_ids tensor must have shape [1, S]");
    const size_t seq_len = shape[1];
    const auto* data = ids_tensor.data<const int64_t>();
    return std::vector<int64_t>(data, data + seq_len);
}

ov::Tensor make_ids_tensor(const std::vector<int64_t>& ids) {
    ov::Tensor tensor(ov::element::i64, {1, ids.size()});
    std::memcpy(tensor.data(), ids.data(), ids.size() * sizeof(int64_t));
    return tensor;
}

ov::Tensor make_attention_mask(size_t seq_len) {
    ov::Tensor mask(ov::element::i64, {1, seq_len});
    auto* data = mask.data<int64_t>();
    for (size_t i = 0; i < seq_len; ++i) data[i] = 1;
    return mask;
}

// 3D mRoPE position_ids [3, 1, seq_len] for text-only (all dims same).
ov::Tensor make_mrope_position_ids(size_t start, size_t count) {
    ov::Tensor ids(ov::element::i64, {3, 1, count});
    auto* data = ids.data<int64_t>();
    for (size_t dim = 0; dim < 3; ++dim)
        for (size_t i = 0; i < count; ++i)
            data[dim * count + i] = static_cast<int64_t>(start + i);
    return ids;
}

ov::Tensor make_beam_idx(size_t batch) {
    ov::Tensor beam_idx(ov::element::i32, {batch});
    auto* data = beam_idx.data<int32_t>();
    for (size_t i = 0; i < batch; ++i) data[i] = static_cast<int32_t>(i);
    return beam_idx;
}

ov::Tensor make_zero_tensor(const ov::element::Type& type, const ov::Shape& shape) {
    ov::Tensor tensor(type, shape);
    std::memset(tensor.data(), 0, tensor.get_byte_size());
    return tensor;
}

ov::Tensor ensure_f32_copy(const ov::Tensor& t) {
    ov::Tensor out(ov::element::f32, t.get_shape());
    t.copy_to(out);
    return out;
}

template <typename T>
int64_t argmax_row(const T* data, size_t vocab) {
    T max_val = data[0];
    size_t max_idx = 0;
    for (size_t i = 1; i < vocab; ++i) {
        if (data[i] > max_val) { max_val = data[i]; max_idx = i; }
    }
    return static_cast<int64_t>(max_idx);
}

std::vector<int64_t> argmax_logits_slice(const ov::Tensor& logits, size_t start, size_t count) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3 || shape[0] != 1)
        throw std::runtime_error("logits tensor must have shape [1, S, V]");
    const size_t seq_len = shape[1];
    const size_t vocab = shape[2];
    if (start + count > seq_len)
        throw std::runtime_error("logits slice out of range");

    std::vector<int64_t> tokens;
    tokens.reserve(count);

    if (logits.get_element_type() == ov::element::f16) {
        const auto* data = logits.data<const ov::float16>();
        for (size_t i = 0; i < count; ++i)
            tokens.push_back(argmax_row(data + (start + i) * vocab, vocab));
        return tokens;
    }
    if (logits.get_element_type() == ov::element::bf16) {
        const auto* data = logits.data<const ov::bfloat16>();
        for (size_t i = 0; i < count; ++i)
            tokens.push_back(argmax_row(data + (start + i) * vocab, vocab));
        return tokens;
    }
    if (logits.get_element_type() == ov::element::f32) {
        const auto* data = logits.data<const float>();
        for (size_t i = 0; i < count; ++i)
            tokens.push_back(argmax_row(data + (start + i) * vocab, vocab));
        return tokens;
    }
    throw std::runtime_error("Unsupported logits dtype");
}

int64_t argmax_last_token(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3 || shape[0] != 1)
        throw std::runtime_error("logits tensor must have shape [1, S, V]");
    return argmax_logits_slice(logits, shape[1] - 1, 1).front();
}

int64_t resolve_mask_token_id(ov::genai::Tokenizer tokenizer) {
    const auto vocab = tokenizer.get_vocab();
    const auto it = vocab.find("<|MASK|>");
    if (it != vocab.end()) return it->second;
    auto try_single_token = [&](const std::string& token) -> std::optional<int64_t> {
        try {
            auto encoded = tokenizer.encode(token, {ov::genai::add_special_tokens(false)}).input_ids;
            if (encoded.get_shape() == ov::Shape{1, 1})
                return encoded.data<const int64_t>()[0];
        } catch (...) {}
        return std::nullopt;
    };
    if (auto fim_pad = try_single_token("<|fim_pad|>")) return *fim_pad;
    if (auto vision_pad = try_single_token("<|vision_pad|>")) return *vision_pad;
    const int64_t pad_id = tokenizer.get_pad_token_id();
    if (pad_id != -1) return pad_id;
    const int64_t eos_id = tokenizer.get_eos_token_id();
    if (eos_id != -1) return eos_id;
    throw std::runtime_error("Tokenizer has no suitable mask/pad/eos token");
}

// ---------------------------------------------------------------------------
// Quantization helpers
// ---------------------------------------------------------------------------

auto parse_quant_mode = [](const std::string& s)
    -> ov::genai::modeling::weights::QuantizationConfig::Mode {
    if (s == "INT4_ASYM") return ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_ASYM;
    if (s == "INT4_SYM")  return ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_SYM;
    if (s == "INT8_ASYM") return ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_ASYM;
    if (s == "INT8_SYM")  return ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_SYM;
    return ov::genai::modeling::weights::QuantizationConfig::Mode::NONE;
};

auto quant_mode_name = [](ov::genai::modeling::weights::QuantizationConfig::Mode m)
    -> const char* {
    switch (m) {
        case ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_ASYM: return "INT4_ASYM";
        case ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_SYM:  return "INT4_SYM";
        case ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_ASYM: return "INT8_ASYM";
        case ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_SYM:  return "INT8_SYM";
        default: return "NONE";
    }
};

std::string quant_mode_cache_token(ov::genai::modeling::weights::QuantizationConfig::Mode m) {
    switch (m) {
        case ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_ASYM: return "4a";
        case ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_SYM:  return "4s";
        case ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_ASYM: return "8a";
        case ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_SYM:  return "8s";
        default: return "fp";
    }
}

std::string quant_ir_suffix(const ov::genai::modeling::weights::QuantizationConfig& cfg) {
    if (!cfg.enabled()) return "";
    return "_q" + quant_mode_cache_token(cfg.mode)
         + "_b" + quant_mode_cache_token(cfg.backup_mode)
         + "_g" + std::to_string(cfg.group_size);
}

// ---------------------------------------------------------------------------
// IR cache helpers
// ---------------------------------------------------------------------------

bool has_ir_model_pair(const std::filesystem::path& xml_path, const std::filesystem::path& bin_path) {
    return std::filesystem::exists(xml_path) && std::filesystem::is_regular_file(xml_path) &&
           std::filesystem::exists(bin_path) && std::filesystem::is_regular_file(bin_path);
}

bool is_export_ir_mode() {
    const char* raw = std::getenv("OV_GENAI_EXPORT_DFLASH_IR");
    return raw && std::string(raw) == "1";
}

// ---------------------------------------------------------------------------
// VL helpers
// ---------------------------------------------------------------------------

std::string build_vl_prompt(const std::string& user_prompt, int64_t image_tokens) {
    std::string prompt = "<|im_start|>user\n<|vision_start|>";
    prompt.reserve(prompt.size() + static_cast<size_t>(image_tokens) * 12 + user_prompt.size() + 64);
    for (int64_t i = 0; i < image_tokens; ++i)
        prompt += "<|image_pad|>";
    prompt += "<|vision_end|>";
    prompt += user_prompt;
    prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

std::string resolve_pos_embed_name(ov::genai::modeling::weights::WeightSource& source) {
    const std::vector<std::string> candidates = {
        "thinker.model.visual.pos_embed.weight",
        "model.visual.pos_embed.weight",
        "visual.pos_embed.weight",
        "pos_embed.weight",
    };
    for (const auto& name : candidates)
        if (source.has(name)) return name;
    for (const auto& name : source.keys())
        if (name.find("pos_embed.weight") != std::string::npos) return name;
    throw std::runtime_error("Failed to locate visual pos_embed.weight in safetensors");
}

ov::Tensor build_vision_attention_mask(const ov::Tensor& grid_thw) {
    if (grid_thw.get_element_type() != ov::element::i64 || grid_thw.get_shape().size() != 2 ||
        grid_thw.get_shape()[1] != 3)
        throw std::runtime_error("grid_thw must be i64 tensor with shape [N,3]");

    const auto* g = grid_thw.data<const int64_t>();
    const size_t rows = grid_thw.get_shape()[0];
    std::vector<size_t> segments;
    size_t total_tokens = 0;
    for (size_t i = 0; i < rows; ++i) {
        const int64_t t = g[i * 3], h = g[i * 3 + 1], w = g[i * 3 + 2];
        const size_t hw = static_cast<size_t>(h * w);
        for (int64_t f = 0; f < t; ++f) {
            segments.push_back(hw);
            total_tokens += hw;
        }
    }
    ov::Tensor mask(ov::element::f32, {1, 1, total_tokens, total_tokens});
    auto* data = mask.data<float>();
    std::fill_n(data, mask.get_size(), -std::numeric_limits<float>::infinity());
    size_t start = 0;
    for (size_t len : segments) {
        for (size_t r = start; r < start + len; ++r)
            std::fill(data + r * total_tokens + start, data + r * total_tokens + start + len, 0.0f);
        start += len;
    }
    return mask;
}

}  // namespace


// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[]) try {
#ifdef _WIN32
    int wargc = 0;
    wchar_t** wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    std::vector<std::string> utf8_args(wargc);
    std::vector<char*> utf8_argv(wargc);
    for (int i = 0; i < wargc; ++i) {
        int len = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        utf8_args[i].resize(len - 1);
        WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, utf8_args[i].data(), len, nullptr, nullptr);
        utf8_argv[i] = utf8_args[i].data();
    }
    LocalFree(wargv);
    argc = wargc;
    argv = utf8_argv.data();
#endif

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <TARGET_MODEL_DIR> <DRAFT_MODEL_DIR> [PROMPT] [DEVICE] [MAX_NEW_TOKENS]"
                  << " [BLOCK_SIZE] [TARGET_QUANT] [DRAFT_QUANT] [IMAGE_PATH]\n"
                  << "\n"
                  << "  TARGET_MODEL_DIR  Path to Qwen3-Omni-4B-Instruct-multilingual HF model\n"
                  << "  DRAFT_MODEL_DIR   Path to DFlash draft model (Qwen3-4B-DFlash-b16)\n"
                  << "  PROMPT            Input prompt (default: 'Tell me a short story about a robot.')\n"
                  << "  DEVICE            OpenVINO device (default: CPU)\n"
                  << "  MAX_NEW_TOKENS    Max tokens to generate (default: 500)\n"
                  << "  BLOCK_SIZE        DFlash block size override (default: from config)\n"
                  << "  TARGET_QUANT      Target quantization: FP16, INT4_ASYM, INT4_SYM (default: FP16)\n"
                  << "  DRAFT_QUANT       Draft quantization: FP16, INT4_ASYM, INT4_SYM (default: FP16)\n"
                  << "  IMAGE_PATH        Path to image file for VL mode (optional)\n";
        return 1;
    }

    const std::filesystem::path target_dir = argv[1];
    const std::filesystem::path draft_dir = argv[2];
    const std::string prompt = (argc > 3) ? argv[3] : "Tell me a short story about a robot.";
    const std::string device = (argc > 4) ? argv[4] : "CPU";
    const int max_new_tokens = (argc > 5) ? std::stoi(argv[5]) : 500;
    const int block_size_arg = (argc > 6) ? std::stoi(argv[6]) : 0;
    const std::string target_quant_arg = (argc > 7) ? argv[7] : "";
    const std::string draft_quant_arg = (argc > 8) ? argv[8] : "";
    const std::string image_path_arg = (argc > 9) ? argv[9] : "";

    const bool vl_mode = !image_path_arg.empty();

    std::cout << "[Qwen3-Omni DFlash Sample" << (vl_mode ? " VL" : "") << "]" << std::endl;
    std::cout << "  Target model: " << target_dir << std::endl;
    std::cout << "  Draft model:  " << draft_dir << std::endl;
    if (vl_mode) std::cout << "  Image:        " << image_path_arg << std::endl;
    std::cout << "  Device:       " << device << std::endl;
    std::cout << "  Max tokens:   " << max_new_tokens << std::endl;

    // ── Load configs ─────────────────────────────────────────────────────
    auto omni_cfg = ov::genai::modeling::models::Qwen3OmniConfig::from_json_file(target_dir);
    auto vl_cfg = ov::genai::modeling::models::to_qwen3_omni_vl_cfg(omni_cfg);
    auto draft_cfg_raw = ov::genai::loaders::ModelConfig::from_hf_json(draft_dir / "config.json");

    ov::genai::modeling::models::DFlashDraftConfig dflash_cfg;
    dflash_cfg.hidden_size = draft_cfg_raw.hidden_size;
    dflash_cfg.intermediate_size = draft_cfg_raw.intermediate_size;
    dflash_cfg.num_hidden_layers = draft_cfg_raw.num_hidden_layers;
    dflash_cfg.num_target_layers = (draft_cfg_raw.num_target_layers > 0)
                                       ? draft_cfg_raw.num_target_layers
                                       : omni_cfg.text.num_hidden_layers;
    dflash_cfg.num_attention_heads = draft_cfg_raw.num_attention_heads;
    dflash_cfg.num_key_value_heads = draft_cfg_raw.num_key_value_heads > 0
                                         ? draft_cfg_raw.num_key_value_heads
                                         : draft_cfg_raw.num_attention_heads;
    dflash_cfg.head_dim = draft_cfg_raw.head_dim > 0
                              ? draft_cfg_raw.head_dim
                              : (draft_cfg_raw.hidden_size / draft_cfg_raw.num_attention_heads);
    dflash_cfg.block_size = block_size_arg > 0 ? block_size_arg : draft_cfg_raw.block_size;
    dflash_cfg.rms_norm_eps = draft_cfg_raw.rms_norm_eps;
    dflash_cfg.rope_theta = draft_cfg_raw.rope_theta;
    dflash_cfg.hidden_act = draft_cfg_raw.hidden_act;
    dflash_cfg.attention_bias = draft_cfg_raw.attention_bias;

    if (dflash_cfg.block_size <= 0) dflash_cfg.block_size = 16;
    if (dflash_cfg.block_size < 2) throw std::runtime_error("block_size must be >= 2");

    const auto target_layer_ids = ov::genai::modeling::models::build_target_layer_ids(
        dflash_cfg.num_target_layers, dflash_cfg.num_hidden_layers);

    std::cout << "  Block size:   " << dflash_cfg.block_size << std::endl;
    std::cout << "  Target layers: [";
    for (size_t i = 0; i < target_layer_ids.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << target_layer_ids[i];
    }
    std::cout << "]" << std::endl;

    // ── Quantization ─────────────────────────────────────────────────────
    ov::genai::modeling::weights::QuantizationConfig target_quant_config;
    if (!target_quant_arg.empty() && target_quant_arg != "FP16") {
        target_quant_config.mode = parse_quant_mode(target_quant_arg);
        target_quant_config.group_size = 128;
        target_quant_config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_ASYM;
    }
    std::cout << "[quant] target: " << (target_quant_config.enabled() ? quant_mode_name(target_quant_config.mode) : "FP16") << std::endl;

    ov::genai::modeling::weights::QuantizationConfig draft_quant_config;
    if (!draft_quant_arg.empty() && draft_quant_arg != "FP16") {
        draft_quant_config.mode = parse_quant_mode(draft_quant_arg);
        draft_quant_config.group_size = 128;
        draft_quant_config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_ASYM;
    }
    std::cout << "[quant] draft:  " << (draft_quant_config.enabled() ? quant_mode_name(draft_quant_config.mode) : "FP16") << std::endl;

    // ── IR cache / export ────────────────────────────────────────────────
    const bool export_ir = is_export_ir_mode();
    const std::filesystem::path ir_dir = draft_dir;
    const std::string target_ir_stem = std::string("dflash_omni_target") + (vl_mode ? "_vl" : "") + quant_ir_suffix(target_quant_config);
    const std::string draft_ir_stem = std::string("dflash_draft_combined") + quant_ir_suffix(draft_quant_config);
    const auto target_xml = ir_dir / (target_ir_stem + ".xml");
    const auto target_bin = ir_dir / (target_ir_stem + ".bin");
    const auto draft_xml = ir_dir / (draft_ir_stem + ".xml");
    const auto draft_bin = ir_dir / (draft_ir_stem + ".bin");

    const bool load_target_from_ir = !export_ir && has_ir_model_pair(target_xml, target_bin);
    const bool load_draft_from_ir = !export_ir && has_ir_model_pair(draft_xml, draft_bin);

    // ── Build models ─────────────────────────────────────────────────────
    PerfStats perf;
    std::shared_ptr<ov::Model> target_model;
    std::shared_ptr<ov::Model> combined_draft_model;
    std::shared_ptr<ov::Model> vision_model;
    ov::Tensor vl_pos_embed_weight;

    ov::Core core;

    if (load_target_from_ir) {
        std::cout << "[cache-ir] Loading target model from IR: " << target_xml << std::endl;
        target_model = core.read_model(target_xml.string(), target_bin.string());
    }
    if (load_draft_from_ir) {
        std::cout << "[cache-ir] Loading draft model from IR: " << draft_xml << std::endl;
        combined_draft_model = core.read_model(draft_xml.string(), draft_bin.string());
    }

    if (!target_model || !combined_draft_model || (vl_mode && !vision_model)) {
        // Load safetensors weights
        auto target_data = ov::genai::safetensors::load_safetensors(target_dir);
        ov::genai::safetensors::SafetensorsWeightSource target_source(std::move(target_data));
        ov::genai::safetensors::SafetensorsWeightFinalizer target_finalizer(
            target_quant_config.enabled() ? target_quant_config
                                          : ov::genai::modeling::weights::QuantizationConfig{});

        auto draft_data = ov::genai::safetensors::load_safetensors(draft_dir);
        ov::genai::safetensors::SafetensorsWeightSource draft_source(std::move(draft_data));
        ov::genai::safetensors::SafetensorsWeightFinalizer draft_finalizer(
            draft_quant_config.enabled() ? draft_quant_config
                                         : ov::genai::modeling::weights::QuantizationConfig{});

        // Build target model: Qwen3-Omni thinker with DFlash target_hidden output.
        // Uses create_qwen3_omni_dflash_target_model which captures hidden states
        // at the selected target layers and concatenates them for draft model context.

        if (!target_model) {
            std::cout << "[Building Qwen3-Omni DFlash target model (thinker"
                      << (vl_mode ? " + vision" : "") << ")...]" << std::endl;

            // Build the DFlash variant that outputs both logits and target_hidden
            // (concatenated hidden states at selected layers for draft model context).
            target_model = ov::genai::modeling::models::create_qwen3_omni_dflash_target_model(
                omni_cfg,
                target_layer_ids,
                target_source,
                target_finalizer,
                vl_mode);  // enable_multimodal_inputs
        }

        if (vl_mode && !vision_model) {
            std::cout << "[Building Qwen3-Omni vision model...]" << std::endl;
            ov::genai::modeling::weights::QuantizationConfig no_quant;
            ov::genai::safetensors::SafetensorsWeightFinalizer vision_finalizer(no_quant);
            vision_model = ov::genai::modeling::models::create_qwen3_omni_vision_model(
                omni_cfg, target_source, vision_finalizer);

            const std::string pos_embed_name = resolve_pos_embed_name(target_source);
            vl_pos_embed_weight = target_source.get_tensor(pos_embed_name);
        }

        if (!combined_draft_model) {
            // Build combined draft model (embed + draft layers + lm_head).
            // The draft model uses target's embed_tokens and lm_head weights,
            // plus draft model's own layer weights.
            // Qwen3-Omni safetensors use "thinker." prefix for the text decoder,
            // so we create a PrefixMappedWeightSource.
            std::cout << "[Building combined draft model (embed+draft+lm_head)...]" << std::endl;

            // Create a prefix-mapped source so that "model.embed_tokens.weight"
            // maps to "thinker.model.embed_tokens.weight" in safetensors.
            ov::genai::modeling::models::PrefixMappedWeightSource thinker_source(target_source, "thinker.");

            combined_draft_model = ov::genai::modeling::models::create_qwen3_omni_dflash_combined_draft_model(
                omni_cfg, dflash_cfg, thinker_source, target_finalizer,
                draft_source, draft_finalizer);
        }

        // Export IR if requested
        if (export_ir) {
            if (target_model) {
                std::cout << "[export-ir] Serializing target model to " << target_xml << " ..." << std::endl;
                ov::serialize(target_model, target_xml.string(), target_bin.string());
            }
            std::cout << "[export-ir] Serializing draft model to " << draft_xml << " ..." << std::endl;
            ov::serialize(combined_draft_model, draft_xml.string(), draft_bin.string());
            if (vision_model) {
                const auto vision_xml = ir_dir / "dflash_omni_vision.xml";
                const auto vision_bin = ir_dir / "dflash_omni_vision.bin";
                ov::serialize(vision_model, vision_xml.string(), vision_bin.string());
            }
            std::cout << "[export-ir] Export complete. Exiting." << std::endl;
            return 0;
        }
    }

    // ── Compile models ───────────────────────────────────────────────────
    ov::AnyMap compile_cfg = {
        {ov::hint::inference_precision.name(), ov::element::f16},
        {ov::hint::kv_cache_precision.name(), ov::element::f16},
    };

    ov::genai::Tokenizer tokenizer(target_dir);
    const int64_t mask_token_id = (draft_cfg_raw.mask_token_id > 0)
                                     ? draft_cfg_raw.mask_token_id
                                     : resolve_mask_token_id(tokenizer);
    std::cerr << "[DFlash] mask_token_id=" << mask_token_id << std::endl;
    const int64_t eos_token_id = tokenizer.get_eos_token_id();

    // ── VL preprocessing ─────────────────────────────────────────────────
    ov::Tensor visual_embeds_padded;
    ov::Tensor visual_pos_mask_tensor;
    ov::Tensor vl_position_ids;
    ov::Tensor vl_input_ids;
    ov::Tensor vl_rope_deltas;
    ov::CompiledModel compiled_vision;
    std::vector<ov::Tensor> deepstack_padded;

    if (vl_mode) {
        std::cout << "[VL] Loading and preprocessing image: " << image_path_arg << std::endl;

        ov::genai::modeling::models::Qwen3OmniVisionPreprocessConfig pre_cfg;
        const auto pre_cfg_path = target_dir / "preprocessor_config.json";
        if (std::filesystem::exists(pre_cfg_path))
            pre_cfg = ov::genai::modeling::models::Qwen3OmniVisionPreprocessConfig::from_json_file(pre_cfg_path);

        compiled_vision = core.compile_model(vision_model, device, compile_cfg);
        auto vision_request = compiled_vision.create_infer_request();

        auto image = utils::load_image(image_path_arg);

        ov::genai::modeling::models::Qwen3OmniVisionPreprocessor preprocessor(omni_cfg, pre_cfg);
        auto vision_inputs = preprocessor.preprocess(image, vl_pos_embed_weight);

        vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kPixelValues, vision_inputs.pixel_values);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kGridThw, vision_inputs.grid_thw);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kPosEmbeds, vision_inputs.pos_embeds);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kRotaryCos, vision_inputs.rotary_cos);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kRotarySin, vision_inputs.rotary_sin);
        vision_request.set_tensor("attention_mask", build_vision_attention_mask(vision_inputs.grid_thw));

        std::cout << "[VL] Running vision encoder..." << std::endl;
        vision_request.infer();

        auto visual_embeds_raw = vision_request.get_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kVisualEmbeds);

        // Deepstack embeds
        std::vector<ov::Tensor> deepstack_embeds;
        for (size_t i = 0; i < vl_cfg.vision.deepstack_visual_indexes.size(); ++i) {
            const std::string name =
                std::string(ov::genai::modeling::models::Qwen3VLVisionIO::kDeepstackEmbedsPrefix) + "." +
                std::to_string(i);
            deepstack_embeds.push_back(vision_request.get_tensor(name));
        }

        const int64_t image_tokens =
            ov::genai::modeling::models::Qwen3OmniVisionPreprocessor::count_visual_tokens(
                vision_inputs.grid_thw, vl_cfg.vision.spatial_merge_size);
        std::cout << "[VL] Image tokens: " << image_tokens << std::endl;

        std::string vl_prompt = build_vl_prompt(prompt, image_tokens);
        auto vl_encoded = tokenizer.encode(vl_prompt, {ov::genai::add_special_tokens(false)});

        ov::genai::modeling::models::Qwen3VLInputPlanner planner(vl_cfg);
        auto plan = planner.build_plan(vl_encoded.input_ids, &vl_encoded.attention_mask,
                                       &vision_inputs.grid_thw);

        visual_embeds_padded = ov::genai::modeling::models::Qwen3VLInputPlanner::scatter_visual_embeds(
            visual_embeds_raw, plan.visual_pos_mask);
        visual_pos_mask_tensor = plan.visual_pos_mask;
        vl_position_ids = plan.position_ids;
        vl_input_ids = vl_encoded.input_ids;
        vl_rope_deltas = plan.rope_deltas;

        deepstack_padded = ov::genai::modeling::models::Qwen3VLInputPlanner::scatter_deepstack_embeds(
            deepstack_embeds, plan.visual_pos_mask);

        std::cout << "[VL] Vision preprocessing complete." << std::endl;
    }

    // ── Tokenize prompt ──────────────────────────────────────────────────
    ov::Tensor prompt_input_ids;

    bool enable_thinking = true;
    {
        const char* raw = std::getenv("OV_GENAI_DISABLE_THINKING");
        if (raw && std::string(raw) == "1") {
            enable_thinking = false;
            std::cout << "[DFlash] Thinking mode DISABLED" << std::endl;
        }
    }

    if (vl_mode) {
        prompt_input_ids = vl_input_ids;
    } else {
        std::string formatted_prompt = prompt;
        bool add_special_tokens = true;
        try {
            if (!tokenizer.get_chat_template().empty()) {
                ov::genai::ChatHistory history({{{"role", "user"}, {"content", prompt}}});
                ov::genai::JsonContainer extra({{"enable_thinking", enable_thinking}});
                formatted_prompt = tokenizer.apply_chat_template(
                    history, true, {}, std::nullopt, extra);
                add_special_tokens = false;
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: chat template apply failed: " << e.what() << ", using raw prompt" << std::endl;
        }
        auto encoded = tokenizer.encode(formatted_prompt, {ov::genai::add_special_tokens(add_special_tokens)});
        prompt_input_ids = encoded.input_ids;
    }

    const std::vector<int64_t> prompt_ids = tensor_to_ids(prompt_input_ids);
    std::vector<int64_t> output_ids = prompt_ids;
    const size_t prompt_len = output_ids.size();
    const size_t max_length = prompt_len + static_cast<size_t>(max_new_tokens);

    std::cout << "[INFO] prompt_len=" << prompt_len << " max_length=" << max_length << std::endl;

    // ── Compile ──────────────────────────────────────────────────────────
    std::cout << "[Compiling models on " << device << "...]" << std::endl;
    std::cout << "[Compiling target model (Qwen3-Omni thinker)...]" << std::endl;
    auto compiled_target = core.compile_model(target_model, device, compile_cfg);
    // Draft model: use f32 precision to avoid BF16/F16 instabilities in the
    // large fc matmul (ctx_dim=12800).  The graph is already f32 but
    // inference_precision=f16 can cause intermediates to overflow.
    ov::AnyMap draft_compile_cfg = compile_cfg;
    {
        const char* draft_prec_env = std::getenv("OV_GENAI_DRAFT_PRECISION");
        if (draft_prec_env && std::string(draft_prec_env) == "f16") {
            std::cout << "[Compiling combined draft model with INFERENCE_PRECISION=f16...]" << std::endl;
        } else {
            draft_compile_cfg[ov::hint::inference_precision.name()] = ov::element::f32;
            std::cout << "[Compiling combined draft model with INFERENCE_PRECISION=f32...]" << std::endl;
        }
    }
    auto compiled_draft = core.compile_model(combined_draft_model, device, draft_compile_cfg);
    std::cout << "[All models compiled.]" << std::endl;

    auto target_request = compiled_target.create_infer_request();
    auto draft_request = compiled_draft.create_infer_request();

    // Free model graphs
    target_model.reset();
    combined_draft_model.reset();
    if (vision_model) vision_model.reset();

    target_request.reset_state();
    ov::genai::utils::KVCacheState target_kv_state;
    const auto kv_pos = ov::genai::utils::get_kv_axes_pos(compiled_target.get_runtime_model());
    target_kv_state.seq_length_axis = kv_pos.seq_len;
    const auto beam_idx = make_beam_idx(1);

    // ── Zero tensors for decode steps (multimodal) ───────────────────────
    const size_t hidden_size = static_cast<size_t>(omni_cfg.text.hidden_size);
    ov::Tensor zero_visual_embeds;
    ov::Tensor zero_visual_pos_mask;
    ov::Tensor zero_audio_features;
    ov::Tensor zero_audio_pos_mask;
    std::vector<ov::Tensor> zero_deepstack;

    auto make_decode_zero_tensors = [&](size_t seq_len) {
        zero_visual_embeds = make_zero_tensor(ov::element::f32, {1, seq_len, hidden_size});
        zero_visual_pos_mask = make_zero_tensor(ov::element::boolean, {1, seq_len});
        zero_audio_features = make_zero_tensor(ov::element::f32, {1, seq_len, hidden_size});
        zero_audio_pos_mask = make_zero_tensor(ov::element::boolean, {1, seq_len});
        zero_deepstack.clear();
        for (size_t i = 0; i < vl_cfg.vision.deepstack_visual_indexes.size(); ++i) {
            zero_deepstack.push_back(make_zero_tensor(ov::element::f32, {1, seq_len, hidden_size}));
        }
    };

    // Helper to set multimodal inputs on target request
    auto set_target_multimodal_inputs = [&](ov::InferRequest& req,
                                            const ov::Tensor& vis_embeds,
                                            const ov::Tensor& vis_mask,
                                            const ov::Tensor& aud_feat,
                                            const ov::Tensor& aud_mask,
                                            const std::vector<ov::Tensor>& ds_embeds) {
        if (vl_mode) {
            req.set_tensor(ov::genai::modeling::models::Qwen3OmniTextIO::kVisualEmbeds, vis_embeds);
            req.set_tensor(ov::genai::modeling::models::Qwen3OmniTextIO::kVisualPosMask, vis_mask);
            req.set_tensor(ov::genai::modeling::models::Qwen3OmniTextIO::kAudioFeatures, aud_feat);
            req.set_tensor(ov::genai::modeling::models::Qwen3OmniTextIO::kAudioPosMask, aud_mask);
            for (size_t i = 0; i < ds_embeds.size(); ++i) {
                const std::string name =
                    std::string(ov::genai::modeling::models::Qwen3OmniTextIO::kDeepstackEmbedsPrefix) +
                    "." + std::to_string(i);
                req.set_tensor(name, ds_embeds[i]);
            }
        }
    };

    // ── Prefill ──────────────────────────────────────────────────────────
    target_request.get_tensor("attention_mask").set_shape({1, 0});

    auto prefill_start = Clock::now();
    target_request.set_tensor("input_ids", make_ids_tensor(output_ids));
    target_request.set_tensor("attention_mask", make_attention_mask(output_ids.size()));
    if (vl_mode) {
        target_request.set_tensor("position_ids", vl_position_ids);
        set_target_multimodal_inputs(target_request,
                                     visual_embeds_padded, visual_pos_mask_tensor,
                                     make_zero_tensor(ov::element::f32, {1, prompt_len, hidden_size}),
                                     make_zero_tensor(ov::element::boolean, {1, prompt_len}),
                                     deepstack_padded);
    } else {
        target_request.set_tensor("position_ids", make_mrope_position_ids(0, output_ids.size()));
    }
    target_request.set_tensor("beam_idx", beam_idx);
    target_request.infer();
    auto logits = target_request.get_tensor("logits");

    // The DFlash target model outputs both "logits" and "target_hidden"
    // (concatenated hidden states at selected target layers).
    // Verify the target_hidden output is present.
    bool has_target_hidden = false;
    for (const auto& out : compiled_target.outputs()) {
        for (const auto& name : out.get_names()) {
            if (name == "target_hidden") {
                has_target_hidden = true;
                break;
            }
        }
        if (has_target_hidden) break;
    }
    if (!has_target_hidden) {
        std::cerr << "[WARNING] Target model does not have target_hidden output. "
                  << "Draft model will operate with zero context (low acceptance rate)." << std::endl;
    }

    target_kv_state.add_inputs(make_ids_tensor(output_ids));

    // Create target_hidden storage for accumulating hidden states across steps.
    const size_t ctx_dim = static_cast<size_t>(dflash_cfg.hidden_size) *
                           static_cast<size_t>(dflash_cfg.num_hidden_layers);
    ov::Tensor target_hidden_storage(ov::element::f32,
                                     {1, max_length + static_cast<size_t>(dflash_cfg.block_size), ctx_dim});
    std::memset(target_hidden_storage.data(), 0, target_hidden_storage.get_byte_size());
    size_t target_hidden_len = prompt_len;

    if (has_target_hidden) {
        ov::Tensor target_hidden_block = ensure_f32_copy(target_request.get_tensor("target_hidden"));
        ov::Tensor target_hidden_init(target_hidden_storage, {0, 0, 0}, {1, prompt_len, ctx_dim});
        target_hidden_block.copy_to(target_hidden_init);
    }

    int64_t next_token = argmax_last_token(logits);
    auto prefill_end = Clock::now();
    perf.ttft_ms = duration_ms(prefill_start, prefill_end);
    output_ids.push_back(next_token);

    if (next_token == eos_token_id) {
        std::cout << "\n[Early Stop] EOS after prefill" << std::endl;
    }

    perf.prefill_wall.add(duration_ms(prefill_start, prefill_end));

    std::cout << "\n[Generating...]" << std::flush;

    const auto generation_start = Clock::now();

    // Prepare decode-step zero tensors
    make_decode_zero_tensors(1);

    // mRoPE rope_deltas for decode (VL mode)
    int64_t past_len = static_cast<int64_t>(prompt_len);

    bool stopped_by_eos = false;
    while (output_ids.size() < max_length) {
        if (next_token == eos_token_id) {
            stopped_by_eos = true;
            std::cout << "\n[Early Stop] EOS at position " << output_ids.size() << std::endl;
            break;
        }
        const auto step_start = Clock::now();
        double step_tracked_ms = 0.0;

        // ── Draft ────────────────────────────────────────────────────────
        std::vector<int64_t> block_ids(static_cast<size_t>(dflash_cfg.block_size), mask_token_id);
        block_ids[0] = output_ids.back();

        auto draft_start = Clock::now();

        // Draft position_ids (2D): [0..T-1, T..T+B-1] — contiguous context + draft
        // Must be contiguous to match the Python DFlash benchmark behaviour.
        const size_t total_pos = target_hidden_len + block_ids.size();
        ov::Tensor draft_pos(ov::element::i64, {1, total_pos});
        {
            auto* pd = draft_pos.data<int64_t>();
            for (size_t i = 0; i < total_pos; ++i)
                pd[i] = static_cast<int64_t>(i);
        }

        ov::Tensor hidden_view(target_hidden_storage, {0, 0, 0}, {1, target_hidden_len, ctx_dim});
        draft_request.set_tensor("target_hidden", hidden_view);
        draft_request.set_tensor("input_ids", make_ids_tensor(block_ids));
        draft_request.set_tensor("position_ids", draft_pos);
        draft_request.infer();

        auto draft_logits = draft_request.get_tensor("logits");
        const size_t draft_len = block_ids.size() - 1;
        auto draft_tokens = argmax_logits_slice(draft_logits, 1, draft_len);

        auto draft_end = Clock::now();
        perf.draft_wall.add(duration_ms(draft_start, draft_end));
        step_tracked_ms += duration_ms(draft_start, draft_end);

        // ── Verify ───────────────────────────────────────────────────────
        std::vector<int64_t> block_output_ids;
        block_output_ids.reserve(block_ids.size());
        block_output_ids.push_back(output_ids.back());
        block_output_ids.insert(block_output_ids.end(), draft_tokens.begin(), draft_tokens.end());

        const size_t verify_len = block_output_ids.size();

        auto verify_start = Clock::now();
        target_request.set_tensor("input_ids", make_ids_tensor(block_output_ids));
        target_request.set_tensor("attention_mask", make_attention_mask(target_hidden_len + verify_len));
        if (vl_mode) {
            auto decode_pos = ov::genai::modeling::models::Qwen3VLInputPlanner::build_decode_position_ids(
                vl_rope_deltas, past_len, static_cast<int64_t>(verify_len));
            target_request.set_tensor("position_ids", decode_pos);
            make_decode_zero_tensors(verify_len);
            set_target_multimodal_inputs(target_request,
                                         zero_visual_embeds, zero_visual_pos_mask,
                                         zero_audio_features, zero_audio_pos_mask,
                                         zero_deepstack);
        } else {
            target_request.set_tensor("position_ids",
                                      make_mrope_position_ids(target_hidden_len, verify_len));
        }
        target_request.set_tensor("beam_idx", beam_idx);
        target_request.infer();
        auto verify_end = Clock::now();
        perf.verify_wall.add(duration_ms(verify_start, verify_end));
        step_tracked_ms += duration_ms(verify_start, verify_end);

        logits = target_request.get_tensor("logits");
        auto posterior_tokens = argmax_logits_slice(logits, 0, verify_len);

        // ── Accept/reject ────────────────────────────────────────────────
        size_t accepted = 0;
        for (size_t i = 0; i < draft_tokens.size(); ++i) {
            if (draft_tokens[i] == posterior_tokens[i]) ++accepted;
            else break;
        }

        int64_t posterior_next = posterior_tokens[accepted];
        const size_t num_accepted = accepted + 1;

        // Trim KV cache for rejected tokens
        const size_t tokens_to_trim = verify_len - num_accepted;
        if (tokens_to_trim > 0) {
            target_kv_state.num_tokens_to_trim = tokens_to_trim;
            ov::genai::utils::trim_kv_cache(target_request, target_kv_state, std::nullopt);
            target_kv_state.num_tokens_to_trim = 0;
        }

        // Update target_hidden if available
        if (has_target_hidden) {
            ov::Tensor target_hidden_block = ensure_f32_copy(target_request.get_tensor("target_hidden"));
            if (num_accepted > 0 &&
                target_hidden_len + num_accepted <= max_length + static_cast<size_t>(dflash_cfg.block_size)) {
                ov::Tensor src_slice(target_hidden_block, {0, 0, 0}, {1, num_accepted, ctx_dim});
                ov::Tensor dst_slice(target_hidden_storage,
                                     {0, target_hidden_len, 0},
                                     {1, target_hidden_len + num_accepted, ctx_dim});
                src_slice.copy_to(dst_slice);
            }
        }
        target_hidden_len += num_accepted;
        past_len += static_cast<int64_t>(num_accepted);

        auto postproc_start = Clock::now();

        for (size_t i = 0; i < accepted && output_ids.size() < max_length; ++i)
            output_ids.push_back(draft_tokens[i]);

        ++perf.draft_steps;
        perf.accepted_tokens += accepted;
        perf.accepted_per_step.push_back(accepted);

        // Print step info
        {
            std::vector<int64_t> step_toks;
            for (size_t i = 0; i < accepted; ++i) step_toks.push_back(draft_tokens[i]);
            step_toks.push_back(posterior_next);
            auto text = tokenizer.decode(step_toks, {ov::genai::skip_special_tokens(true)});
            std::cout << "[Step " << perf.draft_steps << "] accepted=" << accepted
                      << " ids=[";
            for (size_t i = 0; i < step_toks.size(); ++i) {
                if (i) std::cout << ",";
                std::cout << step_toks[i];
            }
            std::cout << "] [" << text << "]" << std::endl;
        }
        perf.postproc_wall.add(duration_ms(postproc_start, Clock::now()));

        if (output_ids.size() >= max_length) break;

        next_token = posterior_next;
        output_ids.push_back(next_token);
        if (posterior_next == eos_token_id) {
            stopped_by_eos = true;
            std::cout << "\n[Early Stop] EOS in posterior at position " << output_ids.size() << std::endl;
            break;
        }

        const double step_ms = duration_ms(step_start, Clock::now());
        perf.other_wall.add(std::max(0.0, step_ms - step_tracked_ms));
    }

    const auto generation_end = Clock::now();
    perf.total_generate_ms = duration_ms(generation_start, generation_end);
    perf.generated_tokens = output_ids.size() - prompt_len;

    // ── Output ───────────────────────────────────────────────────────────
    std::vector<int64_t> generated_ids(output_ids.begin() + static_cast<ptrdiff_t>(prompt_len),
                                       output_ids.end());
    auto output_text = tokenizer.decode(generated_ids, {ov::genai::skip_special_tokens(true)});
    std::cout << "\n\n[Output]\n" << output_text << std::endl;
    std::cout << "\n[Generation Complete]" << std::endl;
    std::cout << "[Stop Reason] " << (stopped_by_eos ? "EOS token detected" : "Max length reached") << "\n" << std::endl;

    const double dflash_throughput = perf.total_generate_ms > 0
        ? (static_cast<double>(perf.generated_tokens) * 1000.0) / perf.total_generate_ms : 0.0;
    const double avg_accept = perf.draft_steps > 0
        ? static_cast<double>(perf.accepted_tokens) / static_cast<double>(perf.draft_steps) : 0.0;
    const double acceptance_rate = perf.generated_tokens > 0
        ? static_cast<double>(perf.accepted_tokens) / static_cast<double>(perf.generated_tokens) : 0.0;
    const size_t tokens_after_first = perf.generated_tokens > 0 ? perf.generated_tokens - 1 : 0;
    const double tpot_ms = tokens_after_first > 0
        ? perf.total_generate_ms / static_cast<double>(tokens_after_first) : 0.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Mode: dflash / " << (vl_mode ? "vl" : "text") << std::endl;
    std::cout << "Prompt token size: " << prompt_len << std::endl;
    std::cout << "Output token size: " << perf.generated_tokens << std::endl;
    std::cout << "TTFT: " << perf.ttft_ms << " ms" << std::endl;
    std::cout << "Decode time: " << perf.total_generate_ms << " ms" << std::endl;
    std::cout << "TPOT: " << tpot_ms << " ms/token" << std::endl;
    std::cout << "Throughput: " << dflash_throughput << " tokens/s" << std::endl;
    std::cout << "Draft steps: " << perf.draft_steps << std::endl;
    std::cout << "Accepted draft tokens: " << perf.accepted_tokens << std::endl;
    std::cout << "Acceptance rate: " << std::setprecision(4) << acceptance_rate << std::endl;
    std::cout << std::setprecision(2);
    std::cout << "Avg accepted per step: " << avg_accept << std::endl;

    std::cout << std::setprecision(3);
    std::cout << "[Stage timings] wall-clock (ms):" << std::endl;
    print_stage_stats("prefill target", perf.prefill_wall);
    print_stage_stats("draft (embed+draft+lm_head)", perf.draft_wall);
    print_stage_stats("target verify", perf.verify_wall);
    print_stage_stats("postproc", perf.postproc_wall);
    print_stage_stats("other (untracked)", perf.other_wall);
    if (!perf.accepted_per_step.empty()) {
        std::cout << "[Draft acceptance per step] [";
        for (size_t i = 0; i < perf.accepted_per_step.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << perf.accepted_per_step[i];
        }
        std::cout << "]" << std::endl;
    }

    return 0;
} catch (const std::exception& ex) {
    std::cerr << "Qwen3-Omni DFlash sample failed: " << ex.what() << std::endl;
    return 1;
}
