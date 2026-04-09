// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/openvino.hpp>

#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/generation_config.hpp"
#include "load_image.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_vision.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "modeling/models/qwen3_5/qwen3_5_weight_specs.hpp"
#include "modeling/weights/quantization_config.hpp"
#include "modeling/weights/synthetic_weight_source.hpp"
#include "sampling/logit_processor.hpp"

namespace {

struct SampleOptions {
    std::optional<std::filesystem::path> model_dir;
    std::string mode;
    std::filesystem::path image_path;
    std::string user_prompt;
    std::optional<std::filesystem::path> prompt_file;
    std::string device = "GPU";
    int max_new_tokens = 64;
    bool cache_model = false;

    std::string dummy_model = "dense";

    std::optional<int> num_layers;
    int max_pixels = 0;

    // Sampling parameters – defaults follow Qwen3.5 official recommendations
    // for "thinking mode, general tasks".
    float temperature = 1.0f;
    float top_p = 0.95f;
    size_t top_k = 20;
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 1.5f;
    size_t rng_seed = 0;  // 0 = use random_device
    bool enable_thinking = true;  // --think 0/1
    bool enable_mtp = false;  // --mtp 0/1: enable Multi-Token Prediction speculative decoding
    int mtp_num_layers = 1;   // --mtp-layers N: number of MTP decoder layers (default: 1)
    int mtp_k = 1;            // --mtp-k N: number of draft tokens per speculation step (default: 1)
    bool seq_verify = false;  // --seq-verify 0/1: use sequential single-token verify (avoids multi-token SDPA)
    bool pure_batch = false;  // --pure-batch 0/1: batch verify with KV trim only (no linear_states rollback)
    int refresh_interval = 0; // --refresh N: periodic state refresh every N tokens (0=disabled)
    bool adaptive_k = false;  // --adaptive-k 0/1: dynamically adjust K based on rolling accept rate
};

bool has_safetensors_file(const std::filesystem::path& model_dir) {
    if (!std::filesystem::exists(model_dir) || !std::filesystem::is_directory(model_dir)) {
        return false;
    }
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() == ".safetensors") {
            return true;
        }
    }
    return false;
}

bool has_ir_model_pair(const std::filesystem::path& xml_path, const std::filesystem::path& bin_path) {
    return std::filesystem::exists(xml_path) && std::filesystem::is_regular_file(xml_path) &&
           std::filesystem::exists(bin_path) && std::filesystem::is_regular_file(bin_path);
}

bool has_model_input_name(const std::shared_ptr<ov::Model>& model, const std::string& input_name) {
    if (!model) {
        return false;
    }
    for (const auto& input : model->inputs()) {
        const auto names = input.get_names();
        if (names.find(input_name) != names.end()) {
            return true;
        }
    }
    return false;
}

bool is_vl_text_ir_compatible(const std::shared_ptr<ov::Model>& model) {
    return has_model_input_name(model, ov::genai::modeling::models::Qwen3_5TextIO::kVisualEmbeds) &&
           has_model_input_name(model, ov::genai::modeling::models::Qwen3_5TextIO::kVisualPosMask);
}

int parse_i32(const std::string& raw, const char* option_name) {
    try {
        return std::stoi(raw);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid integer for ") + option_name + ": " + raw);
    }
}

float parse_float(const std::string& raw, const char* option_name) {
    try {
        return std::stof(raw);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid float for ") + option_name + ": " + raw);
    }
}

std::string to_lower(std::string value) {
    for (auto& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open prompt file: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void print_usage(const char* argv0) {
    std::cout
        << "Qwen3.5 Modeling Sample CLI Helper\n"
        << "==================================\n\n"
        << "Overview:\n"
        << "  - Run Qwen3.5 text-only or vision-language generation.\n"
        << "  - If --model is set: load real HF weights from that directory.\n"
        << "  - If --model is not set: use synthetic dummy weights.\n\n"
        << "Primary Usage:\n"
        << "  " << argv0 << " [--model PATH] [--mode text|vl] [--image IMAGE_PATH] [--prompt TEXT]\n\n"
        << "Examples:\n"
        << "  1) Dummy text:\n"
        << "     " << argv0 << " --mode text --prompt \"Hello\"\n"
        << "  2) Dummy VL (uses built-in 256x256 dummy image):\n"
        << "     " << argv0 << " --mode vl --dummy-model moe --prompt \"Describe this image\"\n"
        << "  3) Real HF text:\n"
        << "     " << argv0 << " --model C:\\models\\Qwen3.5 --mode text\n"
        << "  4) Real HF VL:\n"
        << "     " << argv0 << " --model C:\\models\\Qwen3.5 --mode vl --image C:\\img\\a.jpg\n\n"
        << "Options:\n"
        << "  --model PATH                    HF model directory (config.json + *.safetensors). Omit for dummy weights\n"
        << "  --mode text|vl                  Run path (default: text)\n"
        << "  --image PATH                    Image path for vl mode. Required with --model --mode vl\n"
        << "                                  In dummy vl mode, if omitted, built-in 256x256 dummy image is used\n"
        << "  --prompt TEXT                   User prompt\n"
        << "  --prompt-file PATH              Read prompt text from file\n"
        << "  --device NAME                   OpenVINO device name (default: GPU)\n"
        << "  --output-tokens N               Number of generated tokens (default: 64)\n"
        << "  --cache-model                   Enable model caching behavior.\n"
        << "                                  HF mode: load cached IR from --model dir if exists, else build+save there\n"
        << "                                  Dummy mode: save built IR to app executable folder\n"
        << "                                  IR filename includes concise quant signature when quantization is enabled\n"
        << "  Quantization for both real and dummy mode is controlled only by environment variables:\n"
        << "    OV_GENAI_INFLIGHT_QUANT_MODE\n"
        << "    OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE\n"
        << "    OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE\n"
        << "  --dummy-model MODE              Dummy mode only: dense | moe (default: dense)\n"
    << "  --num-layers N                  Run only the first N text transformer layers (dummy + real model)\n"
    << "  --max-pixels N                  Limit vision input to N pixels (default: from preprocessor_config.json)\n"
    << "                                  Recommended: 602112 (3072 tokens * 14^2 patch) for ARL-H GPU\n"
    << "  --temperature FLOAT             Sampling temperature (default: 1.0, 0 = greedy argmax)\n"
    << "  --top-p FLOAT                   Nucleus sampling threshold (default: 0.95)\n"
    << "  --top-k INT                     Top-K filtering (default: 20)\n"
    << "  --repetition-penalty FLOAT      Penalty for repeating tokens (default: 1.0)\n"
    << "  --frequency-penalty FLOAT       Subtract penalty * token_count from logit (default: 0.0)\n"
    << "  --presence-penalty FLOAT        Subtract penalty if token appeared (default: 1.5)\n"
    << "  --rng-seed INT                  Random seed for sampling (default: 0 = random)\n"
    << "  --think 0|1                     Enable/disable thinking mode (default: 1 = enabled)\n"
    << "  --mtp 0|1                       Enable MTP (Multi-Token Prediction) speculative decoding (default: 0)\n"
    << "  --mtp-layers N                  Number of MTP decoder layers (default: 1)\n"
    << "  --mtp-k N                       Number of draft tokens per speculation step (default: 1)\n"
    << "  --seq-verify 0|1                Sequential single-token verify (default: 0)\n"
    << "  --pure-batch 0|1                Batch verify with KV trim only, no linear_states rollback (default: 0)\n"
    << "  --refresh N                     Periodic state refresh every N tokens in pure-batch mode (default: 0=off)\n"
        << "  -h, --help                      Show this helper\n";
}

std::string quant_mode_cache_token(ov::genai::modeling::weights::QuantizationConfig::Mode mode) {
    using Mode = ov::genai::modeling::weights::QuantizationConfig::Mode;
    switch (mode) {
        case Mode::INT4_SYM:
            return "4s";
        case Mode::INT4_ASYM:
            return "4a";
        case Mode::INT8_SYM:
            return "8s";
        case Mode::INT8_ASYM:
            return "8a";
        case Mode::NONE:
        default:
            return "n";
    }
}

std::string quant_cache_suffix(const ov::genai::modeling::weights::QuantizationConfig& cfg) {
    if (!cfg.enabled()) {
        return "";
    }
    return "_q" + quant_mode_cache_token(cfg.mode) + "_b" + quant_mode_cache_token(cfg.backup_mode) +
           "_g" + std::to_string(cfg.group_size);
}

SampleOptions parse_cli(int argc, char* argv[]) {
    SampleOptions opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg.rfind("--", 0) != 0) {
            throw std::runtime_error("Positional arguments are not supported. Use --help and pass named options.");
        }

        auto take_value = [&](const char* option_name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + option_name);
            }
            return argv[++i];
        };

        if (arg == "--model") {
            opts.model_dir = take_value("--model");
        } else if (arg == "--mode") {
            opts.mode = take_value("--mode");
        } else if (arg == "--image") {
            opts.image_path = take_value("--image");
        } else if (arg == "--prompt") {
            opts.user_prompt = take_value("--prompt");
        } else if (arg == "--prompt-file") {
            opts.prompt_file = take_value("--prompt-file");
        } else if (arg == "--device") {
            opts.device = take_value("--device");
        } else if (arg == "--output-tokens") {
            opts.max_new_tokens = parse_i32(take_value("--output-tokens"), "--output-tokens");
        } else if (arg == "--cache-model") {
            opts.cache_model = true;
        } else if (arg == "--dummy-model") {
            opts.dummy_model = take_value("--dummy-model");
        } else if (arg == "--num-layers") {
            opts.num_layers = parse_i32(take_value("--num-layers"), "--num-layers");
        } else if (arg == "--max-pixels") {
            opts.max_pixels = parse_i32(take_value("--max-pixels"), "--max-pixels");
        } else if (arg == "--temperature") {
            opts.temperature = parse_float(take_value("--temperature"), "--temperature");
        } else if (arg == "--top-p") {
            opts.top_p = parse_float(take_value("--top-p"), "--top-p");
        } else if (arg == "--top-k") {
            opts.top_k = static_cast<size_t>(parse_i32(take_value("--top-k"), "--top-k"));
        } else if (arg == "--repetition-penalty") {
            opts.repetition_penalty = parse_float(take_value("--repetition-penalty"), "--repetition-penalty");
        } else if (arg == "--frequency-penalty") {
            opts.frequency_penalty = parse_float(take_value("--frequency-penalty"), "--frequency-penalty");
        } else if (arg == "--presence-penalty") {
            opts.presence_penalty = parse_float(take_value("--presence-penalty"), "--presence-penalty");
        } else if (arg == "--rng-seed") {
            opts.rng_seed = static_cast<size_t>(parse_i32(take_value("--rng-seed"), "--rng-seed"));
        } else if (arg == "--think") {
            int val = parse_i32(take_value("--think"), "--think");
            opts.enable_thinking = (val != 0);
        } else if (arg == "--mtp") {
            int val = parse_i32(take_value("--mtp"), "--mtp");
            opts.enable_mtp = (val != 0);
        } else if (arg == "--mtp-layers") {
            opts.mtp_num_layers = parse_i32(take_value("--mtp-layers"), "--mtp-layers");
        } else if (arg == "--mtp-k") {
            opts.mtp_k = parse_i32(take_value("--mtp-k"), "--mtp-k");
        } else if (arg == "--seq-verify") {
            int val = parse_i32(take_value("--seq-verify"), "--seq-verify");
            opts.seq_verify = (val != 0);
        } else if (arg == "--pure-batch") {
            int val = parse_i32(take_value("--pure-batch"), "--pure-batch");
            opts.pure_batch = (val != 0);
        } else if (arg == "--refresh") {
            opts.refresh_interval = parse_i32(take_value("--refresh"), "--refresh");
        } else if (arg == "--adaptive-k") {
            int val = parse_i32(take_value("--adaptive-k"), "--adaptive-k");
            opts.adaptive_k = (val != 0);
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    if (opts.mode.empty()) {
        opts.mode = "text";
    }
    opts.mode = to_lower(opts.mode);
    if (opts.mode != "text" && opts.mode != "vl") {
        throw std::runtime_error("Mode must be 'text' or 'vl'");
    }
    opts.dummy_model = to_lower(opts.dummy_model);
    if (opts.dummy_model != "dense" && opts.dummy_model != "moe") {
        throw std::runtime_error("dummy-model must be 'dense' or 'moe'");
    }
    if (!opts.image_path.empty() && opts.mode != "vl") {
        throw std::runtime_error("--image can only be used with --mode vl");
    }

    if (opts.max_new_tokens <= 0) {
        throw std::runtime_error("max_new_tokens must be > 0");
    }
    const int prompt_sources = (!opts.user_prompt.empty() ? 1 : 0) + (opts.prompt_file.has_value() ? 1 : 0);
    if (prompt_sources > 1) {
        throw std::runtime_error("Use only one of --prompt or --prompt-file");
    }
    if (opts.prompt_file.has_value()) {
        opts.user_prompt = read_text_file(*opts.prompt_file);
    }
    if (opts.user_prompt.empty()) {
        opts.user_prompt = (opts.mode == "vl") ? "Describe the image." : "Write one sentence about OpenVINO.";
    }

    return opts;
}

void apply_text_config_overrides(ov::genai::modeling::models::Qwen3_5Config& cfg, const SampleOptions& opts) {
    if (!opts.num_layers.has_value()) {
        return;
    }
    if (*opts.num_layers <= 0) {
        throw std::runtime_error("--num-layers must be > 0");
    }

    if (*opts.num_layers > cfg.text.num_hidden_layers) {
        throw std::runtime_error("--num-layers must be <= model num_hidden_layers (" +
                                 std::to_string(cfg.text.num_hidden_layers) + "), got: " +
                                 std::to_string(*opts.num_layers));
    }

    cfg.text.num_hidden_layers = *opts.num_layers;
    if (!cfg.text.layer_types.empty()) {
        if (cfg.text.layer_types.size() >= static_cast<size_t>(*opts.num_layers)) {
            cfg.text.layer_types.resize(static_cast<size_t>(*opts.num_layers));
        } else {
            cfg.text.layer_types.clear();
        }
    }

    cfg.finalize();
    cfg.validate();
}

std::string build_vl_prompt(const std::string& user_prompt, int64_t image_tokens, bool enable_thinking = true) {
    std::string prompt;
    // When thinking is disabled, add a system message to suppress <think> generation.
    // This mirrors Qwen3.5's chat template behavior with enable_thinking=false.
    if (!enable_thinking) {
        prompt += "<|im_start|>system\n/no_think<|im_end|>\n";
    }
    prompt += "<|im_start|>user\n<|vision_start|>";
    prompt.reserve(prompt.size() + static_cast<size_t>(image_tokens) * 12 + user_prompt.size() + 64);
    for (int64_t i = 0; i < image_tokens; ++i) {
        prompt += "<|image_pad|>";
    }
    prompt += "<|vision_end|>\n";
    prompt += user_prompt;
    prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

std::set<int64_t> resolve_stop_token_ids(const std::filesystem::path& model_dir,
                                         const ov::genai::Tokenizer* tokenizer) {
    std::set<int64_t> stop_token_ids;

    if (!model_dir.empty()) {
        const auto generation_config_path = model_dir / "generation_config.json";
        if (std::filesystem::exists(generation_config_path) && std::filesystem::is_regular_file(generation_config_path)) {
            try {
                ov::genai::GenerationConfig generation_config(generation_config_path);
                stop_token_ids = generation_config.stop_token_ids;
            } catch (const std::exception& e) {
                std::cerr << "Failed to load generation_config.json: " << e.what() << std::endl;
                std::cerr << "Falling back to tokenizer eos_token_id..." << std::endl;
            }
        }
    }

    if (stop_token_ids.empty() && tokenizer) {
        const int64_t eos_token_id = tokenizer->get_eos_token_id();
        if (eos_token_id >= 0) {
            stop_token_ids.insert(eos_token_id);
        }
    }

    return stop_token_ids;
}


// Extract logits at a specific sequence position from [1, S, V] into a float32 scratch buffer.
void extract_logits_at_pos_f32(const ov::Tensor& logits, size_t pos, std::vector<float>& out) {
    const auto shape = logits.get_shape();
    const size_t vocab = shape[2];
    const size_t offset = pos * vocab;
    out.resize(vocab);
    if (logits.get_element_type() == ov::element::f32) {
        std::memcpy(out.data(), logits.data<const float>() + offset, vocab * sizeof(float));
    } else if (logits.get_element_type() == ov::element::f16) {
        const auto* src = logits.data<const ov::float16>() + offset;
        for (size_t i = 0; i < vocab; ++i)
            out[i] = static_cast<float>(src[i]);
    } else if (logits.get_element_type() == ov::element::bf16) {
        const auto* src = logits.data<const ov::bfloat16>() + offset;
        for (size_t i = 0; i < vocab; ++i)
            out[i] = static_cast<float>(src[i]);
    } else {
        throw std::runtime_error("Unsupported logits dtype for logit processing");
    }
}

// Trim the KV cache of an infer request by removing the last `num_tokens` entries.
// The KV cache states have shape [batch, num_kv_heads, seq_len, head_dim] with seq_length_axis=2.
void trim_kv_cache_states(ov::InferRequest& request, size_t num_tokens, size_t seq_length_axis = 2) {
    if (num_tokens == 0) return;
    for (auto& state : request.query_state()) {
        const auto& name = state.get_name();
        // Only trim attention KV states
        if (name.find("past_key_values.") == std::string::npos &&
            name.find(".key_cache") == std::string::npos &&
            name.find(".value_cache") == std::string::npos) {
            continue;
        }
        ov::Tensor old_tensor = state.get_state();
        auto shape = old_tensor.get_shape();
        if (seq_length_axis >= shape.size()) continue;
        const size_t old_seq_len = shape[seq_length_axis];
        const size_t trim = std::min<size_t>(old_seq_len, num_tokens);
        if (trim == 0) continue;
        shape[seq_length_axis] = old_seq_len - trim;
        ov::Coordinate begin(shape.size(), 0);
        ov::Coordinate end(shape.begin(), shape.end());
        auto trimmed = ov::Tensor(old_tensor, begin, end);
        ov::Tensor new_tensor(old_tensor.get_element_type(), shape);
        trimmed.copy_to(new_tensor);
        state.set_state(new_tensor);
    }
}

// GPU-optimized KV cache trim: zero-copy metadata-only operation.
// Uses the GPU plugin's trim_variable_state API to reduce the logical sequence
// length without any data movement.  Falls back to the CPU-based trim on error.
void trim_kv_cache_states_gpu(ov::InferRequest& request, size_t num_tokens, size_t seq_length_axis = 2) {
    if (num_tokens == 0) return;
    for (auto& state : request.query_state()) {
        const auto& name = state.get_name();
        if (name.find("past_key_values.") == std::string::npos &&
            name.find(".key_cache") == std::string::npos &&
            name.find(".value_cache") == std::string::npos) {
            continue;
        }
        try {
            request.trim_variable_state(name, num_tokens, seq_length_axis);
        } catch (...) {
            // Fallback to CPU-based trim for this state
            trim_kv_cache_states(request, num_tokens, seq_length_axis);
            return;
        }
    }
}

// Initialize snapshot buffers for linear_states (recurrent/linear attention layers).
// Returns pre-allocated tensors matching each linear_states tensor's shape.
std::vector<ov::Tensor> init_linear_state_snapshot(ov::InferRequest& request) {
    std::vector<ov::Tensor> snap;
    for (auto& state : request.query_state()) {
        if (state.get_name().find("linear_states.") == std::string::npos) continue;
        ov::Tensor t = state.get_state();
        snap.emplace_back(t.get_element_type(), t.get_shape());
    }
    return snap;
}

// Save current linear_states into pre-allocated snapshot buffers.
void save_linear_states(ov::InferRequest& request, std::vector<ov::Tensor>& snap) {
    size_t idx = 0;
    for (auto& state : request.query_state()) {
        if (state.get_name().find("linear_states.") == std::string::npos) continue;
        if (idx < snap.size()) {
            state.get_state().copy_to(snap[idx]);
            ++idx;
        }
    }
}

// Restore linear_states from snapshot buffers.
void restore_linear_states(ov::InferRequest& request,
                           const std::vector<ov::Tensor>& snap) {
    size_t idx = 0;
    for (auto& state : request.query_state()) {
        if (state.get_name().find("linear_states.") == std::string::npos) continue;
        if (idx < snap.size()) {
            state.set_state(snap[idx]);
            ++idx;
        }
    }
}

// Check if the model has per-token linear_states snapshot outputs.
// Returns the output names in order (empty if not available).
std::vector<std::string> find_all_linear_states_outputs(ov::InferRequest& request) {
    std::vector<std::string> names;
    auto model = request.get_compiled_model();
    for (size_t i = 0; i < model.outputs().size(); ++i) {
        auto out_names = model.output(i).get_names();
        for (const auto& n : out_names) {
            if (n.find("all_linear_states.layer") != std::string::npos) {
                names.push_back(n);
                break;
            }
        }
    }
    // Sort by layer index: "all_linear_states.layer17" -> 17
    std::sort(names.begin(), names.end(), [](const std::string& a, const std::string& b) {
        auto idx_a = std::stoi(a.substr(a.find("layer") + 5));
        auto idx_b = std::stoi(b.substr(b.find("layer") + 5));
        return idx_a < idx_b;
    });
    return names;
}

// Find all conv state snapshot outputs from the compiled model.
std::vector<std::string> find_all_conv_states_outputs(ov::InferRequest& request) {
    std::vector<std::string> names;
    auto model = request.get_compiled_model();
    for (size_t i = 0; i < model.outputs().size(); ++i) {
        auto out_names = model.output(i).get_names();
        for (const auto& n : out_names) {
            if (n.find("all_conv_states.layer") != std::string::npos) {
                names.push_back(n);
                break;
            }
        }
    }
    std::sort(names.begin(), names.end(), [](const std::string& a, const std::string& b) {
        auto idx_a = std::stoi(a.substr(a.find("layer") + 5));
        auto idx_b = std::stoi(b.substr(b.find("layer") + 5));
        return idx_a < idx_b;
    });
    return names;
}

// After batch verify with K+1 tokens, select conv states at the
// num_accepted-th token position from the per-token snapshot outputs.
// all_states_names: output names like "all_linear_states.layer0", ...
// Each output tensor has shape [B, T, num_v_heads, K_HEAD_DIMS, K_HEAD_DIMS].
// We select [:, num_accepted, :, :, :] and write it to the corresponding variable.
//
// If use_gpu_restore is true, uses the GPU-side restore_variable_from_output API
// to copy directly from the internal GPU output buffer to variable memory
// without any CPU round-trip.  Falls back to CPU path if the API throws.
void select_and_restore_linear_states(ov::InferRequest& request,
                                      const std::vector<std::string>& all_states_names,
                                      int num_accepted,
                                      bool use_gpu_restore = true) {
    // Build a map from layer_index -> output tensor name.
    // Output names: "all_linear_states.layer0", "all_linear_states.layer1", ...
    std::unordered_map<int, std::string> output_by_layer;
    for (const auto& name : all_states_names) {
        auto pos = name.find("layer");
        if (pos != std::string::npos) {
            int layer_idx = std::stoi(name.substr(pos + 5));
            output_by_layer[layer_idx] = name;
        }
    }

    for (auto& state : request.query_state()) {
        const auto& sname = state.get_name();
        // Match "linear_states.{N}.recurrent"
        if (sname.find("linear_states.") == std::string::npos) continue;
        if (sname.find(".recurrent") == std::string::npos) continue;

        // Extract layer index from "linear_states.17.recurrent" -> 17
        auto dot1 = sname.find('.') + 1;  // after first '.'
        auto dot2 = sname.find('.', dot1);
        int layer_idx = std::stoi(sname.substr(dot1, dot2 - dot1));

        auto it = output_by_layer.find(layer_idx);
        if (it == output_by_layer.end()) continue;

        if (use_gpu_restore) {
            // GPU-side: direct GPU-to-GPU copy from internal output memory to variable memory.
            // No get_tensor (GPU->CPU) + memcpy + set_state (CPU->GPU) round-trip.
            try {
                request.restore_variable_from_output(sname, it->second, static_cast<size_t>(num_accepted));
                continue;  // success — skip CPU fallback
            } catch (const std::exception&) {
                // Fall through to CPU path
            } catch (...) {
                // Fall through to CPU path
            }
        }

        // CPU fallback path
        ov::Tensor all_states = request.get_tensor(it->second);
        const auto shape = all_states.get_shape();
        // shape = [B, T, H_v, K_HEAD_DIMS, K_HEAD_DIMS]
        const size_t B = shape[0];
        const size_t H = shape[2];
        const size_t K_dim = shape[3];
        const size_t V_dim = shape[4];
        const size_t state_size = H * K_dim * V_dim;
        const size_t token_stride = state_size;

        ov::Tensor target(all_states.get_element_type(), {B, H, K_dim, V_dim});
        const size_t elem_size = all_states.get_element_type().size();
        const auto* src = reinterpret_cast<const uint8_t*>(all_states.data());
        auto* dst = reinterpret_cast<uint8_t*>(target.data());

        for (size_t b = 0; b < B; ++b) {
            const size_t src_off = (b * shape[1] + static_cast<size_t>(num_accepted)) * token_stride * elem_size;
            const size_t dst_off = b * state_size * elem_size;
            std::memcpy(dst + dst_off, src + src_off, state_size * elem_size);
        }

        state.set_state(target);
    }
}

// After batch verify, select conv states at num_accepted-th token position
// from the per-token conv snapshot outputs.
// all_conv_states_names: output names like "all_conv_states.layer0", ...
// Each output tensor has shape [B, T, conv_dim, kernel_size].
// We select [:, num_accepted, :, :] and write it to the corresponding ".conv" variable.
void select_and_restore_conv_states(ov::InferRequest& request,
                                    const std::vector<std::string>& all_conv_states_names,
                                    int num_accepted,
                                    bool use_gpu_restore = true) {
    std::unordered_map<int, std::string> output_by_layer;
    for (const auto& name : all_conv_states_names) {
        auto pos = name.find("layer");
        if (pos != std::string::npos) {
            int layer_idx = std::stoi(name.substr(pos + 5));
            output_by_layer[layer_idx] = name;
        }
    }

    static int conv_dbg_count = 0;

    for (auto& state : request.query_state()) {
        const auto& sname = state.get_name();
        // Match "linear_states.{N}.conv"
        if (sname.find("linear_states.") == std::string::npos) continue;
        if (sname.find(".conv") == std::string::npos) continue;

        auto dot1 = sname.find('.') + 1;
        auto dot2 = sname.find('.', dot1);
        int layer_idx = std::stoi(sname.substr(dot1, dot2 - dot1));

        auto it = output_by_layer.find(layer_idx);
        if (it == output_by_layer.end()) continue;

        // Debug: dump snapshot data before restore
        if (conv_dbg_count < 2) {
            ov::Tensor snap_tensor = request.get_tensor(it->second);
            const auto snap_shape = snap_tensor.get_shape();
            // snap_shape = [B, T, conv_dim, kernel_size]
            size_t T = snap_shape[1];
            size_t cD = snap_shape[2];
            size_t kS = snap_shape[3];

            // Get current variable state for comparison
            ov::Tensor cur_state = state.get_state();

            fprintf(stderr, "[CONV_SNAP_DBG] %s: snap_shape=[%zu,%zu,%zu,%zu] var_shape=[",
                    sname.c_str(), snap_shape[0], snap_shape[1], snap_shape[2], snap_shape[3]);
            auto vs = cur_state.get_shape();
            for (size_t i = 0; i < vs.size(); i++) fprintf(stderr, "%s%zu", i?",":"", vs[i]);
            fprintf(stderr, "] snap_et=%s var_et=%s num_accepted=%d\n",
                    snap_tensor.get_element_type().to_string().c_str(),
                    cur_state.get_element_type().to_string().c_str(),
                    num_accepted);

            // Print first 8 values of snapshot at position num_accepted
            if (snap_tensor.get_element_type() == ov::element::f32) {
                auto* p = snap_tensor.data<float>();
                size_t slice_off = static_cast<size_t>(num_accepted) * cD * kS;
                fprintf(stderr, "[CONV_SNAP_DBG] snap[%d] first 8 f32: ", num_accepted);
                for (int i = 0; i < 8 && i < (int)(cD * kS); i++)
                    fprintf(stderr, "%.4f ", p[slice_off + i]);
                fprintf(stderr, "\n");
                // Also print last row (last channel)
                fprintf(stderr, "[CONV_SNAP_DBG] snap[%d] last ch f32: ", num_accepted);
                size_t last_ch_off = slice_off + (cD - 1) * kS;
                for (size_t i = 0; i < kS; i++)
                    fprintf(stderr, "%.4f ", p[last_ch_off + i]);
                fprintf(stderr, "\n");
            } else if (snap_tensor.get_element_type() == ov::element::f16) {
                auto* p = reinterpret_cast<const ov::float16*>(snap_tensor.data());
                size_t slice_off = static_cast<size_t>(num_accepted) * cD * kS;
                fprintf(stderr, "[CONV_SNAP_DBG] snap[%d] first 8 f16: ", num_accepted);
                for (int i = 0; i < 8 && i < (int)(cD * kS); i++)
                    fprintf(stderr, "%.4f ", float(p[slice_off + i]));
                fprintf(stderr, "\n");
            }

            // Print current variable state for comparison
            if (cur_state.get_element_type() == ov::element::f16) {
                auto* p = reinterpret_cast<const ov::float16*>(cur_state.data());
                fprintf(stderr, "[CONV_SNAP_DBG] var first 8 f16: ");
                for (int i = 0; i < 8 && i < (int)cur_state.get_size(); i++)
                    fprintf(stderr, "%.4f ", float(p[i]));
                fprintf(stderr, "\n");
            }

            // Print snap at position 1 (if T>=2) for comparison
            if (T >= 2 && snap_tensor.get_element_type() == ov::element::f32) {
                auto* p = snap_tensor.data<float>();
                size_t slice_off1 = 1 * cD * kS;
                fprintf(stderr, "[CONV_SNAP_DBG] snap[1] first 8 f32: ");
                for (int i = 0; i < 8 && i < (int)(cD * kS); i++)
                    fprintf(stderr, "%.4f ", p[slice_off1 + i]);
                fprintf(stderr, "\n");
            }

            conv_dbg_count++;
        }

        if (use_gpu_restore) {
            try {
                request.restore_variable_from_output(sname, it->second, static_cast<size_t>(num_accepted));
                continue;
            } catch (const std::exception&) {
                // Fall through to CPU path
            } catch (...) {
                // Fall through to CPU path
            }
        }

        // CPU fallback path
        ov::Tensor all_states = request.get_tensor(it->second);
        const auto shape = all_states.get_shape();
        // shape = [B, T, conv_dim, kernel_size]
        const size_t B = shape[0];
        const size_t conv_dim = shape[2];
        const size_t kernel_size = shape[3];
        const size_t state_size = conv_dim * kernel_size;

        ov::Tensor target(all_states.get_element_type(), {B, conv_dim, kernel_size});
        const size_t elem_size = all_states.get_element_type().size();
        const auto* src = reinterpret_cast<const uint8_t*>(all_states.data());
        auto* dst = reinterpret_cast<uint8_t*>(target.data());

        for (size_t b = 0; b < B; ++b) {
            const size_t src_off = (b * shape[1] + static_cast<size_t>(num_accepted)) * state_size * elem_size;
            const size_t dst_off = b * state_size * elem_size;
            std::memcpy(dst + dst_off, src + src_off, state_size * elem_size);
        }

        state.set_state(target);
    }
}

// Initialize snapshot buffers for conv states (linear attention layers).
std::vector<ov::Tensor> init_conv_state_snapshot(ov::InferRequest& request) {
    std::vector<ov::Tensor> snap;
    for (auto& state : request.query_state()) {
        if (state.get_name().find("linear_states.") == std::string::npos) continue;
        if (state.get_name().find(".conv") == std::string::npos) continue;
        ov::Tensor t = state.get_state();
        snap.emplace_back(t.get_element_type(), t.get_shape());
    }
    return snap;
}

// Save current conv states into pre-allocated snapshot buffers.
void save_conv_states(ov::InferRequest& request, std::vector<ov::Tensor>& snap) {
    size_t idx = 0;
    for (auto& state : request.query_state()) {
        if (state.get_name().find("linear_states.") == std::string::npos) continue;
        if (state.get_name().find(".conv") == std::string::npos) continue;
        if (idx < snap.size()) {
            state.get_state().copy_to(snap[idx]);
            ++idx;
        }
    }
}

// Restore conv states from snapshot buffers.
void restore_conv_states(ov::InferRequest& request,
                         const std::vector<ov::Tensor>& snap) {
    size_t idx = 0;
    for (auto& state : request.query_state()) {
        if (state.get_name().find("linear_states.") == std::string::npos) continue;
        if (state.get_name().find(".conv") == std::string::npos) continue;
        if (idx < snap.size()) {
            state.set_state(snap[idx]);
            ++idx;
        }
    }
}

// Extract the last token's logits from [1, S, V] into a float32 scratch buffer.
// Handles f32, f16, and bf16 logit tensors.
void extract_last_logits_f32(const ov::Tensor& logits, std::vector<float>& out) {
    const auto shape = logits.get_shape();
    const size_t seq_len = shape[1];
    const size_t vocab = shape[2];
    const size_t offset = (seq_len - 1) * vocab;
    out.resize(vocab);
    if (logits.get_element_type() == ov::element::f32) {
        std::memcpy(out.data(), logits.data<const float>() + offset, vocab * sizeof(float));
    } else if (logits.get_element_type() == ov::element::f16) {
        const auto* src = logits.data<const ov::float16>() + offset;
        for (size_t i = 0; i < vocab; ++i)
            out[i] = static_cast<float>(src[i]);
    } else if (logits.get_element_type() == ov::element::bf16) {
        const auto* src = logits.data<const ov::bfloat16>() + offset;
        for (size_t i = 0; i < vocab; ++i)
            out[i] = static_cast<float>(src[i]);
    } else {
        throw std::runtime_error("Unsupported logits dtype for logit processing");
    }
}

int64_t argmax_f32(const std::vector<float>& data) {
    return static_cast<int64_t>(std::max_element(data.begin(), data.end()) - data.begin());
}

// ---------------------------------------------------------------------------
// Fast multinomial sampling — avoids full-vocab softmax and sort.
//
// Strategy (for the common case: temperature>0, top_k>0, top_p<1):
//   1. nth_element to partition top-K logits            — O(V) avg
//   2. Sort only the K candidates                       — O(K log K)
//   3. Apply temperature + softmax on K candidates only — O(K)
//   4. Apply top-P cutoff on K candidates               — O(K)
//   5. Direct CDF sampling                              — O(K)
// Total: O(V) + O(K log K), vs original O(5V + V log V).
// ---------------------------------------------------------------------------

struct SamplingContext {
    // Scratch buffers reused across decode steps.
    std::vector<std::pair<float, int64_t>> ranked;  // top-K candidates (logit, token_id), size K
    std::vector<std::pair<float, int64_t>> candidates;  // full-vocab buffer (only for Case 3: top-P without top-K)
    std::vector<float> probs;                            // softmax probs of selected candidates
};

int64_t sample_fast(const float* logits,
                    size_t vocab_size,
                    float temperature,
                    float top_p,
                    size_t top_k,
                    std::mt19937& rng,
                    SamplingContext& ctx) {
    OPENVINO_ASSERT(vocab_size > 0, "logits must not be empty");
    OPENVINO_ASSERT(temperature > 0.0f, "temperature must be positive for sampling");

    const bool use_top_k = (top_k > 0 && top_k < vocab_size);
    const bool use_top_p = (top_p > 0.0f && top_p < 1.0f);

    // --- Case 1: No top-K, no top-P — full-vocab softmax + CDF sampling (single pass) ---
    if (!use_top_k && !use_top_p) {
        // Find max for numerical stability
        float max_logit = *std::max_element(logits, logits + vocab_size);
        float inv_temp = 1.0f / temperature;

        // Compute exp and accumulate sum in one pass
        ctx.probs.resize(vocab_size);
        float total = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            float val = std::exp((logits[i] - max_logit) * inv_temp);
            ctx.probs[i] = val;
            total += val;
        }

        // Guard against degenerate distributions (NaN/Inf/zero)
        if (!(total > 0.0f) || !std::isfinite(total)) {
            return static_cast<int64_t>(std::max_element(logits, logits + vocab_size) - logits);
        }

        // Sample via CDF
        std::uniform_real_distribution<float> udist(0.0f, total);
        float dart = udist(rng);
        float cumsum = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            cumsum += ctx.probs[i];
            if (cumsum >= dart) {
                return static_cast<int64_t>(i);
            }
        }
        return static_cast<int64_t>(vocab_size - 1);
    }

    // --- Case 2: top-K (with optional top-P) — the common path ---
    // Uses a single sequential read pass over logits (same cache pattern as argmax)
    // with a tiny K-element buffer that stays in L1 cache.
    // Memory: reads 600KB (logit_buf), writes ~240 bytes (K=20 buffer).
    if (use_top_k) {
        size_t k = std::min(top_k, vocab_size);
        ctx.ranked.resize(k);

        // Initialize with first K logits
        for (size_t i = 0; i < k; ++i) {
            ctx.ranked[i] = {logits[i], static_cast<int64_t>(i)};
        }

        // Find current minimum in the K-element buffer
        size_t min_pos = 0;
        float min_val = ctx.ranked[0].first;
        for (size_t i = 1; i < k; ++i) {
            if (ctx.ranked[i].first < min_val) {
                min_val = ctx.ranked[i].first;
                min_pos = i;
            }
        }

        // Single sequential scan over remaining logits — O(V) read-only
        // Only touches the K-element buffer when a new top-K candidate is found.
        for (size_t i = k; i < vocab_size; ++i) {
            if (logits[i] > min_val) {
                ctx.ranked[min_pos] = {logits[i], static_cast<int64_t>(i)};
                // Rescan tiny buffer for new minimum (~20 comparisons, L1-resident)
                min_val = ctx.ranked[0].first;
                min_pos = 0;
                for (size_t j = 1; j < k; ++j) {
                    if (ctx.ranked[j].first < min_val) {
                        min_val = ctx.ranked[j].first;
                        min_pos = j;
                    }
                }
            }
        }

        // Sort only K elements descending — O(K log K), trivial for K=20
        std::sort(ctx.ranked.begin(), ctx.ranked.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        // Apply temperature + softmax on K candidates only
        float max_logit = ctx.ranked[0].first;
        float inv_temp = 1.0f / temperature;
        ctx.probs.resize(k);
        float total = 0.0f;
        for (size_t i = 0; i < k; ++i) {
            float val = std::exp((ctx.ranked[i].first - max_logit) * inv_temp);
            ctx.probs[i] = val;
            total += val;
        }

        // Guard against degenerate distributions (NaN/Inf/zero)
        if (!(total > 0.0f) || !std::isfinite(total)) {
            return ctx.ranked[0].second;  // fallback to highest logit
        }

        // Apply top-P cutoff if needed (ensure at least 1 candidate survives)
        size_t num_candidates = k;
        if (use_top_p) {
            float threshold = top_p * total;
            float cumsum = 0.0f;
            for (size_t i = 0; i < k; ++i) {
                cumsum += ctx.probs[i];
                if (cumsum >= threshold) {
                    num_candidates = i + 1;
                    total = cumsum;
                    break;
                }
            }
        }
        num_candidates = std::max<size_t>(1, num_candidates);

        // Direct CDF sampling from the surviving candidates
        std::uniform_real_distribution<float> udist(0.0f, total);
        float dart = udist(rng);
        float cumsum = 0.0f;
        for (size_t i = 0; i < num_candidates; ++i) {
            cumsum += ctx.probs[i];
            if (cumsum >= dart) {
                return ctx.ranked[i].second;
            }
        }
        return ctx.ranked[num_candidates - 1].second;
    }

    // --- Case 3: top-P only (no top-K) ---
    // Use partial sort with adaptive step, similar to original TopPFilter.
    ctx.candidates.resize(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        ctx.candidates[i] = {logits[i], static_cast<int64_t>(i)};
    }

    // Apply temperature + softmax on full vocab to get probabilities
    float max_logit = std::max_element(ctx.candidates.begin(), ctx.candidates.end(),
                                       [](const auto& a, const auto& b) { return a.first < b.first; })->first;
    float inv_temp = 1.0f / temperature;
    float total = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        ctx.candidates[i].first = std::exp((ctx.candidates[i].first - max_logit) * inv_temp);
        total += ctx.candidates[i].first;
    }

    // Guard against degenerate distributions (NaN/Inf/zero)
    if (!(total > 0.0f) || !std::isfinite(total)) {
        // Find the candidate with the highest original logit (max_logit)
        for (size_t i = 0; i < vocab_size; ++i) {
            if (logits[i] == max_logit) return static_cast<int64_t>(i);
        }
        return static_cast<int64_t>(0);
    }

    float threshold = top_p * total;

    // Try partial sort with increasing step sizes to find top-P nucleus
    size_t num_candidates = vocab_size;
    for (size_t step = 16; step <= 1024; step *= 2) {
        if (vocab_size <= step) break;
        std::partial_sort(ctx.candidates.begin(),
                          ctx.candidates.begin() + static_cast<ptrdiff_t>(step),
                          ctx.candidates.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        float cumsum = 0.0f;
        for (size_t i = 0; i < step; ++i) {
            cumsum += ctx.candidates[i].first;
            if (cumsum >= threshold) {
                num_candidates = i + 1;
                total = cumsum;
                goto nucleus_found;
            }
        }
    }
    // Fallback: full sort
    std::sort(ctx.candidates.begin(), ctx.candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    {
        float cumsum = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            cumsum += ctx.candidates[i].first;
            if (cumsum >= threshold) {
                num_candidates = i + 1;
                total = cumsum;
                break;
            }
        }
    }

nucleus_found:
    num_candidates = std::max<size_t>(1, num_candidates);
    // Direct CDF sampling
    std::uniform_real_distribution<float> udist(0.0f, total);
    float dart = udist(rng);
    float cumsum = 0.0f;
    for (size_t i = 0; i < num_candidates; ++i) {
        cumsum += ctx.candidates[i].first;
        if (cumsum >= dart) {
            return ctx.candidates[i].second;
        }
    }
    return ctx.candidates[num_candidates - 1].second;
}

ov::Tensor make_beam_idx(size_t batch) {
    ov::Tensor beam_idx(ov::element::i32, {batch});
    auto* data = beam_idx.data<int32_t>();
    for (size_t i = 0; i < batch; ++i) {
        data[i] = static_cast<int32_t>(i);
    }
    return beam_idx;
}

ov::Tensor make_zero_tensor(const ov::element::Type& type, const ov::Shape& shape) {
    ov::Tensor tensor(type, shape);
    std::memset(tensor.data(), 0, tensor.get_byte_size());
    return tensor;
}

// ---------------------------------------------------------------------------
// USM-host tensor helpers — iGPU zero-copy optimization
// ---------------------------------------------------------------------------
// On iGPU, GPU can access USM-host memory directly without H2D copy.
// This eliminates the costly wait_for_events overhead for each input tensor.
// Falls back to standard ov::Tensor when the context doesn't support it.

/// Try to get the GPU RemoteContext from a CompiledModel.
std::optional<ov::RemoteContext> try_get_gpu_context(ov::CompiledModel& compiled) {
    try {
        return compiled.get_context();
    } catch (...) {
        return std::nullopt;
    }
}

/// Create a USM-host tensor of given type/shape via the GPU context.
ov::Tensor make_usm_host_tensor(std::optional<ov::RemoteContext>& ctx,
                                const ov::element::Type& type,
                                const ov::Shape& shape) {
    if (ctx.has_value()) {
        try {
            return ctx->create_host_tensor(type, shape);
        } catch (...) {}
    }
    return ov::Tensor(type, shape);
}

/// Create a USM-host tensor and copy data from an existing tensor.
ov::Tensor clone_as_usm_host(std::optional<ov::RemoteContext>& ctx,
                              const ov::Tensor& src) {
    auto dst = make_usm_host_tensor(ctx, src.get_element_type(), src.get_shape());
    std::memcpy(dst.data(), src.data(), src.get_byte_size());
    return dst;
}

std::string resolve_pos_embed_name(ov::genai::modeling::weights::WeightSource& source) {
    const std::vector<std::string> candidates = {
        "model.visual.pos_embed.weight",
        "visual.pos_embed.weight",
        "pos_embed.weight",
    };
    for (const auto& name : candidates) {
        if (source.has(name)) {
            return name;
        }
    }
    for (const auto& name : source.keys()) {
        if (name.find("pos_embed.weight") != std::string::npos) {
            return name;
        }
    }
    throw std::runtime_error("Failed to locate visual.pos_embed.weight");
}

// Name used for the extra pos_embed Result in cached vision IR
static constexpr const char* kPosEmbedCacheResultName = "__pos_embed_cache__";

// Embed pos_embed weight as a Constant->Result in the vision model so that
// it is serialized into the .bin alongside other vision weights.
void embed_pos_embed_in_vision_model(std::shared_ptr<ov::Model>& model,
                                     const ov::Tensor& pos_embed) {
    auto constant = std::make_shared<ov::op::v0::Constant>(pos_embed);
    auto result = std::make_shared<ov::op::v0::Result>(constant);
    result->set_friendly_name(kPosEmbedCacheResultName);
    model->add_results({result});
}

// Extract the cached pos_embed Constant from a vision model loaded from IR,
// then remove the extra Result so it doesn't affect inference.
ov::Tensor extract_pos_embed_from_vision_model(std::shared_ptr<ov::Model>& model) {
    for (const auto& result : model->get_results()) {
        if (result->get_friendly_name() == kPosEmbedCacheResultName) {
            auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                result->input_value(0).get_node_shared_ptr());
            if (!const_node) {
                throw std::runtime_error("pos_embed cache result is not a Constant");
            }
            ov::Tensor tensor(const_node->get_element_type(), const_node->get_shape());
            std::memcpy(tensor.data(), const_node->get_data_ptr(), tensor.get_byte_size());
            model->remove_result(result);
            return tensor;
        }
    }
    throw std::runtime_error("Cached vision IR does not contain pos_embed data");
}

double elapsed_ms(const std::chrono::steady_clock::time_point& start,
                  const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<int64_t> build_dummy_prompt_ids(const ov::genai::modeling::models::Qwen3_5Config& cfg,
                                            const std::string& prompt,
                                            bool with_vision,
                                            int64_t image_tokens) {
    std::vector<int64_t> ids;
    ids.reserve(static_cast<size_t>(prompt.size()) + 16 + static_cast<size_t>(std::max<int64_t>(0, image_tokens)));

    if (with_vision) {
        ids.push_back(cfg.vision_start_token_id);
        for (int64_t i = 0; i < image_tokens; ++i) {
            ids.push_back(cfg.image_token_id);
        }
        ids.push_back(cfg.vision_end_token_id);
    }

    const int64_t vocab_base = std::max<int64_t>(100, cfg.text.vocab_size - 100);
    for (unsigned char c : prompt) {
        ids.push_back(static_cast<int64_t>(c % vocab_base));
    }
    if (ids.empty()) {
        ids = {1, 2, 3, 4, 5, 6};
    }
    return ids;
}

std::pair<ov::Tensor, ov::Tensor> to_batch_tensors(const std::vector<int64_t>& ids) {
    ov::Tensor input_ids(ov::element::i64, {1, ids.size()});
    std::memcpy(input_ids.data(), ids.data(), ids.size() * sizeof(int64_t));

    ov::Tensor attention_mask(ov::element::i64, {1, ids.size()});
    auto* mask = attention_mask.data<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) {
        mask[i] = 1;
    }
    return {input_ids, attention_mask};
}

ov::Tensor make_dummy_image() {
    ov::Tensor image(ov::element::u8, {1, 256, 256, 3});
    std::memset(image.data(), 127, image.get_byte_size());
    return image;
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc == 1) {
        print_usage(argv[0]);
        return 0;
    }

    SampleOptions opts;
    try {
        opts = parse_cli(argc, argv);
    } catch (const std::exception& cli_error) {
        std::cerr << "Command line error: " << cli_error.what() << '\n' << '\n';
        print_usage(argv[0]);
        return 1;
    }

    const bool use_vl = (opts.mode == "vl");
    const bool use_dummy_mode_flag = !opts.model_dir.has_value();
    if (!use_dummy_mode_flag && use_vl && opts.image_path.empty()) {
        std::cerr << "Command line error: Real weights + --mode vl requires --image PATH." << '\n' << '\n';
        print_usage(argv[0]);
        return 1;
    }
    const std::filesystem::path model_dir = opts.model_dir.value_or(std::filesystem::path{});
    if (!use_dummy_mode_flag) {
        if (!std::filesystem::exists(model_dir) || !std::filesystem::is_directory(model_dir)) {
            std::cerr << "Command line error: --model must point to an existing directory." << '\n' << '\n';
            print_usage(argv[0]);
            return 1;
        }
        const bool has_config = std::filesystem::exists(model_dir / "config.json");
        if (!has_config || !has_safetensors_file(model_dir)) {
            std::cerr << "Command line error: --model must contain config.json and at least one *.safetensors file."
                      << '\n'
                      << '\n';
            print_usage(argv[0]);
            return 1;
        }
    }
    if (use_vl && !opts.image_path.empty() && !std::filesystem::exists(opts.image_path)) {
        std::cerr << "Command line error: Image path does not exist: " << opts.image_path.string() << '\n' << '\n';
        print_usage(argv[0]);
        return 1;
    }

    ov::genai::modeling::models::Qwen3_5Config cfg;
    if (use_dummy_mode_flag) {
        cfg = (opts.dummy_model == "moe")
                  ? ov::genai::modeling::models::Qwen3_5Config::make_dummy_moe35b_config()
                  : ov::genai::modeling::models::Qwen3_5Config::make_dummy_dense9b_config();
    } else {
        cfg = ov::genai::modeling::models::Qwen3_5Config::from_json_file(model_dir);
    }
    apply_text_config_overrides(cfg, opts);

    // MTP: override mtp_num_hidden_layers from CLI if --mtp is used
    const bool use_mtp = opts.enable_mtp;
    const bool use_seq_verify = opts.seq_verify;
    const bool use_pure_batch = opts.pure_batch;
    if (use_mtp) {
        if (cfg.text.mtp_num_hidden_layers <= 0) {
            cfg.text.mtp_num_hidden_layers = opts.mtp_num_layers;
        }
        std::cout << "[mtp] MTP enabled, mtp_num_hidden_layers=" << cfg.text.mtp_num_hidden_layers << std::endl;
        if (use_seq_verify) {
            std::cout << "[mtp] Sequential verify enabled (single-token SDPA per position)" << std::endl;
        } else if (use_pure_batch) {
            std::cout << "[mtp] Pure batch verify (KV trim only, no linear_states rollback)";
            if (opts.refresh_interval > 0) {
                std::cout << " + state refresh every " << opts.refresh_interval << " tokens";
            }
            std::cout << std::endl;
        } else {
            std::cout << "[mtp] Batch verify with linear_states snapshot/rollback" << std::endl;
        }
    }

    ov::genai::modeling::weights::QuantizationConfig vision_quant_config;
    ov::genai::modeling::weights::QuantizationConfig text_quant_config;
    const auto shared_quant_config = ov::genai::modeling::weights::parse_quantization_config_from_env();
    if (shared_quant_config.enabled() && shared_quant_config.group_size <= 0) {
        throw std::runtime_error(
            "OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE must be > 0 when OV_GENAI_INFLIGHT_QUANT_MODE is enabled");
    }
    // vision_quant_config = shared_quant_config;  // disabled: GPU compile_model hangs with many INT4 dequant subgraphs in vision encoder
    text_quant_config = shared_quant_config;
    std::cout << "[quant] env mode=" << quant_mode_cache_token(shared_quant_config.mode)
              << ", backup=" << quant_mode_cache_token(shared_quant_config.backup_mode)
              << ", group_size=" << shared_quant_config.group_size << std::endl;

    ov::genai::modeling::models::Qwen3_5VisionPreprocessConfig pre_cfg;
    const auto pre_cfg_path = model_dir / "preprocessor_config.json";
    if (!use_dummy_mode_flag && std::filesystem::exists(pre_cfg_path)) {
        pre_cfg = ov::genai::modeling::models::Qwen3_5VisionPreprocessConfig::from_json_file(pre_cfg_path);
    }
    if (opts.max_pixels > 0) {
        std::cout << "[max-pixels] Overriding pre_cfg.max_pixels: " << pre_cfg.max_pixels
                  << " -> " << opts.max_pixels << std::endl;
        pre_cfg.max_pixels = static_cast<int64_t>(opts.max_pixels);
    }

    const std::filesystem::path app_dir = [&]() {
        std::filesystem::path exe_path = argv[0];
        if (exe_path.is_relative()) {
            exe_path = std::filesystem::absolute(exe_path);
        }
        if (exe_path.has_parent_path()) {
            return exe_path.parent_path();
        }
        return std::filesystem::current_path();
    }();
    const std::filesystem::path ir_dir = use_dummy_mode_flag ? app_dir : model_dir;
    std::string text_ir_stem = (use_vl ? "qwen3_5_text_vl" : "qwen3_5_text") + quant_cache_suffix(text_quant_config);
    if (opts.num_layers.has_value()) {
        text_ir_stem += "_l" + std::to_string(*opts.num_layers);
    }
    std::string vision_ir_stem = "qwen3_5_vision" + quant_cache_suffix(vision_quant_config);
    const auto text_xml_path = ir_dir / (text_ir_stem + ".xml");
    const auto text_bin_path = ir_dir / (text_ir_stem + ".bin");
    const auto vision_xml_path = ir_dir / (vision_ir_stem + ".xml");
    const auto vision_bin_path = ir_dir / (vision_ir_stem + ".bin");
    const std::string mtp_ir_stem = "openvino_mtp_head_int4";
    const auto mtp_xml_path = ir_dir / (mtp_ir_stem + ".xml");
    const auto mtp_bin_path = ir_dir / (mtp_ir_stem + ".bin");

    const bool load_text_from_ir = opts.cache_model && !use_dummy_mode_flag && has_ir_model_pair(text_xml_path, text_bin_path);
    const bool load_vision_from_ir =
        opts.cache_model && !use_dummy_mode_flag && use_vl && has_ir_model_pair(vision_xml_path, vision_bin_path);
    const bool load_mtp_from_ir =
        opts.cache_model && !use_dummy_mode_flag && use_mtp && has_ir_model_pair(mtp_xml_path, mtp_bin_path);

    ov::Core core;
    std::unique_ptr<ov::genai::modeling::weights::WeightSource> source;
    auto ensure_weight_source = [&]() -> ov::genai::modeling::weights::WeightSource& {
        if (!source) {
            if (use_dummy_mode_flag) {
                constexpr uint32_t kDummySeed = 2026u;
                constexpr float kDummyInitRange = 0.02f;
                auto specs = use_vl ? ov::genai::modeling::models::build_qwen3_5_vlm_weight_specs(cfg)
                                    : ov::genai::modeling::models::build_qwen3_5_text_weight_specs(cfg.text);
                // Add MTP weight specs if MTP is enabled
                if (use_mtp) {
                    auto mtp_specs = ov::genai::modeling::models::build_qwen3_5_mtp_weight_specs(cfg.text);
                    specs.insert(specs.end(), mtp_specs.begin(), mtp_specs.end());
                }
                source = std::make_unique<ov::genai::modeling::weights::SyntheticWeightSource>(
                    std::move(specs),
                    kDummySeed,
                    -kDummyInitRange,
                    kDummyInitRange);
            } else {
                auto data = ov::genai::safetensors::load_safetensors(model_dir);
                source = std::make_unique<ov::genai::safetensors::SafetensorsWeightSource>(std::move(data));
            }
        }
        return *source;
    };

    std::shared_ptr<ov::Model> vision_model;
    std::shared_ptr<ov::Model> text_model;
    ov::Tensor cached_pos_embed;  // extracted from cached vision IR, or obtained from weight_source
    if (load_vision_from_ir) {
        std::cout << "[cache-model] Reusing cached vision IR: " << vision_xml_path << std::endl;
        vision_model = core.read_model(vision_xml_path.string(), vision_bin_path.string());
        // Extract pos_embed from the cached vision IR and remove the extra Result
        cached_pos_embed = extract_pos_embed_from_vision_model(vision_model);
        std::cout << "[cache-model] Extracted pos_embed from vision IR" << std::endl;
    } else if (use_vl) {
        auto& weight_source = ensure_weight_source();
        ov::genai::safetensors::SafetensorsWeightFinalizer vision_finalizer(vision_quant_config);
        vision_model = ov::genai::modeling::models::create_qwen3_5_vision_model(cfg, weight_source, vision_finalizer);
        if (opts.cache_model) {
            // Embed pos_embed as a Constant->Result in the vision model so it's saved in the .bin
            const std::string pe_name = resolve_pos_embed_name(weight_source);
            cached_pos_embed = weight_source.get_tensor(pe_name);
            embed_pos_embed_in_vision_model(vision_model, cached_pos_embed);
            ov::serialize(vision_model, vision_xml_path.string(), vision_bin_path.string());
            std::cout << "[cache-model] Saved vision IR (with pos_embed): " << vision_xml_path << std::endl;
            // Remove the extra Result before compile_model
            for (const auto& result : vision_model->get_results()) {
                if (result->get_friendly_name() == kPosEmbedCacheResultName) {
                    vision_model->remove_result(result);
                    break;
                }
            }
        }
    }

    if (load_text_from_ir) {
        std::cout << "[cache-model] Reusing cached text IR: " << text_xml_path << std::endl;
        text_model = core.read_model(text_xml_path.string(), text_bin_path.string());
        if (use_vl && !is_vl_text_ir_compatible(text_model)) {
            std::cout << "[cache-model] Cached text IR is not VL-compatible (missing visual inputs), rebuilding: "
                      << text_xml_path << std::endl;
            text_model.reset();
        }
    }
    if (!text_model) {
        auto& weight_source = ensure_weight_source();
        ov::genai::safetensors::SafetensorsWeightFinalizer text_finalizer(text_quant_config);
        text_model = ov::genai::modeling::models::create_qwen3_5_text_model(
            cfg,
            weight_source,
            text_finalizer,
            false,
            use_vl,
            use_mtp /* output_hidden_states */);
        if (opts.cache_model) {
            ov::serialize(text_model, text_xml_path.string(), text_bin_path.string());
            std::cout << "[cache-model] Saved text IR: " << text_xml_path << std::endl;
        }
    }

    // Build MTP model if enabled
    std::shared_ptr<ov::Model> mtp_model;
    if (use_mtp) {
        if (load_mtp_from_ir) {
            std::cout << "[cache-model] Reusing cached MTP IR: " << mtp_xml_path << std::endl;
            mtp_model = core.read_model(mtp_xml_path.string(), mtp_bin_path.string());
        } else {
            auto& weight_source = ensure_weight_source();
            ov::genai::safetensors::SafetensorsWeightFinalizer mtp_finalizer(text_quant_config);
            mtp_model = ov::genai::modeling::models::create_qwen3_5_mtp_model(
                cfg,
                weight_source,
                mtp_finalizer);
            std::cout << "[mtp] MTP model built successfully" << std::endl;
            if (opts.cache_model) {
                ov::serialize(mtp_model, mtp_xml_path.string(), mtp_bin_path.string());
                std::cout << "[cache-model] Saved MTP IR: " << mtp_xml_path << std::endl;
            }
        }
    }

    if (use_dummy_mode_flag && source) {
        source->release_all_cached_tensors();
        if (!use_vl && !use_mtp) {
            source.reset();
        }
    }

    std::optional<ov::CompiledModel> compiled_vision;
    if (use_vl) {
        std::string vision_device = opts.device;
        if (const char* env_vision_dev = std::getenv("OV_GENAI_VISION_DEVICE")) {
            vision_device = env_vision_dev;
        }
        std::cout << "[vision] Compiling vision model on device: " << vision_device << std::endl;
        compiled_vision = core.compile_model(vision_model, vision_device);
    }

    // oneDNN INT4 FC batch-invariance: The GPU plugin's oneDNN FC implementation
    // now includes a "batch-1 loop" that splits small verify batches (M=2..8) into
    // M individual M=1 matmul calls, ensuring bit-identical computation to decode.
    // This eliminates the batch-dependent GEMM tiling divergence that previously
    // required disabling oneDNN entirely for INT4 MTP.
    // OV_GPU_ONEDNN_FC_BATCH1_MAX controls the threshold (default 8, set 0 to disable).
    //
    // Legacy auto-disable: If OV_GPU_USE_ONEDNN is not set, we now leave it enabled
    // (relying on the batch-1 loop).  Set OV_GPU_USE_ONEDNN=0 to force OCL path.
    if (use_mtp && use_pure_batch && text_quant_config.is_primary_4bit()) {
        auto* existing = std::getenv("OV_GPU_USE_ONEDNN");
        if (!existing || std::string(existing).empty()) {
            std::cout << "[mtp] oneDNN enabled for INT4 MTP (batch-1 loop ensures decode/verify identity)" << std::endl;
        } else {
            std::cout << "[mtp] OV_GPU_USE_ONEDNN=" << existing << " (user override)" << std::endl;
        }
    }

    // Auto-set SDPA single-token threshold for MTP pure-batch mode.
    // SDPA multi-token kernel uses different tiling than single-token kernel,
    // causing batch-size-dependent numerical divergence (same root cause class
    // as the oneDNN FC issue).  Force single-token SDPA kernel for batch sizes
    // up to K+1 to ensure batch=1 draft and batch=K+1 verify produce identical
    // hidden states at each token position.
    //
    // Applied for all quantization modes: INT4/INT8/f16 all exhibit cumulative
    // divergence in Mamba recurrent states when multi-token SDPA causes
    // batch-vs-sequential discrepancy.  Especially pronounced with longer KV
    // caches (e.g. VL mode) and K>=2 where the error accumulates over many
    // reject-and-restore cycles.
    if (use_mtp && use_pure_batch) {
        const int sdpa_threshold = opts.mtp_k + 1;  // K+1 tokens in verify batch
        auto* existing = std::getenv("OV_GPU_SDPA_SINGLE_TOKEN_THRESHOLD");
        const int current = existing ? std::atoi(existing) : 0;
        if (sdpa_threshold > current) {
#ifdef _WIN32
            _putenv_s("OV_GPU_SDPA_SINGLE_TOKEN_THRESHOLD", std::to_string(sdpa_threshold).c_str());
#else
            setenv("OV_GPU_SDPA_SINGLE_TOKEN_THRESHOLD", std::to_string(sdpa_threshold).c_str(), 1);
#endif
            std::cout << "[mtp] Auto-set OV_GPU_SDPA_SINGLE_TOKEN_THRESHOLD=" << sdpa_threshold
                      << " for batch-size-invariant SDPA" << std::endl;
        } else if (existing) {
            std::cout << "[mtp] OV_GPU_SDPA_SINGLE_TOKEN_THRESHOLD=" << current
                      << " (user override, K+1=" << sdpa_threshold << ")" << std::endl;
        }
    }

    auto compiled_text = core.compile_model(text_model, opts.device);

    // Compile MTP model if enabled
    std::optional<ov::CompiledModel> compiled_mtp;
    if (use_mtp && mtp_model) {
        compiled_mtp = core.compile_model(mtp_model, opts.device);
        std::cout << "[mtp] MTP model compiled on device: " << opts.device << std::endl;
    }

    ov::Tensor visual_embeds;
    ov::Tensor grid_thw;
    if (use_vl) {
        ov::Tensor image = opts.image_path.empty() ? make_dummy_image() : utils::load_image(opts.image_path);
        ov::Tensor pos_embed_weight;
        if (cached_pos_embed) {
            pos_embed_weight = cached_pos_embed;
        } else {
            auto& weight_source = ensure_weight_source();
            const std::string pos_embed_name = resolve_pos_embed_name(weight_source);
            pos_embed_weight = weight_source.get_tensor(pos_embed_name);
        }

        ov::genai::modeling::models::Qwen3_5VisionPreprocessor preprocessor(cfg.vision, pre_cfg);
        const auto preprocess_start = std::chrono::steady_clock::now();
        auto vision_inputs = preprocessor.preprocess(image, pos_embed_weight);
        const auto preprocess_end = std::chrono::steady_clock::now();

        auto vision_request = compiled_vision->create_infer_request();
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kPixelValues, vision_inputs.pixel_values);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kGridThw, vision_inputs.grid_thw);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kPosEmbeds, vision_inputs.pos_embeds);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kRotaryCos, vision_inputs.rotary_cos);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kRotarySin, vision_inputs.rotary_sin);
        const auto vision_start = std::chrono::steady_clock::now();
        vision_request.infer();
        const auto vision_end = std::chrono::steady_clock::now();

        visual_embeds = vision_request.get_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kVisualEmbeds);
        grid_thw = vision_inputs.grid_thw;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "[vision] preprocess: " << elapsed_ms(preprocess_start, preprocess_end) << " ms" << std::endl;
        std::cout << "[vision] encode: " << elapsed_ms(vision_start, vision_end) << " ms" << std::endl;

        if (use_dummy_mode_flag && source) {
            source->release_all_cached_tensors();
            source.reset();
        }
    }

    std::unique_ptr<ov::genai::Tokenizer> tokenizer;
    if (!use_dummy_mode_flag) {
        try {
            tokenizer = std::make_unique<ov::genai::Tokenizer>(model_dir);
        } catch (const std::exception&) {
            tokenizer.reset();
        }
    }

    ov::Tensor input_ids;
    ov::Tensor attention_mask;
    bool use_dummy_tokenization = !tokenizer;
    if (tokenizer) {
        try {
            std::string prompt;
            bool add_special_tokens = true;
            if (use_vl) {
                const int64_t image_tokens =
                    ov::genai::modeling::models::Qwen3_5VisionPreprocessor::count_visual_tokens(
                        grid_thw,
                        cfg.vision.spatial_merge_size);
                prompt = build_vl_prompt(opts.user_prompt, image_tokens, opts.enable_thinking);
                add_special_tokens = false;
            } else {
                prompt = opts.user_prompt;
                if (!tokenizer->get_chat_template().empty()) {
                    ov::genai::ChatHistory history({{{"role", "user"}, {"content", prompt}}});
                    constexpr bool add_generation_prompt = true;
                    ov::genai::JsonContainer extra({{"enable_thinking", opts.enable_thinking}});
                    prompt = tokenizer->apply_chat_template(history, add_generation_prompt, {}, std::nullopt, extra);
                    add_special_tokens = false;
                }
            }
            auto tokenized = tokenizer->encode(prompt, ov::genai::add_special_tokens(add_special_tokens));
            input_ids = tokenized.input_ids;
            attention_mask = tokenized.attention_mask;
            
            // For VL mode, verify token IDs match config - fall back to dummy if mismatch
            if (use_vl) {
                const int64_t* ids = input_ids.data<const int64_t>();
                size_t img_tok_count = 0;
                for (size_t i = 0; i < input_ids.get_shape()[1]; ++i) {
                    if (ids[i] == cfg.image_token_id) img_tok_count++;
                }
                if (img_tok_count == 0) {
                    std::cerr << "Tokenizer image_token_id mismatch (expected " << cfg.image_token_id 
                              << " but none found in input_ids)" << std::endl;
                    std::cerr << "Falling back to dummy tokenization for VL mode..." << std::endl;
                    use_dummy_tokenization = true;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Tokenizer encode failed: " << e.what() << std::endl;
            std::cerr << "Falling back to dummy tokenization..." << std::endl;
            use_dummy_tokenization = true;
            tokenizer.reset();
        }
    }
    if (use_dummy_tokenization) {
        const int64_t image_tokens = use_vl
                                         ? ov::genai::modeling::models::Qwen3_5VisionPreprocessor::count_visual_tokens(
                                               grid_thw,
                                               cfg.vision.spatial_merge_size)
                                         : 0;
        auto ids = build_dummy_prompt_ids(cfg, opts.user_prompt, use_vl, image_tokens);
        auto tensors = to_batch_tensors(ids);
        input_ids = tensors.first;
        attention_mask = tensors.second;
    }

    const size_t batch = input_ids.get_shape().at(0);
    const int64_t prompt_len = static_cast<int64_t>(input_ids.get_shape().at(1));

    // Collect all prompt token IDs (flat, for LogitProcessor repetition tracking).
    std::vector<int64_t> prompt_token_ids(
        input_ids.data<const int64_t>(),
        input_ids.data<const int64_t>() + input_ids.get_size());

    ov::genai::modeling::models::Qwen3_5InputPlanner planner(cfg);
    auto plan = planner.build_plan(input_ids, &attention_mask, use_vl ? &grid_thw : nullptr);

    ov::Tensor visual_padded;
    if (use_vl) {
        visual_padded = ov::genai::modeling::models::Qwen3_5InputPlanner::scatter_visual_embeds(
            visual_embeds,
            plan.visual_pos_mask);
    }

    auto beam_idx = make_beam_idx(batch);
    auto text_request = compiled_text.create_infer_request();

    // Get GPU context for USM-host tensors (iGPU zero-copy optimization)
    auto gpu_ctx = try_get_gpu_context(compiled_text);

    // Clone prefill inputs as USM-host for zero-copy on iGPU
    auto usm_input_ids = clone_as_usm_host(gpu_ctx, input_ids);
    auto usm_attention_mask = clone_as_usm_host(gpu_ctx, attention_mask);
    auto usm_position_ids = clone_as_usm_host(gpu_ctx, plan.position_ids);
    auto usm_beam_idx = clone_as_usm_host(gpu_ctx, beam_idx);

    text_request.reset_state();
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kInputIds, usm_input_ids);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kAttentionMask, usm_attention_mask);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kPositionIds, usm_position_ids);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kBeamIdx, usm_beam_idx);
    if (use_vl) {
        auto usm_visual_padded = clone_as_usm_host(gpu_ctx, visual_padded);
        auto usm_visual_pos_mask = clone_as_usm_host(gpu_ctx, plan.visual_pos_mask);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualEmbeds, usm_visual_padded);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualPosMask, usm_visual_pos_mask);
    }

    // Build GenerationConfig for penalty-only LogitProcessor.
    // Temperature / TopP / TopK are handled by sample_fast() to avoid
    // O(V log V) full-vocab softmax+sort on every decode step.
    const bool use_sampling = opts.temperature > 0.0f;
    ov::genai::GenerationConfig penalty_config;
    penalty_config.do_sample = false;  // penalties only — no Temperature/TopP/TopK transforms
    penalty_config.repetition_penalty = opts.repetition_penalty;
    penalty_config.frequency_penalty = opts.frequency_penalty;
    penalty_config.presence_penalty = opts.presence_penalty;
    ov::genai::LogitProcessor penalty_processor(penalty_config, prompt_token_ids);

    // RNG for multinomial sampling.
    std::mt19937 rng(opts.rng_seed != 0
                     ? static_cast<std::mt19937::result_type>(opts.rng_seed)
                     : std::random_device{}());

    // Reusable float32 scratch buffer for logit processing across all decode steps.
    std::vector<float> logit_buf;

    // Pre-allocated scratch buffers for fast sampling (reused across decode steps).
    SamplingContext sampling_ctx;

    const auto prefill_start = std::chrono::steady_clock::now();
    text_request.infer();
    ov::Tensor logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kLogits);
    const auto prefill_end = std::chrono::steady_clock::now();

    // If MTP enabled, also capture hidden_states from the main model
    ov::Tensor main_hidden_states;
    if (use_mtp) {
        main_hidden_states = text_request.get_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kHiddenStates);
    }

    // Process prefill logits and select first token.
    extract_last_logits_f32(logits, logit_buf);
    int64_t next_id;
    {
        ov::genai::Logits lw(logit_buf.data(), logit_buf.size());
        penalty_processor.apply(lw);  // penalties only — O(|generated|)
        next_id = use_sampling
                      ? sample_fast(logit_buf.data(), logit_buf.size(),
                                    opts.temperature, opts.top_p, opts.top_k, rng, sampling_ctx)
                      : argmax_f32(logit_buf);
    }
    penalty_processor.register_new_generated_token(next_id);
    penalty_processor.update_generated_len(1);

    std::vector<int64_t> generated;
    generated.reserve(static_cast<size_t>(opts.max_new_tokens));
    generated.push_back(next_id);

    const auto stop_token_ids = resolve_stop_token_ids(use_dummy_mode_flag ? std::filesystem::path{} : model_dir,
                                                       tokenizer.get());

    ov::Tensor step_ids = make_usm_host_tensor(gpu_ctx, ov::element::i64, {batch, 1});
    ov::Tensor step_mask = make_usm_host_tensor(gpu_ctx, ov::element::i64, {batch, 1});
    auto* step_mask_data = step_mask.data<int64_t>();
    for (size_t b = 0; b < batch; ++b) {
        step_mask_data[b] = 1;
    }

    ov::Tensor decode_visual;
    ov::Tensor decode_visual_mask;
    if (use_vl) {
        decode_visual = make_usm_host_tensor(gpu_ctx, ov::element::f32, {batch, 1, static_cast<size_t>(cfg.text.hidden_size)});
        std::memset(decode_visual.data(), 0, decode_visual.get_byte_size());
        decode_visual_mask = make_usm_host_tensor(gpu_ctx, ov::element::boolean, {batch, 1});
        std::memset(decode_visual_mask.data(), 0, decode_visual_mask.get_byte_size());
    }

    int64_t past_len = prompt_len;
    if (attention_mask.get_element_type() == ov::element::i64) {
        const int64_t* mask_data = attention_mask.data<const int64_t>();
        const size_t mask_batch = attention_mask.get_shape().at(0);
        const size_t mask_seq = attention_mask.get_shape().at(1);
        if (mask_batch > 0 && mask_seq > 0) {
            int64_t first_active_tokens = -1;
            for (size_t b = 0; b < mask_batch; ++b) {
                int64_t active_tokens = 0;
                for (size_t s = 0; s < mask_seq; ++s) {
                    if (mask_data[b * mask_seq + s] != 0) {
                        active_tokens += 1;
                    }
                }

                if (first_active_tokens < 0) {
                    first_active_tokens = active_tokens;
                } else if (first_active_tokens != active_tokens) {
                    OPENVINO_THROW("Batch decode with different active prompt lengths is not supported in this sample: first=",
                                   first_active_tokens,
                                   ", current=",
                                   active_tokens,
                                   ".");
                }
            }

            if (first_active_tokens >= 0) {
                past_len = first_active_tokens;
            }
        }
    }
    // Pre-allocate USM-host tensor for decode position_ids - reuse across steps
    // to avoid per-step create_host_tensor() + memcpy overhead.
    // Shape: [3, batch, 1] (3 planes of identical position values per batch element)
    ov::Tensor usm_decode_pos = make_usm_host_tensor(gpu_ctx, ov::element::i64, {3, batch, 1});
    const int64_t* rope_deltas_data = plan.rope_deltas.data<const int64_t>();

    // ---------------------------------------------------------------------------
    // MTP inference request setup (speculative decoding — K+1 verify)
    // ---------------------------------------------------------------------------
    std::optional<ov::InferRequest> mtp_request;
    ov::Tensor mtp_step_ids;
    ov::Tensor mtp_hidden_in;
    ov::Tensor mtp_mask;
    ov::Tensor mtp_pos;
    ov::Tensor mtp_beam;
    std::optional<ov::RemoteContext> mtp_gpu_ctx;
    size_t mtp_hits = 0;
    size_t mtp_attempts = 0;
    size_t mtp_main_infers = 0;  // number of main model infer calls during decode
    std::vector<float> mtp_logit_buf;

    const int max_K = opts.mtp_k;  // max speculation depth from CLI
    int K = max_K;                  // current effective K (mutable for adaptive mode)
    std::vector<int64_t> drafts(static_cast<size_t>(max_K), -1);
    bool drafts_ready = false;

    // Phase C: Adaptive K selection state
    // When enabled (--adaptive-k 1), K adapts between 1 and max_K based on
    // a rolling accept rate. Low accept rates (creative text) → K=1 to minimize
    // wasted drafts. High accept rates (structured/VL) → higher K.
    const bool adaptive_k_enabled = opts.adaptive_k && max_K > 1;
    const int adaptive_window = 16;            // rolling window size (steps)
    const double adaptive_up_threshold = 0.70; // increase K when hit rate > 70%
    const double adaptive_down_threshold = 0.40; // decrease K when hit rate < 40%
    const int adaptive_warmup = 8;             // collect data before adapting
    const int adaptive_cooldown = 4;           // min steps between K changes
    std::deque<std::pair<int, int>> adaptive_history; // (hits, attempts) per step
    int adaptive_cooldown_counter = 0;
    size_t adaptive_k_changes = 0;             // count of K adaptation events

    if (use_mtp && compiled_mtp) {
        mtp_request = compiled_mtp->create_infer_request();
        mtp_gpu_ctx = try_get_gpu_context(*compiled_mtp);

        mtp_step_ids = make_usm_host_tensor(mtp_gpu_ctx, ov::element::i64, {batch, 1});
        mtp_hidden_in = make_usm_host_tensor(mtp_gpu_ctx, ov::element::f32, {batch, 1, static_cast<size_t>(cfg.text.hidden_size)});
        mtp_mask = make_usm_host_tensor(mtp_gpu_ctx, ov::element::i64, {batch, 1});
        auto* mtp_mask_ptr = mtp_mask.data<int64_t>();
        for (size_t b = 0; b < batch; ++b) {
            mtp_mask_ptr[b] = 1;
        }
        mtp_pos = make_usm_host_tensor(mtp_gpu_ctx, ov::element::i64, {3, batch, 1});
        mtp_beam = make_usm_host_tensor(mtp_gpu_ctx, ov::element::i32, {batch});
        auto* mtp_beam_ptr = mtp_beam.data<int32_t>();
        for (size_t b = 0; b < batch; ++b) {
            mtp_beam_ptr[b] = static_cast<int32_t>(b);
        }

        // Phase A: Pre-bind all MTP input tensors once. Shapes are constant
        // ([batch,1] for ids/mask, [3,batch,1] for pos, [batch] for beam),
        // so we write directly to pre-bound data pointers on each step
        // instead of calling set_tensor() 5 times per MTP inference.
        mtp_request->set_tensor(ov::genai::modeling::models::Qwen3_5MtpIO::kInputIds, mtp_step_ids);
        mtp_request->set_tensor(ov::genai::modeling::models::Qwen3_5MtpIO::kHiddenStates, mtp_hidden_in);
        mtp_request->set_tensor(ov::genai::modeling::models::Qwen3_5MtpIO::kAttentionMask, mtp_mask);
        mtp_request->set_tensor(ov::genai::modeling::models::Qwen3_5MtpIO::kPositionIds, mtp_pos);
        mtp_request->set_tensor(ov::genai::modeling::models::Qwen3_5MtpIO::kBeamIdx, mtp_beam);
    }

    // Phase A: Cached MTP output tensor references. Populated after the
    // first MTP inference to avoid per-call get_tensor() overhead.
    // Output shapes are constant ([1,1,vocab] for logits, [1,1,hidden] for hs),
    // so the runtime reuses the same output buffers across inferences.
    ov::Tensor mtp_logits_cached;
    ov::Tensor mtp_hs_cached;
    bool mtp_outputs_bound = false;

    // Phase D: MTP draft sub-step timing instrumentation.
    // Quantifies the lm_head bottleneck (80.7% of MTP compute = 4096×248320 matmul)
    // to justify future speculative head investment (reduced lm_head → 4096×32K).
    double mtp_sub_memcpy_ms = 0;   // hidden_states copy to MTP input
    double mtp_sub_infer_ms = 0;    // GPU infer (decoder layer + lm_head)
    double mtp_sub_extract_ms = 0;  // logits extraction from GPU tensor
    double mtp_sub_sample_ms = 0;   // argmax/sampling over vocab_size logits
    size_t mtp_sub_count = 0;

    // Lambda: single MTP inference. Writes to pre-bound input buffers, infers,
    // reads from cached output tensors. No set_tensor/get_tensor per call.
    // Returns sampled draft token.
    auto run_mtp_single = [&](int64_t token_id, const float* hs_src, int64_t position) -> int64_t {
        const size_t hidden_size = static_cast<size_t>(cfg.text.hidden_size);

        auto t_mc0 = std::chrono::steady_clock::now();
        std::memcpy(mtp_hidden_in.data<float>(), hs_src, hidden_size * sizeof(float));
        mtp_step_ids.data<int64_t>()[0] = token_id;
        auto* pos_ptr = mtp_pos.data<int64_t>();
        const int64_t mtp_pos_val = position + rope_deltas_data[0];
        pos_ptr[0] = mtp_pos_val;
        pos_ptr[batch] = mtp_pos_val;
        pos_ptr[2 * batch] = mtp_pos_val;
        auto t_mc1 = std::chrono::steady_clock::now();
        mtp_sub_memcpy_ms += elapsed_ms(t_mc0, t_mc1);

        // Input tensors are pre-bound — just infer directly.
        auto t_inf0 = std::chrono::steady_clock::now();
        mtp_request->infer();
        auto t_inf1 = std::chrono::steady_clock::now();
        mtp_sub_infer_ms += elapsed_ms(t_inf0, t_inf1);

        // Cache output tensor handles after first inference
        if (!mtp_outputs_bound) {
            mtp_logits_cached = mtp_request->get_tensor(ov::genai::modeling::models::Qwen3_5MtpIO::kLogits);
            mtp_hs_cached = mtp_request->get_tensor(ov::genai::modeling::models::Qwen3_5MtpIO::kMtpHiddenStates);
            mtp_outputs_bound = true;
        }

        auto t_ext0 = std::chrono::steady_clock::now();
        extract_last_logits_f32(mtp_logits_cached, mtp_logit_buf);
        auto t_ext1 = std::chrono::steady_clock::now();
        mtp_sub_extract_ms += elapsed_ms(t_ext0, t_ext1);

        auto t_smp0 = std::chrono::steady_clock::now();
        int64_t result = use_sampling
                   ? sample_fast(mtp_logit_buf.data(), mtp_logit_buf.size(),
                                 opts.temperature, opts.top_p, opts.top_k, rng, sampling_ctx)
                   : argmax_f32(mtp_logit_buf);
        auto t_smp1 = std::chrono::steady_clock::now();
        mtp_sub_sample_ms += elapsed_ms(t_smp0, t_smp1);
        mtp_sub_count++;
        return result;
    };

    // Dead position tracker for virtual trim (pure-batch mode).
    // Instead of physically trimming KV cache (expensive GPU→CPU→GPU copy),
    // we leave rejected entries in the cache and mask them with 0 in
    // attention_mask. SDPA adds -inf at masked positions (ignored).
    // Linear attention multiplies hidden_states by mask, zeroing dead slots.
    // Declared here so both generate_k_drafts and run_kp1_verify can capture it.
    std::vector<int64_t> dead_positions;

    // Lambda: generate K draft tokens via autoregressive MTP.
    // First draft uses main model hidden_states at hs_pos. Subsequent drafts
    // use MTP's own hidden_states output (autoregressive).
    // Per-k MTP timing (declared before generate_k_drafts so lambda can capture)
    std::vector<double> step_mtp_k_ms(static_cast<size_t>(max_K), 0.0);  // filled per-step in-place
    double step_mtp_reset_ms = 0.0;

    auto generate_k_drafts = [&](const ov::Tensor& main_hs, size_t hs_pos) {
        const size_t hidden_size = static_cast<size_t>(cfg.text.hidden_size);

        auto t_reset0 = std::chrono::steady_clock::now();
        mtp_request->reset_state();
        auto t_reset1 = std::chrono::steady_clock::now();
        step_mtp_reset_ms = elapsed_ms(t_reset0, t_reset1);

        // Use past_len for RoPE position_ids. Do NOT subtract dead_positions:
        // KV cache entries retain their original RoPE encoding at their physical
        // position, so new tokens must use contiguous positions starting at past_len.

        // Draft 0: from main model hidden_states
        const float* hs_ptr = main_hs.data<const float>() + hs_pos * hidden_size;
        {
            auto t_k0 = std::chrono::steady_clock::now();
            drafts[0] = run_mtp_single(next_id, hs_ptr, past_len);
            auto t_k1 = std::chrono::steady_clock::now();
            step_mtp_k_ms[0] = elapsed_ms(t_k0, t_k1);
        }

        // Drafts 1..K-1: from MTP's own hidden_states (autoregressive).
        // Use cached mtp_hs_cached (populated by run_mtp_single after first infer)
        // instead of calling get_tensor() each iteration.
        for (int k = 1; k < K; ++k) {
            auto t_k0 = std::chrono::steady_clock::now();
            const float* mtp_hs_ptr = mtp_hs_cached.data<const float>();
            drafts[k] = run_mtp_single(drafts[k - 1], mtp_hs_ptr, past_len + k);
            auto t_k1 = std::chrono::steady_clock::now();
            step_mtp_k_ms[static_cast<size_t>(k)] = elapsed_ms(t_k0, t_k1);
        }
    };

    // Generate initial K drafts from prefill hidden_states
    if (use_mtp && mtp_request) {
        const auto hs_shape = main_hidden_states.get_shape();
        const size_t prefill_last_pos = hs_shape[1] - 1;
        generate_k_drafts(main_hidden_states, prefill_last_pos);
        drafts_ready = true;
    }

    size_t decode_steps = 0;
    // Timing accumulators for profiling speculative decode overhead
    double time_verify_ms = 0, time_draft_ms = 0, time_trim_ms = 0;
    double time_snapshot_ms = 0, time_restore_ms = 0, time_reforward_ms = 0;
    size_t count_verify_infers = 0, count_draft_infers = 0, count_trims = 0;
    size_t count_restores = 0, count_reforwards = 0;

    // Per-step profiling: fine-grained sub-step timing (enabled by OV_GENAI_STEP_PROFILE=1)
    static const bool step_profile_enabled = []() {
        auto* env = std::getenv("OV_GENAI_STEP_PROFILE");
        return env && std::string(env) != "0";
    }();
    // Sub-step accumulators (always tracked, printed in summary when step_profile enabled)
    double time_snapshot_save_ms = 0;    // linear_states + conv_states save before verify
    double time_main_infer_ms = 0;       // main model GPU infer (K+1 batch)
    double time_accept_check_ms = 0;     // CPU-side logits extract + argmax + compare
    double time_state_restore_ms = 0;    // linear/conv state restore on rejection
    double time_kv_trim_ms = 0;          // physical KV trim on rejection
    double time_mtp_reset_ms = 0;        // MTP model reset_state() call
    std::vector<double> time_mtp_each_ms; // per-k MTP head infer times within draft phase
    time_mtp_each_ms.resize(static_cast<size_t>(max_K), 0.0);
    std::vector<size_t> count_mtp_each(static_cast<size_t>(max_K), 0);
    size_t step_counter = 0;

    // Pre-allocate snapshot buffers for linear_states rollback (batch verify path)
    auto linear_snap = init_linear_state_snapshot(text_request);
    if (!linear_snap.empty()) {
        std::cout << "[mtp] linear_states snapshot: " << linear_snap.size()
                  << " tensors pre-allocated for batch verify rollback" << std::endl;
    }

    // Detect per-token linear_states snapshot outputs (kernel-level snapshots).
    // When available (OV_GENAI_MTP_SNAPSHOT=1), the model returns per-token
    // intermediate recurrent states, allowing precise state selection after
    // batch verify without re-forward.
    auto all_linear_states_names = find_all_linear_states_outputs(text_request);
    auto all_conv_states_names = find_all_conv_states_outputs(text_request);
    const bool has_kernel_snapshot_real = !all_linear_states_names.empty();
    const bool has_kernel_snapshot = [&]() -> bool {
        auto* env = std::getenv("OV_GENAI_USE_KERNEL_SNAPSHOT");
        if (env && std::string(env) == "0") {
            fprintf(stderr, "[MTP_DBG] Kernel snapshot restore DISABLED by OV_GENAI_USE_KERNEL_SNAPSHOT=0\n");
            return false;
        }
        return has_kernel_snapshot_real;
    }();
    if (has_kernel_snapshot_real) {
        std::cout << "[mtp] Kernel-level per-token state snapshots: "
                  << all_linear_states_names.size() << " linear + "
                  << all_conv_states_names.size() << " conv outputs detected" << std::endl;
    }

    // Conv state snapshot for batch verify: conv states are small (~128KB × 24 layers)
    // and need rollback on rejection since conv is a sliding window that advances.
    auto conv_snap = init_conv_state_snapshot(text_request);
    if (!conv_snap.empty() && has_kernel_snapshot) {
        std::cout << "[mtp] conv_states snapshot: " << conv_snap.size()
                  << " tensors pre-allocated for batch verify rollback" << std::endl;
    }

    // Periodic state refresh for pure-batch mode: rolling checkpoint to
    // correct accumulated state drift from batch=K+1 inference.
    // With oneDNN batch-1 loop + SDPA threshold, K=1 and K=2 are fully
    // batch-size-invariant and need NO refresh.  K≥3 still has residual
    // drift from other batch-dependent kernels and needs modest refresh.
    const int REFRESH_INTERVAL = [&]() -> int {
        if (opts.refresh_interval > 0) return opts.refresh_interval;
        auto* env = std::getenv("OV_GENAI_SNAPSHOT_RESTORE");
        int mode = env ? std::atoi(env) : 3;
        if (mode & 8) return 32;  // re-forward mode: frequent refresh
        if (has_kernel_snapshot && max_K >= 3) {
            // K≥3: batch=4+, residual drift beyond FC/SDPA → 32 tokens
            return 32;
        }
        return 0;  // K=1/K=2: no refresh needed with batch-1 loop + SDPA threshold
    }();
    int64_t checkpoint_past_len = 0;
    std::vector<int64_t> tokens_since_checkpoint;
    double time_refresh_ms = 0;
    size_t count_refreshes = 0;

    const auto decode_start = std::chrono::steady_clock::now();

    // =======================================================================
    // Speculative decode loop — vLLM-style K+1 verify
    // =======================================================================
    // Architecture (matches vLLM + NVIDIA flow):
    //   1. DRAFT PHASE: MTP proposer generates K draft tokens autoregressively
    //   2. VERIFY PHASE: Feed [next_id, d1, ..., dk] as K+1 tokens to main model
    //   3. ACCEPT/REJECT: Sequential verification (stop at first mismatch)
    //      - All accepted: emit K drafts + bonus (K+1 tokens), no KV trim
    //      - Rejected at position k: emit k accepted + 1 correction, trim K-k
    //   4. DRAFT for next step: generate K new drafts from main model hs
    //
    // E[Tok/Infer] = sum_{i=0}^{K} P(accept i) * (i+1) / 1  (since 1 main infer)
    // For K=1: E = h*2 + (1-h)*1 = 1+h   (identical to v3)
    // For K=2: E = h²*3 + h(1-h)*2 + (1-h)*1 = 1+h+h²
    // =======================================================================
    if (use_mtp && mtp_request && drafts_ready) {
        const size_t max_kp1 = static_cast<size_t>(max_K + 1);  // max K+1 for pre-allocation

        // Snapshot restore mode bitmask:
        //   1=linear from kernel snapshot, 2=conv from kernel snapshot, 3=both
        //   4=conv from CPU pre-batch, 8=re-forward (trim all K+1 + replay accepted)
        static const int snapshot_restore_mode = []() -> int {
            auto* env = std::getenv("OV_GENAI_SNAPSHOT_RESTORE");
            return env ? std::atoi(env) : 3;
        }();

        // Initialize rolling checkpoint for pure-batch state refresh.
        // Needed when:
        //   - No kernel snapshots (fallback): re-forward on rejection + periodic refresh
        //   - Mode 8 (kernel snapshot + re-forward): periodic refresh to prevent
        //     KV cache drift from batch=K+1 vs batch=1 numerical differences
        //   - REFRESH_INTERVAL > 0: periodic refresh as safety net (e.g. K>=3)
        const bool needs_refresh_tracking = !has_kernel_snapshot || (snapshot_restore_mode & 8) || REFRESH_INTERVAL > 0;
        if (needs_refresh_tracking) {
            save_linear_states(text_request, linear_snap);
            checkpoint_past_len = past_len;
            tokens_since_checkpoint.clear();
            tokens_since_checkpoint.push_back(next_id);
        }

        // Pre-allocate K+1 token tensors for main model verify (max_K for allocation)
        ov::Tensor step_ids_kp1 = make_usm_host_tensor(gpu_ctx, ov::element::i64, {batch, max_kp1});
        ov::Tensor usm_decode_pos_kp1 = make_usm_host_tensor(gpu_ctx, ov::element::i64, {3, batch, max_kp1});

        ov::Tensor decode_visual_kp1;
        ov::Tensor decode_visual_mask_kp1;
        if (use_vl) {
            decode_visual_kp1 = make_usm_host_tensor(gpu_ctx, ov::element::f32,
                {batch, max_kp1, static_cast<size_t>(cfg.text.hidden_size)});
            std::memset(decode_visual_kp1.data(), 0, decode_visual_kp1.get_byte_size());
            decode_visual_mask_kp1 = make_usm_host_tensor(gpu_ctx, ov::element::boolean, {batch, max_kp1});
            std::memset(decode_visual_mask_kp1.data(), 0, decode_visual_mask_kp1.get_byte_size());
        }

        // Helper: run K+1 token main model verify (batch) and advance past_len.
        // Uses current effective K (may change with adaptive K).
        auto run_kp1_verify = [&]() {
            const size_t kp1 = static_cast<size_t>(K + 1);

            // Resize pre-allocated tensors to current K+1 (no realloc if ≤ max_K+1)
            step_ids_kp1.set_shape({batch, kp1});
            usm_decode_pos_kp1.set_shape({3, batch, kp1});
            if (use_vl) {
                decode_visual_kp1.set_shape({batch, kp1, static_cast<size_t>(cfg.text.hidden_size)});
                std::memset(decode_visual_kp1.data(), 0, decode_visual_kp1.get_byte_size());
                decode_visual_mask_kp1.set_shape({batch, kp1});
                std::memset(decode_visual_mask_kp1.data(), 0, decode_visual_mask_kp1.get_byte_size());
            }

            auto* ids_data = step_ids_kp1.data<int64_t>();
            for (size_t b = 0; b < batch; ++b) {
                ids_data[b * kp1] = next_id;
                for (int k = 0; k < K; ++k) {
                    ids_data[b * kp1 + static_cast<size_t>(k + 1)] = drafts[k];
                }
            }
            auto* pos_data = usm_decode_pos_kp1.data<int64_t>();
            for (size_t b = 0; b < batch; ++b) {
                for (size_t j = 0; j < kp1; ++j) {
                    const int64_t pos_val = (past_len + static_cast<int64_t>(j)) + rope_deltas_data[b];
                    for (size_t plane = 0; plane < 3; ++plane) {
                        pos_data[plane * batch * kp1 + b * kp1 + j] = pos_val;
                    }
                }
            }
            const size_t full_mask_len = static_cast<size_t>(past_len) + kp1;
            ov::Tensor step_mask_kp1 = make_usm_host_tensor(gpu_ctx, ov::element::i64, {batch, full_mask_len});
            {
                auto* p = step_mask_kp1.data<int64_t>();
                for (size_t i = 0; i < batch * full_mask_len; ++i) p[i] = 1;
                for (int64_t dpos : dead_positions) {
                    if (dpos >= 0 && static_cast<size_t>(dpos) < full_mask_len) {
                        for (size_t b = 0; b < batch; ++b) {
                            p[b * full_mask_len + static_cast<size_t>(dpos)] = 0;
                        }
                    }
                }
            }

            text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kInputIds, step_ids_kp1);
            text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kAttentionMask, step_mask_kp1);
            text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kPositionIds, usm_decode_pos_kp1);
            text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kBeamIdx, usm_beam_idx);
            if (use_vl) {
                text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualEmbeds, decode_visual_kp1);
                text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualPosMask, decode_visual_mask_kp1);
            }
            text_request.infer();
            mtp_main_infers++;
            past_len += static_cast<int64_t>(kp1);
        };

        // Sequential verify: K+1 individual single-token inferences.
        // Each uses q_len=1 → SINGLE_TOKEN SDPA kernel → baseline precision.
        // Stores logits and hidden_states per position for accept/reject.
        const size_t hidden_size = static_cast<size_t>(cfg.text.hidden_size);
        std::vector<std::vector<float>> seq_logits(max_kp1);
        std::vector<std::vector<float>> seq_hs(max_kp1);
        for (size_t j = 0; j < max_kp1; ++j) {
            seq_hs[j].resize(hidden_size);
        }

        // Inline sequential verify with early stopping.
        // Key insight: Qwen3.5 has recurrent/linear attention layers (linear_states)
        // that accumulate across tokens. KV trim only handles past_key_values but NOT
        // linear_states, so feeding rejected tokens corrupts the recurrence.
        // Solution: interleave inference and accept/reject. Stop as soon as a draft
        // is rejected. This way, only accepted tokens (and the correction) update
        // the linear_states, and no trim is ever needed.
        auto run_inline_seq_verify = [&](int& num_accepted, bool& stopped,
                                         std::vector<float>& last_hs) {
            // Tokens to verify: [next_id, drafts[0], ..., drafts[K-1]]
            const size_t kp1 = static_cast<size_t>(K + 1);
            std::vector<int64_t> verify_tokens(kp1);
            verify_tokens[0] = next_id;
            for (int k = 0; k < K; ++k) {
                verify_tokens[static_cast<size_t>(k + 1)] = drafts[k];
            }

            auto do_single_infer = [&](int64_t token) {
                auto* ids_data = step_ids.data<int64_t>();
                for (size_t b = 0; b < batch; ++b) ids_data[b] = token;

                auto* pos_data = usm_decode_pos.data<int64_t>();
                for (size_t b = 0; b < batch; ++b) {
                    const int64_t value = past_len + rope_deltas_data[b];
                    pos_data[b] = value;
                    pos_data[batch + b] = value;
                    pos_data[2 * batch + b] = value;
                }

                text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kInputIds, step_ids);
                text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kAttentionMask, step_mask);
                text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kPositionIds, usm_decode_pos);
                text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kBeamIdx, usm_beam_idx);
                if (use_vl) {
                    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualEmbeds, decode_visual);
                    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualPosMask, decode_visual_mask);
                }
                text_request.infer();
                past_len += 1;
            };

            num_accepted = 0;
            stopped = false;

            // Step 0: infer next_id (always needed)
            do_single_infer(next_id);
            extract_last_logits_f32(
                text_request.get_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kLogits), logit_buf);
            {
                ov::Tensor hs_t = text_request.get_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kHiddenStates);
                std::memcpy(last_hs.data(), hs_t.data<const float>(), hidden_size * sizeof(float));
            }

            // Verify drafts[0..K-1] with inline accept/reject
            for (int k = 0; k < K && !stopped; ++k) {
                // logit_buf already has the logits from the previous inference
                int64_t verified;
                {
                    ov::genai::Logits lw(logit_buf.data(), logit_buf.size());
                    penalty_processor.apply(lw);
                    verified = use_sampling
                                   ? sample_fast(logit_buf.data(), logit_buf.size(),
                                                 opts.temperature, opts.top_p, opts.top_k, rng, sampling_ctx)
                                   : argmax_f32(logit_buf);
                }
                mtp_attempts++;

                if (verified == drafts[k]) {
                    // ACCEPT: draft was correct
                    mtp_hits++;
                    penalty_processor.register_new_generated_token(verified);
                    generated.push_back(verified);
                    penalty_processor.update_generated_len(generated.size());
                    decode_steps++;
                    num_accepted++;

                    if (static_cast<int>(generated.size()) >= opts.max_new_tokens) { stopped = true; break; }
                    if (!stop_token_ids.empty() && stop_token_ids.count(verified) > 0) { stopped = true; break; }

                    // Infer the accepted draft token to get logits for next position
                    do_single_infer(drafts[k]);
                    extract_last_logits_f32(
                        text_request.get_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kLogits), logit_buf);
                    {
                        ov::Tensor hs_t = text_request.get_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kHiddenStates);
                        std::memcpy(last_hs.data(), hs_t.data<const float>(), hidden_size * sizeof(float));
                    }
                } else {
                    // REJECT: emit correction token, stop verification.
                    // We do NOT infer the wrong draft — linear_states stay clean.
                    penalty_processor.register_new_generated_token(verified);
                    generated.push_back(verified);
                    penalty_processor.update_generated_len(generated.size());
                    decode_steps++;
                    next_id = verified;

                    if (static_cast<int>(generated.size()) >= opts.max_new_tokens) { stopped = true; }
                    if (!stop_token_ids.empty() && stop_token_ids.count(verified) > 0) { stopped = true; }
                    break;
                }
            }

            // BONUS token: if all K drafts accepted, sample from last logits
            if (num_accepted == K && !stopped) {
                // logit_buf has the logits from the last accepted draft inference
                int64_t bonus;
                {
                    ov::genai::Logits lw(logit_buf.data(), logit_buf.size());
                    penalty_processor.apply(lw);
                    bonus = use_sampling
                                ? sample_fast(logit_buf.data(), logit_buf.size(),
                                              opts.temperature, opts.top_p, opts.top_k, rng, sampling_ctx)
                                : argmax_f32(logit_buf);
                }
                penalty_processor.register_new_generated_token(bonus);
                generated.push_back(bonus);
                penalty_processor.update_generated_len(generated.size());
                decode_steps++;
                next_id = bonus;

                if (static_cast<int>(generated.size()) >= opts.max_new_tokens) { stopped = true; }
                if (!stop_token_ids.empty() && stop_token_ids.count(bonus) > 0) { stopped = true; }
            }

            mtp_main_infers++;
        };

        // Buffer for hidden_states from inline verify
        std::vector<float> verify_hs(hidden_size);

        while (static_cast<int>(generated.size()) < opts.max_new_tokens) {
            if (!stop_token_ids.empty() && stop_token_ids.count(next_id) > 0) break;

            // === VERIFY PHASE ===
            auto t_v0 = std::chrono::steady_clock::now();

            int num_accepted = 0;
            bool stopped = false;

            // Per-step sub-timing (populated by batch verify path below)
            double step_snap_save = 0, step_main_infer = 0, step_accept = 0;
            double step_state_restore = 0, step_kv_trim = 0, step_reforward = 0;
            int trim_count = 0;

            if (use_seq_verify) {
                // Inline sequential verify: interleaves inference + accept/reject.
                // Early stop on rejection → no KV trim, no linear_state corruption.
                run_inline_seq_verify(num_accepted, stopped, verify_hs);
            } else {
                // Batch verify: send K+1 tokens in one main model inference.
                // Two sub-modes for state fixup on rejection:
                //   kernel snapshot: per-token state selection from GPU kernel + physical KV trim
                //                    (vLLM-style: spec_state_indices_tensor equivalent)
                //   fallback:        save/restore linear_states + full KV rollback + re-forward
                const int64_t past_len_before = past_len;
                const int64_t original_next_id = next_id;

                // Snapshot linear_states before batch verify.
                // With kernel snapshots: not needed (per-token restore from output).
                // Without kernel snapshots: needed for rollback+replay on rejection.
                // Mode 8 (re-forward): needs pre-batch snapshot even with kernel snapshots,
                // because re-forward must start from pre-batch state (not post-accepted).
                // Periodic refresh also needs pre-batch snapshot for checkpoint rollback.
                auto t_snap_save0 = std::chrono::steady_clock::now();
                const bool needs_linear_snap = !has_kernel_snapshot || (snapshot_restore_mode & 8) || REFRESH_INTERVAL > 0;
                if (needs_linear_snap) {
                    save_linear_states(text_request, linear_snap);
                }

                // Save conv states before batch verify for CPU-based restore on rejection.
                // Only needed when mode 4 is active (CPU conv restore). In mode 3
                // (kernel snapshot restore for both linear + conv), the CPU conv_snap
                // is never read — conv states are restored from GPU kernel snapshots.
                // Skipping saves ~1ms/step of GPU→CPU sync overhead (2 conv layers).
                const bool needs_conv_snap = has_kernel_snapshot && !conv_snap.empty() &&
                                             (snapshot_restore_mode & 4);
                if (needs_conv_snap) {
                    save_conv_states(text_request, conv_snap);
                }
                auto t_snap_save1 = std::chrono::steady_clock::now();
                step_snap_save = elapsed_ms(t_snap_save0, t_snap_save1);
                time_snapshot_ms += step_snap_save;
                time_snapshot_save_ms += step_snap_save;

                // Batch verify: send K+1 tokens at once (main model GPU inference)
                auto t_main_infer0 = std::chrono::steady_clock::now();
                run_kp1_verify();
                auto t_main_infer1 = std::chrono::steady_clock::now();
                step_main_infer = elapsed_ms(t_main_infer0, t_main_infer1);
                time_main_infer_ms += step_main_infer;
                logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kLogits);
                main_hidden_states = text_request.get_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kHiddenStates);

                // DIAGNOSTIC: compare snapshot[last] vs final state variable for first linear layer
                {
                    static int diag_count = 0;
                    if (diag_count < 5 && !all_linear_states_names.empty()) {
                        try {
                            // Get snapshot tensor (f32, post-reorder)
                            ov::Tensor snap_t = text_request.get_tensor(all_linear_states_names[0]);
                            auto snap_shape = snap_t.get_shape();
                            size_t T_dim = snap_shape[1];
                            size_t state_elems = 1;
                            for (size_t d = 2; d < snap_shape.size(); ++d) state_elems *= snap_shape[d];

                            // Find the matching recurrent state variable
                            auto lpos = all_linear_states_names[0].find("layer");
                            int snap_layer_idx = std::stoi(all_linear_states_names[0].substr(lpos + 5));
                            std::string target_var = "linear_states." + std::to_string(snap_layer_idx) + ".recurrent";
                            ov::Tensor var_t;
                            for (auto& state : text_request.query_state()) {
                                if (state.get_name() == target_var) {
                                    var_t = state.get_state();
                                    break;
                                }
                            }

                            if (var_t) {
                                auto var_et = var_t.get_element_type();
                                auto var_shape = var_t.get_shape();
                                size_t var_elems = 1;
                                for (auto d : var_shape) var_elems *= d;
                                size_t cmp_elems = std::min(state_elems, var_elems);

                                // Convert variable to f32 for comparison
                                std::vector<float> var_f32(cmp_elems);
                                if (var_et == ov::element::f16) {
                                    const auto* src = reinterpret_cast<const ov::float16*>(var_t.data());
                                    for (size_t e = 0; e < cmp_elems; ++e)
                                        var_f32[e] = float(src[e]);
                                } else if (var_et == ov::element::f32) {
                                    const float* src = var_t.data<float>();
                                    std::copy(src, src + cmp_elems, var_f32.begin());
                                } else {
                                    fprintf(stderr, "[SNAP_DIAG] unsupported var type: %s\n", var_et.get_type_name().c_str());
                                    diag_count++;
                                    goto diag_done;
                                }

                                const float* snap_data = snap_t.data<float>();
                                size_t last_pos = T_dim - 1;
                                const float* snap_last = snap_data + last_pos * state_elems;

                                double max_diff = 0.0, sum_sq = 0.0;
                                size_t mismatches = 0;
                                for (size_t e = 0; e < cmp_elems; ++e) {
                                    double d = std::abs((double)snap_last[e] - (double)var_f32[e]);
                                    if (d > 0.0) mismatches++;
                                    max_diff = std::max(max_diff, d);
                                    sum_sq += d * d;
                                }
                                double rmse = std::sqrt(sum_sq / cmp_elems);

                                fprintf(stderr, "[SNAP_DIAG] iter=%d layer=%d var_et=%s "
                                    "snap[last=%zu] vs var: max_diff=%.6e rmse=%.6e mismatches=%zu/%zu\n",
                                    diag_count, snap_layer_idx, var_et.get_type_name().c_str(),
                                    last_pos, max_diff, rmse, mismatches, cmp_elems);

                                // Print first 8 values from each
                                fprintf(stderr, "[SNAP_DIAG] snap[last]: ");
                                for (size_t e = 0; e < std::min((size_t)8, cmp_elems); ++e)
                                    fprintf(stderr, "%.6f ", snap_last[e]);
                                fprintf(stderr, "\n[SNAP_DIAG] var(f32):   ");
                                for (size_t e = 0; e < std::min((size_t)8, cmp_elems); ++e)
                                    fprintf(stderr, "%.6f ", var_f32[e]);
                                fprintf(stderr, "\n");

                                // Compare snap[0] vs snap[last]
                                if (T_dim > 1) {
                                    double max01 = 0.0;
                                    for (size_t e = 0; e < cmp_elems; ++e) {
                                        double d = std::abs((double)snap_data[e] - (double)snap_last[e]);
                                        max01 = std::max(max01, d);
                                    }
                                    fprintf(stderr, "[SNAP_DIAG] snap[0] vs snap[last]: max_diff=%.6e\n", max01);
                                }
                            }
                            diag_done:;
                        } catch (const std::exception& e) {
                            fprintf(stderr, "[SNAP_DIAG] EXCEPTION: %s\n", e.what());
                        }
                        diag_count++;
                    }
                }

                // Accept/reject phase for batch verify
                auto t_accept0 = std::chrono::steady_clock::now();
                for (int k = 0; k < K && !stopped; ++k) {
                    extract_logits_at_pos_f32(logits, static_cast<size_t>(k), logit_buf);
                    int64_t verified;
                    {
                        ov::genai::Logits lw(logit_buf.data(), logit_buf.size());
                        penalty_processor.apply(lw);
                        verified = use_sampling
                                       ? sample_fast(logit_buf.data(), logit_buf.size(),
                                                     opts.temperature, opts.top_p, opts.top_k, rng, sampling_ctx)
                                       : argmax_f32(logit_buf);
                    }
                    mtp_attempts++;

                    // Debug: trace token-level accept/reject
                    {
                        static int dbg_step = 0;
                        if (dbg_step < 300) {
                            fprintf(stderr, "[MTP_TRACE] step=%d k=%d verified=%lld draft=%lld %s past_len=%lld\n",
                                dbg_step, k, (long long)verified, (long long)drafts[k],
                                (verified == drafts[k]) ? "ACCEPT" : "REJECT",
                                (long long)past_len);
                        }
                        dbg_step++;
                    }

                    if (verified == drafts[k]) {
                        mtp_hits++;
                        penalty_processor.register_new_generated_token(verified);
                        generated.push_back(verified);
                        penalty_processor.update_generated_len(generated.size());
                        decode_steps++;
                        num_accepted++;
                        if (static_cast<int>(generated.size()) >= opts.max_new_tokens) { stopped = true; }
                        if (!stop_token_ids.empty() && stop_token_ids.count(verified) > 0) { stopped = true; }
                    } else {
                        penalty_processor.register_new_generated_token(verified);
                        generated.push_back(verified);
                        penalty_processor.update_generated_len(generated.size());
                        decode_steps++;
                        next_id = verified;
                        if (static_cast<int>(generated.size()) >= opts.max_new_tokens) { stopped = true; }
                        if (!stop_token_ids.empty() && stop_token_ids.count(verified) > 0) { stopped = true; }
                        break;
                    }
                }

                // Bonus token for batch verify
                if (num_accepted == K && !stopped) {
                    extract_logits_at_pos_f32(logits, static_cast<size_t>(K), logit_buf);
                    int64_t bonus;
                    {
                        ov::genai::Logits lw(logit_buf.data(), logit_buf.size());
                        penalty_processor.apply(lw);
                        bonus = use_sampling
                                    ? sample_fast(logit_buf.data(), logit_buf.size(),
                                                  opts.temperature, opts.top_p, opts.top_k, rng, sampling_ctx)
                                    : argmax_f32(logit_buf);
                    }
                    fprintf(stderr, "[MTP_TRACE] bonus=%lld (all K accepted) gen_size=%d\n",
                        (long long)bonus, (int)generated.size());
                    penalty_processor.register_new_generated_token(bonus);
                    generated.push_back(bonus);
                    penalty_processor.update_generated_len(generated.size());
                    decode_steps++;
                    next_id = bonus;
                    if (static_cast<int>(generated.size()) >= opts.max_new_tokens) { stopped = true; }
                    if (!stop_token_ids.empty() && stop_token_ids.count(bonus) > 0) { stopped = true; }
                }

                // State fixup on rejection
                auto t_accept1 = std::chrono::steady_clock::now();
                step_accept = elapsed_ms(t_accept0, t_accept1);
                time_accept_check_ms += step_accept;

                bool did_reforward_this_cycle = false;
                trim_count = K - num_accepted;
                if (trim_count > 0) {
                    if (has_kernel_snapshot) {
                        // vLLM-style per-token state restore + physical KV trim.
                        // The GPU LinearAttention kernel wrote intermediate recurrent states
                        // for each token in the K+1 batch (like vLLM's spec_state_indices_tensor).
                        // Select the state at num_accepted position (= state after processing
                        // next_id + num_accepted accepted drafts), then trim rejected KV entries.
                        auto tt0 = std::chrono::steady_clock::now();

                        if (snapshot_restore_mode & 1) {
                            static const bool gpu_restore = []() {
                                auto* env = std::getenv("OV_GENAI_GPU_RESTORE");
                                return env == nullptr || std::string(env) != "0";
                            }();
                            select_and_restore_linear_states(text_request, all_linear_states_names, num_accepted, gpu_restore);
                        }
                        if ((snapshot_restore_mode & 2) && !all_conv_states_names.empty()) {
                            static const bool gpu_restore_conv = []() {
                                auto* env = std::getenv("OV_GENAI_GPU_RESTORE_CONV");
                                if (env) return std::string(env) != "0";
                                env = std::getenv("OV_GENAI_GPU_RESTORE");
                                return env == nullptr || std::string(env) != "0";
                            }();
                            select_and_restore_conv_states(text_request, all_conv_states_names, num_accepted, gpu_restore_conv);
                        }
                        // Mode 4: restore conv states from CPU pre-batch snapshot (instead of kernel snapshot)
                        if ((snapshot_restore_mode & 4) && !conv_snap.empty()) {
                            restore_conv_states(text_request, conv_snap);
                        }

                        auto tt_restore1 = std::chrono::steady_clock::now();
                        step_state_restore = elapsed_ms(tt0, tt_restore1);
                        time_restore_ms += step_state_restore;
                        time_state_restore_ms += step_state_restore;

                        // Mode 8: re-forward after restoring PRE-BATCH states to regenerate
                        // KV entries and recurrent states from scratch (like fallback path).
                        // Must use pre-batch linear_snap (not kernel snapshot at num_accepted),
                        // because re-forward reprocesses accepted tokens from scratch.
                        if (snapshot_restore_mode & 8) {
                            // Restore linear states to PRE-BATCH state (overrides any bit-1 restore above)
                            restore_linear_states(text_request, linear_snap);

                            // Trim ALL K+1 entries from KV cache, then re-forward accepted tokens.
                            {
                                auto tt_trim0 = std::chrono::steady_clock::now();
                                const size_t step_kp1 = static_cast<size_t>(K + 1);
                                trim_kv_cache_states_gpu(text_request, step_kp1);
                                auto tt_trim1 = std::chrono::steady_clock::now();
                                double dt = elapsed_ms(tt_trim0, tt_trim1);
                                time_trim_ms += dt; count_trims++;
                                step_kv_trim += dt;
                                time_kv_trim_ms += dt;
                            }
                            past_len = past_len_before;

                            const size_t replay_n = static_cast<size_t>(num_accepted) + 1;
                            {
                                ov::Tensor replay_ids = make_usm_host_tensor(gpu_ctx, ov::element::i64, {batch, replay_n});
                                auto* rids = replay_ids.data<int64_t>();
                                for (size_t b = 0; b < batch; ++b) {
                                    rids[b * replay_n] = original_next_id;
                                    for (int k = 0; k < num_accepted; ++k) {
                                        rids[b * replay_n + static_cast<size_t>(k + 1)] = drafts[k];
                                    }
                                }

                                ov::Tensor replay_pos = make_usm_host_tensor(gpu_ctx, ov::element::i64, {3, batch, replay_n});
                                auto* rpos = replay_pos.data<int64_t>();
                                for (size_t b = 0; b < batch; ++b) {
                                    for (size_t j = 0; j < replay_n; ++j) {
                                        const int64_t val = (past_len + static_cast<int64_t>(j)) + rope_deltas_data[b];
                                        for (size_t plane = 0; plane < 3; ++plane) {
                                            rpos[plane * batch * replay_n + b * replay_n + j] = val;
                                        }
                                    }
                                }

                                const size_t replay_mask_len = static_cast<size_t>(past_len) + replay_n;
                                ov::Tensor replay_mask = make_usm_host_tensor(gpu_ctx, ov::element::i64, {batch, replay_mask_len});
                                {
                                    auto* p = replay_mask.data<int64_t>();
                                    for (size_t i = 0; i < batch * replay_mask_len; ++i) p[i] = 1;
                                }

                                text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kInputIds, replay_ids);
                                text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kAttentionMask, replay_mask);
                                text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kPositionIds, replay_pos);
                                text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kBeamIdx, usm_beam_idx);
                                if (use_vl) {
                                    ov::Tensor replay_vis = make_usm_host_tensor(gpu_ctx, ov::element::f32,
                                        {batch, replay_n, static_cast<size_t>(cfg.text.hidden_size)});
                                    std::memset(replay_vis.data(), 0, replay_vis.get_byte_size());
                                    ov::Tensor replay_vis_mask = make_usm_host_tensor(gpu_ctx, ov::element::boolean, {batch, replay_n});
                                    std::memset(replay_vis_mask.data(), 0, replay_vis_mask.get_byte_size());
                                    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualEmbeds, replay_vis);
                                    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualPosMask, replay_vis_mask);
                                }

                                text_request.infer();
                                past_len += static_cast<int64_t>(replay_n);
                                mtp_main_infers++;

                                main_hidden_states = text_request.get_tensor(
                                    ov::genai::modeling::models::Qwen3_5TextIO::kHiddenStates);
                            }
                            count_reforwards++;

                            // Re-forward refreshed KV + states — reset checkpoint.
                            // Accepted drafts are already in re-forward KV, so only
                            // add next_id (correction/bonus) to tracking below.
                            if (needs_refresh_tracking) {
                                save_linear_states(text_request, linear_snap);
                                checkpoint_past_len = past_len;
                                tokens_since_checkpoint.clear();
                                did_reforward_this_cycle = true;
                            }
                        } else {
                            // Physical KV trim: remove rejected entries (like vLLM's seq_lens
                            // adjustment). Avoids RoPE position gaps from dead entries.
                            // Uses GPU-side zero-copy trim (metadata reinterpret, no data copy).
                            {
                                auto tt_trim0 = std::chrono::steady_clock::now();
                                trim_kv_cache_states_gpu(text_request, static_cast<size_t>(trim_count));
                                auto tt_trim1 = std::chrono::steady_clock::now();
                                double dt = elapsed_ms(tt_trim0, tt_trim1);
                                time_trim_ms += dt; count_trims++;
                                step_kv_trim += dt;
                                time_kv_trim_ms += dt;
                            }
                            past_len = past_len_before + 1 + static_cast<int64_t>(num_accepted);
                        }

                        // VALIDATE_SNAPSHOT: compare kernel-snapshot states vs fallback re-forward states
                        static const bool validate_snapshot = []() {
                            auto* env = std::getenv("OV_GENAI_VALIDATE_SNAPSHOT");
                            return env && std::string(env) == "1";
                        }();
                        if (validate_snapshot) {
                            // 1. Save snapshot-restored states to CPU
                            std::vector<std::pair<std::string, ov::Tensor>> snap_states;
                            for (auto& state : text_request.query_state()) {
                                const auto& sname = state.get_name();
                                if (sname.find("linear_states.") == std::string::npos) continue;
                                if (sname.find(".recurrent") == std::string::npos && sname.find(".conv") == std::string::npos) continue;
                                ov::Tensor t = state.get_state();
                                ov::Tensor copy(t.get_element_type(), t.get_shape());
                                t.copy_to(copy);
                                snap_states.emplace_back(sname, copy);
                            }

                            // 2. Undo: restore pre-batch linear states + full KV rollback + re-forward
                            restore_linear_states(text_request, linear_snap);
                            // Undo the physical KV trim: we need to trim all kp1 entries
                            // But we already trimmed `trim_count`, so now trim the remaining `num_accepted+1`
                            trim_kv_cache_states(text_request, static_cast<size_t>(num_accepted) + 1);
                            auto refw_past_len = past_len_before;
                            const size_t replay_n = static_cast<size_t>(num_accepted) + 1;
                            {
                                ov::Tensor replay_ids = make_usm_host_tensor(gpu_ctx, ov::element::i64, {batch, replay_n});
                                auto* rids = replay_ids.data<int64_t>();
                                for (size_t b2 = 0; b2 < batch; ++b2) {
                                    rids[b2 * replay_n] = original_next_id;
                                    for (int k2 = 0; k2 < num_accepted; ++k2) {
                                        rids[b2 * replay_n + static_cast<size_t>(k2 + 1)] = drafts[k2];
                                    }
                                }
                                ov::Tensor replay_pos = make_usm_host_tensor(gpu_ctx, ov::element::i64, {3, batch, replay_n});
                                auto* rpos = replay_pos.data<int64_t>();
                                for (size_t b2 = 0; b2 < batch; ++b2) {
                                    for (size_t j2 = 0; j2 < replay_n; ++j2) {
                                        const int64_t val = (refw_past_len + static_cast<int64_t>(j2)) + rope_deltas_data[b2];
                                        for (size_t plane = 0; plane < 3; ++plane) {
                                            rpos[plane * batch * replay_n + b2 * replay_n + j2] = val;
                                        }
                                    }
                                }
                                const size_t replay_mask_len = static_cast<size_t>(refw_past_len) + replay_n;
                                ov::Tensor replay_mask = make_usm_host_tensor(gpu_ctx, ov::element::i64, {batch, replay_mask_len});
                                {
                                    auto* p = replay_mask.data<int64_t>();
                                    for (size_t i2 = 0; i2 < batch * replay_mask_len; ++i2) p[i2] = 1;
                                }
                                text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kInputIds, replay_ids);
                                text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kAttentionMask, replay_mask);
                                text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kPositionIds, replay_pos);
                                text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kBeamIdx, usm_beam_idx);
                                if (use_vl) {
                                    ov::Tensor replay_vis = make_usm_host_tensor(gpu_ctx, ov::element::f32,
                                        {batch, replay_n, static_cast<size_t>(cfg.text.hidden_size)});
                                    std::memset(replay_vis.data(), 0, replay_vis.get_byte_size());
                                    ov::Tensor replay_vis_mask = make_usm_host_tensor(gpu_ctx, ov::element::boolean, {batch, replay_n});
                                    std::memset(replay_vis_mask.data(), 0, replay_vis_mask.get_byte_size());
                                    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualEmbeds, replay_vis);
                                    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualPosMask, replay_vis_mask);
                                }
                                text_request.infer();
                                past_len = refw_past_len + static_cast<int64_t>(replay_n);
                                main_hidden_states = text_request.get_tensor(
                                    ov::genai::modeling::models::Qwen3_5TextIO::kHiddenStates);
                            }

                            // 3. Compare re-forward states vs snapshot states
                            size_t si = 0;
                            for (auto& state : text_request.query_state()) {
                                const auto& sname = state.get_name();
                                if (sname.find("linear_states.") == std::string::npos) continue;
                                if (sname.find(".recurrent") == std::string::npos && sname.find(".conv") == std::string::npos) continue;
                                if (si >= snap_states.size()) break;
                                const auto& [s_name, s_tensor] = snap_states[si];
                                if (s_name != sname) { si++; continue; }
                                ov::Tensor refw_tensor = state.get_state();
                                // Compare f16 values
                                const size_t n = s_tensor.get_size();
                                const auto* sp = reinterpret_cast<const ov::float16*>(s_tensor.data());
                                const auto* rp = reinterpret_cast<const ov::float16*>(refw_tensor.data());
                                float max_d = 0; size_t mismatch = 0;
                                for (size_t e = 0; e < n; ++e) {
                                    float diff = std::abs(float(sp[e]) - float(rp[e]));
                                    if (diff > max_d) max_d = diff;
                                    if (sp[e] != rp[e]) mismatch++;
                                }
                                if (mismatch > 0) {
                                    fprintf(stderr, "[VALIDATE] %s: max_diff=%.6e mismatches=%zu/%zu\n",
                                        sname.c_str(), max_d, mismatch, n);
                                }
                                si++;
                            }
                        }

                        auto tt1 = std::chrono::steady_clock::now();
                        time_restore_ms += elapsed_ms(tt0, tt1); count_restores++;
                    } else {
                        // Snapshot mode: restore linear_states + full KV rollback + re-forward
                        {
                            auto tr0 = std::chrono::steady_clock::now();
                            restore_linear_states(text_request, linear_snap);
                            auto tr1 = std::chrono::steady_clock::now();
                            time_restore_ms += elapsed_ms(tr0, tr1);
                        }
                        count_restores++;

                        {
                            auto tt0 = std::chrono::steady_clock::now();
                            trim_kv_cache_states(text_request, static_cast<size_t>(K + 1));
                            auto tt1 = std::chrono::steady_clock::now();
                            time_trim_ms += elapsed_ms(tt0, tt1); count_trims++;
                        }
                        past_len = past_len_before;

                        const size_t replay_n = static_cast<size_t>(num_accepted) + 1;
                        {
                            auto trf0 = std::chrono::steady_clock::now();

                            ov::Tensor replay_ids = make_usm_host_tensor(gpu_ctx, ov::element::i64, {batch, replay_n});
                            auto* rids = replay_ids.data<int64_t>();
                            for (size_t b = 0; b < batch; ++b) {
                                rids[b * replay_n] = original_next_id;
                                for (int k = 0; k < num_accepted; ++k) {
                                    rids[b * replay_n + static_cast<size_t>(k + 1)] = drafts[k];
                                }
                            }

                            ov::Tensor replay_pos = make_usm_host_tensor(gpu_ctx, ov::element::i64, {3, batch, replay_n});
                            auto* rpos = replay_pos.data<int64_t>();
                            for (size_t b = 0; b < batch; ++b) {
                                for (size_t j = 0; j < replay_n; ++j) {
                                    const int64_t val = (past_len + static_cast<int64_t>(j)) + rope_deltas_data[b];
                                    for (size_t plane = 0; plane < 3; ++plane) {
                                        rpos[plane * batch * replay_n + b * replay_n + j] = val;
                                    }
                                }
                            }

                            const size_t replay_mask_len = static_cast<size_t>(past_len) + replay_n;
                            ov::Tensor replay_mask = make_usm_host_tensor(gpu_ctx, ov::element::i64, {batch, replay_mask_len});
                            {
                                auto* p = replay_mask.data<int64_t>();
                                for (size_t i = 0; i < batch * replay_mask_len; ++i) p[i] = 1;
                            }

                            text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kInputIds, replay_ids);
                            text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kAttentionMask, replay_mask);
                            text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kPositionIds, replay_pos);
                            text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kBeamIdx, usm_beam_idx);
                            if (use_vl) {
                                ov::Tensor replay_vis = make_usm_host_tensor(gpu_ctx, ov::element::f32,
                                    {batch, replay_n, static_cast<size_t>(cfg.text.hidden_size)});
                                std::memset(replay_vis.data(), 0, replay_vis.get_byte_size());
                                ov::Tensor replay_vis_mask = make_usm_host_tensor(gpu_ctx, ov::element::boolean, {batch, replay_n});
                                std::memset(replay_vis_mask.data(), 0, replay_vis_mask.get_byte_size());
                                text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualEmbeds, replay_vis);
                                text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualPosMask, replay_vis_mask);
                            }

                            text_request.infer();
                            past_len += static_cast<int64_t>(replay_n);
                            mtp_main_infers++;

                            main_hidden_states = text_request.get_tensor(
                                ov::genai::modeling::models::Qwen3_5TextIO::kHiddenStates);

                            auto trf1 = std::chrono::steady_clock::now();
                            double dt = elapsed_ms(trf0, trf1);
                            time_reforward_ms += dt;
                            step_reforward += dt;
                        }
                        count_reforwards++;
                    }
                }

                // Track emitted tokens for periodic state refresh
                if (needs_refresh_tracking) {
                    // After mode-8 re-forward, checkpoint was reset and accepted drafts
                    // are already in the refreshed KV — only add next_id (correction/bonus).
                    if (!did_reforward_this_cycle) {
                        for (int k = 0; k < num_accepted; ++k) {
                            tokens_since_checkpoint.push_back(drafts[k]);
                        }
                    }
                    tokens_since_checkpoint.push_back(next_id);
                }
            }

            auto t_v1 = std::chrono::steady_clock::now();
            time_verify_ms += elapsed_ms(t_v0, t_v1); count_verify_infers++;

            if (stopped) break;

            // === STATE REFRESH: periodic linear_states + KV correction ===
            // Without kernel snapshots: corrects linear_states drift from rejected tokens.
            // With mode 8: corrects KV cache drift from batch=K+1 numerical differences.
            // Every N generated tokens, restore linear_states to last checkpoint,
            // trim KV, re-forward all tokens since checkpoint as a batch.
            if (needs_refresh_tracking && REFRESH_INTERVAL > 0 &&
                !tokens_since_checkpoint.empty() &&
                static_cast<int>(tokens_since_checkpoint.size()) >= REFRESH_INTERVAL) {
                auto t_rf0 = std::chrono::steady_clock::now();

                // 1. Restore linear_states to last checkpoint
                restore_linear_states(text_request, linear_snap);

                // 2. Trim KV back to checkpoint position
                const size_t trim_amount = static_cast<size_t>(past_len - checkpoint_past_len);
                if (trim_amount > 0) {
                    if (has_kernel_snapshot)
                        trim_kv_cache_states_gpu(text_request, trim_amount);
                    else
                        trim_kv_cache_states(text_request, trim_amount);
                    past_len = checkpoint_past_len;
                }
                // After physical trim, all dead entries are removed — clear tracker.
                dead_positions.clear();

                // 3. Re-forward all COMMITTED tokens (exclude last = next_id,
                //    which will be fed by the next batch verify cycle).
                //    tokens_since_checkpoint layout: [t0, t1, ..., tN-2, next_id]
                //    replay: [t0, t1, ..., tN-2]
                const size_t replay_n = tokens_since_checkpoint.size() - 1;
                if (replay_n > 0) {
                    ov::Tensor replay_ids = make_usm_host_tensor(gpu_ctx, ov::element::i64, {batch, replay_n});
                    auto* rids = replay_ids.data<int64_t>();
                    for (size_t b = 0; b < batch; ++b) {
                        for (size_t j = 0; j < replay_n; ++j) {
                            rids[b * replay_n + j] = tokens_since_checkpoint[j];
                        }
                    }

                    ov::Tensor replay_pos = make_usm_host_tensor(gpu_ctx, ov::element::i64, {3, batch, replay_n});
                    auto* rpos = replay_pos.data<int64_t>();
                    for (size_t b = 0; b < batch; ++b) {
                        for (size_t j = 0; j < replay_n; ++j) {
                            const int64_t val = (past_len + static_cast<int64_t>(j)) + rope_deltas_data[b];
                            for (size_t plane = 0; plane < 3; ++plane) {
                                rpos[plane * batch * replay_n + b * replay_n + j] = val;
                            }
                        }
                    }

                    const size_t replay_mask_len = static_cast<size_t>(past_len) + replay_n;
                    ov::Tensor replay_mask = make_usm_host_tensor(gpu_ctx, ov::element::i64, {batch, replay_mask_len});
                    {
                        auto* p = replay_mask.data<int64_t>();
                        for (size_t i = 0; i < batch * replay_mask_len; ++i) p[i] = 1;
                    }

                    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kInputIds, replay_ids);
                    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kAttentionMask, replay_mask);
                    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kPositionIds, replay_pos);
                    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kBeamIdx, usm_beam_idx);
                    if (use_vl) {
                        ov::Tensor replay_vis = make_usm_host_tensor(gpu_ctx, ov::element::f32,
                            {batch, replay_n, static_cast<size_t>(cfg.text.hidden_size)});
                        std::memset(replay_vis.data(), 0, replay_vis.get_byte_size());
                        ov::Tensor replay_vis_mask = make_usm_host_tensor(gpu_ctx, ov::element::boolean, {batch, replay_n});
                        std::memset(replay_vis_mask.data(), 0, replay_vis_mask.get_byte_size());
                        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualEmbeds, replay_vis);
                        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualPosMask, replay_vis_mask);
                    }

                    text_request.infer();
                    past_len += static_cast<int64_t>(replay_n);
                    mtp_main_infers++;

                    // Update hidden_states for draft generation
                    main_hidden_states = text_request.get_tensor(
                        ov::genai::modeling::models::Qwen3_5TextIO::kHiddenStates);
                }

                // 4. New checkpoint: save corrected linear_states
                save_linear_states(text_request, linear_snap);
                checkpoint_past_len = past_len;
                tokens_since_checkpoint.clear();
                tokens_since_checkpoint.push_back(next_id);

                // Update num_accepted so draft generation uses correct hs position
                // from the re-forward output (last position = replay_n - 1)
                if (replay_n > 0) {
                    num_accepted = static_cast<int>(replay_n) - 1;
                }

                auto t_rf1 = std::chrono::steady_clock::now();
                time_refresh_ms += elapsed_ms(t_rf0, t_rf1);
                count_refreshes++;
            }

            // === DRAFT PHASE: generate K drafts for next iteration ===
            auto t_d0 = std::chrono::steady_clock::now();
            if (use_seq_verify) {
                // In inline sequential mode, verify_hs has the hidden_states from the
                // last inference step (which processed the last accepted draft or next_id).
                ov::Tensor hs_for_draft(ov::element::f32, {batch, 1, hidden_size}, verify_hs.data());
                generate_k_drafts(hs_for_draft, 0);
            } else {
                generate_k_drafts(main_hidden_states, static_cast<size_t>(num_accepted));
            }
            auto t_d1 = std::chrono::steady_clock::now();
            double step_draft = elapsed_ms(t_d0, t_d1);
            time_draft_ms += step_draft; count_draft_infers += K;
            time_mtp_reset_ms += step_mtp_reset_ms;
            for (int k = 0; k < K; ++k) {
                time_mtp_each_ms[static_cast<size_t>(k)] += step_mtp_k_ms[static_cast<size_t>(k)];
                count_mtp_each[static_cast<size_t>(k)]++;
            }
            step_counter++;

            // === Phase C: Adaptive K selection ===
            // After each step, update rolling accept history and possibly adjust K.
            if (adaptive_k_enabled) {
                // Record this step's accept stats
                adaptive_history.push_back({num_accepted, K});
                if (static_cast<int>(adaptive_history.size()) > adaptive_window) {
                    adaptive_history.pop_front();
                }
                if (adaptive_cooldown_counter > 0) {
                    adaptive_cooldown_counter--;
                }

                // Only adapt after warmup and cooldown
                if (static_cast<int>(adaptive_history.size()) >= adaptive_warmup &&
                    adaptive_cooldown_counter == 0) {
                    // Compute rolling accept rate over window
                    int total_hits = 0, total_drafted = 0;
                    for (const auto& [h, d] : adaptive_history) {
                        total_hits += h;
                        total_drafted += d;
                    }
                    const double rolling_rate = total_drafted > 0
                        ? static_cast<double>(total_hits) / static_cast<double>(total_drafted) : 0.0;

                    const int old_K = K;
                    if (rolling_rate > adaptive_up_threshold && K < max_K) {
                        K++;
                        adaptive_cooldown_counter = adaptive_cooldown;
                        adaptive_k_changes++;
                    } else if (rolling_rate < adaptive_down_threshold && K > 1) {
                        K--;
                        adaptive_cooldown_counter = adaptive_cooldown;
                        adaptive_k_changes++;
                    }
                    if (K != old_K) {
                        fprintf(stderr, "[ADAPTIVE_K] step=%zu rate=%.1f%% K=%d→%d (window=%zu)\n",
                            step_counter, rolling_rate * 100.0, old_K, K,
                            adaptive_history.size());
                    }
                }
            }

            // Per-step profiling output (to stderr, gated by OV_GENAI_STEP_PROFILE)
            if (step_profile_enabled) {
                auto t_step_end = std::chrono::steady_clock::now();
                double step_total = elapsed_ms(t_v0, t_step_end);
                double step_verify = elapsed_ms(t_v0, t_d0);  // everything before draft
                fprintf(stderr, "[STEP %3zu] total=%.1fms | VERIFY=%.1fms (save=%.1f infer=%.1f accept=%.1f",
                    step_counter, step_total, step_verify, step_snap_save, step_main_infer, step_accept);
                if (trim_count > 0) {
                    fprintf(stderr, " restore=%.1f trim=%.1f", step_state_restore, step_kv_trim);
                    if (step_reforward > 0) fprintf(stderr, " refwd=%.1f", step_reforward);
                }
                fprintf(stderr, ") | DRAFT=%.1fms (reset=%.1f", step_draft, step_mtp_reset_ms);
                for (int k = 0; k < K; ++k) {
                    fprintf(stderr, " k%d=%.1f", k, step_mtp_k_ms[static_cast<size_t>(k)]);
                }
                fprintf(stderr, ") | accepted=%d/%d gen=%zu\n",
                    num_accepted, K, generated.size());
            }
        }
    } else {
        // =======================================================================
        // Normal decode loop (no MTP)
        // =======================================================================
        for (int step = 1; step < opts.max_new_tokens; ++step) {
            if (!stop_token_ids.empty() && stop_token_ids.count(next_id) > 0) {
                break;
            }
            auto* step_data = step_ids.data<int64_t>();
            for (size_t b = 0; b < batch; ++b) {
                step_data[b] = next_id;
            }

            auto* pos_data = usm_decode_pos.data<int64_t>();
            for (size_t b = 0; b < batch; ++b) {
                const int64_t value = past_len + rope_deltas_data[b];
                pos_data[b] = value;
                pos_data[batch + b] = value;
                pos_data[2 * batch + b] = value;
            }

            text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kInputIds, step_ids);
            text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kAttentionMask, step_mask);
            text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kPositionIds, usm_decode_pos);
            text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kBeamIdx, usm_beam_idx);
            if (use_vl) {
                text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualEmbeds, decode_visual);
                text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualPosMask, decode_visual_mask);
            }

            text_request.infer();
            logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kLogits);
            extract_last_logits_f32(logits, logit_buf);
            {
                ov::genai::Logits lw(logit_buf.data(), logit_buf.size());
                penalty_processor.apply(lw);
                next_id = use_sampling
                              ? sample_fast(logit_buf.data(), logit_buf.size(),
                                            opts.temperature, opts.top_p, opts.top_k, rng, sampling_ctx)
                              : argmax_f32(logit_buf);
            }

            penalty_processor.register_new_generated_token(next_id);
            generated.push_back(next_id);
            penalty_processor.update_generated_len(generated.size());
            decode_steps += 1;
            past_len += 1;
        }
    }
    const auto decode_end = std::chrono::steady_clock::now();

    const double ttft_ms = elapsed_ms(prefill_start, prefill_end);
    const double decode_ms = elapsed_ms(decode_start, decode_end);
    const double tpot_ms = decode_steps > 0 ? (decode_ms / static_cast<double>(decode_steps)) : 0.0;
    const double throughput = decode_steps > 0 && decode_ms > 0.0
                                  ? (static_cast<double>(decode_steps) * 1000.0 / decode_ms)
                                  : 0.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Mode: " << (use_dummy_mode_flag ? "dummy" : "hf") << " / " << opts.mode << std::endl;
    if (use_dummy_mode_flag) {
        std::cout << "Dummy text arch: " << (cfg.text.is_moe_enabled() ? "moe" : "dense") << std::endl;
    }
    std::cout << "Prompt token size: " << prompt_len << std::endl;
    std::cout << "Output token size: " << generated.size() << std::endl;
    std::cout << "TTFT: " << ttft_ms << " ms" << std::endl;
    std::cout << "Decode time: " << decode_ms << " ms" << std::endl;
    if (decode_steps > 0) {
        std::cout << "TPOT: " << tpot_ms << " ms/token" << std::endl;
        std::cout << "Throughput: " << throughput << " tokens/s" << std::endl;
    } else {
        std::cout << "TPOT: N/A" << std::endl;
        std::cout << "Throughput: N/A" << std::endl;
    }

    if (use_mtp && mtp_attempts > 0) {
        const double hit_rate = 100.0 * static_cast<double>(mtp_hits) / static_cast<double>(mtp_attempts);
        std::cout << "MTP hits: " << mtp_hits << "/" << mtp_attempts
                  << " (" << hit_rate << "%)" << std::endl;
        // Absolute draft acceptance rate: accepted / total_drafted
        if (count_draft_infers > 0) {
            const double abs_rate = 100.0 * static_cast<double>(mtp_hits) / static_cast<double>(count_draft_infers);
            std::cout << "MTP draft acceptance: " << mtp_hits << "/" << count_draft_infers
                      << " (" << abs_rate << "%)" << std::endl;
        }
        // Mean accepted tokens per step
        if (mtp_main_infers > 0) {
            const double mean_accepted = static_cast<double>(mtp_hits) / static_cast<double>(mtp_main_infers);
            std::cout << "MTP mean accepted/step: " << mean_accepted << " (of K=" << K
                      << (adaptive_k_enabled ? ", adaptive" : "") << ")" << std::endl;
        }
        std::cout << "MTP main model infers: " << mtp_main_infers << std::endl;
        if (mtp_main_infers > 0) {
            const double tokens_per_infer = static_cast<double>(decode_steps) / static_cast<double>(mtp_main_infers);
            std::cout << "MTP tokens/infer: " << tokens_per_infer << std::endl;
        }
        if (adaptive_k_enabled) {
            std::cout << "MTP adaptive K: max=" << max_K << " final=" << K
                      << " changes=" << adaptive_k_changes << std::endl;
        }
        // Profiling breakdown
        const char* verify_mode_str = use_seq_verify ? " [sequential]"
            : (has_kernel_snapshot ? " [kernel-snapshot]"
            : (use_pure_batch ? " [pure-batch]" : " [batch+rollback]"));
        std::cout << "--- Spec decode profiling (K+1 verify" << verify_mode_str << ", K=" << K
                  << (adaptive_k_enabled ? " adaptive" : "") << ") ---" << std::endl;
        std::cout << "  Main verify (K+1):  " << time_verify_ms << " ms (" << count_verify_infers << " calls)"
                  << (count_verify_infers > 0 ? (", avg " + std::to_string(time_verify_ms / count_verify_infers) + " ms") : "")
                  << std::endl;
        std::cout << "  MTP draft (x" << max_K << "):    " << time_draft_ms << " ms (" << count_draft_infers << " calls)"
                  << (count_draft_infers > 0 ? (", avg " + std::to_string(time_draft_ms / count_draft_infers) + " ms") : "")
                  << std::endl;
        std::cout << "  KV trim:            " << time_trim_ms << " ms (" << count_trims << " calls)"
                  << (count_trims > 0 ? (", avg " + std::to_string(time_trim_ms / count_trims) + " ms") : "")
                  << std::endl;
        if (!use_seq_verify) {
            std::cout << "  Snapshot save:      " << time_snapshot_ms << " ms (" << count_verify_infers << " calls)"
                      << (count_verify_infers > 0 ? (", avg " + std::to_string(time_snapshot_ms / count_verify_infers) + " ms") : "")
                      << std::endl;
            std::cout << "  Restore+re-fwd:     " << (time_restore_ms + time_reforward_ms) << " ms (" << count_restores << " restores, "
                      << count_reforwards << " re-forwards)"
                      << std::endl;
            if (count_refreshes > 0) {
                std::cout << "  State refresh:      " << time_refresh_ms << " ms (" << count_refreshes << " refreshes"
                          << ", avg " << std::to_string(time_refresh_ms / count_refreshes) << " ms)"
                          << std::endl;
            }
            if (use_pure_batch || has_kernel_snapshot) {
                std::cout << "  Dead KV positions:  " << dead_positions.size()
                          << " (virtual trim, no physical KV copies)" << std::endl;
            }
            // Sub-step breakdown (always printed for batch verify)
            if (step_counter > 0) {
                std::cout << "  --- Per-step avg breakdown (" << step_counter << " steps) ---" << std::endl;
                std::cout << "    Snapshot save:     " << std::to_string(time_snapshot_save_ms / step_counter) << " ms/step" << std::endl;
                std::cout << "    Main GPU infer:    " << std::to_string(time_main_infer_ms / step_counter) << " ms/step" << std::endl;
                std::cout << "    Accept/reject:     " << std::to_string(time_accept_check_ms / step_counter) << " ms/step" << std::endl;
                std::cout << "    State restore:     " << std::to_string(time_state_restore_ms / step_counter) << " ms/step (on rejection)" << std::endl;
                std::cout << "    KV trim:           " << std::to_string(time_kv_trim_ms / step_counter) << " ms/step (on rejection)" << std::endl;
                std::cout << "    MTP reset:         " << std::to_string(time_mtp_reset_ms / step_counter) << " ms/step" << std::endl;
                for (int k = 0; k < max_K; ++k) {
                    auto kidx = static_cast<size_t>(k);
                    if (count_mtp_each[kidx] > 0) {
                        std::cout << "    MTP draft k=" << k << ":     "
                                  << std::to_string(time_mtp_each_ms[kidx] / count_mtp_each[kidx]) << " ms/call"
                                  << " (" << count_mtp_each[kidx] << " calls, total " << std::to_string(time_mtp_each_ms[kidx]) << " ms)"
                                  << std::endl;
                    }
                }
                double avg_verify_total = time_verify_ms / step_counter;
                double avg_draft_total = time_draft_ms / step_counter;
                double avg_step_total = avg_verify_total + avg_draft_total;
                std::cout << "    ---------------------" << std::endl;
                std::cout << "    Avg step total:    " << std::to_string(avg_step_total)
                          << " ms (verify=" << std::to_string(avg_verify_total)
                          << " + draft=" << std::to_string(avg_draft_total) << ")" << std::endl;
                // Phase D: MTP draft sub-step breakdown
                if (mtp_sub_count > 0) {
                    const double total_draft_sub = mtp_sub_memcpy_ms + mtp_sub_infer_ms +
                                                   mtp_sub_extract_ms + mtp_sub_sample_ms;
                    std::cout << "  --- MTP draft sub-step breakdown (" << mtp_sub_count << " inferences) ---" << std::endl;
                    std::cout << "    Input setup:       " << std::to_string(mtp_sub_memcpy_ms / mtp_sub_count) << " ms/call"
                              << " (" << std::to_string(100.0 * mtp_sub_memcpy_ms / total_draft_sub) << "%)" << std::endl;
                    std::cout << "    GPU infer:         " << std::to_string(mtp_sub_infer_ms / mtp_sub_count) << " ms/call"
                              << " (" << std::to_string(100.0 * mtp_sub_infer_ms / total_draft_sub) << "%)  ← lm_head [4096×"
                              << cfg.text.vocab_size << "] dominates" << std::endl;
                    std::cout << "    Logits extract:    " << std::to_string(mtp_sub_extract_ms / mtp_sub_count) << " ms/call"
                              << " (" << std::to_string(100.0 * mtp_sub_extract_ms / total_draft_sub) << "%)" << std::endl;
                    std::cout << "    Argmax/sample:     " << std::to_string(mtp_sub_sample_ms / mtp_sub_count) << " ms/call"
                              << " (" << std::to_string(100.0 * mtp_sub_sample_ms / total_draft_sub) << "%)" << std::endl;
                    // Speculative head potential: estimate time with reduced vocab
                    const size_t reduced_vocab = 32768;
                    if (cfg.text.vocab_size > 0) {
                        const double ratio = static_cast<double>(reduced_vocab) / cfg.text.vocab_size;
                        const double est_reduced_infer = mtp_sub_infer_ms / mtp_sub_count * ratio;
                        std::cout << "    [Spec head est]:   " << std::to_string(est_reduced_infer) << " ms/call"
                                  << " with " << reduced_vocab << " vocab (vs " << cfg.text.vocab_size
                                  << "), " << std::to_string((1.0 - ratio) * 100.0) << "% reduction" << std::endl;
                    }
                }
            }
        }
    }

    if (tokenizer) {
        std::cout << tokenizer->decode(generated, ov::genai::skip_special_tokens(true)) << std::endl;
    } else {
        std::cout << "Generated token ids:";
        for (const auto id : generated) {
            std::cout << ' ' << id;
        }
        std::cout << std::endl;
    }

    return 0;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << std::endl;
        std::cerr.flush();
        std::cout << "[ERROR] " << error.what() << std::endl;
        std::cout.flush();
    } catch (const std::ios_base::failure&) {
    }
    return 1;
}
