// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/openvino.hpp>

#include "load_image.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "safetensors_utils/quantization_utils.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_vision.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "modeling/models/qwen3_5/qwen3_5_weight_specs.hpp"
#include "modeling/weights/synthetic_weight_source.hpp"

namespace {

enum class WeightModeSelection {
    Auto,
    Dummy,
    Real,
};

struct SampleOptions {
    std::optional<std::filesystem::path> model_dir;
    std::string mode;
    std::filesystem::path image_path;
    std::string user_prompt;
    std::string device = "CPU";
    int max_new_tokens = 64;

    std::string vision_quant_mode;
    int vision_group_size = 128;
    std::string vision_backup_mode;
    std::string text_quant_mode;
    int text_group_size = 128;
    std::string text_backup_mode;

    WeightModeSelection weight_mode = WeightModeSelection::Auto;

    uint32_t dummy_seed = 2026u;
    float dummy_init_range = 0.02f;
    std::string dummy_weight_mode = "INT4_ASYM";
    int dummy_group_size = 128;

    int dummy_num_layers = 0;
    int dummy_hidden_size = 0;
    int dummy_num_heads = 0;
    int dummy_num_kv_heads = 0;
    int dummy_head_dim = 0;
    int dummy_intermediate_size = 0;
    int dummy_vocab_size = 0;
    int dummy_max_position_embeddings = 0;
};

const char* get_env(const char* name) {
    const char* value = std::getenv(name);
    return (value && value[0] != '\0') ? value : nullptr;
}

bool env_enabled(const char* name) {
    const char* value = get_env(name);
    if (!value) {
        return false;
    }
    const std::string v(value);
    return v == "1" || v == "true" || v == "TRUE" || v == "on" || v == "ON";
}

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

bool should_use_dummy_mode(const std::optional<std::filesystem::path>& model_dir) {
    if (env_enabled("OV_GENAI_QWEN3_5_DUMMY_ENABLE")) {
        return true;
    }
    if (!model_dir.has_value()) {
        return true;
    }
    const bool has_config = std::filesystem::exists(*model_dir / "config.json");
    const bool has_safetensors = has_safetensors_file(*model_dir);
    return !(has_config && has_safetensors);
}

uint32_t read_env_u32(const char* name, uint32_t default_value) {
    const char* raw = get_env(name);
    if (!raw) {
        return default_value;
    }
    try {
        return static_cast<uint32_t>(std::stoul(raw));
    } catch (const std::exception&) {
        return default_value;
    }
}

float read_env_float(const char* name, float default_value) {
    const char* raw = get_env(name);
    if (!raw) {
        return default_value;
    }
    try {
        return std::stof(raw);
    } catch (const std::exception&) {
        return default_value;
    }
}

int read_env_i32(const char* name, int default_value) {
    const char* raw = get_env(name);
    if (!raw) {
        return default_value;
    }
    try {
        return std::stoi(raw);
    } catch (const std::exception&) {
        return default_value;
    }
}

int parse_i32(const std::string& raw, const char* option_name) {
    try {
        return std::stoi(raw);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid integer for ") + option_name + ": " + raw);
    }
}

uint32_t parse_u32(const std::string& raw, const char* option_name) {
    try {
        return static_cast<uint32_t>(std::stoul(raw));
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid unsigned integer for ") + option_name + ": " + raw);
    }
}

float parse_f32(const std::string& raw, const char* option_name) {
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

void print_usage(const char* argv0) {
    std::cout
        << "Usage:\n"
        << "  " << argv0 << " --dummy [--mode text|vl] [--prompt TEXT] [--device CPU] [--max-new-tokens N]\n"
        << "  " << argv0 << " --real --model-dir <MODEL_DIR> [--mode text|vl] [--image IMAGE_PATH]\n"
        << "  " << argv0 << " [legacy positional args are still supported]\n\n"
        << "Options:\n"
        << "  --dummy                         Force synthetic random weights (no model dir required)\n"
        << "  --real                          Force real HF safetensors+config loading\n"
        << "  --model-dir PATH                HF model directory (config.json + *.safetensors)\n"
        << "  --mode text|vl                  Run text-only or vision-language path\n"
        << "  --image PATH                    Image path for vl mode (if omitted: built-in dummy image)\n"
        << "  --prompt TEXT                   User prompt\n"
        << "  --device NAME                   OpenVINO device name (default: CPU)\n"
        << "  --max-new-tokens N              Number of generated tokens (default: 64)\n"
        << "  --vision-quant MODE             Vision quant mode (real mode only)\n"
        << "  --vision-gs N                   Vision quant group size (real mode only)\n"
        << "  --vision-backup MODE            Vision backup quant mode (real mode only)\n"
        << "  --text-quant MODE               Text quant mode (real mode only)\n"
        << "  --text-gs N                     Text quant group size (real mode only)\n"
        << "  --text-backup MODE              Text backup quant mode (real mode only)\n"
        << "  --dummy-seed N                  Synthetic weight RNG seed\n"
        << "  --dummy-init-range F            Synthetic init range, sampled in [-F, F]\n"
        << "  --dummy-weight-mode MODE        FP32 | INT4_ASYM | INT4_SYM (default: INT4_ASYM)\n"
        << "  --dummy-group-size N            Dummy quant group size\n"
        << "  --dummy-num-layers N            Override dummy text num_hidden_layers\n"
        << "  --dummy-hidden-size N           Override dummy text hidden_size\n"
        << "  --dummy-num-heads N             Override dummy text num_attention_heads\n"
        << "  --dummy-num-kv-heads N          Override dummy text num_key_value_heads\n"
        << "  --dummy-head-dim N              Override dummy text head_dim\n"
        << "  --dummy-intermediate-size N     Override dummy text intermediate_size\n"
        << "  --dummy-vocab-size N            Override dummy text vocab_size\n"
        << "  --dummy-max-position N          Override dummy text max_position_embeddings\n"
        << "  -h, --help                      Show this help\n";
}

SampleOptions parse_cli(int argc, char* argv[]) {
    SampleOptions opts;

    if (const char* env_mode = get_env("OV_GENAI_QWEN3_5_SAMPLE_MODE")) {
        opts.mode = env_mode;
    }
    opts.dummy_seed = read_env_u32("OV_GENAI_QWEN3_5_DUMMY_SEED", opts.dummy_seed);
    opts.dummy_init_range = read_env_float("OV_GENAI_QWEN3_5_DUMMY_INIT_RANGE", opts.dummy_init_range);
    if (const char* mode = get_env("OV_GENAI_QWEN3_5_DUMMY_WEIGHT_MODE")) {
        opts.dummy_weight_mode = mode;
    }
    opts.dummy_group_size = read_env_i32("OV_GENAI_QWEN3_5_DUMMY_GROUP_SIZE", opts.dummy_group_size);
    opts.vision_group_size = read_env_i32("OV_GENAI_QWEN3_5_VISION_GROUP_SIZE", opts.vision_group_size);
    opts.text_group_size = read_env_i32("OV_GENAI_QWEN3_5_TEXT_GROUP_SIZE", opts.text_group_size);

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg.rfind("--", 0) != 0) {
            positional.push_back(arg);
            continue;
        }

        auto take_value = [&](const char* option_name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + option_name);
            }
            return argv[++i];
        };

        if (arg == "--dummy") {
            if (opts.weight_mode == WeightModeSelection::Real) {
                throw std::runtime_error("--dummy conflicts with --real");
            }
            opts.weight_mode = WeightModeSelection::Dummy;
        } else if (arg == "--real") {
            if (opts.weight_mode == WeightModeSelection::Dummy) {
                throw std::runtime_error("--real conflicts with --dummy");
            }
            opts.weight_mode = WeightModeSelection::Real;
        } else if (arg == "--model-dir") {
            opts.model_dir = take_value("--model-dir");
        } else if (arg == "--mode") {
            opts.mode = take_value("--mode");
        } else if (arg == "--image") {
            opts.image_path = take_value("--image");
        } else if (arg == "--prompt") {
            opts.user_prompt = take_value("--prompt");
        } else if (arg == "--device") {
            opts.device = take_value("--device");
        } else if (arg == "--max-new-tokens") {
            opts.max_new_tokens = parse_i32(take_value("--max-new-tokens"), "--max-new-tokens");
        } else if (arg == "--vision-quant") {
            opts.vision_quant_mode = take_value("--vision-quant");
        } else if (arg == "--vision-gs") {
            opts.vision_group_size = parse_i32(take_value("--vision-gs"), "--vision-gs");
        } else if (arg == "--vision-backup") {
            opts.vision_backup_mode = take_value("--vision-backup");
        } else if (arg == "--text-quant") {
            opts.text_quant_mode = take_value("--text-quant");
        } else if (arg == "--text-gs") {
            opts.text_group_size = parse_i32(take_value("--text-gs"), "--text-gs");
        } else if (arg == "--text-backup") {
            opts.text_backup_mode = take_value("--text-backup");
        } else if (arg == "--dummy-seed") {
            opts.dummy_seed = parse_u32(take_value("--dummy-seed"), "--dummy-seed");
        } else if (arg == "--dummy-init-range") {
            opts.dummy_init_range = parse_f32(take_value("--dummy-init-range"), "--dummy-init-range");
        } else if (arg == "--dummy-weight-mode") {
            opts.dummy_weight_mode = take_value("--dummy-weight-mode");
        } else if (arg == "--dummy-group-size") {
            opts.dummy_group_size = parse_i32(take_value("--dummy-group-size"), "--dummy-group-size");
        } else if (arg == "--dummy-num-layers") {
            opts.dummy_num_layers = parse_i32(take_value("--dummy-num-layers"), "--dummy-num-layers");
        } else if (arg == "--dummy-hidden-size") {
            opts.dummy_hidden_size = parse_i32(take_value("--dummy-hidden-size"), "--dummy-hidden-size");
        } else if (arg == "--dummy-num-heads") {
            opts.dummy_num_heads = parse_i32(take_value("--dummy-num-heads"), "--dummy-num-heads");
        } else if (arg == "--dummy-num-kv-heads") {
            opts.dummy_num_kv_heads = parse_i32(take_value("--dummy-num-kv-heads"), "--dummy-num-kv-heads");
        } else if (arg == "--dummy-head-dim") {
            opts.dummy_head_dim = parse_i32(take_value("--dummy-head-dim"), "--dummy-head-dim");
        } else if (arg == "--dummy-intermediate-size") {
            opts.dummy_intermediate_size = parse_i32(take_value("--dummy-intermediate-size"), "--dummy-intermediate-size");
        } else if (arg == "--dummy-vocab-size") {
            opts.dummy_vocab_size = parse_i32(take_value("--dummy-vocab-size"), "--dummy-vocab-size");
        } else if (arg == "--dummy-max-position") {
            opts.dummy_max_position_embeddings = parse_i32(take_value("--dummy-max-position"), "--dummy-max-position");
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    size_t pos = 0;
    if (pos < positional.size()) {
        const std::string first = positional[pos];
        const bool looks_like_mode = (first == "text" || first == "vl");
        if (!looks_like_mode && !opts.model_dir.has_value()) {
            opts.model_dir = first;
            pos++;
        }
    }
    if (pos < positional.size() && opts.mode.empty()) {
        const std::string maybe_mode = positional[pos];
        if (maybe_mode == "text" || maybe_mode == "vl") {
            opts.mode = maybe_mode;
            pos++;
        }
    }
    if (pos < positional.size() && opts.mode == "vl" && opts.image_path.empty()) {
        if (positional[pos] != "-" && std::filesystem::exists(positional[pos])) {
            opts.image_path = positional[pos];
            pos++;
        }
    }
    if (pos < positional.size() && opts.user_prompt.empty()) {
        opts.user_prompt = positional[pos++];
    }
    if (pos < positional.size()) {
        opts.device = positional[pos++];
    }
    if (pos < positional.size()) {
        opts.max_new_tokens = parse_i32(positional[pos++], "MAX_NEW_TOKENS");
    }
    if (pos < positional.size() && opts.vision_quant_mode.empty()) {
        opts.vision_quant_mode = positional[pos++];
    }
    if (pos < positional.size()) {
        opts.vision_group_size = parse_i32(positional[pos++], "VISION_GS");
    }
    if (pos < positional.size() && opts.vision_backup_mode.empty()) {
        opts.vision_backup_mode = positional[pos++];
    }
    if (pos < positional.size() && opts.text_quant_mode.empty()) {
        opts.text_quant_mode = positional[pos++];
    }
    if (pos < positional.size()) {
        opts.text_group_size = parse_i32(positional[pos++], "TEXT_GS");
    }
    if (pos < positional.size() && opts.text_backup_mode.empty()) {
        opts.text_backup_mode = positional[pos++];
    }
    if (pos != positional.size()) {
        std::ostringstream oss;
        oss << "Too many positional arguments. Unexpected token: " << positional[pos];
        throw std::runtime_error(oss.str());
    }

    if (opts.mode.empty()) {
        opts.mode = opts.image_path.empty() ? "text" : "vl";
    }
    opts.mode = to_lower(opts.mode);
    if (opts.mode != "text" && opts.mode != "vl") {
        throw std::runtime_error("Mode must be 'text' or 'vl'");
    }

    if (opts.max_new_tokens <= 0) {
        throw std::runtime_error("max_new_tokens must be > 0");
    }
    if (opts.user_prompt.empty()) {
        opts.user_prompt = (opts.mode == "vl") ? "Describe the image." : "Write one sentence about OpenVINO.";
    }

    return opts;
}

bool use_dummy_mode(const SampleOptions& opts) {
    switch (opts.weight_mode) {
    case WeightModeSelection::Dummy:
        return true;
    case WeightModeSelection::Real:
        return false;
    case WeightModeSelection::Auto:
        return should_use_dummy_mode(opts.model_dir);
    }
    return should_use_dummy_mode(opts.model_dir);
}

void apply_dummy_config_overrides(ov::genai::modeling::models::Qwen3_5Config& cfg, const SampleOptions& opts) {
    bool layer_schedule_needs_refresh = false;
    bool hidden_size_changed = false;
    if (opts.dummy_num_layers > 0) {
        cfg.text.num_hidden_layers = opts.dummy_num_layers;
        layer_schedule_needs_refresh = true;
    }
    if (opts.dummy_hidden_size > 0) {
        cfg.text.hidden_size = opts.dummy_hidden_size;
        hidden_size_changed = true;
    }
    if (opts.dummy_num_heads > 0) {
        cfg.text.num_attention_heads = opts.dummy_num_heads;
    }
    if (opts.dummy_num_kv_heads > 0) {
        cfg.text.num_key_value_heads = opts.dummy_num_kv_heads;
    }
    if (opts.dummy_head_dim > 0) {
        cfg.text.head_dim = opts.dummy_head_dim;
    }
    if (opts.dummy_intermediate_size > 0) {
        cfg.text.intermediate_size = opts.dummy_intermediate_size;
    }
    if (opts.dummy_vocab_size > 0) {
        cfg.text.vocab_size = opts.dummy_vocab_size;
    }
    if (opts.dummy_max_position_embeddings > 0) {
        cfg.text.max_position_embeddings = opts.dummy_max_position_embeddings;
    }
    if (hidden_size_changed) {
        cfg.vision.out_hidden_size = cfg.text.hidden_size;
    }
    if (layer_schedule_needs_refresh) {
        cfg.text.layer_types.clear();
    }
    cfg.finalize();
    cfg.validate();
}

std::string build_vl_prompt(const std::string& user_prompt, int64_t image_tokens) {
    std::string prompt = "<|im_start|>user\n<|vision_start|>";
    prompt.reserve(prompt.size() + static_cast<size_t>(image_tokens) * 12 + user_prompt.size() + 64);
    for (int64_t i = 0; i < image_tokens; ++i) {
        prompt += "<|image_pad|>";
    }
    prompt += "<|vision_end|>\n";
    prompt += user_prompt;
    prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

int64_t argmax_last_token(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3 || shape[0] != 1) {
        throw std::runtime_error("logits must have shape [1, S, V]");
    }
    const size_t seq_len = shape[1];
    const size_t vocab = shape[2];
    const size_t offset = (seq_len - 1) * vocab;

    if (logits.get_element_type() == ov::element::f16) {
        const auto* data = logits.data<const ov::float16>() + offset;
        ov::float16 max_val = data[0];
        size_t max_idx = 0;
        for (size_t i = 1; i < vocab; ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }
        return static_cast<int64_t>(max_idx);
    }
    if (logits.get_element_type() == ov::element::bf16) {
        const auto* data = logits.data<const ov::bfloat16>() + offset;
        ov::bfloat16 max_val = data[0];
        size_t max_idx = 0;
        for (size_t i = 1; i < vocab; ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }
        return static_cast<int64_t>(max_idx);
    }
    if (logits.get_element_type() != ov::element::f32) {
        throw std::runtime_error("Unsupported logits dtype");
    }
    const auto* data = logits.data<const float>() + offset;
    float max_val = data[0];
    size_t max_idx = 0;
    for (size_t i = 1; i < vocab; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return static_cast<int64_t>(max_idx);
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
    ov::Tensor image(ov::element::u8, {1, 224, 224, 3});
    std::memset(image.data(), 127, image.get_byte_size());
    return image;
}

ov::genai::modeling::weights::QuantizationConfig build_dummy_quant_config(const SampleOptions& opts) {
    using Mode = ov::genai::modeling::weights::QuantizationConfig::Mode;
    ov::genai::modeling::weights::QuantizationConfig cfg;

    std::string mode = opts.dummy_weight_mode;
    for (auto& c : mode) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }

    if (mode == "INT4_SYM") {
        cfg.mode = Mode::INT4_SYM;
    } else if (mode == "INT4_ASYM" || mode == "INT4") {
        cfg.mode = Mode::INT4_ASYM;
    } else {
        cfg.mode = Mode::NONE;
    }

    cfg.group_size = opts.dummy_group_size;
    cfg.backup_mode = cfg.mode;
    cfg.selection.exclude_patterns.clear();
    return cfg;
}

}  // namespace

int main(int argc, char* argv[]) try {
    const SampleOptions opts = parse_cli(argc, argv);
    const bool use_vl = (opts.mode == "vl");
    const bool use_dummy_mode_flag = use_dummy_mode(opts);
    if (!use_dummy_mode_flag && !opts.model_dir.has_value()) {
        throw std::runtime_error("Real mode requires --model-dir (or legacy MODEL_DIR positional argument).");
    }
    const std::filesystem::path model_dir = opts.model_dir.value_or(std::filesystem::path{});
    if (use_vl && !opts.image_path.empty() && !std::filesystem::exists(opts.image_path)) {
        throw std::runtime_error("Image path does not exist: " + opts.image_path.string());
    }

    ov::genai::modeling::models::Qwen3_5Config cfg =
        use_dummy_mode_flag ? ov::genai::modeling::models::Qwen3_5Config::make_dummy_dense9b_config()
                            : ov::genai::modeling::models::Qwen3_5Config::from_json_file(model_dir);
    if (use_dummy_mode_flag) {
        apply_dummy_config_overrides(cfg, opts);
    }

    ov::genai::modeling::weights::QuantizationConfig vision_quant_config;
    ov::genai::modeling::weights::QuantizationConfig text_quant_config;
    if (use_dummy_mode_flag) {
        const auto dummy_quant = build_dummy_quant_config(opts);
        vision_quant_config = dummy_quant;
        text_quant_config = dummy_quant;
    } else {
        vision_quant_config = create_quantization_config(opts.vision_quant_mode, opts.vision_group_size, opts.vision_backup_mode);
        text_quant_config = create_quantization_config(opts.text_quant_mode, opts.text_group_size, opts.text_backup_mode);
    }

    std::unique_ptr<ov::genai::modeling::weights::WeightSource> source;
    if (use_dummy_mode_flag) {
        const uint32_t seed = opts.dummy_seed;
        const float init_range = opts.dummy_init_range;
        auto specs = use_vl ? ov::genai::modeling::models::build_qwen3_5_vlm_weight_specs(cfg)
                            : ov::genai::modeling::models::build_qwen3_5_text_weight_specs(cfg.text);
        source = std::make_unique<ov::genai::modeling::weights::SyntheticWeightSource>(
            std::move(specs),
            seed,
            -init_range,
            init_range);
    } else {
        auto data = ov::genai::safetensors::load_safetensors(model_dir);
        source = std::make_unique<ov::genai::safetensors::SafetensorsWeightSource>(std::move(data));
    }

    ov::genai::modeling::models::Qwen3_5VisionPreprocessConfig pre_cfg;
    const auto pre_cfg_path = model_dir / "preprocessor_config.json";
    if (!use_dummy_mode_flag && std::filesystem::exists(pre_cfg_path)) {
        pre_cfg = ov::genai::modeling::models::Qwen3_5VisionPreprocessConfig::from_json_file(pre_cfg_path);
    }

    std::shared_ptr<ov::Model> vision_model;
    if (use_vl) {
        ov::genai::safetensors::SafetensorsWeightFinalizer vision_finalizer(vision_quant_config);
        vision_model = ov::genai::modeling::models::create_qwen3_5_vision_model(cfg, *source, vision_finalizer);
    }

    std::shared_ptr<ov::Model> text_model;
    {
        ov::genai::safetensors::SafetensorsWeightFinalizer text_finalizer(text_quant_config);
        text_model = ov::genai::modeling::models::create_qwen3_5_text_model(
            cfg,
            *source,
            text_finalizer,
            false,
            use_vl);
    }

    if (use_dummy_mode_flag && source) {
        source->release_all_cached_tensors();
        if (!use_vl) {
            source.reset();
        }
    }

    ov::Core core;
    std::optional<ov::CompiledModel> compiled_vision;
    if (use_vl) {
        compiled_vision = core.compile_model(vision_model, opts.device);
    }
    auto compiled_text = core.compile_model(text_model, opts.device);

    ov::Tensor visual_embeds;
    ov::Tensor grid_thw;
    if (use_vl) {
        ov::Tensor image = opts.image_path.empty() ? make_dummy_image() : utils::load_image(opts.image_path);
        const std::string pos_embed_name = resolve_pos_embed_name(*source);
        const ov::Tensor pos_embed_weight = source->get_tensor(pos_embed_name);

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
    if (tokenizer) {
        std::string prompt;
        if (use_vl) {
            const int64_t image_tokens =
                ov::genai::modeling::models::Qwen3_5VisionPreprocessor::count_visual_tokens(
                    grid_thw,
                    cfg.vision.spatial_merge_size);
            prompt = build_vl_prompt(opts.user_prompt, image_tokens);
        } else {
            prompt = opts.user_prompt;
        }
        auto tokenized = tokenizer->encode(prompt, ov::genai::add_special_tokens(false));
        input_ids = tokenized.input_ids;
        attention_mask = tokenized.attention_mask;
    } else {
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
    text_request.reset_state();
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kInputIds, input_ids);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kAttentionMask, attention_mask);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kPositionIds, plan.position_ids);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kBeamIdx, beam_idx);
    if (use_vl) {
        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualEmbeds, visual_padded);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualPosMask, plan.visual_pos_mask);
    }

    const auto prefill_start = std::chrono::steady_clock::now();
    text_request.infer();
    ov::Tensor logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kLogits);
    int64_t next_id = argmax_last_token(logits);
    const auto prefill_end = std::chrono::steady_clock::now();

    std::vector<int64_t> generated;
    generated.reserve(static_cast<size_t>(opts.max_new_tokens));
    generated.push_back(next_id);

    int64_t eos_token_id = -1;
    if (tokenizer) {
        eos_token_id = tokenizer->get_eos_token_id();
    }

    ov::Tensor step_ids(ov::element::i64, {batch, 1});
    ov::Tensor step_mask(ov::element::i64, {batch, 1});
    auto* step_mask_data = step_mask.data<int64_t>();
    for (size_t b = 0; b < batch; ++b) {
        step_mask_data[b] = 1;
    }

    ov::Tensor decode_visual;
    ov::Tensor decode_visual_mask;
    if (use_vl) {
        decode_visual = make_zero_tensor(ov::element::f32, {batch, 1, static_cast<size_t>(cfg.text.hidden_size)});
        decode_visual_mask = make_zero_tensor(ov::element::boolean, {batch, 1});
    }

    int64_t past_len = prompt_len;
    size_t decode_steps = 0;
    const auto decode_start = std::chrono::steady_clock::now();
    for (int step = 1; step < opts.max_new_tokens; ++step) {
        if (eos_token_id >= 0 && next_id == eos_token_id) {
            break;
        }
        auto* step_data = step_ids.data<int64_t>();
        for (size_t b = 0; b < batch; ++b) {
            step_data[b] = next_id;
        }

        auto position_ids = ov::genai::modeling::models::Qwen3_5InputPlanner::build_decode_position_ids(
            plan.rope_deltas,
            past_len,
            1);

        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kInputIds, step_ids);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kAttentionMask, step_mask);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kPositionIds, position_ids);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kBeamIdx, beam_idx);
        if (use_vl) {
            text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualEmbeds, decode_visual);
            text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualPosMask, decode_visual_mask);
        }

        text_request.infer();
        logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kLogits);
        next_id = argmax_last_token(logits);
        generated.push_back(next_id);
        decode_steps += 1;
        past_len += 1;
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
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return 1;
}
