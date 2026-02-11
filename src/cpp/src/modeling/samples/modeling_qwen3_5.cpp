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

struct SampleOptions {
    std::optional<std::filesystem::path> model_dir;
    std::string mode;
    std::filesystem::path image_path;
    std::string user_prompt;
    std::string device = "GPU";
    int max_new_tokens = 64;

    std::string vision_quant_mode;
    int vision_group_size = 128;
    std::string text_quant_mode;
    int text_group_size = 128;

    std::string dummy_model = "dense";
    std::string dummy_weight_mode = "INT4_ASYM";
    int dummy_group_size = 128;

    int dummy_num_layers = 0;
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

int parse_i32(const std::string& raw, const char* option_name) {
    try {
        return std::stoi(raw);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid integer for ") + option_name + ": " + raw);
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
        << "  --device NAME                   OpenVINO device name (default: GPU)\n"
        << "  --output-tokens N               Number of generated tokens (default: 64)\n"
        << "  --vision-quant MODE             Real mode only. MODE: none|int4_asym|int4_sym (default: none)\n"
        << "  --vision-gs N                   Real mode only. Vision quant group size, integer > 0 (default: 128)\n"
        << "  --text-quant MODE               Real mode only. MODE: none|int4_asym|int4_sym (default: none)\n"
        << "  --text-gs N                     Real mode only. Text quant group size, integer > 0 (default: 128)\n"
        << "  --dummy-model MODE              Dummy mode only: dense | moe (default: dense)\n"
        << "  --dummy-weight-mode MODE        FP32 | INT4_ASYM | INT4_SYM (default: INT4_ASYM)\n"
        << "  --dummy-group-size N            Dummy quant group size, integer > 0 (default: 128)\n"
        << "  --dummy-num-layers N            Override dummy text num_hidden_layers\n"
        << "  -h, --help                      Show this helper\n";
}

bool is_valid_quant_mode(std::string mode) {
    mode = to_lower(std::move(mode));
    return mode.empty() || mode == "none" || mode == "int4_asym" || mode == "int4_sym";
}

bool is_valid_dummy_weight_mode(std::string mode) {
    for (auto& c : mode) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return mode == "FP32" || mode == "INT4_ASYM" || mode == "INT4_SYM" || mode == "INT4";
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
        } else if (arg == "--device") {
            opts.device = take_value("--device");
        } else if (arg == "--output-tokens") {
            opts.max_new_tokens = parse_i32(take_value("--output-tokens"), "--output-tokens");
        } else if (arg == "--vision-quant") {
            opts.vision_quant_mode = take_value("--vision-quant");
        } else if (arg == "--vision-gs") {
            opts.vision_group_size = parse_i32(take_value("--vision-gs"), "--vision-gs");
        } else if (arg == "--text-quant") {
            opts.text_quant_mode = take_value("--text-quant");
        } else if (arg == "--text-gs") {
            opts.text_group_size = parse_i32(take_value("--text-gs"), "--text-gs");
        } else if (arg == "--dummy-model") {
            opts.dummy_model = take_value("--dummy-model");
        } else if (arg == "--dummy-weight-mode") {
            opts.dummy_weight_mode = take_value("--dummy-weight-mode");
        } else if (arg == "--dummy-group-size") {
            opts.dummy_group_size = parse_i32(take_value("--dummy-group-size"), "--dummy-group-size");
        } else if (arg == "--dummy-num-layers") {
            opts.dummy_num_layers = parse_i32(take_value("--dummy-num-layers"), "--dummy-num-layers");
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
    if (opts.vision_group_size <= 0) {
        throw std::runtime_error("vision-gs must be > 0");
    }
    if (opts.text_group_size <= 0) {
        throw std::runtime_error("text-gs must be > 0");
    }
    if (opts.dummy_group_size <= 0) {
        throw std::runtime_error("dummy-group-size must be > 0");
    }
    if (!is_valid_quant_mode(opts.vision_quant_mode)) {
        throw std::runtime_error("vision-quant must be one of: none|int4_asym|int4_sym");
    }
    if (!is_valid_quant_mode(opts.text_quant_mode)) {
        throw std::runtime_error("text-quant must be one of: none|int4_asym|int4_sym");
    }
    if (!is_valid_dummy_weight_mode(opts.dummy_weight_mode)) {
        throw std::runtime_error("dummy-weight-mode must be one of: FP32|INT4_ASYM|INT4_SYM");
    }
    if (opts.user_prompt.empty()) {
        opts.user_prompt = (opts.mode == "vl") ? "Describe the image." : "Write one sentence about OpenVINO.";
    }

    return opts;
}

void apply_dummy_config_overrides(ov::genai::modeling::models::Qwen3_5Config& cfg, const SampleOptions& opts) {
    bool layer_schedule_needs_refresh = false;
    if (opts.dummy_num_layers > 0) {
        cfg.text.num_hidden_layers = opts.dummy_num_layers;
        layer_schedule_needs_refresh = true;
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
    ov::Tensor image(ov::element::u8, {1, 256, 256, 3});
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
        vision_quant_config = create_quantization_config(opts.vision_quant_mode, opts.vision_group_size, "none");
        text_quant_config = create_quantization_config(opts.text_quant_mode, opts.text_group_size, "none");
    }

    std::unique_ptr<ov::genai::modeling::weights::WeightSource> source;
    if (use_dummy_mode_flag) {
        constexpr uint32_t kDummySeed = 2026u;
        constexpr float kDummyInitRange = 0.02f;
        auto specs = use_vl ? ov::genai::modeling::models::build_qwen3_5_vlm_weight_specs(cfg)
                            : ov::genai::modeling::models::build_qwen3_5_text_weight_specs(cfg.text);
        source = std::make_unique<ov::genai::modeling::weights::SyntheticWeightSource>(
            std::move(specs),
            kDummySeed,
            -kDummyInitRange,
            kDummyInitRange);
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
