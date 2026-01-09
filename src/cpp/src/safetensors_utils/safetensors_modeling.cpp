// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "safetensors_utils/safetensors_modeling.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/hf_config.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <sstream>

#include <openvino/openvino.hpp>
#include "openvino/runtime/core.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/serialize.hpp"

// Include GGUF building blocks - we'll reuse these for legacy path
#include "gguf_utils/building_blocks.hpp"
#include "gguf_utils/gguf.hpp"
#include "gguf_utils/rtn_quantize.hpp"  // C++ RTN quantization
#include "utils.hpp"

// Include modeling API components
#include "modeling/models/qwen3_dense.hpp"
#include "modeling/models/smollm3.hpp"

using namespace ov;
using namespace ov::op::v13;
using namespace ov::op;

namespace ov {
namespace genai {
namespace safetensors {

// Forward declaration for internal function
std::shared_ptr<ov::Model> create_from_safetensors_compressed(
    const std::filesystem::path& model_dir,
    const InFlightCompressionConfig& compression_config,
    bool enable_save_ov_model);

// Implementation of InFlightCompressionConfig::to_gguf_type()
int InFlightCompressionConfig::to_gguf_type() const {
    switch (mode) {
        case Mode::INT4_SYM:  return ov_extended_types::GGUF_TYPE_INFLIGHT_INT4_SYM;
        case Mode::INT4_ASYM: return ov_extended_types::GGUF_TYPE_INFLIGHT_INT4_ASYM;
        case Mode::INT8_SYM:  return ov_extended_types::GGUF_TYPE_INFLIGHT_INT8_SYM;
        case Mode::INT8_ASYM: return ov_extended_types::GGUF_TYPE_INFLIGHT_INT8_ASYM;
        default: return 0;
    }
}

namespace {

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

/**
 * @brief Check if new modeling API should be used
 *
 * Controlled by OV_GENAI_USE_MODELING_API environment variable
 */
bool use_modeling_api() {
    auto is_truthy = [](std::string v) {
        std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return v == "1" || v == "true" || v == "on" || v == "yes";
    };

    if (const char* v = std::getenv("OV_GENAI_USE_MODELING_API")) {
        return is_truthy(v);
    }
    return false;
}

/**
 * @brief Get in-flight compression configuration from environment variables
 *
 * Environment variables:
 *   OV_GENAI_INFLIGHT_QUANT_MODE: int4_sym, int4_asym, int8_sym, int8_asym (empty = disabled)
 *   OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE: group size (default: 128)
 */
InFlightCompressionConfig get_compression_config_from_env() {
    InFlightCompressionConfig config;
    
    // Check if quantization mode is specified
    if (const char* mode_str = std::getenv("OV_GENAI_INFLIGHT_QUANT_MODE")) {
        std::string mode(mode_str);
        if (!mode.empty() && mode != "none") {
            config.enabled = true;
            config.mode = InFlightCompressionConfig::parse_mode(mode);
        }
    }
    
    // Get group size
    if (const char* gs_str = std::getenv("OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE")) {
        try {
            config.group_size = std::stoi(gs_str);
        } catch (...) {
            // Use default
        }
    }
    
    return config;
}

/**
 * @brief Convert HuggingFace weight name to building_blocks format
 * 
 * HF uses: model.layers.0.self_attn.q_proj.weight
 * building_blocks uses: model.layers[0].self_attn.q_proj.weight
 */
std::string convert_weight_name(const std::string& name) {
    std::string result = name;
    
    // Find "model.layers." and convert to "model.layers[N]."
    const std::string prefix = "model.layers.";
    size_t pos = result.find(prefix);
    if (pos != std::string::npos) {
        size_t num_start = pos + prefix.length();
        size_t num_end = result.find('.', num_start);
        if (num_end != std::string::npos) {
            std::string layer_num = result.substr(num_start, num_end - num_start);
            // Check if it's actually a number
            bool is_number = !layer_num.empty() && std::all_of(layer_num.begin(), layer_num.end(), ::isdigit);
            if (is_number) {
                result = result.substr(0, pos) + "model.layers[" + layer_num + "]" + result.substr(num_end);
            }
        }
    }
    
    return result;
}

/**
 * @brief Map HuggingFace weight names to internal format
 * 
 * HF uses: model.layers.{i}.self_attn.q_proj.weight
 * We need to create qtype entries for compatibility with building_blocks
 * 
 * @param compression_mode The compression mode (NONE = no compression)
 */
void create_qtype_entries(
    std::unordered_map<std::string, gguf_tensor_type>& qtypes,
    const std::unordered_map<std::string, ov::Tensor>& weights,
    int num_layers,
    InFlightCompressionConfig::Mode compression_mode = InFlightCompressionConfig::Mode::NONE) {
    
    // Determine dtype from first weight tensor for non-compressed weights
    gguf_tensor_type default_qtype = gguf_tensor_type::GGUF_TYPE_F16;
    
    for (const auto& [name, tensor] : weights) {
        if (tensor.get_element_type() == ov::element::bf16) {
            default_qtype = gguf_tensor_type::GGUF_TYPE_BF16;
            break;
        } else if (tensor.get_element_type() == ov::element::f32) {
            default_qtype = gguf_tensor_type::GGUF_TYPE_F32;
            break;
        }
    }
    
    // Determine compressed qtype from compression mode
    InFlightCompressionConfig config;
    config.mode = compression_mode;
    int gguf_type = config.to_gguf_type();
    gguf_tensor_type compressed_qtype = (gguf_type != 0) 
        ? static_cast<gguf_tensor_type>(gguf_type) 
        : default_qtype;
    
    // Set embedding qtype (embeddings are not compressed)
    qtypes["model.embed_tokens.qtype"] = default_qtype;
    qtypes["lm_head.qtype"] = default_qtype;
    
    // Set layer qtypes
    for (int i = 0; i < num_layers; i++) {
        std::string prefix = "model.layers[" + std::to_string(i) + "]";
        
        // Attention projections
        qtypes[prefix + ".self_attn.q_proj.qtype"] = compressed_qtype;
        qtypes[prefix + ".self_attn.k_proj.qtype"] = compressed_qtype;
        qtypes[prefix + ".self_attn.v_proj.qtype"] = compressed_qtype;
        qtypes[prefix + ".self_attn.o_proj.qtype"] = compressed_qtype;
        
        // MLP projections
        qtypes[prefix + ".mlp.gate_proj.qtype"] = compressed_qtype;
        qtypes[prefix + ".mlp.up_proj.qtype"] = compressed_qtype;
        qtypes[prefix + ".mlp.down_proj.qtype"] = compressed_qtype;
    }
}

/**
 * @brief Convert HF config to GGUFMetaData map for compatibility
 */
std::map<std::string, GGUFMetaData> convert_config_to_gguf_format(const HFConfig& config) {
    std::map<std::string, GGUFMetaData> result;
    
    result["architecture"] = config.model_type;
    result["hidden_size"] = config.hidden_size;
    result["intermediate_size"] = config.intermediate_size;
    result["layer_num"] = config.num_hidden_layers;
    result["head_num"] = config.num_attention_heads;      // building_blocks uses head_num
    result["head_num_kv"] = config.kv_heads();            // building_blocks uses head_num_kv
    result["head_size"] = config.head_size();             // building_blocks uses head_size
    result["rms_norm_eps"] = config.rms_norm_eps;         // building_blocks uses rms_norm_eps
    
    return result;
}

/**
 * @brief Create model using new modeling API
 *
 * This uses the Qwen3ForCausalLM class from modeling/models/qwen3_dense.hpp
 */
std::shared_ptr<ov::Model> create_model_with_modeling_api(
    const HFConfig& hf_config,
    std::unordered_map<std::string, ov::Tensor>& tensors) {
    SafetensorsWeightSource source(tensors);
    SafetensorsWeightFinalizer finalizer;

    std::shared_ptr<ov::Model> ov_model;
    if (hf_config.model_type == "qwen3") {
        ov::genai::modeling::models::Qwen3DenseConfig cfg;
        cfg.architecture = hf_config.model_type;
        cfg.hidden_size = hf_config.hidden_size;
        cfg.num_hidden_layers = hf_config.num_hidden_layers;
        cfg.num_attention_heads = hf_config.num_attention_heads;
        cfg.num_key_value_heads = hf_config.kv_heads();
        cfg.head_dim = hf_config.head_size();
        cfg.rope_theta = hf_config.rope_theta;
        cfg.attention_bias = tensors.count("model.layers[0].self_attn.q_proj.bias") > 0;
        cfg.rms_norm_eps = hf_config.rms_norm_eps;
        cfg.tie_word_embeddings = tensors.count("lm_head.weight") == 0;
        ov_model = ov::genai::modeling::models::create_qwen3_dense_model(cfg, source, finalizer);
    } else if (hf_config.model_type == "smollm3") {
        ov::genai::modeling::models::SmolLM3Config cfg;
        cfg.architecture = hf_config.model_type;
        cfg.hidden_size = hf_config.hidden_size;
        cfg.num_hidden_layers = hf_config.num_hidden_layers;
        cfg.num_attention_heads = hf_config.num_attention_heads;
        cfg.num_key_value_heads = hf_config.kv_heads();
        cfg.head_dim = hf_config.head_size();
        cfg.rope_theta = hf_config.rope_theta;
        cfg.rms_norm_eps = hf_config.rms_norm_eps;
        cfg.attention_bias = tensors.count("model.layers[0].self_attn.q_proj.bias") > 0;
        cfg.mlp_bias = tensors.count("model.layers[0].mlp.gate_proj.bias") > 0;
        cfg.tie_word_embeddings = tensors.count("lm_head.weight") == 0;
        ov_model = ov::genai::modeling::models::create_smollm3_model(cfg, source, finalizer);
    } else {
        throw std::runtime_error("Unsupported model architecture '" + hf_config.model_type + "'");
    }

    // Set runtime options for optimal performance
    ov_model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    ov_model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});

    return ov_model;
}

/**
 * @brief Create the language model from HuggingFace weights (legacy building_blocks path)
 */
std::shared_ptr<ov::Model> create_language_model(
    const HFConfig& hf_config,
    std::unordered_map<std::string, ov::Tensor>& consts,
    std::unordered_map<std::string, gguf_tensor_type>& qtypes) {
    
    // Convert config to GGUF format for compatibility with building_blocks
    auto configs = convert_config_to_gguf_format(hf_config);
    
    // Create input parameters
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    set_name(input_ids, "input_ids");

    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    set_name(attention_mask, "attention_mask");

    auto position_ids = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    set_name(position_ids, "position_ids");

    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i32, ov::PartialShape{-1});
    set_name(beam_idx, "beam_idx");

    // Create embedding layer
    auto [inputs_embeds, embeddings] = make_embedding(
        "model.embed_tokens",
        input_ids->output(0),
        consts,
        qtypes.at("model.embed_tokens.qtype"));

    auto hidden_states = inputs_embeds;

    // Initialize RoPE
    auto rope_const = init_rope(
        hf_config.head_size(),
        hf_config.max_position_embeddings,
        hf_config.rope_theta);

    // Get input shape components
    auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(input_ids);
    auto batch_axis = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{1}, 0);
    auto batch_size = std::make_shared<ov::op::v8::Gather>(
        input_shape, batch_axis, batch_axis);

    auto hidden_dim = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{1}, 3);

    // Process layers
    ov::SinkVector sinks;
    ov::Output<ov::Node> causal_mask;
    std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> cos_sin_cached;
    std::shared_ptr<ov::Node> output_shape = nullptr;

    std::cout << "[Safetensors] Building " << hf_config.num_hidden_layers << " transformer layers..." << std::endl;
    
    for (int i = 0; i < hf_config.num_hidden_layers; ++i) {
        auto [new_hidden, layer_sinks, new_mask, new_cos_sin, new_shape] = layer(
            configs,
            consts,
            qtypes,
            i,
            hidden_states,
            attention_mask,
            causal_mask,
            position_ids,
            rope_const,
            beam_idx,
            batch_size,
            hidden_dim,
            cos_sin_cached,
            output_shape);

        hidden_states = new_hidden;
        causal_mask = new_mask;
        cos_sin_cached = new_cos_sin;
        output_shape = new_shape;

        sinks.insert(sinks.end(), layer_sinks.begin(), layer_sinks.end());
        
        if ((i + 1) % 10 == 0 || i == hf_config.num_hidden_layers - 1) {
            std::cout << "[Safetensors] Built layer " << (i + 1) << "/" << hf_config.num_hidden_layers << std::endl;
        }
    }

    // Final layer norm
    auto final_norm = make_rms_norm(
        "model.norm",
        hidden_states,
        consts,
        hf_config.rms_norm_eps);

    // LM head
    auto embed_out = make_lm_head(
        "lm_head",
        final_norm,
        consts,
        embeddings,
        qtypes.at("lm_head.qtype"));

    // Create results
    auto logits = std::make_shared<ov::op::v0::Result>(embed_out);
    set_name(logits, "logits");

    // Create model
    ov::ParameterVector inputs{input_ids, attention_mask, position_ids, beam_idx};
    auto model = std::make_shared<ov::Model>(ov::OutputVector({logits->output(0)}), sinks, inputs);

    // Set runtime options
    model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});

    return model;
}

}  // anonymous namespace

/**
 * @brief Convert HuggingFace tokenizer to OpenVINO format if needed
 * 
 * This function checks if openvino_tokenizer.xml exists in the model directory.
 * If not, it calls Python to convert the tokenizer using openvino_tokenizers.
 */
void convert_tokenizer_if_needed(const std::filesystem::path& model_dir) {
    auto tokenizer_path = model_dir / "openvino_tokenizer.xml";
    auto detokenizer_path = model_dir / "openvino_detokenizer.xml";
    
    // Check if tokenizer already exists
    if (std::filesystem::exists(tokenizer_path) && std::filesystem::exists(detokenizer_path)) {
        std::cout << "[Safetensors] OpenVINO tokenizer already exists, skipping conversion" << std::endl;
        return;
    }
    
    // Check if HuggingFace tokenizer exists
    auto hf_tokenizer_path = model_dir / "tokenizer.json";
    if (!std::filesystem::exists(hf_tokenizer_path)) {
        std::cout << "[Safetensors] Warning: No tokenizer.json found, skipping tokenizer conversion" << std::endl;
        return;
    }
    
    std::cout << "[Safetensors] Converting HuggingFace tokenizer to OpenVINO format..." << std::endl;
    
    // Build Python command
    // Escape backslashes for Windows paths
    std::string model_dir_str = model_dir.string();
    std::string tokenizer_path_str = tokenizer_path.string();
    std::string detokenizer_path_str = detokenizer_path.string();
    
    // Replace backslashes with forward slashes for Python compatibility
    std::replace(model_dir_str.begin(), model_dir_str.end(), '\\', '/');
    std::replace(tokenizer_path_str.begin(), tokenizer_path_str.end(), '\\', '/');
    std::replace(detokenizer_path_str.begin(), detokenizer_path_str.end(), '\\', '/');
    
    std::string python_cmd = 
        "python -c \""
        "from transformers import AutoTokenizer; "
        "from openvino_tokenizers import convert_tokenizer; "
        "from openvino import save_model; "
        "t = AutoTokenizer.from_pretrained('" + model_dir_str + "'); "
        "tok, detok = convert_tokenizer(t, with_detokenizer=True); "
        "save_model(tok, '" + tokenizer_path_str + "'); "
        "save_model(detok, '" + detokenizer_path_str + "'); "
        "print('Tokenizer conversion successful')\"";
    
    std::cout << "[Safetensors] Running: python -c \"...\"" << std::endl;
    
    int result = std::system(python_cmd.c_str());
    
    if (result != 0) {
        std::cerr << "[Safetensors] Warning: Tokenizer conversion failed (exit code: " << result << ")" << std::endl;
        std::cerr << "[Safetensors] Please ensure Python, transformers, and openvino_tokenizers are installed" << std::endl;
        std::cerr << "[Safetensors] You can manually convert with:" << std::endl;
        std::cerr << "  python -c \"from transformers import AutoTokenizer; "
                  << "from openvino_tokenizers import convert_tokenizer; "
                  << "from openvino import save_model; "
                  << "t = AutoTokenizer.from_pretrained('" << model_dir_str << "'); "
                  << "tok, detok = convert_tokenizer(t, with_detokenizer=True); "
                  << "save_model(tok, '" << tokenizer_path_str << "'); "
                  << "save_model(detok, '" << detokenizer_path_str << "')\"" << std::endl;
    } else {
        std::cout << "[Safetensors] Tokenizer conversion completed successfully" << std::endl;
    }
}

std::shared_ptr<ov::Model> create_from_safetensors(
    const std::filesystem::path& model_dir,
    bool enable_save_ov_model) {
    
    // Check if in-flight compression is enabled via environment variable
    auto compression_config = get_compression_config_from_env();
    if (compression_config.enabled) {
        std::cout << "[Safetensors] In-flight compression enabled via OV_GENAI_INFLIGHT_QUANT_MODE" << std::endl;
        return create_from_safetensors_compressed(model_dir, compression_config, enable_save_ov_model);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "[Safetensors] Loading model from: " << model_dir << std::endl;
    
    // Step 0: Convert tokenizer if needed (auto-conversion)
    convert_tokenizer_if_needed(model_dir);
    
    // Step 1: Load config
    std::cout << "[Safetensors] Loading config.json..." << std::endl;
    HFConfig config = load_hf_config(model_dir);
    
    std::cout << "[Safetensors] Model: " << config.model_type << std::endl;
    std::cout << "[Safetensors] Hidden size: " << config.hidden_size << std::endl;
    std::cout << "[Safetensors] Layers: " << config.num_hidden_layers << std::endl;
    std::cout << "[Safetensors] Attention heads: " << config.num_attention_heads << std::endl;
    std::cout << "[Safetensors] KV heads: " << config.kv_heads() << std::endl;
    
    // Step 2: Load weights
    std::cout << "[Safetensors] Loading safetensors weights..." << std::endl;
    SafetensorsData st_data = load_safetensors(model_dir);
    
    // Note: HuggingFace weight names already include ".weight" suffix (e.g., "model.embed_tokens.weight")
    // building_blocks functions append ".weight" to the key when looking up weights
    // So we keep the original names - no preprocessing needed
    // e.g., key="model.embed_tokens" -> looks for consts["model.embed_tokens.weight"]
    
    // Debug: Print first few weight names
    std::cout << "[Safetensors] First 10 weight names:" << std::endl;
    int count = 0;
    for (const auto& [name, tensor] : st_data.tensors) {
        if (count++ < 10) {
            std::cout << "  - " << name << " (shape: ";
            for (size_t i = 0; i < tensor.get_shape().size(); ++i) {
                std::cout << tensor.get_shape()[i];
                if (i < tensor.get_shape().size() - 1) std::cout << "x";
            }
            std::cout << ")" << std::endl;
        }
    }
    
    std::cout << "[Safetensors] Loaded " << st_data.tensors.size() << " weight tensors" << std::endl;

    // Step 2.5: Convert weight names from HF format to internal format
    // HF uses: model.layers.0.xxx, internal uses: model.layers[0].xxx
    // This conversion is required for both modeling API and building_blocks paths
    std::unordered_map<std::string, ov::Tensor> converted_weights;
    for (auto& [name, tensor] : st_data.tensors) {
        std::string converted_name = convert_weight_name(name);
        converted_weights[converted_name] = std::move(tensor);
    }
    st_data.tensors = std::move(converted_weights);

    auto load_finish_time = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        load_finish_time - start_time).count();
    std::cout << "[Safetensors] Weight loading done. Time: " << load_duration << "ms" << std::endl;
    
    // Step 4: Build model
    std::cout << "[Safetensors] Building OpenVINO model..." << std::endl;
    
    std::shared_ptr<ov::Model> model;
    const std::string& model_type = config.model_type;

    // Check if new modeling API should be used
    if ((model_type == "qwen3" || model_type == "smollm3")&& use_modeling_api()) {
        std::cout << "[Safetensors] Using new modeling API" << std::endl;
        model = create_model_with_modeling_api(config, st_data.tensors);
    } else if (model_type == "llama" || model_type == "qwen2" || model_type == "qwen3" || 
               model_type == "mistral" || model_type == "mixtral") {
        std::cout << "[Safetensors] Using legacy building_blocks" << std::endl;
        // Create qtype entries for building_blocks compatibility (only needed for legacy path)
        std::unordered_map<std::string, gguf_tensor_type> qtypes;
        create_qtype_entries(qtypes, st_data.tensors, config.num_hidden_layers);
        model = create_language_model(config, st_data.tensors, qtypes);
    } else {
        throw std::runtime_error("Unsupported model architecture '" + model_type + "'");
    }
    
    auto build_finish_time = std::chrono::high_resolution_clock::now();
    auto build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        build_finish_time - load_finish_time).count();
    std::cout << "[Safetensors] Model building done. Time: " << build_duration << "ms" << std::endl;
    
    // Step 5: Optionally save the model
    if (enable_save_ov_model) {
        std::filesystem::path save_path = model_dir / "openvino_model.xml";
        std::cout << "[Safetensors] Saving OpenVINO model to: " << save_path << std::endl;
        ov::genai::utils::save_openvino_model(model, save_path.string(), true);
    }
    
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "[Safetensors] Total time: " << total_duration << "ms" << std::endl;
    
    return model;
}

std::shared_ptr<ov::Model> create_from_safetensors_compressed(
    const std::filesystem::path& model_dir,
    const InFlightCompressionConfig& compression_config,
    bool enable_save_ov_model) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "[Safetensors] Loading model from: " << model_dir << std::endl;
    std::cout << "[Safetensors] In-flight compression:" << std::endl;
    std::cout << "[Safetensors]   Mode: " << compression_config.mode_string() << std::endl;
    std::cout << "[Safetensors]   Bits: " << compression_config.bits() << std::endl;
    std::cout << "[Safetensors]   Symmetric: " << (compression_config.symmetric() ? "yes" : "no") << std::endl;
    std::cout << "[Safetensors]   Group size: " << compression_config.group_size << std::endl;
    
    // Step 0: Convert tokenizer if needed
    convert_tokenizer_if_needed(model_dir);
    
    // Step 1: Load config
    std::cout << "[Safetensors] Loading config.json..." << std::endl;
    HFConfig config = load_hf_config(model_dir);
    
    std::cout << "[Safetensors] Model: " << config.model_type << std::endl;
    std::cout << "[Safetensors] Hidden size: " << config.hidden_size << std::endl;
    std::cout << "[Safetensors] Layers: " << config.num_hidden_layers << std::endl;
    std::cout << "[Safetensors] Attention heads: " << config.num_attention_heads << std::endl;
    std::cout << "[Safetensors] KV heads: " << config.kv_heads() << std::endl;
    
    // Step 2: Load weights
    std::cout << "[Safetensors] Loading safetensors weights..." << std::endl;
    SafetensorsData st_data = load_safetensors(model_dir);
    
    // Debug: Print first few weight names
    std::cout << "[Safetensors] First 10 weight names:" << std::endl;
    int count = 0;
    for (const auto& [name, tensor] : st_data.tensors) {
        if (count++ < 10) {
            std::cout << "  - " << name << " (shape: ";
            for (size_t i = 0; i < tensor.get_shape().size(); ++i) {
                std::cout << tensor.get_shape()[i];
                if (i < tensor.get_shape().size() - 1) std::cout << "x";
            }
            std::cout << ", dtype: " << tensor.get_element_type() << ")" << std::endl;
        }
    }
    
    std::cout << "[Safetensors] Loaded " << st_data.tensors.size() << " weight tensors" << std::endl;

    // Step 2.5: Convert weight names from HF format to internal format
    std::unordered_map<std::string, ov::Tensor> converted_weights;
    for (auto& [name, tensor] : st_data.tensors) {
        std::string converted_name = convert_weight_name(name);
        converted_weights[converted_name] = std::move(tensor);
    }
    st_data.tensors = std::move(converted_weights);

    auto load_finish_time = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        load_finish_time - start_time).count();
    std::cout << "[Safetensors] Weight loading done. Time: " << load_duration << "ms" << std::endl;
    
    // Step 3: Apply in-flight compression using C++ RTN quantization
    std::cout << "[Safetensors] Applying in-flight quantization (C++ RTN)..." << std::endl;
    std::cout << "[Safetensors] Expected memory reduction: ~" 
              << (16.0 / compression_config.bits()) << "x for compressed layers" << std::endl;
    
    // Identify weights to quantize (attention and MLP projections)
    std::vector<std::string> weight_keys_to_quantize;
    for (const auto& [name, tensor] : st_data.tensors) {
        // Only quantize 2D weight matrices, not biases/layernorms/embeddings
        if (name.find(".weight") != std::string::npos &&
            tensor.get_shape().size() == 2 &&
            (name.find("q_proj") != std::string::npos ||
             name.find("k_proj") != std::string::npos ||
             name.find("v_proj") != std::string::npos ||
             name.find("o_proj") != std::string::npos ||
             name.find("gate_proj") != std::string::npos ||
             name.find("up_proj") != std::string::npos ||
             name.find("down_proj") != std::string::npos)) {
            weight_keys_to_quantize.push_back(name);
        }
    }
    
    std::cout << "[Safetensors] Quantizing " << weight_keys_to_quantize.size() << " weight tensors..." << std::endl;
    
    int quantized_count = 0;
    
    // C++ RTN quantization
    for (const auto& key : weight_keys_to_quantize) {
        auto it = st_data.tensors.find(key);
        if (it == st_data.tensors.end()) continue;
        
        const ov::Tensor& original_weight = it->second;
        
        try {
            // Quantize using C++ RTN algorithm
            rtn::QuantizedWeight quantized;
            
            if (compression_config.mode == InFlightCompressionConfig::Mode::INT4_SYM) {
                quantized = rtn::quantize_int4_sym(original_weight, compression_config.group_size);
            } else if (compression_config.mode == InFlightCompressionConfig::Mode::INT4_ASYM) {
                quantized = rtn::quantize_int4_asym(original_weight, compression_config.group_size);
            } else if (compression_config.mode == InFlightCompressionConfig::Mode::INT8_SYM) {
                quantized = rtn::quantize_int8_sym(original_weight, compression_config.group_size);
            } else {
                // For asymmetric modes, fall back to symmetric for now
                // TODO: Implement asymmetric quantization
                if (compression_config.bits() == 4) {
                    quantized = rtn::quantize_int4_sym(original_weight, compression_config.group_size);
                } else {
                    quantized = rtn::quantize_int8_sym(original_weight, compression_config.group_size);
                }
            }
            
            // Extract base name (remove .weight suffix)
            std::string base_key = key.substr(0, key.length() - 7);  // Remove ".weight"
            
            // Replace original weight with compressed version
            st_data.tensors[key] = quantized.compressed;
            
            // Add scale tensor
            st_data.tensors[base_key + ".scales"] = quantized.scale;
            
            // Add zero point tensor (for asymmetric modes)
            if (quantized.has_zero_point) {
                st_data.tensors[base_key + ".biases"] = quantized.zero_point;
            }
            
            quantized_count++;
            if (quantized_count % 50 == 0 || quantized_count == static_cast<int>(weight_keys_to_quantize.size())) {
                std::cout << "[Safetensors] Quantized " << quantized_count << "/" 
                          << weight_keys_to_quantize.size() << " weights" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[Safetensors] WARNING: Failed to quantize " << key << ": " << e.what() << std::endl;
        }
    }
    
    auto quant_finish_time = std::chrono::high_resolution_clock::now();
    auto quant_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        quant_finish_time - load_finish_time).count();
    std::cout << "[Safetensors] Quantization done. Time: " << quant_duration << "ms" << std::endl;
    
    // Step 4: Build model with compression
    std::cout << "[Safetensors] Building OpenVINO model..." << std::endl;
    
    const std::string& model_type = config.model_type;
    if (model_type != "llama" && model_type != "qwen2" && model_type != "qwen3" && 
        model_type != "mistral" && model_type != "mixtral") {
        throw std::runtime_error(
            "In-flight compression only supported for llama, qwen2, qwen3, mistral, mixtral architectures. "
            "Got: '" + model_type + "'");
    }
    
    // Create qtype entries with compression enabled
    std::unordered_map<std::string, gguf_tensor_type> qtypes;
    create_qtype_entries(qtypes, st_data.tensors, config.num_hidden_layers, compression_config.mode);
    
    // Build model with compression-aware qtypes
    auto model = create_language_model(config, st_data.tensors, qtypes);
    
    auto build_finish_time = std::chrono::high_resolution_clock::now();
    auto build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        build_finish_time - load_finish_time).count();
    std::cout << "[Safetensors] Model building done. Time: " << build_duration << "ms" << std::endl;
    
    // Step 5: Optionally save the model
    if (enable_save_ov_model) {
        std::filesystem::path save_path = model_dir / "openvino_model.xml";
        std::cout << "[Safetensors] Saving OpenVINO model to: " << save_path << std::endl;
        ov::genai::utils::save_openvino_model(model, save_path.string(), true);
    }
    
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "[Safetensors] Total time: " << total_duration << "ms" << std::endl;
    
    return model;
}

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
