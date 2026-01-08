// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_vl_spec.hpp"

#include <fstream>

#include <openvino/core/except.hpp>

#include "json_utils.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

namespace {

void read_json_file(const std::filesystem::path& path, nlohmann::json& data) {
    std::ifstream file(path);
    if (!file.is_open()) {
        OPENVINO_THROW("Failed to open config file: ", path.string());
    }
    file >> data;
}

std::filesystem::path resolve_config_path(const std::filesystem::path& path) {
    if (std::filesystem::is_directory(path)) {
        return path / "config.json";
    }
    return path;
}

void parse_rope_config(const nlohmann::json& data, Qwen3VLRopeConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "mrope_interleaved", cfg.mrope_interleaved);
    read_json_param(data, "mrope_section", cfg.mrope_section);
    read_json_param(data, "rope_type", cfg.rope_type);
}

void parse_text_config(const nlohmann::json& data, Qwen3VLTextConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "model_type", cfg.model_type);
    read_json_param(data, "vocab_size", cfg.vocab_size);
    read_json_param(data, "hidden_size", cfg.hidden_size);
    read_json_param(data, "intermediate_size", cfg.intermediate_size);
    read_json_param(data, "num_hidden_layers", cfg.num_hidden_layers);
    read_json_param(data, "num_attention_heads", cfg.num_attention_heads);
    read_json_param(data, "num_key_value_heads", cfg.num_key_value_heads);
    read_json_param(data, "head_dim", cfg.head_dim);
    read_json_param(data, "max_position_embeddings", cfg.max_position_embeddings);
    read_json_param(data, "rms_norm_eps", cfg.rms_norm_eps);
    read_json_param(data, "rope_theta", cfg.rope_theta);
    read_json_param(data, "hidden_act", cfg.hidden_act);
    read_json_param(data, "attention_bias", cfg.attention_bias);
    read_json_param(data, "attention_dropout", cfg.attention_dropout);
    read_json_param(data, "tie_word_embeddings", cfg.tie_word_embeddings);
    read_json_param(data, "dtype", cfg.dtype);

    if (data.contains("rope_scaling")) {
        parse_rope_config(data.at("rope_scaling"), cfg.rope);
    }
    if (data.contains("rope_parameters")) {
        parse_rope_config(data.at("rope_parameters"), cfg.rope);
    }

    cfg.finalize();
}

void parse_vision_config(const nlohmann::json& data, Qwen3VLVisionConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "model_type", cfg.model_type);
    read_json_param(data, "depth", cfg.depth);
    read_json_param(data, "hidden_size", cfg.hidden_size);
    read_json_param(data, "hidden_act", cfg.hidden_act);
    read_json_param(data, "intermediate_size", cfg.intermediate_size);
    read_json_param(data, "num_heads", cfg.num_heads);
    read_json_param(data, "in_channels", cfg.in_channels);
    read_json_param(data, "patch_size", cfg.patch_size);
    read_json_param(data, "spatial_merge_size", cfg.spatial_merge_size);
    read_json_param(data, "temporal_patch_size", cfg.temporal_patch_size);
    read_json_param(data, "out_hidden_size", cfg.out_hidden_size);
    read_json_param(data, "num_position_embeddings", cfg.num_position_embeddings);
    read_json_param(data, "deepstack_visual_indexes", cfg.deepstack_visual_indexes);
    read_json_param(data, "initializer_range", cfg.initializer_range);

    cfg.finalize();
}

void validate_index_range(const std::vector<int32_t>& indexes,
                          int32_t upper_bound,
                          const std::string& name) {
    for (int32_t idx : indexes) {
        if (idx < 0 || idx >= upper_bound) {
            OPENVINO_THROW("Invalid ", name, " index: ", idx);
        }
    }
}

}  // namespace

int32_t Qwen3VLTextConfig::kv_heads() const {
    return num_key_value_heads > 0 ? num_key_value_heads : num_attention_heads;
}

int32_t Qwen3VLTextConfig::resolved_head_dim() const {
    if (head_dim > 0) {
        return head_dim;
    }
    if (hidden_size > 0 && num_attention_heads > 0) {
        return hidden_size / num_attention_heads;
    }
    return 0;
}

void Qwen3VLTextConfig::finalize() {
    if (num_key_value_heads <= 0) {
        num_key_value_heads = num_attention_heads;
    }
    if (head_dim <= 0) {
        head_dim = resolved_head_dim();
    }
    if (rope.mrope_section.empty()) {
        rope.mrope_section = {24, 20, 20};
    }
}

void Qwen3VLTextConfig::validate() const {
    if (hidden_size <= 0) {
        OPENVINO_THROW("Qwen3VLTextConfig.hidden_size must be > 0");
    }
    if (num_hidden_layers <= 0) {
        OPENVINO_THROW("Qwen3VLTextConfig.num_hidden_layers must be > 0");
    }
    if (num_attention_heads <= 0) {
        OPENVINO_THROW("Qwen3VLTextConfig.num_attention_heads must be > 0");
    }
    if (kv_heads() <= 0) {
        OPENVINO_THROW("Qwen3VLTextConfig.num_key_value_heads must be > 0");
    }
    if (num_attention_heads % kv_heads() != 0) {
        OPENVINO_THROW("Qwen3VLTextConfig.num_attention_heads must be divisible by num_key_value_heads");
    }
    if (resolved_head_dim() <= 0) {
        OPENVINO_THROW("Qwen3VLTextConfig.head_dim must be > 0");
    }
    if (hidden_size % num_attention_heads != 0) {
        OPENVINO_THROW("Qwen3VLTextConfig.hidden_size must be divisible by num_attention_heads");
    }
    if (rope.mrope_interleaved && rope.mrope_section.size() != 3) {
        OPENVINO_THROW("Qwen3VLTextConfig.mrope_section must have 3 elements");
    }
}

int32_t Qwen3VLVisionConfig::head_dim() const {
    if (num_heads <= 0) {
        return 0;
    }
    return hidden_size / num_heads;
}

void Qwen3VLVisionConfig::finalize() {
    if (out_hidden_size <= 0) {
        out_hidden_size = hidden_size;
    }
}

void Qwen3VLVisionConfig::validate() const {
    if (depth <= 0) {
        OPENVINO_THROW("Qwen3VLVisionConfig.depth must be > 0");
    }
    if (hidden_size <= 0) {
        OPENVINO_THROW("Qwen3VLVisionConfig.hidden_size must be > 0");
    }
    if (num_heads <= 0) {
        OPENVINO_THROW("Qwen3VLVisionConfig.num_heads must be > 0");
    }
    if (hidden_size % num_heads != 0) {
        OPENVINO_THROW("Qwen3VLVisionConfig.hidden_size must be divisible by num_heads");
    }
    if (patch_size <= 0 || spatial_merge_size <= 0 || temporal_patch_size <= 0) {
        OPENVINO_THROW("Qwen3VLVisionConfig patch/merge sizes must be > 0");
    }
    if (out_hidden_size <= 0) {
        OPENVINO_THROW("Qwen3VLVisionConfig.out_hidden_size must be > 0");
    }
    if (num_position_embeddings <= 0) {
        OPENVINO_THROW("Qwen3VLVisionConfig.num_position_embeddings must be > 0");
    }
    if (!deepstack_visual_indexes.empty()) {
        validate_index_range(deepstack_visual_indexes, depth, "deepstack_visual_indexes");
    }
}

void Qwen3VLConfig::finalize() {
    if (model_type.empty()) {
        model_type = "qwen3_vl";
    }
    text.finalize();
    vision.finalize();
    if (text.tie_word_embeddings) {
        tie_word_embeddings = true;
    }
    if (tie_word_embeddings) {
        text.tie_word_embeddings = true;
    }
}

void Qwen3VLConfig::validate() const {
    if (model_type != "qwen3_vl") {
        OPENVINO_THROW("Unsupported model_type: ", model_type);
    }
    text.validate();
    vision.validate();
    if (image_token_id < 0 || vision_start_token_id < 0 || vision_end_token_id < 0) {
        OPENVINO_THROW("Invalid token ids in Qwen3VLConfig");
    }
}

Qwen3VLConfig Qwen3VLConfig::from_json(const nlohmann::json& data) {
    using ov::genai::utils::read_json_param;
    Qwen3VLConfig cfg;
    read_json_param(data, "model_type", cfg.model_type);
    read_json_param(data, "architectures", cfg.architectures);
    read_json_param(data, "image_token_id", cfg.image_token_id);
    read_json_param(data, "video_token_id", cfg.video_token_id);
    read_json_param(data, "vision_start_token_id", cfg.vision_start_token_id);
    read_json_param(data, "vision_end_token_id", cfg.vision_end_token_id);
    read_json_param(data, "tie_word_embeddings", cfg.tie_word_embeddings);

    if (data.contains("text_config")) {
        parse_text_config(data.at("text_config"), cfg.text);
    } else {
        OPENVINO_THROW("Qwen3VLConfig is missing text_config");
    }

    if (data.contains("vision_config")) {
        parse_vision_config(data.at("vision_config"), cfg.vision);
    } else {
        OPENVINO_THROW("Qwen3VLConfig is missing vision_config");
    }

    cfg.finalize();
    cfg.validate();
    return cfg;
}

Qwen3VLConfig Qwen3VLConfig::from_json_file(const std::filesystem::path& config_path) {
    auto resolved = resolve_config_path(config_path);
    if (!std::filesystem::exists(resolved)) {
        OPENVINO_THROW("Config file not found: ", resolved.string());
    }
    nlohmann::json data;
    read_json_file(resolved, data);
    return from_json(data);
}

std::string Qwen3VLModuleNames::vision_block(int32_t index) {
    return std::string("blocks.") + std::to_string(index);
}

std::string Qwen3VLModuleNames::deepstack_merger(int32_t index) {
    return std::string("deepstack_merger_list.") + std::to_string(index);
}

std::string Qwen3VLModuleNames::text_layer(int32_t index) {
    return std::string("layers.") + std::to_string(index);
}

std::vector<std::string> Qwen3VLGraphSpec::vision_required_inputs(bool use_external_pos_embeds) {
    std::vector<std::string> inputs = {
        Qwen3VLVisionIO::kPixelValues,
        Qwen3VLVisionIO::kGridThw,
    };
    if (use_external_pos_embeds) {
        inputs.emplace_back(Qwen3VLVisionIO::kPosEmbeds);
        inputs.emplace_back(Qwen3VLVisionIO::kRotaryCos);
        inputs.emplace_back(Qwen3VLVisionIO::kRotarySin);
    }
    return inputs;
}

std::vector<std::string> Qwen3VLGraphSpec::vision_outputs(const Qwen3VLVisionConfig& cfg) {
    std::vector<std::string> outputs = {Qwen3VLVisionIO::kVisualEmbeds};
    for (size_t i = 0; i < cfg.deepstack_visual_indexes.size(); ++i) {
        outputs.push_back(std::string(Qwen3VLVisionIO::kDeepstackEmbedsPrefix) + "." + std::to_string(i));
    }
    return outputs;
}

std::vector<std::string> Qwen3VLGraphSpec::text_required_inputs(bool use_inputs_embeds) {
    std::vector<std::string> inputs = {
        Qwen3VLTextIO::kAttentionMask,
        Qwen3VLTextIO::kPositionIds,
        Qwen3VLTextIO::kBeamIdx,
    };
    if (use_inputs_embeds) {
        inputs.emplace_back(Qwen3VLTextIO::kInputsEmbeds);
    } else {
        inputs.emplace_back(Qwen3VLTextIO::kInputIds);
    }
    return inputs;
}

std::vector<std::string> Qwen3VLGraphSpec::text_optional_inputs() {
    return {
        Qwen3VLTextIO::kInputsEmbeds,
        Qwen3VLTextIO::kVisualEmbeds,
        Qwen3VLTextIO::kVisualPosMask,
        Qwen3VLTextIO::kDeepstackEmbedsPrefix,
    };
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
