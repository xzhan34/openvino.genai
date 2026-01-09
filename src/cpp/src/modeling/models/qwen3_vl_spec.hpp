// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "openvino/genai/visibility.hpp"

#include <nlohmann/json.hpp>

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct OPENVINO_GENAI_EXPORTS Qwen3VLRopeConfig {
    bool mrope_interleaved = false;
    std::vector<int32_t> mrope_section = {24, 20, 20};
    std::string rope_type = "default";
};

struct OPENVINO_GENAI_EXPORTS Qwen3VLTextConfig {
    std::string model_type = "qwen3_vl_text";
    int32_t vocab_size = 0;
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t head_dim = 0;
    int32_t max_position_embeddings = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    std::string hidden_act = "silu";
    bool attention_bias = false;
    float attention_dropout = 0.0f;
    bool tie_word_embeddings = false;
    std::string dtype = "float16";
    Qwen3VLRopeConfig rope;

    int32_t kv_heads() const;
    int32_t resolved_head_dim() const;
    void finalize();
    void validate() const;
};

struct OPENVINO_GENAI_EXPORTS Qwen3VLVisionConfig {
    std::string model_type = "qwen3_vl";
    int32_t depth = 0;
    int32_t hidden_size = 0;
    std::string hidden_act = "gelu_pytorch_tanh";
    int32_t intermediate_size = 0;
    int32_t num_heads = 0;
    int32_t in_channels = 3;
    int32_t patch_size = 16;
    int32_t spatial_merge_size = 2;
    int32_t temporal_patch_size = 2;
    int32_t out_hidden_size = 0;
    int32_t num_position_embeddings = 0;
    std::vector<int32_t> deepstack_visual_indexes;
    float initializer_range = 0.02f;

    int32_t head_dim() const;
    void finalize();
    void validate() const;
};

struct OPENVINO_GENAI_EXPORTS Qwen3VLConfig {
    std::string model_type = "qwen3_vl";
    std::vector<std::string> architectures;
    Qwen3VLTextConfig text;
    Qwen3VLVisionConfig vision;
    int32_t image_token_id = 0;
    int32_t video_token_id = 0;
    int32_t vision_start_token_id = 0;
    int32_t vision_end_token_id = 0;
    bool tie_word_embeddings = false;

    void finalize();
    void validate() const;

    static Qwen3VLConfig from_json(const nlohmann::json& data);
    static Qwen3VLConfig from_json_file(const std::filesystem::path& config_path);
};

struct OPENVINO_GENAI_EXPORTS Qwen3VLModuleNames {
    static constexpr const char* kRoot = "model";
    static constexpr const char* kVision = "visual";
    static constexpr const char* kText = "language_model";
    static constexpr const char* kLmHead = "lm_head";
    static std::string vision_block(int32_t index);
    static std::string deepstack_merger(int32_t index);
    static std::string text_layer(int32_t index);
};

struct OPENVINO_GENAI_EXPORTS Qwen3VLVisionIO {
    static constexpr const char* kPixelValues = "pixel_values";
    static constexpr const char* kGridThw = "grid_thw";
    static constexpr const char* kPosEmbeds = "pos_embeds";
    static constexpr const char* kRotaryCos = "rotary_cos";
    static constexpr const char* kRotarySin = "rotary_sin";
    static constexpr const char* kVisualEmbeds = "visual_embeds";
    static constexpr const char* kDeepstackEmbedsPrefix = "deepstack_embeds";
};

struct OPENVINO_GENAI_EXPORTS Qwen3VLTextIO {
    static constexpr const char* kInputIds = "input_ids";
    static constexpr const char* kInputsEmbeds = "inputs_embeds";
    static constexpr const char* kAttentionMask = "attention_mask";
    static constexpr const char* kPositionIds = "position_ids";
    static constexpr const char* kBeamIdx = "beam_idx";
    static constexpr const char* kVisualEmbeds = "visual_embeds";
    static constexpr const char* kVisualPosMask = "visual_pos_mask";
    static constexpr const char* kDeepstackEmbedsPrefix = "deepstack_embeds";
    static constexpr const char* kLogits = "logits";
};

struct OPENVINO_GENAI_EXPORTS Qwen3VLGraphSpec {
    static std::vector<std::string> vision_required_inputs(bool use_external_pos_embeds);
    static std::vector<std::string> vision_outputs(const Qwen3VLVisionConfig& cfg);
    static std::vector<std::string> text_required_inputs(bool use_inputs_embeds);
    static std::vector<std::string> text_optional_inputs();
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
