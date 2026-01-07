// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file model_config.hpp
 * @brief Unified model configuration structure
 * 
 * ModelConfig provides a format-agnostic representation of model configuration.
 * It can be constructed from different source formats (GGUF metadata, HuggingFace JSON).
 */

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include <filesystem>

#include <openvino/openvino.hpp>

// Forward declarations for format-specific types
namespace ov {
namespace genai {

// GGUF metadata type (from gguf.hpp)
using GGUFMetaData = std::variant<
    std::monostate,
    float,
    int,
    ov::Tensor,
    std::string,
    std::vector<std::string>,
    std::vector<int32_t>
>;

}  // namespace genai
}  // namespace ov

namespace ov {
namespace genai {
namespace loaders {

/**
 * @brief Unified model configuration
 * 
 * This structure contains all configuration fields needed to build a model,
 * regardless of the source format. Factory methods convert format-specific
 * configurations to this unified format.
 */
struct ModelConfig {
    // ========== Model identification ==========
    
    /// Architecture name (e.g., "llama", "qwen3", "mistral")
    std::string architecture;
    
    /// Model type from HuggingFace (e.g., "qwen3", "llama")
    std::string model_type;
    
    // ========== Model dimensions ==========
    
    /// Hidden layer size (e.g., 2560 for Qwen3-4B)
    int32_t hidden_size = 0;
    
    /// Intermediate size in MLP (e.g., 9728 for Qwen3-4B)
    int32_t intermediate_size = 0;
    
    /// Number of transformer layers (e.g., 36 for Qwen3-4B)
    int32_t num_hidden_layers = 0;
    
    /// Number of attention heads (e.g., 32 for Qwen3-4B)
    int32_t num_attention_heads = 0;
    
    /// Number of key-value heads for GQA (e.g., 8 for Qwen3-4B)
    int32_t num_key_value_heads = 0;
    
    /// Head dimension (hidden_size / num_attention_heads)
    int32_t head_dim = 0;
    
    /// Vocabulary size
    int32_t vocab_size = 0;
    
    /// Maximum sequence length
    int32_t max_position_embeddings = 0;
    
    // ========== Normalization ==========
    
    /// RMS normalization epsilon
    float rms_norm_eps = 1e-6f;
    
    // ========== RoPE (Rotary Position Embedding) ==========
    
    /// RoPE base frequency
    float rope_theta = 10000.0f;
    
    /// RoPE scaling factor
    float rope_scaling_factor = 1.0f;
    
    // ========== Other configurations ==========
    
    /// Whether to tie embedding and output weights
    bool tie_word_embeddings = false;
    
    /// Whether attention has bias
    bool attention_bias = false;
    
    /// Whether MLP has bias (SmolLM3)
    bool mlp_bias = false;
    
    /// Activation function (e.g., "silu", "gelu")
    std::string hidden_act = "silu";
    
    // ========== SmolLM3-specific ==========
    
    /// Interval for layers without RoPE (SmolLM3: every 4th layer has no RoPE)
    int32_t no_rope_layer_interval = 0;
    
    /// Explicit list of layers without RoPE (alternative to interval)
    std::vector<int32_t> no_rope_layers;
    
    /// Model data type (e.g., bf16, fp16)
    ov::element::Type dtype = ov::element::bf16;
    
    /// GGUF file type (for quantization info)
    int32_t file_type = 0;
    
    // ========== Factory methods ==========
    
    /**
     * @brief Create ModelConfig from GGUF metadata
     * 
     * @param meta GGUF metadata map
     * @return ModelConfig with values from GGUF
     */
    static ModelConfig from_gguf(const std::map<std::string, GGUFMetaData>& meta);
    
    /**
     * @brief Create ModelConfig from HuggingFace config.json
     * 
     * @param config_path Path to config.json file
     * @return ModelConfig with values from JSON
     */
    static ModelConfig from_hf_json(const std::filesystem::path& config_path);
    
    /**
     * @brief Get configuration value as GGUF-compatible format
     * 
     * This is used for compatibility with existing building_blocks code.
     * 
     * @return Map in GGUF metadata format
     */
    std::map<std::string, GGUFMetaData> to_gguf_format() const;
    
    /**
     * @brief Validate configuration completeness
     * 
     * @throws std::runtime_error if required fields are missing
     */
    void validate() const;
    
    /**
     * @brief Get a human-readable summary of the configuration
     */
    std::string summary() const;
};

}  // namespace loaders
}  // namespace genai
}  // namespace ov
