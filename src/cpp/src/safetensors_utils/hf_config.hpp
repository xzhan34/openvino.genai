// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <filesystem>
#include <map>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ov {
namespace genai {
namespace safetensors {

/**
 * @brief HuggingFace model configuration parsed from config.json
 */
struct HFConfig {
    // Model architecture
    std::string model_type;           // "qwen3", "llama", "mistral", etc.
    std::vector<std::string> architectures;  // ["Qwen3ForCausalLM"]
    
    // Model dimensions
    int hidden_size = 0;              // 2560 for Qwen3-4B
    int intermediate_size = 0;        // 9728 for Qwen3-4B (FFN hidden size)
    int num_hidden_layers = 0;        // 36 for Qwen3-4B
    int num_attention_heads = 0;      // 32 for Qwen3-4B
    int num_key_value_heads = 0;      // 8 for Qwen3-4B (GQA)
    int head_dim = 0;                 // 128 for Qwen3-4B
    int vocab_size = 0;               // 151936 for Qwen3-4B
    int max_position_embeddings = 0;  // 40960 for Qwen3-4B
    
    // Normalization
    float rms_norm_eps = 1e-6f;       // RMS normalization epsilon
    
    // RoPE (Rotary Position Embedding)
    float rope_theta = 10000.0f;      // 1000000 for Qwen3
    // TODO: Add rope_scaling support
    
    // Activation
    std::string hidden_act = "silu";  // "silu", "gelu", etc.
    
    // Token IDs
    int bos_token_id = 0;
    int eos_token_id = 0;
    int pad_token_id = -1;
    
    // Weight tying
    bool tie_word_embeddings = false;
    
    // Data type
    std::string torch_dtype = "float16";  // "bfloat16", "float16", "float32"
    
    // Computed properties
    int kv_heads() const { 
        return num_key_value_heads > 0 ? num_key_value_heads : num_attention_heads; 
    }
    
    int head_size() const {
        return head_dim > 0 ? head_dim : (hidden_size / num_attention_heads);
    }
};

/**
 * @brief Load HuggingFace config from config.json
 * 
 * @param model_dir Path to the model directory containing config.json
 * @return HFConfig Parsed configuration
 * @throws std::runtime_error if config.json is not found or invalid
 */
HFConfig load_hf_config(const std::filesystem::path& model_dir);

/**
 * @brief Convert HFConfig to a map for compatibility with existing code
 * 
 * This allows reusing building_blocks.cpp which expects a config map
 */
using ConfigValue = std::variant<std::monostate, float, int, std::string, std::vector<std::string>>;
std::map<std::string, ConfigValue> config_to_map(const HFConfig& config);

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
