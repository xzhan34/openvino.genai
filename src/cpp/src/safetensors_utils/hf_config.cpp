// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "safetensors_utils/hf_config.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

// Simple JSON parsing without external dependencies
// For production, consider using nlohmann/json or rapidjson

namespace ov {
namespace genai {
namespace safetensors {

namespace {

// Simple JSON value extraction helpers
std::string extract_string(const std::string& json, const std::string& key) {
    std::string search_key = "\"" + key + "\"";
    size_t pos = json.find(search_key);
    if (pos == std::string::npos) return "";
    
    pos = json.find(":", pos);
    if (pos == std::string::npos) return "";
    
    // Skip whitespace
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n')) pos++;
    
    if (pos >= json.size()) return "";
    
    // Check for null
    if (json.substr(pos, 4) == "null") return "";
    
    // Check for string value
    if (json[pos] == '"') {
        pos++;
        size_t end = json.find('"', pos);
        if (end == std::string::npos) return "";
        return json.substr(pos, end - pos);
    }
    
    return "";
}

int extract_int(const std::string& json, const std::string& key, int default_value = 0) {
    std::string search_key = "\"" + key + "\"";
    size_t pos = json.find(search_key);
    if (pos == std::string::npos) return default_value;
    
    pos = json.find(":", pos);
    if (pos == std::string::npos) return default_value;
    
    // Skip whitespace
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n')) pos++;
    
    if (pos >= json.size()) return default_value;
    
    // Check for null
    if (json.substr(pos, 4) == "null") return default_value;
    
    // Parse number
    try {
        size_t end_pos;
        int value = std::stoi(json.substr(pos), &end_pos);
        return value;
    } catch (...) {
        return default_value;
    }
}

float extract_float(const std::string& json, const std::string& key, float default_value = 0.0f) {
    std::string search_key = "\"" + key + "\"";
    size_t pos = json.find(search_key);
    if (pos == std::string::npos) return default_value;
    
    pos = json.find(":", pos);
    if (pos == std::string::npos) return default_value;
    
    // Skip whitespace
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n')) pos++;
    
    if (pos >= json.size()) return default_value;
    
    // Check for null
    if (json.substr(pos, 4) == "null") return default_value;
    
    // Parse number (handle scientific notation like 1e-06)
    try {
        size_t end_pos;
        float value = std::stof(json.substr(pos), &end_pos);
        return value;
    } catch (...) {
        return default_value;
    }
}

bool extract_bool(const std::string& json, const std::string& key, bool default_value = false) {
    std::string search_key = "\"" + key + "\"";
    size_t pos = json.find(search_key);
    if (pos == std::string::npos) return default_value;
    
    pos = json.find(":", pos);
    if (pos == std::string::npos) return default_value;
    
    // Skip whitespace
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n')) pos++;
    
    if (pos >= json.size()) return default_value;
    
    if (json.substr(pos, 4) == "true") return true;
    if (json.substr(pos, 5) == "false") return false;
    
    return default_value;
}

std::vector<std::string> extract_string_array(const std::string& json, const std::string& key) {
    std::vector<std::string> result;
    
    std::string search_key = "\"" + key + "\"";
    size_t pos = json.find(search_key);
    if (pos == std::string::npos) return result;
    
    pos = json.find("[", pos);
    if (pos == std::string::npos) return result;
    
    size_t end = json.find("]", pos);
    if (end == std::string::npos) return result;
    
    std::string array_str = json.substr(pos + 1, end - pos - 1);
    
    // Extract strings from array
    size_t start = 0;
    while ((start = array_str.find('"', start)) != std::string::npos) {
        start++;
        size_t str_end = array_str.find('"', start);
        if (str_end == std::string::npos) break;
        result.push_back(array_str.substr(start, str_end - start));
        start = str_end + 1;
    }
    
    return result;
}

}  // anonymous namespace

HFConfig load_hf_config(const std::filesystem::path& model_dir) {
    std::filesystem::path config_path = model_dir / "config.json";
    
    if (!std::filesystem::exists(config_path)) {
        throw std::runtime_error("config.json not found in " + model_dir.string());
    }
    
    // Read file content
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open " + config_path.string());
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();
    
    // Parse config
    HFConfig config;
    
    // Model architecture
    config.model_type = extract_string(json, "model_type");
    config.architectures = extract_string_array(json, "architectures");
    
    // Model dimensions
    config.hidden_size = extract_int(json, "hidden_size");
    config.intermediate_size = extract_int(json, "intermediate_size");
    config.num_hidden_layers = extract_int(json, "num_hidden_layers");
    config.num_attention_heads = extract_int(json, "num_attention_heads");
    config.num_key_value_heads = extract_int(json, "num_key_value_heads", config.num_attention_heads);
    config.head_dim = extract_int(json, "head_dim", 0);
    config.vocab_size = extract_int(json, "vocab_size");
    config.max_position_embeddings = extract_int(json, "max_position_embeddings", 2048);
    
    // Normalization
    config.rms_norm_eps = extract_float(json, "rms_norm_eps", 1e-6f);
    
    // RoPE
    config.rope_theta = extract_float(json, "rope_theta", 10000.0f);
    
    // Activation
    config.hidden_act = extract_string(json, "hidden_act");
    if (config.hidden_act.empty()) {
        config.hidden_act = "silu";  // Default for most models
    }
    
    // Token IDs
    config.bos_token_id = extract_int(json, "bos_token_id");
    config.eos_token_id = extract_int(json, "eos_token_id");
    config.pad_token_id = extract_int(json, "pad_token_id", -1);
    
    // Weight tying
    config.tie_word_embeddings = extract_bool(json, "tie_word_embeddings");
    
    // Data type
    config.torch_dtype = extract_string(json, "torch_dtype");
    if (config.torch_dtype.empty()) {
        config.torch_dtype = "float16";
    }
    
    // Validation
    if (config.hidden_size == 0) {
        throw std::runtime_error("Invalid config: hidden_size is 0");
    }
    if (config.num_hidden_layers == 0) {
        throw std::runtime_error("Invalid config: num_hidden_layers is 0");
    }
    if (config.num_attention_heads == 0) {
        throw std::runtime_error("Invalid config: num_attention_heads is 0");
    }
    
    return config;
}

std::map<std::string, ConfigValue> config_to_map(const HFConfig& config) {
    std::map<std::string, ConfigValue> result;
    
    // Map to keys expected by building_blocks.cpp
    result["architecture"] = config.model_type;
    result["hidden_size"] = config.hidden_size;
    result["intermediate_size"] = config.intermediate_size;
    result["layer_num"] = config.num_hidden_layers;
    result["head_count"] = config.num_attention_heads;
    result["head_count_kv"] = config.kv_heads();
    result["head_size"] = config.head_size();
    result["vocab_size"] = config.vocab_size;
    result["max_position_embeddings"] = config.max_position_embeddings;
    result["rms_norm_eps"] = config.rms_norm_eps;
    result["rope_freq_base"] = config.rope_theta;
    result["tie_word_embeddings"] = config.tie_word_embeddings ? 1 : 0;
    
    return result;
}

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
