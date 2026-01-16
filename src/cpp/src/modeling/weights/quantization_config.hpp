// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <limits>

namespace ov {
namespace genai {
namespace modeling {
namespace weights {

/**
 * @brief Custom weight selection configuration for quantization
 * 
 * Provides flexible control over which weights to quantize using multiple strategies:
 * - Pattern-based (wildcard matching)
 * - Layer-based (layer index ranges)
 * - Type-based (attention, mlp, embeddings, etc.)
 * - Explicit lists (include/exclude specific weights)
 * - Size-based (minimum/maximum weight size)
 * 
 * Selection priority (highest to lowest):
 * 1. Size thresholds (applied first)
 * 2. Explicit exclude list
 * 3. Explicit include list
 * 4. Exclude patterns
 * 5. Include patterns
 * 6. Layer range
 * 7. Type-based flags
 */
struct WeightSelectionConfig {
    // Pattern-based selection (supports wildcards: *, ?)
    std::vector<std::string> include_patterns;  // Quantize weights matching these patterns
    std::vector<std::string> exclude_patterns;  // Don't quantize weights matching these patterns
    
    // Layer-based selection (e.g., only quantize layers 5-15)
    std::optional<std::pair<int, int>> layer_range;  // {start_layer, end_layer} inclusive
    
    // Type-based selection (default behavior if no patterns specified)
    bool quantize_attention = true;     // q_proj, k_proj, v_proj, o_proj
    bool quantize_mlp = true;           // gate_proj, up_proj, down_proj
    bool quantize_moe = true;           // MoE expert weights
    bool quantize_embeddings = false;   // embed_tokens
    bool quantize_lm_head = false;      // lm_head
    bool quantize_norm = false;         // Normalization layers (usually not beneficial)
    bool quantize_routers = false;      // MoE router/gate weights (usually not beneficial)
    
    // Explicit weight lists (highest priority)
    std::vector<std::string> include_weights;  // Exact weight names to quantize
    std::vector<std::string> exclude_weights;  // Exact weight names to skip
    
    // Size-based filtering
    size_t min_weight_size = 0;         // Only quantize weights with >= this many elements
    size_t max_weight_size = std::numeric_limits<size_t>::max();  // Only quantize weights with <= this many elements
    
    // Advanced options
    bool only_2d_weights = false;        // Only quantize 2D weight matrices
    bool verbose = false;               // Print quantization decisions
    
    /**
     * @brief Helper: Check if a weight name matches any pattern
     * @param name Weight name to check
     * @param patterns List of wildcard patterns (* = any sequence, ? = single char)
     * @return true if name matches any pattern
     */
    bool matches_pattern(const std::string& name, const std::vector<std::string>& patterns) const;
    
    /**
     * @brief Helper: Extract layer index from weight name (e.g., "layers[5].mlp.weight" -> 5)
     * @param name Weight name
     * @return Layer index if found, std::nullopt otherwise
     */
    std::optional<int> extract_layer_index(const std::string& name) const;
};

/**
 * @brief Quantization configuration for inflight quantization
 * 
 * This configuration is format-agnostic and can be used by any weight finalizer
 * that supports quantization (Safetensors, GGUF, etc.)
 */
struct QuantizationConfig {
    enum class Mode {
        NONE,          // No quantization
        INT4_SYM,      // INT4 symmetric
        INT4_ASYM,     // INT4 asymmetric  
        INT8_SYM,      // INT8 symmetric
        INT8_ASYM      // INT8 asymmetric
    };
    
    Mode mode = Mode::NONE;
    int group_size = 128;              // Group size for group-wise quantization
    
    // Legacy options (kept for backward compatibility)
    bool quantize_embeddings = false;  // Whether to quantize embeddings
    bool quantize_lm_head = false;     // Whether to quantize LM head
    
    // Custom weight selection (new!)
    WeightSelectionConfig selection;
    
    bool enabled() const { return mode != Mode::NONE; }
    int bits() const {
        return (mode == Mode::INT4_SYM || mode == Mode::INT4_ASYM) ? 4 : 8;
    }
    
    /**
     * @brief Convenience: Set selection from legacy options
     */
    void apply_legacy_options() {
        selection.quantize_embeddings = quantize_embeddings;
        selection.quantize_lm_head = quantize_lm_head;
    }
};

/**
 * @brief Parse quantization configuration from environment variables
 * 
 * Environment variables:
 *   OV_GENAI_INFLIGHT_QUANT_MODE: Quantization mode (INT4_SYM, INT4_ASYM, INT8_SYM, INT8_ASYM)
 *   OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE: Group size for quantization (default: 128)
 *   OV_GENAI_INFLIGHT_QUANT_INCLUDE: Comma-separated include patterns
 *   OV_GENAI_INFLIGHT_QUANT_EXCLUDE: Comma-separated exclude patterns
 *   OV_GENAI_INFLIGHT_QUANT_LAYER_RANGE: Layer range (e.g., "10-20")
 *   OV_GENAI_INFLIGHT_QUANT_WEIGHT_NAMES: Comma-separated explicit weight names
 *   OV_GENAI_INFLIGHT_QUANT_MIN_SIZE: Minimum weight size in bytes
 *   OV_GENAI_INFLIGHT_QUANT_MAX_SIZE: Maximum weight size in bytes
 * 
 * @return QuantizationConfig parsed from environment variables
 */
QuantizationConfig parse_quantization_config_from_env();

}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov
