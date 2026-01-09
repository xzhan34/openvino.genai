// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/genai/visibility.hpp"
#include <openvino/openvino.hpp>

#include "modeling/weights/weight_source.hpp"
#include "safetensors_utils/safetensors_loader.hpp"

namespace ov {
namespace genai {
namespace safetensors {

/**
 * @brief Weight source for safetensors format
 *
 * This class provides access to weights loaded from HuggingFace safetensors files.
 * It handles name conversion from HF format (layers.N) to canonical format (layers[N])
 * for seamless integration with the modeling API.
 * 
 * Usage:
 * @code
 * auto data = safetensors::load_safetensors(model_dir);
 * auto source = std::make_shared<SafetensorsWeightSource>(std::move(data));
 * // source->keys() returns canonical names like "layers[0].self_attn.q_proj.weight"
 * @endcode
 */
class OPENVINO_GENAI_EXPORTS SafetensorsWeightSource : public ov::genai::modeling::weights::WeightSource {
public:
    /**
     * @brief Construct from loaded SafetensorsData (takes ownership)
     * 
     * @param data Loaded safetensors data (will be moved)
     */
    explicit SafetensorsWeightSource(SafetensorsData data);
    
    /**
     * @brief Construct from pre-converted tensor map (legacy compatibility)
     * 
     * This constructor is for backward compatibility with code that already
     * has tensors in canonical name format (e.g., safetensors_modeling.cpp).
     * 
     * @param tensors Tensor map with names already in canonical format (layers[N])
     */
    explicit SafetensorsWeightSource(const std::unordered_map<std::string, ov::Tensor>& tensors);

    std::vector<std::string> keys() const override;
    bool has(const std::string& name) const override;
    const ov::Tensor& get_tensor(const std::string& name) const override;

private:
    SafetensorsData m_data;  // Owns the tensor data (for SafetensorsData constructor)
    const std::unordered_map<std::string, ov::Tensor>* m_tensors_ref = nullptr;  // Reference (for legacy constructor)
    std::vector<std::string> m_keys;  // Canonical names
    std::unordered_map<std::string, std::string> m_canonical_to_hf;  // canonical -> HF name mapping
};

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
