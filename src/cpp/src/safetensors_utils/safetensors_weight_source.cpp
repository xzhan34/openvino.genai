// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "safetensors_utils/safetensors_weight_source.hpp"

#include <stdexcept>

#include "loaders/weight_name_mapper.hpp"

namespace ov {
namespace genai {
namespace safetensors {

SafetensorsWeightSource::SafetensorsWeightSource(SafetensorsData data)
    : m_data(std::move(data)), m_tensors_ref(nullptr) {
    // Build key list with canonical names for load_model to use
    m_keys.reserve(m_data.tensors.size());
    for (const auto& [name, _] : m_data.tensors) {
        // Convert HF format (layers.N) to canonical format (layers[N])
        std::string canonical = loaders::WeightNameMapper::from_hf(name);
        m_keys.push_back(canonical);
        // Store mapping from canonical to original HF name
        m_canonical_to_hf[canonical] = name;
    }
}

SafetensorsWeightSource::SafetensorsWeightSource(const std::unordered_map<std::string, ov::Tensor>& tensors)
    : m_tensors_ref(&tensors) {
    // Legacy constructor: tensors are already in canonical format (layers[N])
    // No name conversion needed
    m_keys.reserve(tensors.size());
    for (const auto& [name, _] : tensors) {
        m_keys.push_back(name);
    }
}

std::vector<std::string> SafetensorsWeightSource::keys() const {
    return m_keys;
}

bool SafetensorsWeightSource::has(const std::string& name) const {
    // Legacy mode: direct lookup
    if (m_tensors_ref) {
        return m_tensors_ref->count(name) > 0;
    }
    
    // New mode: use canonical -> HF mapping
    auto cache_it = m_canonical_to_hf.find(name);
    if (cache_it != m_canonical_to_hf.end()) {
        return m_data.tensors.find(cache_it->second) != m_data.tensors.end();
    }
    // Try direct lookup (for non-layer names like embed_tokens, norm, lm_head)
    if (m_data.tensors.find(name) != m_data.tensors.end()) {
        return true;
    }
    // Try dynamic conversion: canonical layers[N] -> HF layers.N
    std::string hf_name = loaders::WeightNameMapper::to_hf(name);
    return m_data.tensors.find(hf_name) != m_data.tensors.end();
}

const ov::Tensor& SafetensorsWeightSource::get_tensor(const std::string& name) const {
    // Legacy mode: direct lookup
    if (m_tensors_ref) {
        auto it = m_tensors_ref->find(name);
        if (it != m_tensors_ref->end()) {
            return it->second;
        }
        throw std::runtime_error("Weight not found: " + name);
    }
    
    // New mode: use canonical -> HF mapping
    auto cache_it = m_canonical_to_hf.find(name);
    if (cache_it != m_canonical_to_hf.end()) {
        auto it = m_data.tensors.find(cache_it->second);
        if (it != m_data.tensors.end()) {
            return it->second;
        }
    }
    // Try direct lookup
    auto it = m_data.tensors.find(name);
    if (it != m_data.tensors.end()) {
        return it->second;
    }
    // Try dynamic conversion: canonical layers[N] -> HF layers.N
    std::string hf_name = loaders::WeightNameMapper::to_hf(name);
    it = m_data.tensors.find(hf_name);
    if (it != m_data.tensors.end()) {
        return it->second;
    }
    
    throw std::runtime_error("Weight not found: " + name + " (also tried: " + hf_name + ")");
}

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
