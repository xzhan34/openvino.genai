// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "safetensors_utils/safetensors_weight_source.hpp"

#include <stdexcept>
#include <cstring>

#include "loaders/weight_name_mapper.hpp"
#include <openvino/runtime/shared_buffer.hpp>

namespace ov {
namespace genai {
namespace safetensors {

// =============================================================================
// SafetensorsWeightSource Implementation
// =============================================================================

SafetensorsWeightSource::SafetensorsWeightSource(SafetensorsData data)
    : m_data(std::move(data)), m_tensors_ref(nullptr) {
    
    // Build key list with canonical names
    // Use tensor_infos for zero-copy mode (no tensor data stored)
    if (!m_data.tensor_infos.empty()) {
        m_keys.reserve(m_data.tensor_infos.size());
        for (const auto& [name, _] : m_data.tensor_infos) {
            // Convert HF format (layers.N) to canonical format (layers[N])
            std::string canonical = loaders::WeightNameMapper::from_hf(name);
            m_keys.push_back(canonical);
            // Store mapping from canonical to original HF name
            m_canonical_to_hf[canonical] = name;
        }
    } else if (!m_data.tensors.empty()) {
        // Legacy path: tensors were copied
        m_keys.reserve(m_data.tensors.size());
        for (const auto& [name, _] : m_data.tensors) {
            std::string canonical = loaders::WeightNameMapper::from_hf(name);
            m_keys.push_back(canonical);
            m_canonical_to_hf[canonical] = name;
        }
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

std::string SafetensorsWeightSource::get_hf_name(const std::string& canonical_name) const {
    // Check cache first
    auto cache_it = m_canonical_to_hf.find(canonical_name);
    if (cache_it != m_canonical_to_hf.end()) {
        return cache_it->second;
    }
    // Try dynamic conversion
    return loaders::WeightNameMapper::to_hf(canonical_name);
}

bool SafetensorsWeightSource::has(const std::string& name) const {
    // Legacy mode: direct lookup
    if (m_tensors_ref) {
        return m_tensors_ref->count(name) > 0;
    }
    
    // Zero-copy mode: check tensor_infos
    std::string hf_name = get_hf_name(name);
    
    // Try tensor_mmap_info first (zero-copy)
    if (m_data.tensor_mmap_info.find(hf_name) != m_data.tensor_mmap_info.end()) {
        return true;
    }
    // Fallback to tensor_infos
    if (m_data.tensor_infos.find(hf_name) != m_data.tensor_infos.end()) {
        return true;
    }
    // Try direct name lookup
    if (m_data.tensor_mmap_info.find(name) != m_data.tensor_mmap_info.end()) {
        return true;
    }
    return m_data.tensor_infos.find(name) != m_data.tensor_infos.end();
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
    
    // Zero-copy mode: check cache first
    auto cache_it = m_tensor_cache.find(name);
    if (cache_it != m_tensor_cache.end()) {
        return cache_it->second;
    }
    
    // Legacy tensor data path (if tensors were copied)
    if (!m_data.tensors.empty()) {
        std::string hf_name = get_hf_name(name);
        auto it = m_data.tensors.find(hf_name);
        if (it != m_data.tensors.end()) {
            return it->second;
        }
        it = m_data.tensors.find(name);
        if (it != m_data.tensors.end()) {
            return it->second;
        }
    }
    
    // Zero-copy mode: create tensor by copying from mmap (cache it)
    auto [ptr, size] = get_data_ptr(name);
    const auto& info = get_info(name);
    
    ov::Tensor tensor(info.dtype, info.shape);
    std::memcpy(tensor.data(), ptr, size);
    
    // Cache the tensor
    m_tensor_cache[name] = std::move(tensor);
    return m_tensor_cache[name];
}

const TensorInfo& SafetensorsWeightSource::get_info(const std::string& name) const {
    std::string hf_name = get_hf_name(name);
    
    auto it = m_data.tensor_infos.find(hf_name);
    if (it != m_data.tensor_infos.end()) {
        return it->second;
    }
    
    // Try direct name
    it = m_data.tensor_infos.find(name);
    if (it != m_data.tensor_infos.end()) {
        return it->second;
    }
    
    throw std::runtime_error("Weight info not found: " + name + " (tried: " + hf_name + ")");
}

std::pair<const void*, size_t> SafetensorsWeightSource::get_data_ptr(const std::string& name) const {
    std::string hf_name = get_hf_name(name);
    
    auto it = m_data.tensor_mmap_info.find(hf_name);
    if (it == m_data.tensor_mmap_info.end()) {
        // Try direct name
        it = m_data.tensor_mmap_info.find(name);
    }
    
    if (it == m_data.tensor_mmap_info.end()) {
        throw std::runtime_error("Weight mmap info not found: " + name + " (tried: " + hf_name + ")");
    }
    
    const auto& mmap_info = it->second;
    const uint8_t* data_ptr = mmap_info.holder->data_buffer() + mmap_info.offset;
    
    return {data_ptr, mmap_info.size};
}

std::shared_ptr<ov::AlignedBuffer> SafetensorsWeightSource::get_shared_buffer(const std::string& name) const {
    std::string hf_name = get_hf_name(name);
    
    auto it = m_data.tensor_mmap_info.find(hf_name);
    if (it == m_data.tensor_mmap_info.end()) {
        // Try direct name
        it = m_data.tensor_mmap_info.find(name);
    }
    
    if (it == m_data.tensor_mmap_info.end()) {
        throw std::runtime_error("Weight mmap info not found for shared buffer: " + name);
    }
    
    const auto& mmap_info = it->second;
    char* data_ptr = const_cast<char*>(reinterpret_cast<const char*>(
        mmap_info.holder->data_buffer() + mmap_info.offset));
    
    // Create SharedBuffer that holds reference to mmap holder
    // ov::SharedBuffer<T> template keeps shared_ptr<MmapHolder> alive
    return std::make_shared<ov::SharedBuffer<std::shared_ptr<MmapHolder>>>(
        data_ptr,
        mmap_info.size,
        mmap_info.holder  // shared_ptr copy keeps mmap alive
    );
}

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
