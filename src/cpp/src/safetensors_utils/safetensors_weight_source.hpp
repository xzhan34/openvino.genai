// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/genai/visibility.hpp"
#include <openvino/openvino.hpp>
#include <openvino/runtime/aligned_buffer.hpp>

#include "modeling/weights/weight_source.hpp"
#include "safetensors_utils/safetensors_loader.hpp"

namespace ov {
namespace genai {
namespace safetensors {

/**
 * @brief Weight source for safetensors format (Zero-Copy)
 *
 * This class provides access to weights loaded from HuggingFace safetensors files.
 * It handles name conversion from HF format (layers.N) to canonical format (layers[N])
 * for seamless integration with the modeling API.
 * 
 * Zero-Copy Mode:
 * - Uses mmap to map safetensors files into memory
 * - get_shared_buffer() returns SharedBuffer pointing directly to mmap
 * - No memory copy until GPU compile copies to usm_host
 * 
 * Usage:
 * @code
 * auto data = safetensors::load_safetensors(model_dir);  // Zero-copy load
 * auto source = std::make_shared<SafetensorsWeightSource>(std::move(data));
 * 
 * // For Constant creation (zero-copy):
 * auto info = source->get_info("layers[0].self_attn.q_proj.weight");
 * auto shared_buffer = source->get_shared_buffer("layers[0].self_attn.q_proj.weight");
 * auto constant = std::make_shared<ov::op::v0::Constant>(
 *     info.dtype, info.shape, shared_buffer);
 * @endcode
 */
class OPENVINO_GENAI_EXPORTS SafetensorsWeightSource : public ov::genai::modeling::weights::WeightSource {
public:
    /**
     * @brief Construct from loaded SafetensorsData (takes ownership, zero-copy mode)
     * 
     * @param data Loaded safetensors data (will be moved)
     */
    explicit SafetensorsWeightSource(SafetensorsData data);
    
    /**
     * @brief Construct from pre-converted tensor map (legacy compatibility, with copy)
     * 
     * This constructor is for backward compatibility with code that already
     * has tensors in canonical name format (e.g., safetensors_modeling.cpp).
     * 
     * @param tensors Tensor map with names already in canonical format (layers[N])
     */
    explicit SafetensorsWeightSource(const std::unordered_map<std::string, ov::Tensor>& tensors);

    std::vector<std::string> keys() const override;
    bool has(const std::string& name) const override;
    
    /**
     * @brief Get tensor data (legacy API, may trigger copy for zero-copy mode)
     * 
     * For zero-copy mode, this creates an ov::Tensor by copying from mmap.
     * Prefer using get_shared_buffer() for Constant creation to avoid copy.
     */
    const ov::Tensor& get_tensor(const std::string& name) const override;
    
    // =========================================================================
    // Zero-Copy API
    // =========================================================================
    
    /**
     * @brief Get tensor metadata (dtype, shape)
     * @param name Canonical weight name
     * @return TensorInfo with dtype and shape
     */
    const TensorInfo& get_info(const std::string& name) const;
    
    /**
     * @brief Get raw data pointer into mmap (zero-copy)
     * @param name Canonical weight name
     * @return Pair of (data pointer, size in bytes)
     */
    std::pair<const void*, size_t> get_data_ptr(const std::string& name) const;
    
    /**
     * @brief Get SharedBuffer for zero-copy Constant creation
     * 
     * The returned SharedBuffer points directly to the mmap'ed data and
     * keeps the mmap alive through shared_ptr reference counting.
     * 
     * Usage:
     * @code
     * auto info = source->get_info(name);
     * auto buffer = source->get_shared_buffer(name);
     * auto constant = std::make_shared<ov::op::v0::Constant>(
     *     info.dtype, info.shape, buffer);
     * @endcode
     * 
     * @param name Canonical weight name
     * @return SharedBuffer that can be passed to Constant constructor
     */
    std::shared_ptr<ov::AlignedBuffer> get_shared_buffer(const std::string& name) const;
    
    /**
     * @brief Check if running in zero-copy mode
     * 
     * Zero-copy mode is active when:
     * 1. Not using legacy tensors_ref, AND
     * 2. tensor_mmap_info is populated (mmap data available)
     */
    bool is_zero_copy_mode() const { 
        return m_tensors_ref == nullptr && !m_data.tensor_mmap_info.empty(); 
    }

private:
    /**
     * @brief Get HF name from canonical name
     */
    std::string get_hf_name(const std::string& canonical_name) const;
    
    SafetensorsData m_data;  // Owns the mmap data (for zero-copy mode)
    const std::unordered_map<std::string, ov::Tensor>* m_tensors_ref = nullptr;  // Reference (for legacy mode)
    std::vector<std::string> m_keys;  // Canonical names
    std::unordered_map<std::string, std::string> m_canonical_to_hf;  // canonical -> HF name mapping
    
    // Cache for legacy get_tensor() calls in zero-copy mode
    mutable std::unordered_map<std::string, ov::Tensor> m_tensor_cache;
};

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
