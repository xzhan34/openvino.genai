// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "modeling/weights/weight_finalizer.hpp"
#include "modeling/weights/quantization_config.hpp"
#include <openvino/openvino.hpp>

namespace ov {
namespace genai {
namespace modeling {
namespace weights {

/**
 * @brief Helper class for weight finalizers that support quantization
 * 
 * This class provides common selection logic that can be used by any
 * weight finalizer implementation (Safetensors, GGUF, etc.)
 * 
 * Usage:
 * ```cpp
 * class MyWeightFinalizer : public WeightFinalizer {
 * public:
 *     MyWeightFinalizer(const QuantizationConfig& config)
 *         : selector_(config) {}
 *     
 *     Tensor finalize(const std::string& name, WeightSource& source, OpContext& ctx) override {
 *         auto tensor = source.get_tensor(name);
 *         if (selector_.should_quantize(name, tensor.get_shape())) {
 *             // Apply quantization
 *         }
 *         // Return result
 *     }
 * 
 * private:
 *     QuantizationSelector selector_;
 * };
 * ```
 */
class QuantizationSelector {
public:
    QuantizationSelector() = default;
    explicit QuantizationSelector(const QuantizationConfig& config);
    
    /**
     * @brief Check if a weight should be quantized based on selection config
     * @param name Weight name
     * @param shape Weight shape
     * @param dtype Weight data type (optional, for dtype-based filtering)
     * @return true if weight should be quantized
     */
    bool should_quantize(const std::string& name, 
                        const ov::Shape& shape,
                        ov::element::Type dtype = ov::element::undefined) const;
    
    /**
     * @brief Get the quantization configuration
     */
    const QuantizationConfig& config() const { return config_; }
    
    /**
     * @brief Check if quantization is enabled
     */
    bool enabled() const { return config_.enabled(); }
    
private:
    QuantizationConfig config_;
};

}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov
