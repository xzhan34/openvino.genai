// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <memory>

#include "openvino/genai/visibility.hpp"
#include <openvino/openvino.hpp>

#include "modeling/ops/context.hpp"
#include "modeling/weights/weight_finalizer.hpp"
#include "modeling/weights/quantization_config.hpp"
#include "modeling/weights/quantization_selector.hpp"

namespace ov {
namespace genai {

// Forward declare QuantizedWeight
namespace rtn {
struct QuantizedWeight;
}

namespace safetensors {

// Re-export modeling API types for convenience and backward compatibility
using QuantizationConfig = ov::genai::modeling::weights::QuantizationConfig;
using WeightSelectionConfig = ov::genai::modeling::weights::WeightSelectionConfig;

/**
 * @brief Weight finalizer for safetensors format with optional in-flight quantization
 *
 * Supports:
 * - Direct weight loading (F32/F16/BF16)
 * - In-flight INT4/INT8 quantization during loading
 * - Zero-copy mode for efficiency
 * - Custom weight selection for quantization
 */
class OPENVINO_GENAI_EXPORTS SafetensorsWeightFinalizer : public ov::genai::modeling::weights::WeightFinalizer {
public:
    SafetensorsWeightFinalizer();
    explicit SafetensorsWeightFinalizer(const QuantizationConfig& config);
    ~SafetensorsWeightFinalizer() override;

    ov::genai::modeling::weights::FinalizedWeight finalize(
        const std::string& name,
        ov::genai::modeling::weights::WeightSource& source,
        ov::genai::modeling::OpContext& ctx) override;

private:
    /**
     * @brief Perform in-flight quantization on a tensor
     * @return QuantizedWeight struct containing compressed weights, scales, and zero-points
     */
    rtn::QuantizedWeight quantize_weight(
        const std::string& name,
        const ov::Tensor& tensor,
        ov::genai::modeling::weights::WeightSource& source,
        ov::genai::modeling::OpContext& ctx);
    
    /**
     * @brief Create dequantization subgraph from quantized result
     * Supports both INT4 (packed in U8) and INT8 quantization
     */
    ov::Output<ov::Node> create_dequant_subgraph(
        const std::string& name,
        const rtn::QuantizedWeight& quant_result,
        const ov::Shape& original_shape);
    
    /**
     * @brief Check if weight name corresponds to a MoE weight
     */
    bool is_moe_weight(const std::string& name) const;
    
    /**
     * @brief Create MoE-specific subgraph with accessible scales and zero-points
     * Returns FinalizedWeight with dequantized weight as primary and scales/zps in auxiliary
     */
    ov::genai::modeling::weights::FinalizedWeight create_moe_subgraph(
        const std::string& name,
        const rtn::QuantizedWeight& quant_result,
        const ov::Shape& original_shape,
        ov::genai::modeling::OpContext& ctx);

    std::unordered_map<std::string, ov::Output<ov::Node>> cache_;
    ov::genai::modeling::weights::QuantizationSelector selector_;

    // Statistics
    size_t total_weights_ = 0;
    size_t quantized_weights_ = 0;
};

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
