// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>

#include "openvino/genai/visibility.hpp"
#include <openvino/openvino.hpp>

#include "modeling/ops/context.hpp"
#include "modeling/weights/weight_finalizer.hpp"

namespace ov {
namespace genai {
namespace safetensors {

/**
 * @brief Weight finalizer for safetensors format
 *
 * Safetensors weights are typically in BF16/F16/F32 format and don't require
 * dequantization. This finalizer simply creates constant nodes from the tensors.
 */
class OPENVINO_GENAI_EXPORTS SafetensorsWeightFinalizer : public ov::genai::modeling::weights::WeightFinalizer {
public:
    SafetensorsWeightFinalizer();

    ov::genai::modeling::Tensor finalize(const std::string& name,
                                         ov::genai::modeling::weights::WeightSource& source,
                                         ov::genai::modeling::OpContext& ctx) override;

private:
    std::unordered_map<std::string, ov::Output<ov::Node>> cache_;
};

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
