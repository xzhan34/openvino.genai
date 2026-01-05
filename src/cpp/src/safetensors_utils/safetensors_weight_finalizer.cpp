// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "safetensors_utils/safetensors_weight_finalizer.hpp"

#include <openvino/core/except.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/ops.hpp>

namespace ov {
namespace genai {
namespace safetensors {

SafetensorsWeightFinalizer::SafetensorsWeightFinalizer() = default;

ov::genai::modeling::Tensor SafetensorsWeightFinalizer::finalize(
    const std::string& name,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::OpContext& ctx) {
    if (!source.has(name)) {
        OPENVINO_THROW("Missing safetensors tensor: ", name);
    }

    // Check cache first
    const auto it_cached = cache_.find(name);
    if (it_cached != cache_.end()) {
        return ov::genai::modeling::Tensor(it_cached->second, &ctx);
    }

    // Get the tensor from source
    const ov::Tensor& tensor = source.get_tensor(name);

    // Create a constant node from the tensor
    auto constant = std::make_shared<ov::op::v0::Constant>(tensor);
    constant->set_friendly_name(name);
    constant->output(0).set_names({name});

    // Convert to F32 if needed (matching GGUF behavior)
    // This ensures consistency with how GGUF weights are processed
    ov::Output<ov::Node> output;
    if (tensor.get_element_type() == ov::element::bf16 || 
        tensor.get_element_type() == ov::element::f16) {
        auto converted = std::make_shared<ov::op::v0::Convert>(constant, ov::element::f32);
        output = converted->output(0);
    } else {
        output = constant->output(0);
    }
    
    cache_.emplace(name, output);

    return ov::genai::modeling::Tensor(output, &ctx);
}

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
