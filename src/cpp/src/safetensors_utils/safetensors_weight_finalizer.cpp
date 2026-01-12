// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

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

    std::shared_ptr<ov::op::v0::Constant> constant;
    ov::element::Type element_type;
    
    // Try zero-copy path first
    auto* st_source = dynamic_cast<SafetensorsWeightSource*>(&source);
    if (st_source && st_source->is_zero_copy_mode()) {
        // Zero-copy: use SharedBuffer directly
        const auto& info = st_source->get_info(name);
        auto shared_buffer = st_source->get_shared_buffer(name);
        
        constant = std::make_shared<ov::op::v0::Constant>(
            info.dtype, info.shape, shared_buffer);
        element_type = info.dtype;
    } else {
        // Legacy path: copy from tensor
        const ov::Tensor& tensor = source.get_tensor(name);
        constant = std::make_shared<ov::op::v0::Constant>(tensor);
        element_type = tensor.get_element_type();
    }
    
    constant->set_friendly_name(name);
    constant->output(0).set_names({name});

    // Convert to F32 if needed (matching GGUF behavior)
    // This ensures consistency with how GGUF weights are processed
    ov::Output<ov::Node> output;
    if (element_type == ov::element::bf16 || 
        element_type == ov::element::f16) {
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
