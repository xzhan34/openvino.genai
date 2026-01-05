// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>

#include "modeling/weights/weight_source.hpp"

namespace ov {
namespace genai {
namespace safetensors {

/**
 * @brief Weight source for safetensors format
 *
 * This class provides access to weights loaded from HuggingFace safetensors files.
 * Weight names are expected to be in the converted format (model.layers[i].xxx).
 */
class SafetensorsWeightSource : public ov::genai::modeling::weights::WeightSource {
public:
    explicit SafetensorsWeightSource(const std::unordered_map<std::string, ov::Tensor>& tensors);

    std::vector<std::string> keys() const override;
    bool has(const std::string& name) const override;
    const ov::Tensor& get_tensor(const std::string& name) const override;

private:
    const std::unordered_map<std::string, ov::Tensor>& tensors_;
    std::vector<std::string> keys_;
};

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
