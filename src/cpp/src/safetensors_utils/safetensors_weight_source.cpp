// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "safetensors_utils/safetensors_weight_source.hpp"

#include <stdexcept>

namespace ov {
namespace genai {
namespace safetensors {

SafetensorsWeightSource::SafetensorsWeightSource(const std::unordered_map<std::string, ov::Tensor>& tensors)
    : tensors_(tensors) {
    keys_.reserve(tensors_.size());
    for (const auto& [key, _] : tensors_) {
        keys_.push_back(key);
    }
}

std::vector<std::string> SafetensorsWeightSource::keys() const {
    return keys_;
}

bool SafetensorsWeightSource::has(const std::string& name) const {
    return tensors_.count(name) > 0;
}

const ov::Tensor& SafetensorsWeightSource::get_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
