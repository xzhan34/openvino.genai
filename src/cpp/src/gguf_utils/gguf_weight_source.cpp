// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_utils/gguf_weight_source.hpp"

#include <algorithm>

#include <openvino/core/except.hpp>

namespace {

bool ends_with(const std::string& value, const std::string& suffix) {
    if (value.size() < suffix.size()) {
        return false;
    }
    return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

namespace ov {
namespace genai {
namespace gguf {

GGUFWeightSource::GGUFWeightSource(const std::unordered_map<std::string, ov::Tensor>& consts) : consts_(consts) {
    keys_.reserve(consts_.size());
    for (const auto& kv : consts_) {
        if (ends_with(kv.first, ".weight")) {
            keys_.push_back(kv.first);
        }
    }
    std::sort(keys_.begin(), keys_.end());
}

std::vector<std::string> GGUFWeightSource::keys() const {
    return keys_;
}

bool GGUFWeightSource::has(const std::string& name) const {
    return consts_.count(name) > 0;
}

const ov::Tensor& GGUFWeightSource::get_tensor(const std::string& name) const {
    auto it = consts_.find(name);
    if (it == consts_.end()) {
        OPENVINO_THROW("Unknown GGUF tensor: ", name);
    }
    return it->second;
}

}  // namespace gguf
}  // namespace genai
}  // namespace ov
