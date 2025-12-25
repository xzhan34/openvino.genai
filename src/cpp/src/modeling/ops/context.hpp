// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>

#include <openvino/openvino.hpp>

namespace ov {
namespace genai {
namespace modeling {

class OpContext {
public:
    ov::Output<ov::Node> scalar_f32(float v);
    ov::Output<ov::Node> scalar_i64(int64_t v);

private:
    std::unordered_map<float, ov::Output<ov::Node>> f32_cache_;
    std::unordered_map<int64_t, ov::Output<ov::Node>> i64_cache_;
};

}  // namespace modeling
}  // namespace genai
}  // namespace ov
