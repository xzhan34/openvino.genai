// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/context.hpp"

#include <openvino/opsets/opset13.hpp>

namespace ov {
namespace genai {
namespace modeling {

ov::Output<ov::Node> OpContext::scalar_f32(float v) {
    auto it = f32_cache_.find(v);
    if (it != f32_cache_.end()) {
        return it->second;
    }
    auto node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {v});
    f32_cache_.emplace(v, node);
    return node;
}

ov::Output<ov::Node> OpContext::scalar_i64(int64_t v) {
    auto it = i64_cache_.find(v);
    if (it != i64_cache_.end()) {
        return it->second;
    }
    auto node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {v});
    i64_cache_.emplace(v, node);
    return node;
}

}  // namespace modeling
}  // namespace genai
}  // namespace ov
