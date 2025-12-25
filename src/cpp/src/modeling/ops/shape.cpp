// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/shape.hpp"

#include <openvino/opsets/opset13.hpp>

namespace ov {
namespace genai {
namespace modeling {
namespace shape {

ov::Output<ov::Node> axis_i64(int64_t axis) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis});
}

}  // namespace shape
}  // namespace modeling
}  // namespace genai
}  // namespace ov
