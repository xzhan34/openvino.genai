// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <openvino/openvino.hpp>

namespace ov {
namespace genai {
namespace modeling {
namespace shape {

ov::Output<ov::Node> axis_i64(int64_t axis);

}  // namespace shape
}  // namespace modeling
}  // namespace genai
}  // namespace ov
