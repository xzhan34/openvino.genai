// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace ops {
namespace nn {

Tensor conv3d(const Tensor& input,
              const Tensor& weight,
              const std::vector<int64_t>& strides,
              const std::vector<int64_t>& pads_begin,
              const std::vector<int64_t>& pads_end,
              const std::vector<int64_t>& dilations = {1, 1, 1});

Tensor conv3d(const Tensor& input,
              const Tensor& weight,
              const Tensor& bias,
              const std::vector<int64_t>& strides,
              const std::vector<int64_t>& pads_begin,
              const std::vector<int64_t>& pads_end,
              const std::vector<int64_t>& dilations = {1, 1, 1});

Tensor layer_norm(const Tensor& input,
                  const Tensor& weight,
                  const Tensor* bias,
                  float eps,
                  int64_t axis = -1);

Tensor layer_norm(const Tensor& input,
                  const Tensor& weight,
                  float eps,
                  int64_t axis = -1);

Tensor gelu(const Tensor& input, bool approximate = true);

}  // namespace nn
}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
