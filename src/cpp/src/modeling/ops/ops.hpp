// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace ops {

Tensor matmul(const Tensor& a, const Tensor& b, bool ta = false, bool tb = false);
Tensor linear(const Tensor& x, const Tensor& weight);
Tensor reduce_mean(const Tensor& x, int64_t axis, bool keepdim = true);
Tensor gather(const Tensor& data, const Tensor& indices, int64_t axis);
Tensor slice(const Tensor& data, int64_t start, int64_t stop, int64_t step, int64_t axis);
Tensor concat(const std::vector<Tensor>& xs, int64_t axis);
Tensor rms(const Tensor& x, const Tensor& weight, float eps);

}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
