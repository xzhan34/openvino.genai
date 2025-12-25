// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace ops {

Tensor matmul(const Tensor& a, const Tensor& b, bool ta = false, bool tb = false);
Tensor reduce_mean(const Tensor& x, int64_t axis, bool keepdim = true);
Tensor rms(const Tensor& x, const Tensor& weight, float eps);

}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
