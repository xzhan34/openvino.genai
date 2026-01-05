// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace ops {

ov::Output<ov::Node> const_scalar(OpContext* ctx, float value);
ov::Output<ov::Node> const_scalar(OpContext* ctx, int64_t value);
ov::Output<ov::Node> const_scalar(OpContext* ctx, int32_t value);
ov::Output<ov::Node> const_scalar(OpContext* ctx, bool value);
ov::Output<ov::Node> const_vec(OpContext* ctx, const std::vector<float>& values);
ov::Output<ov::Node> const_vec(OpContext* ctx, const std::vector<int64_t>& values);
ov::Output<ov::Node> const_vec(OpContext* ctx, const std::vector<int32_t>& values);
Tensor constant(const ov::Tensor& tensor, OpContext* ctx = nullptr);

Tensor matmul(const Tensor& a, const Tensor& b, bool ta = false, bool tb = false);
Tensor linear(const Tensor& x, const Tensor& weight);
Tensor silu(const Tensor& x);
Tensor reduce_mean(const Tensor& x, int64_t axis, bool keepdim = true);
Tensor gather(const Tensor& data, const Tensor& indices, int64_t axis);
Tensor slice(const Tensor& data, int64_t start, int64_t stop, int64_t step, int64_t axis);
Tensor range(const Tensor& stop, int64_t start, int64_t step, const ov::element::Type& type);
Tensor range(const Tensor& start, const Tensor& stop, int64_t step, const ov::element::Type& type);
Tensor greater_equal(const Tensor& a, const Tensor& b);
Tensor less_equal(const Tensor& a, const Tensor& b);
Tensor where(const Tensor& cond, const Tensor& then_value, const Tensor& else_value);
Tensor concat(const std::vector<Tensor>& xs, int64_t axis);
Tensor rms(const Tensor& x, const Tensor& weight, float eps);

}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
