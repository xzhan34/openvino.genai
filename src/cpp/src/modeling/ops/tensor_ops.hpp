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
namespace tensor {

Tensor repeat(const Tensor& x, const std::vector<int64_t>& repeats);
Tensor repeat(const Tensor& x, std::initializer_list<int64_t> repeats);
Tensor tile(const Tensor& x, const std::vector<int64_t>& repeats);
Tensor tile(const Tensor& x, std::initializer_list<int64_t> repeats);
Tensor stack(const std::vector<Tensor>& xs, int64_t axis);
Tensor masked_scatter(const Tensor& input, const Tensor& mask, const Tensor& updates);
Tensor masked_add(const Tensor& input, const Tensor& mask, const Tensor& updates);

}  // namespace tensor
}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
