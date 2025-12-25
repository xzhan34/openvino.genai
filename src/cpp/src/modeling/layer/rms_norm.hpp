// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {

class RMSNorm {
public:
    RMSNorm(const Tensor& weight, float eps);

    Tensor operator()(const Tensor& x) const;

private:
    Tensor weight_;
    float eps_ = 1e-6f;
};

}  // namespace modeling
}  // namespace genai
}  // namespace ov
