// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/layer/rms_norm.hpp"

#include <openvino/openvino.hpp>

namespace ov {
namespace genai {
namespace modeling {

RMSNorm::RMSNorm(const Tensor& weight, float eps) : weight_(weight), eps_(eps) {}

Tensor RMSNorm::operator()(const Tensor& x) const {
    auto orig_dtype = x.dtype();
    auto xf = x.to(ov::element::f32);
    auto var = xf.pow(2.0f).mean(-1, true);
    auto norm = xf * (var + eps_).rsqrt();
    return norm.to(orig_dtype) * weight_;
}

}  // namespace modeling
}  // namespace genai
}  // namespace ov
