// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/layer/rms_norm.hpp"

#include <openvino/core/except.hpp>
#include <openvino/openvino.hpp>

namespace ov {
namespace genai {
namespace modeling {

RMSNorm::RMSNorm(const Tensor& weight, float eps) : Module(), weight_(weight), eps_(eps) {}

RMSNorm::RMSNorm(BuilderContext& ctx, const std::string& name, float eps, Module* parent)
    : Module(name, ctx, parent), eps_(eps) {
    weight_param_ = &register_parameter("weight");
}

WeightParameter& RMSNorm::weight_param() {
    if (!weight_param_) {
        OPENVINO_THROW("RMSNorm has no registered parameter");
    }
    return *weight_param_;
}

const WeightParameter& RMSNorm::weight_param() const {
    if (!weight_param_) {
        OPENVINO_THROW("RMSNorm has no registered parameter");
    }
    return *weight_param_;
}

const Tensor& RMSNorm::weight() const {
    if (weight_param_) {
        return weight_param_->value();
    }
    return weight_;
}

Tensor RMSNorm::operator()(const Tensor& x) const {
    auto orig_dtype = x.dtype();
    auto xf = x.to(ov::element::f32);
    auto var = xf.pow(2.0f).mean(-1, true);
    auto norm = xf * (var + eps_).rsqrt();
    return norm.to(orig_dtype) * weight();
}

}  // namespace modeling
}  // namespace genai
}  // namespace ov
