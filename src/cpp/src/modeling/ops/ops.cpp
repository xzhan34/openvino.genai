// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/ops.hpp"

#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

namespace {

ov::genai::modeling::OpContext* resolve_context(const ov::genai::modeling::Tensor& a,
                                                const ov::genai::modeling::Tensor& b) {
    auto* a_ctx = a.context();
    auto* b_ctx = b.context();
    if (a_ctx && b_ctx && a_ctx != b_ctx) {
        OPENVINO_THROW("Tensor contexts do not match");
    }
    return a_ctx ? a_ctx : b_ctx;
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace ops {

Tensor matmul(const Tensor& a, const Tensor& b, bool ta, bool tb) {
    auto* ctx = resolve_context(a, b);
    auto node = std::make_shared<ov::op::v0::MatMul>(a.output(), b.output(), ta, tb);
    return Tensor(node, ctx);
}

Tensor reduce_mean(const Tensor& x, int64_t axis, bool keepdim) {
    return x.mean(axis, keepdim);
}

Tensor rms(const Tensor& x, const Tensor& weight, float eps) {
    auto orig_dtype = x.dtype();
    auto xf = x.to(ov::element::f32);
    auto var = xf.pow(2.0f).mean(-1, true);
    auto norm = xf * (var + eps).rsqrt();
    return norm.to(orig_dtype) * weight;
}

}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
