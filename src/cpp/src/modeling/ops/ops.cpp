// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/ops.hpp"

#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/op/placeholder_extension.hpp>
#include <ov_ops/fully_connected.hpp>

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

Tensor linear(const Tensor& x, const Tensor& weight) {
    auto* ctx = resolve_context(x, weight);
    auto no_bias = std::make_shared<ov::op::internal::PlaceholderExtension>();
    auto node = std::make_shared<ov::op::internal::FullyConnected>(x.output(), weight.output(), no_bias);
    return Tensor(node, ctx);
}

Tensor reduce_mean(const Tensor& x, int64_t axis, bool keepdim) {
    return x.mean(axis, keepdim);
}

Tensor gather(const Tensor& data, const Tensor& indices, int64_t axis) {
    auto* ctx = resolve_context(data, indices);
    auto axis_node = ctx ? ctx->scalar_i64(axis) : ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
    auto node = std::make_shared<ov::op::v8::Gather>(data.output(), indices.output(), axis_node, 0);
    return Tensor(node, ctx);
}

Tensor slice(const Tensor& data, int64_t start, int64_t stop, int64_t step, int64_t axis) {
    auto* ctx = data.context();
    auto start_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {start});
    auto stop_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {stop});
    auto step_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {step});
    auto axes_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis});
    auto node = std::make_shared<ov::opset13::Slice>(data.output(), start_node, stop_node, step_node, axes_node);
    return Tensor(node, ctx);
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
