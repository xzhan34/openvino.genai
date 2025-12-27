// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/layer/lm_head.hpp"

#include <limits>

#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/ops.hpp"

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

ov::Output<ov::Node> scalar_i64(ov::genai::modeling::OpContext* ctx, int64_t v) {
    if (ctx) {
        return ctx->scalar_i64(v);
    }
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {v});
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {

LMHead::LMHead(const Tensor& weight) : weight_(weight) {}

Tensor LMHead::operator()(const Tensor& x) const {
    return ops::linear(x, weight_);
}

Tensor LMHead::operator()(const Tensor& x, const Tensor& cu_seqlens_q) const {
    auto* ctx = resolve_context(x, cu_seqlens_q);
    auto cu_i64 = cu_seqlens_q.to(ov::element::i64);

    // last_indices = cu_seqlens_q[1:] - 1
    auto cu_tail = ops::slice(cu_i64, 1, std::numeric_limits<int64_t>::max(), 1, 0);
    auto one = scalar_i64(ctx, 1);
    auto last_indices_node =
        std::make_shared<ov::op::v1::Subtract>(cu_tail.output(), one, ov::op::AutoBroadcastType::NUMPY);
    Tensor last_indices(last_indices_node, ctx);

    // x[last_indices]
    auto x_last = ops::gather(x, last_indices, 0);
    return ops::linear(x_last, weight_);
}

}  // namespace modeling
}  // namespace genai
}  // namespace ov

