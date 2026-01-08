// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/nn.hpp"

#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/ops.hpp"

namespace {

std::vector<size_t> to_size_t(const std::vector<int64_t>& values) {
    std::vector<size_t> out;
    out.reserve(values.size());
    for (auto v : values) {
        out.push_back(static_cast<size_t>(v));
    }
    return out;
}

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
namespace nn {

Tensor conv3d(const Tensor& input,
              const Tensor& weight,
              const std::vector<int64_t>& strides,
              const std::vector<int64_t>& pads_begin,
              const std::vector<int64_t>& pads_end,
              const std::vector<int64_t>& dilations) {
    auto* ctx = resolve_context(input, weight);
    auto node = std::make_shared<ov::op::v1::Convolution>(input.output(),
                                                          weight.output(),
                                                          to_size_t(strides),
                                                          ov::CoordinateDiff(pads_begin.begin(), pads_begin.end()),
                                                          ov::CoordinateDiff(pads_end.begin(), pads_end.end()),
                                                          to_size_t(dilations));
    return Tensor(node, ctx);
}

Tensor conv3d(const Tensor& input,
              const Tensor& weight,
              const Tensor& bias,
              const std::vector<int64_t>& strides,
              const std::vector<int64_t>& pads_begin,
              const std::vector<int64_t>& pads_end,
              const std::vector<int64_t>& dilations) {
    auto conv = conv3d(input, weight, strides, pads_begin, pads_end, dilations);
    auto node = std::make_shared<ov::op::v1::Add>(conv.output(), bias.output(), ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, conv.context());
}

Tensor layer_norm(const Tensor& input,
                  const Tensor& weight,
                  const Tensor* bias,
                  float eps,
                  int64_t axis) {
    auto orig_dtype = input.dtype();
    auto x = input.to(ov::element::f32);
    auto mean = x.mean(axis, true);
    auto diff = x - mean;
    auto var = diff.pow(2.0f).mean(axis, true);
    auto norm = diff * (var + eps).rsqrt();
    auto out = norm.to(orig_dtype) * weight;
    if (bias) {
        out = out + *bias;
    }
    return out;
}

Tensor layer_norm(const Tensor& input,
                  const Tensor& weight,
                  float eps,
                  int64_t axis) {
    return layer_norm(input, weight, nullptr, eps, axis);
}

Tensor gelu(const Tensor& input, bool approximate) {
    auto* ctx = input.context();
    auto mode = approximate ? ov::op::GeluApproximationMode::TANH
                            : ov::op::GeluApproximationMode::ERF;
    auto node = std::make_shared<ov::op::v7::Gelu>(input.output(), mode);
    return Tensor(node, ctx);
}

}  // namespace nn
}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
