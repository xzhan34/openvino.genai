// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/nn.hpp"

#include <openvino/op/interpolate.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"

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

Tensor conv2d(const Tensor& input,
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

Tensor conv2d(const Tensor& input,
              const Tensor& weight,
              const Tensor& bias,
              const std::vector<int64_t>& strides,
              const std::vector<int64_t>& pads_begin,
              const std::vector<int64_t>& pads_end,
              const std::vector<int64_t>& dilations) {
    auto conv = conv2d(input, weight, strides, pads_begin, pads_end, dilations);
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

Tensor group_norm(const Tensor& input,
                  const Tensor& weight,
                  const Tensor* bias,
                  int64_t num_groups,
                  float eps) {
    if (num_groups <= 0) {
        OPENVINO_THROW("group_norm requires num_groups > 0");
    }
    auto* ctx = resolve_context(input, weight);
    auto orig_dtype = input.dtype();

    auto x = input.to(ov::element::f32);
    auto grouped = x.reshape({0, num_groups, -1});
    auto mean = grouped.mean(2, true);
    auto diff = grouped - mean;
    auto var = diff.pow(2.0f).mean(2, true);
    auto norm = diff * (var + eps).rsqrt();
    auto restored = norm.reshape(shape::of(input));
    auto out = restored.to(orig_dtype);

    auto scale = weight.to(orig_dtype).reshape({1, -1, 1, 1});
    out = out * scale;
    if (bias) {
        auto shift = bias->to(orig_dtype).reshape({1, -1, 1, 1});
        out = out + shift;
    }
    return out;
}

Tensor group_norm(const Tensor& input,
                  const Tensor& weight,
                  int64_t num_groups,
                  float eps) {
    return group_norm(input, weight, nullptr, num_groups, eps);
}

Tensor gelu(const Tensor& input, bool approximate) {
    auto* ctx = input.context();
    auto mode = approximate ? ov::op::GeluApproximationMode::TANH
                            : ov::op::GeluApproximationMode::ERF;
    auto node = std::make_shared<ov::op::v7::Gelu>(input.output(), mode);
    return Tensor(node, ctx);
}

Tensor upsample_nearest(const Tensor& input, int64_t scale_h, int64_t scale_w) {
    if (scale_h <= 0 || scale_w <= 0) {
        OPENVINO_THROW("upsample_nearest requires positive scales");
    }
    auto* ctx = input.context();
    auto h = shape::dim(input, 2);
    auto w = shape::dim(input, 3);
    auto h_scale = ops::const_scalar(ctx, static_cast<int64_t>(scale_h));
    auto w_scale = ops::const_scalar(ctx, static_cast<int64_t>(scale_w));
    auto out_h = std::make_shared<ov::op::v1::Multiply>(h, h_scale);
    auto out_w = std::make_shared<ov::op::v1::Multiply>(w, w_scale);
    auto sizes = shape::make({out_h, out_w});
    auto axes = ops::const_vec(ctx, std::vector<int64_t>{2, 3});

    ov::op::v11::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::v11::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ov::op::v11::Interpolate::ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = ov::op::v11::Interpolate::CoordinateTransformMode::ASYMMETRIC;
    attrs.nearest_mode = ov::op::v11::Interpolate::NearestMode::FLOOR;

    auto interp = std::make_shared<ov::op::v11::Interpolate>(input.output(), sizes, axes, attrs);
    return Tensor(interp, ctx);
}

}  // namespace nn
}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
