// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/layer/rms_norm.hpp"
#include "modeling/ops/context.hpp"
#include "modeling/ops/tensor.hpp"

namespace {

std::vector<float> rms_ref(const std::vector<float>& input,
                           const std::vector<float>& weight,
                           size_t rows,
                           size_t cols,
                           float eps) {
    std::vector<float> out(rows * cols, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        float sumsq = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float v = input[r * cols + c];
            sumsq += v * v;
        }
        float mean = sumsq / static_cast<float>(cols);
        float inv = 1.0f / std::sqrt(mean + eps);
        for (size_t c = 0; c < cols; ++c) {
            out[r * cols + c] = input[r * cols + c] * inv * weight[c];
        }
    }
    return out;
}

}  // namespace

TEST(RMSNormLayer, Basic) {
    ov::genai::modeling::OpContext ctx;

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    ov::genai::modeling::Tensor x(param, &ctx);

    std::vector<float> weight_vals = {1.0f, 0.5f, 2.0f, -1.0f};
    auto weight_node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4}, weight_vals);
    ov::genai::modeling::Tensor w(weight_node, &ctx);

    const float eps = 1e-6f;
    ov::genai::modeling::RMSNorm rms(w, eps);
    auto y = rms(x);

    auto result = std::make_shared<ov::op::v0::Result>(y.output());
    auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{param});

    // ov::serialize(model, "rms_ult_model_original.xml");

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");

    // ov::serialize(compiled.get_runtime_model(), "rms_ult_model_compiled.xml");

    auto request = compiled.create_infer_request();

    std::vector<float> input = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    ov::Tensor input_tensor(ov::element::f32, ov::Shape{2, 4}, input.data());
    request.set_input_tensor(input_tensor);
    request.infer();

    ov::Tensor output = request.get_output_tensor();
    const float* out_data = output.data<const float>();

    auto expected = rms_ref(input, weight_vals, 2, 4, eps);
    ASSERT_EQ(output.get_size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(out_data[i], expected[i], 1e-5f);
    }
}
