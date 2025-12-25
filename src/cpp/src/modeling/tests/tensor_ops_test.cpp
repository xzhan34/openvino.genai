// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/context.hpp"
#include "modeling/ops/tensor.hpp"

namespace {

std::vector<float> compute_expected(const std::vector<float>& input, size_t rows, size_t cols) {
    std::vector<float> out(rows, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        float sumsq = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float v = input[r * cols + c];
            sumsq += v * v;
        }
        float mean = sumsq / static_cast<float>(cols);
        out[r] = 1.0f / std::sqrt(mean + 1.0f);
    }
    return out;
}

}  // namespace

TEST(TensorOps, PowMeanRsqrt) {
    ov::genai::modeling::OpContext ctx;

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
    ov::genai::modeling::Tensor x(param, &ctx);

    auto y = x.pow(2.0f).mean(-1, true);
    auto z = (y + 1.0f).rsqrt();

    auto result = std::make_shared<ov::op::v0::Result>(z.output());
    auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{param});

    ov::Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto request = compiled.create_infer_request();

    std::vector<float> input = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    ov::Tensor input_tensor(ov::element::f32, ov::Shape{2, 3}, input.data());
    request.set_input_tensor(input_tensor);
    request.infer();

    ov::Tensor output = request.get_output_tensor();
    const float* out_data = output.data<const float>();

    auto expected = compute_expected(input, 2, 3);
    ASSERT_EQ(output.get_size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(out_data[i], expected[i], 1e-5f);
    }
}
