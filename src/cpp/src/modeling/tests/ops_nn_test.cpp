// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/nn.hpp"
#include "modeling/ops/tensor.hpp"
#include "modeling/tests/test_utils.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

std::shared_ptr<ov::Model> build_model_from_output(const ov::Output<ov::Node>& output,
                                                   const ov::ParameterVector& params) {
    auto result = std::make_shared<ov::op::v0::Result>(output);
    return std::make_shared<ov::Model>(ov::OutputVector{result}, params);
}

void run_model_test(const std::shared_ptr<ov::Model>& model,
                    const ov::Tensor& input_tensor,
                    const std::vector<float>& expected,
                    float tol = 1e-4f) {
    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(input_tensor);
    request.infer();
    ov::Tensor output = request.get_output_tensor();
    test_utils::expect_tensor_near(output, expected, tol);
}

std::vector<float> layer_norm_ref(const std::vector<float>& input,
                                  const std::vector<float>& weight,
                                  const std::vector<float>& bias,
                                  size_t rows,
                                  size_t cols,
                                  float eps) {
    std::vector<float> out(input.size());
    for (size_t r = 0; r < rows; ++r) {
        float mean = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            mean += input[r * cols + c];
        }
        mean /= static_cast<float>(cols);
        float var = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float diff = input[r * cols + c] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(cols);
        float inv = 1.0f / std::sqrt(var + eps);
        for (size_t c = 0; c < cols; ++c) {
            float norm = (input[r * cols + c] - mean) * inv;
            out[r * cols + c] = norm * weight[c] + bias[c];
        }
    }
    return out;
}

float gelu_tanh_ref(float x) {
    const float k0 = 0.7978845608f;
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(k0 * (x + k1 * x3)));
}

}  // namespace

TEST(OpsNN, LayerNorm) {
    ov::genai::modeling::OpContext ctx;
    const ov::Shape shape{1, 2, 3};
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    ov::genai::modeling::Tensor x(param, &ctx);

    std::vector<float> weight_data{1.0f, 0.5f, 2.0f};
    std::vector<float> bias_data{0.1f, -0.2f, 0.3f};
    auto weight = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{3}, weight_data);
    auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{3}, bias_data);
    ov::genai::modeling::Tensor w(weight, &ctx);
    ov::genai::modeling::Tensor b(bias, &ctx);

    auto out = ov::genai::modeling::ops::nn::layer_norm(x, w, &b, 1e-5f, -1);
    auto model = build_model_from_output(out.output(), {param});

    std::vector<float> input_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    ov::Tensor input_tensor(ov::element::f32, shape);
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    auto expected = layer_norm_ref(input_data, weight_data, bias_data, 2, 3, 1e-5f);
    run_model_test(model, input_tensor, expected, 1e-4f);
}

TEST(OpsNN, GeluTanh) {
    ov::genai::modeling::OpContext ctx;
    const ov::Shape shape{3};
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    ov::genai::modeling::Tensor x(param, &ctx);

    auto out = ov::genai::modeling::ops::nn::gelu(x, true);
    auto model = build_model_from_output(out.output(), {param});

    std::vector<float> input_data{-1.0f, 0.0f, 1.0f};
    std::vector<float> expected;
    expected.reserve(input_data.size());
    for (float v : input_data) {
        expected.push_back(gelu_tanh_ref(v));
    }
    ov::Tensor input_tensor(ov::element::f32, shape);
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    run_model_test(model, input_tensor, expected, 1e-4f);
}
