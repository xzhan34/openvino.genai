// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/context.hpp"
#include "modeling/ops/tensor.hpp"

namespace {

std::vector<float> make_seq(size_t n, float start = 0.0f, float step = 1.0f) {
    std::vector<float> out(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        out[i] = start + step * static_cast<float>(i);
    }
    return out;
}

std::vector<float> pow_mean_rsqrt_expected(const std::vector<float>& input, size_t rows, size_t cols) {
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

std::vector<float> mean_expected(const std::vector<float>& input, size_t rows, size_t cols) {
    std::vector<float> out(rows, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        float acc = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            acc += input[r * cols + c];
        }
        out[r] = acc / static_cast<float>(cols);
    }
    return out;
}

std::vector<float> to_heads_ref(const std::vector<float>& x,
                                size_t batch,
                                size_t seq_len,
                                size_t num_heads,
                                size_t head_dim) {
    std::vector<float> out(batch * num_heads * seq_len * head_dim, 0.0f);
    const size_t hidden = num_heads * head_dim;
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const size_t in_base = (b * seq_len + s) * hidden;
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t out_base = ((b * num_heads + h) * seq_len + s) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    out[out_base + d] = x[in_base + h * head_dim + d];
                }
            }
        }
    }
    return out;
}

void expect_tensor_near(const ov::Tensor& output, const std::vector<float>& expected, float tol) {
    ASSERT_EQ(output.get_size(), expected.size());
    const float* out_data = output.data<const float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(out_data[i], expected[i], tol);
    }
}

}  // namespace

TEST(TensorOps, TypeConversions) {
    ov::genai::modeling::OpContext ctx;

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
    ov::genai::modeling::Tensor x(param, &ctx);
    EXPECT_EQ(x.dtype(), ov::element::f32);

    auto y = x.to(ov::element::f16);
    EXPECT_EQ(y.dtype(), ov::element::f16);

    auto z = y.to(ov::element::f32);
    EXPECT_EQ(z.dtype(), ov::element::f32);

    auto result = std::make_shared<ov::op::v0::Result>(z.output());
    auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{param});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto input_data = make_seq(6, 0.0f, 1.0f);
    ov::Tensor input_tensor(ov::element::f32, {2, 3});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);
    request.infer();

    expect_tensor_near(request.get_output_tensor(), input_data, 1e-3f);
}

TEST(TensorOps, PowMeanRsqrtAndMean) {
    ov::genai::modeling::OpContext ctx;

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
    ov::genai::modeling::Tensor x(param, &ctx);

    auto y = x.pow(2.0f).mean(-1, true);
    auto z = (y + 1.0f).rsqrt();
    auto mean_flat = x.mean(1, false);

    auto z_result = std::make_shared<ov::op::v0::Result>(z.output());
    auto mean_result = std::make_shared<ov::op::v0::Result>(mean_flat.output());
    auto model = std::make_shared<ov::Model>(ov::OutputVector{z_result, mean_result},
                                             ov::ParameterVector{param});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto input_data = make_seq(6, 1.0f, 1.0f);
    ov::Tensor input_tensor(ov::element::f32, {2, 3});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);
    request.infer();

    auto expected_z = pow_mean_rsqrt_expected(input_data, 2, 3);
    auto expected_mean = mean_expected(input_data, 2, 3);

    expect_tensor_near(request.get_output_tensor(0), expected_z, 1e-3f);
    expect_tensor_near(request.get_output_tensor(1), expected_mean, 1e-3f);
}

TEST(TensorOps, ArithmeticOperators) {
    ov::genai::modeling::OpContext ctx;

    auto x_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
    auto y_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
    ov::genai::modeling::Tensor x(x_param, &ctx);
    ov::genai::modeling::Tensor y(y_param, &ctx);

    auto sum = x + y;
    auto sum_scalar = sum + 1.0f;
    auto sum_rev = 2.0f + x;
    auto prod = x * y;
    auto div = prod / (x + 1.0f);

    auto sum_result = std::make_shared<ov::op::v0::Result>(sum.output());
    auto sum_scalar_result = std::make_shared<ov::op::v0::Result>(sum_scalar.output());
    auto sum_rev_result = std::make_shared<ov::op::v0::Result>(sum_rev.output());
    auto prod_result = std::make_shared<ov::op::v0::Result>(prod.output());
    auto div_result = std::make_shared<ov::op::v0::Result>(div.output());
    auto model = std::make_shared<ov::Model>(
        ov::OutputVector{sum_result, sum_scalar_result, sum_rev_result, prod_result, div_result},
        ov::ParameterVector{x_param, y_param});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> x_data = {1.0f, 3.0f, 7.0f};
    std::vector<float> y_data = {4.0f, 4.0f, 8.0f};
    ov::Tensor x_tensor(ov::element::f32, {3});
    ov::Tensor y_tensor(ov::element::f32, {3});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));
    std::memcpy(y_tensor.data(), y_data.data(), y_data.size() * sizeof(float));

    request.set_input_tensor(0, x_tensor);
    request.set_input_tensor(1, y_tensor);
    request.infer();

    std::vector<float> expected_sum = {5.0f, 7.0f, 15.0f};
    std::vector<float> expected_sum_scalar = {6.0f, 8.0f, 16.0f};
    std::vector<float> expected_sum_rev = {3.0f, 5.0f, 9.0f};
    std::vector<float> expected_prod = {4.0f, 12.0f, 56.0f};
    std::vector<float> expected_div = {2.0f, 3.0f, 7.0f};

    expect_tensor_near(request.get_output_tensor(0), expected_sum, 1e-3f);
    expect_tensor_near(request.get_output_tensor(1), expected_sum_scalar, 1e-3f);
    expect_tensor_near(request.get_output_tensor(2), expected_sum_rev, 1e-3f);
    expect_tensor_near(request.get_output_tensor(3), expected_prod, 1e-3f);
    expect_tensor_near(request.get_output_tensor(4), expected_div, 1e-3f);
}

TEST(TensorShapeOps, ReshapePermuteMerge) {
    ov::genai::modeling::OpContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 2;
    const size_t num_heads = 2;
    const size_t head_dim = 3;
    const size_t hidden = num_heads * head_dim;

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{batch, seq_len, hidden});
    ov::genai::modeling::Tensor x(param, &ctx);

    auto heads = x.reshape({0, 0, static_cast<int64_t>(num_heads), static_cast<int64_t>(head_dim)})
                    .permute({0, 2, 1, 3});
    auto merged = heads.transpose({0, 2, 1, 3}).reshape({0, 0, static_cast<int64_t>(hidden)});

    auto heads_result = std::make_shared<ov::op::v0::Result>(heads.output());
    auto merged_result = std::make_shared<ov::op::v0::Result>(merged.output());
    auto model = std::make_shared<ov::Model>(ov::OutputVector{heads_result, merged_result},
                                             ov::ParameterVector{param});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto input_data = make_seq(batch * seq_len * hidden);
    ov::Tensor input_tensor(ov::element::f32, {batch, seq_len, hidden});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);
    request.infer();

    auto heads_output = request.get_output_tensor(0);
    auto merged_output = request.get_output_tensor(1);

    EXPECT_EQ(heads_output.get_shape(), (ov::Shape{batch, num_heads, seq_len, head_dim}));
    EXPECT_EQ(merged_output.get_shape(), (ov::Shape{batch, seq_len, hidden}));

    auto expected_heads = to_heads_ref(input_data, batch, seq_len, num_heads, head_dim);
    expect_tensor_near(heads_output, expected_heads, 1e-3f);
    expect_tensor_near(merged_output, input_data, 1e-3f);
}

TEST(TensorShapeOps, UnsqueezeSqueezeChain) {
    ov::genai::modeling::OpContext ctx;

    const size_t rows = 2;
    const size_t cols = 3;

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{rows, cols});
    ov::genai::modeling::Tensor x(param, &ctx);

    auto y = x.unsqueeze({0, 2}).squeeze({0, 2});
    auto result = std::make_shared<ov::op::v0::Result>(y.output());
    auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{param});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto input_data = make_seq(rows * cols);
    ov::Tensor input_tensor(ov::element::f32, {rows, cols});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);
    request.infer();

    auto output = request.get_output_tensor();
    EXPECT_EQ(output.get_shape(), (ov::Shape{rows, cols}));
    expect_tensor_near(output, input_data, 1e-6f);
}
