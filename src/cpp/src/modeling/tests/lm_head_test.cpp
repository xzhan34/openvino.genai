// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/layers/lm_head.hpp"
#include "modeling/ops/context.hpp"
#include "modeling/ops/tensor.hpp"

namespace {

std::vector<float> linear_ref(const std::vector<float>& x,
                              const std::vector<float>& w,
                              size_t rows,
                              size_t in_features,
                              size_t out_features) {
    std::vector<float> y(rows * out_features, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t o = 0; o < out_features; ++o) {
            float acc = 0.0f;
            for (size_t i = 0; i < in_features; ++i) {
                acc += x[r * in_features + i] * w[o * in_features + i];
            }
            y[r * out_features + o] = acc;
        }
    }
    return y;
}

void expect_tensor_near(const ov::Tensor& output, const std::vector<float>& expected, float tol) {
    ASSERT_EQ(output.get_size(), expected.size());
    const float* out_data = output.data<const float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(out_data[i], expected[i], tol);
    }
}

std::shared_ptr<ov::Model> build_model_from_output(const ov::Output<ov::Node>& output, const ov::ParameterVector& params) {
    auto result = std::make_shared<ov::op::v0::Result>(output);
    return std::make_shared<ov::Model>(ov::OutputVector{result}, params);
}

}  // namespace

TEST(LMHeadLayer, Decode) {
    ov::genai::modeling::OpContext ctx;

    const ov::Shape x_shape{2, 4};
    const ov::Shape w_shape{5, 4};

    const std::vector<float> x_data = {
        1.f, 2.f, 3.f, 4.f,  //
        5.f, 6.f, 7.f, 8.f,  //
    };
    const std::vector<float> w_data = {
        1.f, 0.f, 0.f, 0.f,   //
        0.f, 1.f, 0.f, 0.f,   //
        0.f, 0.f, 1.f, 0.f,   //
        0.f, 0.f, 0.f, 1.f,   //
        1.f, 1.f, 1.f, 1.f,   //
    };
    const std::vector<float> expected = linear_ref(x_data, w_data, x_shape[0], x_shape[1], w_shape[0]);

    auto x_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, x_shape);
    ov::genai::modeling::Tensor x(x_param, &ctx);

    auto w_node = ov::op::v0::Constant::create(ov::element::f32, w_shape, w_data);
    ov::genai::modeling::Tensor w(w_node, &ctx);

    ov::genai::modeling::LMHead lm_head(w);
    auto logits = lm_head.forward(x);

    auto model = build_model_from_output(logits.output(), {x_param});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    ov::Tensor input_tensor(ov::element::f32, x_shape);
    std::memcpy(input_tensor.data(), x_data.data(), x_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);
    request.infer();

    expect_tensor_near(request.get_output_tensor(), expected, 1e-3f);
}

TEST(LMHeadLayer, PrefillLastToken) {
    ov::genai::modeling::OpContext ctx;

    // Two sequences with lengths [2, 3] -> cu_seqlens_q = [0, 2, 5]
    const std::vector<int64_t> cu_seqlens = {0, 2, 5};
    const ov::Shape cu_shape{cu_seqlens.size()};

    const ov::Shape x_shape{5, 4};  // total_tokens=5, hidden=4
    const ov::Shape w_shape{5, 4};

    const std::vector<float> x_data = {
        1.f, 2.f, 3.f, 4.f,     // token 0
        5.f, 6.f, 7.f, 8.f,     // token 1 (seq0 last)
        9.f, 10.f, 11.f, 12.f,  // token 2
        13.f, 14.f, 15.f, 16.f, // token 3
        17.f, 18.f, 19.f, 20.f, // token 4 (seq1 last)
    };
    const std::vector<float> w_data = {
        1.f, 0.f, 0.f, 0.f,   //
        0.f, 1.f, 0.f, 0.f,   //
        0.f, 0.f, 1.f, 0.f,   //
        0.f, 0.f, 0.f, 1.f,   //
        1.f, 1.f, 1.f, 1.f,   //
    };

    const std::vector<float> x_last = {
        5.f, 6.f, 7.f, 8.f,      //
        17.f, 18.f, 19.f, 20.f,  //
    };
    const std::vector<float> expected = linear_ref(x_last, w_data, 2, x_shape[1], w_shape[0]);

    auto x_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, x_shape);
    auto cu_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, cu_shape);

    ov::genai::modeling::Tensor x(x_param, &ctx);
    ov::genai::modeling::Tensor cu(cu_param, &ctx);

    auto w_node = ov::op::v0::Constant::create(ov::element::f32, w_shape, w_data);
    ov::genai::modeling::Tensor w(w_node, &ctx);

    ov::genai::modeling::LMHead lm_head(w);
    auto logits = lm_head.forward(x, cu);

    auto model = build_model_from_output(logits.output(), {x_param, cu_param});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    ov::Tensor x_tensor(ov::element::f32, x_shape);
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));
    request.set_input_tensor(0, x_tensor);

    ov::Tensor cu_tensor(ov::element::i64, cu_shape);
    std::memcpy(cu_tensor.data(), cu_seqlens.data(), cu_seqlens.size() * sizeof(int64_t));
    request.set_input_tensor(1, cu_tensor);

    request.infer();

    expect_tensor_near(request.get_output_tensor(), expected, 1e-3f);
}


