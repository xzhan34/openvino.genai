// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include "modeling/builder_context.hpp"
#include "modeling/ops/ops.hpp"

namespace {

std::vector<float> make_seq(size_t n, float start = 0.0f, float step = 1.0f) {
    std::vector<float> out(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        out[i] = start + step * static_cast<float>(i);
    }
    return out;
}

std::vector<float> matmul_ref(const std::vector<float>& a,
                              const std::vector<float>& b,
                              size_t m,
                              size_t k,
                              size_t n) {
    std::vector<float> out(m * n, 0.0f);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t kk = 0; kk < k; ++kk) {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    return out;
}

std::vector<float> matmul_ref_transpose_a(const std::vector<float>& a,
                                          const std::vector<float>& b,
                                          size_t m,
                                          size_t k,
                                          size_t n) {
    std::vector<float> out(m * n, 0.0f);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t kk = 0; kk < k; ++kk) {
                acc += a[kk * m + i] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    return out;
}

std::vector<float> linear_ref_3d(const std::vector<float>& x,
                                 const std::vector<float>& w,
                                 size_t batch,
                                 size_t seq_len,
                                 size_t in_features,
                                 size_t out_features) {
    std::vector<float> out(batch * seq_len * out_features, 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const size_t x_base = (b * seq_len + s) * in_features;
            const size_t y_base = (b * seq_len + s) * out_features;
            for (size_t o = 0; o < out_features; ++o) {
                float acc = 0.0f;
                for (size_t i = 0; i < in_features; ++i) {
                    acc += x[x_base + i] * w[o * in_features + i];
                }
                out[y_base + o] = acc;
            }
        }
    }
    return out;
}

std::vector<float> mean_ref(const std::vector<float>& x, size_t rows, size_t cols) {
    std::vector<float> out(rows, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        float acc = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            acc += x[r * cols + c];
        }
        out[r] = acc / static_cast<float>(cols);
    }
    return out;
}

std::vector<float> rms_ref(const std::vector<float>& x,
                           const std::vector<float>& weight,
                           size_t rows,
                           size_t cols,
                           float eps) {
    std::vector<float> out(x.size(), 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        float sumsq = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float v = x[r * cols + c];
            sumsq += v * v;
        }
        float mean = sumsq / static_cast<float>(cols);
        float inv = 1.0f / std::sqrt(mean + eps);
        for (size_t c = 0; c < cols; ++c) {
            out[r * cols + c] = x[r * cols + c] * inv * weight[c];
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

TEST(Ops, Matmul) {
    ov::genai::modeling::BuilderContext ctx;

    auto a = ctx.parameter("a", ov::element::f32, ov::Shape{2, 3});
    auto b = ctx.parameter("b", ov::element::f32, ov::Shape{3, 2});

    auto out = ov::genai::modeling::ops::matmul(a, b);
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto a_data = make_seq(6, 1.0f, 1.0f);
    auto b_data = make_seq(6, 7.0f, 1.0f);
    ov::Tensor a_tensor(ov::element::f32, {2, 3});
    ov::Tensor b_tensor(ov::element::f32, {3, 2});
    std::memcpy(a_tensor.data(), a_data.data(), a_data.size() * sizeof(float));
    std::memcpy(b_tensor.data(), b_data.data(), b_data.size() * sizeof(float));

    request.set_input_tensor(0, a_tensor);
    request.set_input_tensor(1, b_tensor);
    request.infer();

    auto expected = matmul_ref(a_data, b_data, 2, 3, 2);
    expect_tensor_near(request.get_output_tensor(), expected, 1e-3f);
}

TEST(Ops, MatmulTransposeA) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t m = 2;
    const size_t k = 3;
    const size_t n = 4;

    auto a = ctx.parameter("a", ov::element::f32, ov::Shape{k, m});
    auto b = ctx.parameter("b", ov::element::f32, ov::Shape{k, n});

    auto out = ov::genai::modeling::ops::matmul(a, b, true, false);
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto a_data = make_seq(k * m, 0.5f, 0.5f);
    auto b_data = make_seq(k * n, 1.0f, 0.25f);
    ov::Tensor a_tensor(ov::element::f32, {k, m});
    ov::Tensor b_tensor(ov::element::f32, {k, n});
    std::memcpy(a_tensor.data(), a_data.data(), a_data.size() * sizeof(float));
    std::memcpy(b_tensor.data(), b_data.data(), b_data.size() * sizeof(float));

    request.set_input_tensor(0, a_tensor);
    request.set_input_tensor(1, b_tensor);
    request.infer();

    auto expected = matmul_ref_transpose_a(a_data, b_data, m, k, n);
    expect_tensor_near(request.get_output_tensor(), expected, 1e-3f);
}

TEST(Ops, Linear) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 2;
    const size_t in_features = 3;
    const size_t out_features = 4;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{batch, seq_len, in_features});
    auto w = ctx.parameter("w", ov::element::f32, ov::Shape{out_features, in_features});

    auto out = ov::genai::modeling::ops::linear(x, w);
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto x_data = make_seq(batch * seq_len * in_features, 0.1f, 0.1f);
    auto w_data = make_seq(out_features * in_features, 0.2f, 0.05f);
    ov::Tensor x_tensor(ov::element::f32, {batch, seq_len, in_features});
    ov::Tensor w_tensor(ov::element::f32, {out_features, in_features});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));
    std::memcpy(w_tensor.data(), w_data.data(), w_data.size() * sizeof(float));

    request.set_input_tensor(0, x_tensor);
    request.set_input_tensor(1, w_tensor);
    request.infer();

    auto expected = linear_ref_3d(x_data, w_data, batch, seq_len, in_features, out_features);
    expect_tensor_near(request.get_output_tensor(), expected, 1e-3f);
}

TEST(Ops, ConstHelpers) {
    ov::genai::modeling::BuilderContext ctx;
    auto* op_ctx = &ctx.op_context();

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{2});

    auto two = ov::genai::modeling::Tensor(ov::genai::modeling::ops::const_scalar(op_ctx, 2.0f), op_ctx);
    auto scaled = x * two;

    auto idx =
        ov::genai::modeling::Tensor(ov::genai::modeling::ops::const_vec(op_ctx, std::vector<int64_t>{1}), op_ctx);
    auto picked = ov::genai::modeling::ops::gather(x, idx, 0);

    auto model = ctx.build_model({scaled.output(), picked.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> x_data = {3.0f, 5.0f};
    ov::Tensor x_tensor(ov::element::f32, {2});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));

    request.set_input_tensor(x_tensor);
    request.infer();

    expect_tensor_near(request.get_output_tensor(0), {6.0f, 10.0f}, 1e-3f);
    expect_tensor_near(request.get_output_tensor(1), {5.0f}, 1e-3f);
}

TEST(Ops, ReduceMean) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{2, 3});
    auto out = ov::genai::modeling::ops::reduce_mean(x, 1, false);

    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto x_data = make_seq(6, 1.0f, 1.0f);
    ov::Tensor x_tensor(ov::element::f32, {2, 3});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));

    request.set_input_tensor(x_tensor);
    request.infer();

    auto expected = mean_ref(x_data, 2, 3);
    expect_tensor_near(request.get_output_tensor(), expected, 1e-3f);
}

TEST(Ops, GatherSliceConcat) {
    ov::genai::modeling::BuilderContext ctx;

    auto data = ctx.parameter("data", ov::element::f32, ov::Shape{2, 5});
    auto indices = ctx.parameter("indices", ov::element::i64, ov::Shape{2});

    auto gathered = ov::genai::modeling::ops::gather(data, indices, 1);
    auto sliced = ov::genai::modeling::ops::slice(data, 1, 5, 2, 1);
    auto concatenated = ov::genai::modeling::ops::concat({gathered, sliced}, 1);

    auto model = ctx.build_model({gathered.output(), sliced.output(), concatenated.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto data_values = make_seq(10, 0.0f, 1.0f);
    ov::Tensor data_tensor(ov::element::f32, {2, 5});
    std::memcpy(data_tensor.data(), data_values.data(), data_values.size() * sizeof(float));

    std::vector<int64_t> indices_values = {0, 2};
    ov::Tensor idx_tensor(ov::element::i64, {2});
    std::memcpy(idx_tensor.data(), indices_values.data(), indices_values.size() * sizeof(int64_t));

    request.set_input_tensor(0, data_tensor);
    request.set_input_tensor(1, idx_tensor);
    request.infer();

    std::vector<float> expected_gather = {0.0f, 2.0f, 5.0f, 7.0f};
    std::vector<float> expected_slice = {1.0f, 3.0f, 6.0f, 8.0f};
    std::vector<float> expected_concat = {0.0f, 2.0f, 1.0f, 3.0f, 5.0f, 7.0f, 6.0f, 8.0f};

    expect_tensor_near(request.get_output_tensor(0), expected_gather, 1e-3f);
    expect_tensor_near(request.get_output_tensor(1), expected_slice, 1e-3f);
    expect_tensor_near(request.get_output_tensor(2), expected_concat, 1e-3f);
}

TEST(Ops, Rms) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{2, 3});
    auto w = ctx.parameter("w", ov::element::f32, ov::Shape{3});

    const float eps = 1.0f;
    auto out = ov::genai::modeling::ops::rms(x, w, eps);

    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto x_data = make_seq(6, 1.0f, 1.0f);
    auto w_data = std::vector<float>{0.5f, 1.0f, 1.5f};
    ov::Tensor x_tensor(ov::element::f32, {2, 3});
    ov::Tensor w_tensor(ov::element::f32, {3});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));
    std::memcpy(w_tensor.data(), w_data.data(), w_data.size() * sizeof(float));

    request.set_input_tensor(0, x_tensor);
    request.set_input_tensor(1, w_tensor);
    request.infer();

    auto expected = rms_ref(x_data, w_data, 2, 3, eps);
    expect_tensor_near(request.get_output_tensor(), expected, 1e-3f);
}
