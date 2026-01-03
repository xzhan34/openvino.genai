// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_dense.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

ov::Tensor make_tensor(const std::vector<float>& data, const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::f32, shape);
    std::memcpy(tensor.data(), data.data(), data.size() * sizeof(float));
    return tensor;
}

std::vector<float> make_seq(size_t n, float start, float step) {
    std::vector<float> out(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        out[i] = start + step * static_cast<float>(i);
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

std::vector<float> mul_ref(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(a.size(), 0.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] * b[i];
    }
    return out;
}

std::vector<float> silu_ref(const std::vector<float>& x) {
    std::vector<float> out(x.size(), 0.0f);
    for (size_t i = 0; i < x.size(); ++i) {
        const float v = x[i];
        out[i] = v / (1.0f + std::exp(-v));
    }
    return out;
}

std::vector<float> mlp_ref(const std::vector<float>& x,
                           const std::vector<float>& gate_w,
                           const std::vector<float>& up_w,
                           const std::vector<float>& down_w,
                           size_t batch,
                           size_t seq_len,
                           size_t hidden,
                           size_t intermediate) {
    auto gate = linear_ref_3d(x, gate_w, batch, seq_len, hidden, intermediate);
    auto up = linear_ref_3d(x, up_w, batch, seq_len, hidden, intermediate);
    auto gated = mul_ref(silu_ref(gate), up);
    return linear_ref_3d(gated, down_w, batch, seq_len, intermediate, hidden);
}

void expect_tensor_near(const ov::Tensor& output, const std::vector<float>& expected, float tol) {
    ASSERT_EQ(output.get_size(), expected.size());
    const float* out_data = output.data<const float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(out_data[i], expected[i], tol);
    }
}

}  // namespace

TEST(Qwen3MLP, MatchesReference) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 2;
    const size_t hidden = 3;
    const size_t intermediate = 4;

    const ov::Shape mlp_up_shape{intermediate, hidden};
    const ov::Shape mlp_down_shape{hidden, intermediate};

    const auto gate_w = make_seq(intermediate * hidden, 0.01f, 0.02f);
    const auto up_w = make_seq(intermediate * hidden, 0.015f, 0.02f);
    const auto down_w = make_seq(hidden * intermediate, 0.02f, 0.02f);

    ov::genai::modeling::tests::DummyWeightSource weights;
    weights.add("mlp.gate_proj.weight", make_tensor(gate_w, mlp_up_shape));
    weights.add("mlp.up_proj.weight", make_tensor(up_w, mlp_up_shape));
    weights.add("mlp.down_proj.weight", make_tensor(down_w, mlp_down_shape));

    ov::genai::modeling::tests::DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::Qwen3DenseConfig cfg;
    cfg.hidden_size = static_cast<int32_t>(hidden);
    cfg.intermediate_size = static_cast<int32_t>(intermediate);
    cfg.num_hidden_layers = 1;
    cfg.hidden_act = "silu";

    ov::genai::modeling::models::Qwen3MLP mlp(ctx, "mlp", cfg);
    ov::genai::modeling::weights::load_model(mlp, weights, finalizer);

    auto input = ctx.parameter("input", ov::element::f32, ov::PartialShape{batch, seq_len, hidden});
    auto output = mlp.forward(input);
    auto ov_model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    const std::vector<float> input_data = {
        0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f,
    };
    ov::Tensor input_tensor(ov::element::f32, {batch, seq_len, hidden});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);

    request.infer();

    auto expected = mlp_ref(input_data, gate_w, up_w, down_w, batch, seq_len, hidden, intermediate);
    expect_tensor_near(request.get_output_tensor(), expected, 1e-3f);
}
