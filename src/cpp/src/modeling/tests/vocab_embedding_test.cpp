// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include "modeling/builder_context.hpp"
#include "modeling/layers/vocab_embedding.hpp"
#include "modeling/ops/ops.hpp"

namespace {

std::vector<float> embedding_ref(const std::vector<int64_t>& ids,
                                 const std::vector<float>& weight,
                                 size_t rows,
                                 size_t cols,
                                 size_t embed_dim) {
    std::vector<float> y(rows * cols * embed_dim, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            int64_t idx = ids[r * cols + c];
            for (size_t e = 0; e < embed_dim; ++e) {
                y[(r * cols + c) * embed_dim + e] = weight[static_cast<size_t>(idx) * embed_dim + e];
            }
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

}  // namespace

TEST(VocabEmbeddingLayer, Basic) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t vocab_size = 6;
    const size_t embed_dim = 4;
    const ov::Shape weight_shape{vocab_size, embed_dim};
    const ov::Shape ids_shape{2, 3};

    const std::vector<float> weight = {
        0.f, 1.f, 2.f, 3.f,      //
        10.f, 11.f, 12.f, 13.f,  //
        20.f, 21.f, 22.f, 23.f,  //
        30.f, 31.f, 32.f, 33.f,  //
        40.f, 41.f, 42.f, 43.f,  //
        50.f, 51.f, 52.f, 53.f,  //
    };
    const std::vector<int64_t> ids = {
        0, 2, 5,  //
        3, 1, 4,  //
    };
    const std::vector<float> expected = embedding_ref(ids, weight, ids_shape[0], ids_shape[1], embed_dim);

    auto ids_t = ctx.parameter("ids", ov::element::i64, ids_shape);

    ov::Tensor w_tensor(ov::element::f32, weight_shape);
    std::memcpy(w_tensor.data(), weight.data(), weight.size() * sizeof(float));
    auto w = ov::genai::modeling::ops::constant(w_tensor, &ctx.op_context());

    ov::genai::modeling::VocabEmbedding embed(w);
    auto y = embed.forward(ids_t);

    auto model = ctx.build_model({y.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    ov::Tensor ids_tensor(ov::element::i64, ids_shape);
    std::memcpy(ids_tensor.data(), ids.data(), ids.size() * sizeof(int64_t));
    request.set_input_tensor(ids_tensor);

    request.infer();

    expect_tensor_near(request.get_output_tensor(), expected, 1e-3f);
}


