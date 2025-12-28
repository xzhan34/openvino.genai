// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/models/qwen3_dense.hpp"

namespace {

class FakeWeightProvider : public ov::genai::modeling::weights::IWeightProvider {
public:
    FakeWeightProvider(ov::genai::modeling::OpContext* ctx,
                       const std::vector<float>& embed_weight,
                       const ov::Shape& embed_shape,
                       const std::vector<float>& norm_weight,
                       const ov::Shape& norm_shape,
                       const std::vector<float>& lm_head_weight,
                       const ov::Shape& lm_head_shape) :
        ctx_(ctx),
        embed_weight_(embed_weight),
        embed_shape_(embed_shape),
        norm_weight_(norm_weight),
        norm_shape_(norm_shape),
        lm_head_weight_(lm_head_weight),
        lm_head_shape_(lm_head_shape) {}

    bool has(const std::string& key) const override {
        return key == "model.embed_tokens.weight" || key == "model.norm.weight" || key == "lm_head.weight";
    }

    ov::genai::modeling::Tensor get(const std::string& base_key) override {
        if (base_key == "model.embed_tokens") {
            auto node = ov::op::v0::Constant::create(ov::element::f32, embed_shape_, embed_weight_);
            return ov::genai::modeling::Tensor(node, ctx_);
        }
        if (base_key == "model.norm") {
            auto node = ov::op::v0::Constant::create(ov::element::f32, norm_shape_, norm_weight_);
            return ov::genai::modeling::Tensor(node, ctx_);
        }
        if (base_key == "lm_head") {
            auto node = ov::op::v0::Constant::create(ov::element::f32, lm_head_shape_, lm_head_weight_);
            return ov::genai::modeling::Tensor(node, ctx_);
        }
        OPENVINO_THROW("Unknown base_key: ", base_key);
    }

private:
    ov::genai::modeling::OpContext* ctx_ = nullptr;
    std::vector<float> embed_weight_;
    ov::Shape embed_shape_;
    std::vector<float> norm_weight_;
    ov::Shape norm_shape_;
    std::vector<float> lm_head_weight_;
    ov::Shape lm_head_shape_;
};

std::vector<float> embedding_ref(const std::vector<int64_t>& ids,
                                 const std::vector<float>& weight,
                                 size_t batch,
                                 size_t seq_len,
                                 size_t vocab,
                                 size_t hidden) {
    (void)vocab;
    std::vector<float> out(batch * seq_len * hidden, 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const int64_t token = ids[b * seq_len + s];
            for (size_t h = 0; h < hidden; ++h) {
                out[(b * seq_len + s) * hidden + h] = weight[static_cast<size_t>(token) * hidden + h];
            }
        }
    }
    return out;
}

std::vector<float> rmsnorm_ref(const std::vector<float>& x,
                               const std::vector<float>& weight,
                               size_t batch,
                               size_t seq_len,
                               size_t hidden,
                               float eps) {
    std::vector<float> out(x.size(), 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const size_t base = (b * seq_len + s) * hidden;
            float sumsq = 0.0f;
            for (size_t h = 0; h < hidden; ++h) {
                const float v = x[base + h];
                sumsq += v * v;
            }
            const float mean = sumsq / static_cast<float>(hidden);
            const float inv = 1.0f / std::sqrt(mean + eps);
            for (size_t h = 0; h < hidden; ++h) {
                out[base + h] = x[base + h] * inv * weight[h];
            }
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

void expect_tensor_near(const ov::Tensor& output, const std::vector<float>& expected, float tol) {
    ASSERT_EQ(output.get_size(), expected.size());
    const float* out_data = output.data<const float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(out_data[i], expected[i], tol);
    }
}

}  // namespace

TEST(Qwen3DenseDummy, BuildsAndRuns) {
    ov::genai::modeling::OpContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 3;
    const size_t vocab = 6;
    const size_t hidden = 4;

    const ov::Shape embed_shape{vocab, hidden};
    const ov::Shape norm_shape{hidden};
    const ov::Shape lm_head_shape{vocab, hidden};

    const std::vector<float> embed_weight = {
        0.f, 1.f, 2.f, 3.f,      //
        10.f, 11.f, 12.f, 13.f,  //
        20.f, 21.f, 22.f, 23.f,  //
        30.f, 31.f, 32.f, 33.f,  //
        40.f, 41.f, 42.f, 43.f,  //
        50.f, 51.f, 52.f, 53.f,  //
    };
    const std::vector<float> norm_weight = {1.f, 1.f, 1.f, 1.f};
    const std::vector<float> lm_head_weight = {
        1.f, 0.f, 0.f, 0.f,    //
        0.f, 1.f, 0.f, 0.f,    //
        0.f, 0.f, 1.f, 0.f,    //
        0.f, 0.f, 0.f, 1.f,    //
        1.f, 1.f, 1.f, 1.f,    //
        -1.f, -1.f, -1.f, -1.f //
    };

    FakeWeightProvider weights(&ctx, embed_weight, embed_shape, norm_weight, norm_shape, lm_head_weight, lm_head_shape);

    ov::genai::modeling::models::Qwen3DenseConfig cfg;
    cfg.architecture = "qwen3";
    cfg.hidden_size = static_cast<int32_t>(hidden);
    cfg.rms_norm_eps = 1e-6f;

    auto model = ov::genai::modeling::models::build_qwen3_dense_dummy(cfg, weights, ctx);
    ov::serialize(model, "qwen3_dummy_original.xml");

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();
    ov::serialize(compiled.get_runtime_model(), "qwen3_dummy_compiled.xml");

    const std::vector<int64_t> input_ids = {0, 2, 5};
    const std::vector<int64_t> attention_mask = {1, 1, 0};
    const std::vector<int64_t> position_ids = {0, 1, 2};
    const std::vector<int32_t> beam_idx = {2};

    ov::Tensor input_ids_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(input_ids_tensor.data(), input_ids.data(), input_ids.size() * sizeof(int64_t));
    request.set_input_tensor(0, input_ids_tensor);

    ov::Tensor attention_mask_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(attention_mask_tensor.data(), attention_mask.data(), attention_mask.size() * sizeof(int64_t));
    request.set_input_tensor(1, attention_mask_tensor);

    ov::Tensor position_ids_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(position_ids_tensor.data(), position_ids.data(), position_ids.size() * sizeof(int64_t));
    request.set_input_tensor(2, position_ids_tensor);

    ov::Tensor beam_idx_tensor(ov::element::i32, {beam_idx.size()});
    std::memcpy(beam_idx_tensor.data(), beam_idx.data(), beam_idx.size() * sizeof(int32_t));
    request.set_input_tensor(3, beam_idx_tensor);

    request.infer();

    // Reference: embedding -> *mask -> +pos -> +sum(beam_idx) -> rmsnorm -> linear
    auto hidden0 = embedding_ref(input_ids, embed_weight, batch, seq_len, vocab, hidden);
    for (size_t i = 0; i < batch * seq_len; ++i) {
        const float mask = static_cast<float>(attention_mask[i]);
        const float pos = static_cast<float>(position_ids[i]);
        for (size_t h = 0; h < hidden; ++h) {
            hidden0[i * hidden + h] = hidden0[i * hidden + h] * mask + pos + static_cast<float>(beam_idx[0]);
        }
    }
    auto normed = rmsnorm_ref(hidden0, norm_weight, batch, seq_len, hidden, cfg.rms_norm_eps);
    auto expected = linear_ref_3d(normed, lm_head_weight, batch, seq_len, hidden, vocab);

    expect_tensor_near(request.get_output_tensor(), expected, 1e-3f);
}

