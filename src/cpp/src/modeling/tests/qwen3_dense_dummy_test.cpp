// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/core/except.hpp>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_dense.hpp"
#include "modeling/weights/weight_loader.hpp"
#include "modeling/weights/weight_finalizer.hpp"
#include "modeling/weights/weight_source.hpp"

namespace {

class DummyWeightSource : public ov::genai::modeling::weights::WeightSource {
public:
    void add(const std::string& name, const ov::Tensor& tensor) {
        if (!weights_.count(name)) {
            keys_.push_back(name);
        }
        weights_[name] = tensor;
    }

    std::vector<std::string> keys() const override {
        return keys_;
    }

    bool has(const std::string& name) const override {
        return weights_.count(name) != 0;
    }

    const ov::Tensor& get_tensor(const std::string& name) const override {
        auto it = weights_.find(name);
        if (it == weights_.end()) {
            OPENVINO_THROW("Unknown weight: ", name);
        }
        return it->second;
    }

private:
    std::unordered_map<std::string, ov::Tensor> weights_;
    std::vector<std::string> keys_;
};

class DummyWeightFinalizer : public ov::genai::modeling::weights::WeightFinalizer {
public:
    ov::genai::modeling::Tensor finalize(const std::string& name,
                                            ov::genai::modeling::weights::WeightSource& source,
                                            ov::genai::modeling::OpContext& ctx) override {
        const auto& tensor = source.get_tensor(name);
        auto node = std::make_shared<ov::op::v0::Constant>(tensor);
        return ov::genai::modeling::Tensor(node, &ctx);
    }
};

ov::Tensor make_tensor(const std::vector<float>& data, const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::f32, shape);
    std::memcpy(tensor.data(), data.data(), data.size() * sizeof(float));
    return tensor;
}

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

std::vector<float> make_seq(size_t n, float start, float step) {
    std::vector<float> out(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        out[i] = start + step * static_cast<float>(i);
    }
    return out;
}

std::vector<float> add_ref(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(a.size(), 0.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
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

TEST(Qwen3DenseDummy, BuildsAndRuns) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 3;
    const size_t vocab = 6;
    const size_t hidden = 4;
    const size_t intermediate = 6;
    const size_t num_layers = 2;

    const ov::Shape embed_shape{vocab, hidden};
    const ov::Shape norm_shape{hidden};
    const ov::Shape lm_head_shape{vocab, hidden};
    const ov::Shape mlp_up_shape{intermediate, hidden};
    const ov::Shape mlp_down_shape{hidden, intermediate};

    const std::vector<float> embed_weight = {
        0.f, 1.f, 2.f, 3.f,      //
        10.f, 11.f, 12.f, 13.f,  //
        20.f, 21.f, 22.f, 23.f,  //
        30.f, 31.f, 32.f, 33.f,  //
        40.f, 41.f, 42.f, 43.f,  //
        50.f, 51.f, 52.f, 53.f,  //
    };
    const std::vector<float> post_norm_weight0 = {1.0f, 0.9f, 1.1f, 1.05f};
    const std::vector<float> post_norm_weight1 = {0.95f, 1.05f, 1.0f, 0.9f};
    const std::vector<float> norm_weight = {1.f, 1.f, 1.f, 1.f};
    const std::vector<float> lm_head_weight = {
        1.f, 0.f, 0.f, 0.f,    //
        0.f, 1.f, 0.f, 0.f,    //
        0.f, 0.f, 1.f, 0.f,    //
        0.f, 0.f, 0.f, 1.f,    //
        1.f, 1.f, 1.f, 1.f,    //
        -1.f, -1.f, -1.f, -1.f //
    };
    const auto gate_w0 = make_seq(intermediate * hidden, 0.01f, 0.01f);
    const auto up_w0 = make_seq(intermediate * hidden, 0.02f, 0.01f);
    const auto down_w0 = make_seq(hidden * intermediate, 0.03f, 0.01f);
    const auto gate_w1 = make_seq(intermediate * hidden, 0.015f, 0.01f);
    const auto up_w1 = make_seq(intermediate * hidden, 0.025f, 0.01f);
    const auto down_w1 = make_seq(hidden * intermediate, 0.035f, 0.01f);

    DummyWeightSource weights;
    weights.add("model.embed_tokens.weight", make_tensor(embed_weight, embed_shape));
    weights.add("model.layers[0].post_attention_layernorm.weight", make_tensor(post_norm_weight0, norm_shape));
    weights.add("model.layers[0].mlp.gate_proj.weight", make_tensor(gate_w0, mlp_up_shape));
    weights.add("model.layers[0].mlp.up_proj.weight", make_tensor(up_w0, mlp_up_shape));
    weights.add("model.layers[0].mlp.down_proj.weight", make_tensor(down_w0, mlp_down_shape));
    weights.add("model.layers[1].post_attention_layernorm.weight", make_tensor(post_norm_weight1, norm_shape));
    weights.add("model.layers[1].mlp.gate_proj.weight", make_tensor(gate_w1, mlp_up_shape));
    weights.add("model.layers[1].mlp.up_proj.weight", make_tensor(up_w1, mlp_up_shape));
    weights.add("model.layers[1].mlp.down_proj.weight", make_tensor(down_w1, mlp_down_shape));
    weights.add("model.norm.weight", make_tensor(norm_weight, norm_shape));
    weights.add("lm_head.weight", make_tensor(lm_head_weight, lm_head_shape));

    DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::Qwen3DenseConfig cfg;
    cfg.architecture = "qwen3";
    cfg.hidden_size = static_cast<int32_t>(hidden);
    cfg.intermediate_size = static_cast<int32_t>(intermediate);
    cfg.num_hidden_layers = static_cast<int32_t>(num_layers);
    cfg.rms_norm_eps = 1e-6f;
    cfg.hidden_act = "silu";
    cfg.tie_word_embeddings = false;

    ov::genai::modeling::models::Qwen3ForCausalLM model(ctx, cfg);
    ov::genai::modeling::weights::load_model(model, weights, finalizer);

    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});

    auto logits = model.forward(input_ids, position_ids);
    auto ov_model = ctx.build_model({logits.output()});
    ov::serialize(ov_model, "qwen3_dummy_modular.xml");

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    const std::vector<int64_t> input_ids_data = {0, 2, 5};
    const std::vector<int64_t> position_ids_data = {0, 1, 2};

    ov::Tensor input_ids_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(input_ids_tensor.data(), input_ids_data.data(), input_ids_data.size() * sizeof(int64_t));
    request.set_input_tensor(0, input_ids_tensor);

    ov::Tensor position_ids_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(position_ids_tensor.data(), position_ids_data.data(), position_ids_data.size() * sizeof(int64_t));
    request.set_input_tensor(1, position_ids_tensor);

    request.infer();

    // Reference: embedding -> (post-attn rmsnorm -> mlp) * num_layers -> final rmsnorm -> linear
    auto hidden0 = embedding_ref(input_ids_data, embed_weight, batch, seq_len, vocab, hidden);
    std::vector<float> residual;
    bool has_residual = false;

    std::vector<float> normed;
    std::vector<float> sum;

    // Layer 0
    if (has_residual) {
        sum = add_ref(hidden0, residual);
        normed = rmsnorm_ref(sum, post_norm_weight0, batch, seq_len, hidden, cfg.rms_norm_eps);
        residual = sum;
    } else {
        normed = rmsnorm_ref(hidden0, post_norm_weight0, batch, seq_len, hidden, cfg.rms_norm_eps);
        residual = hidden0;
        has_residual = true;
    }
    hidden0 = mlp_ref(normed, gate_w0, up_w0, down_w0, batch, seq_len, hidden, intermediate);

    // Layer 1
    sum = add_ref(hidden0, residual);
    normed = rmsnorm_ref(sum, post_norm_weight1, batch, seq_len, hidden, cfg.rms_norm_eps);
    residual = sum;
    hidden0 = mlp_ref(normed, gate_w1, up_w1, down_w1, batch, seq_len, hidden, intermediate);

    // Final norm
    if (has_residual) {
        sum = add_ref(hidden0, residual);
        hidden0 = rmsnorm_ref(sum, norm_weight, batch, seq_len, hidden, cfg.rms_norm_eps);
    } else {
        hidden0 = rmsnorm_ref(hidden0, norm_weight, batch, seq_len, hidden, cfg.rms_norm_eps);
    }
    auto expected = linear_ref_3d(hidden0, lm_head_weight, batch, seq_len, hidden, vocab);

    expect_tensor_near(request.get_output_tensor(), expected, 1e-3f);
}

TEST(Qwen3DenseDummy, TiedWeights) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 2;
    const size_t vocab = 4;
    const size_t hidden = 3;
    const size_t intermediate = 4;
    const size_t num_layers = 1;

    const ov::Shape embed_shape{vocab, hidden};
    const ov::Shape norm_shape{hidden};
    const ov::Shape mlp_up_shape{intermediate, hidden};
    const ov::Shape mlp_down_shape{hidden, intermediate};

    const std::vector<float> embed_weight = {
        0.f, 1.f, 2.f,  //
        3.f, 4.f, 5.f,  //
        6.f, 7.f, 8.f,  //
        9.f, 10.f, 11.f //
    };
    const std::vector<float> post_norm_weight0 = {0.9f, 1.1f, 1.0f};
    const std::vector<float> norm_weight = {1.f, 1.f, 1.f};
    const auto gate_w0 = make_seq(intermediate * hidden, 0.01f, 0.02f);
    const auto up_w0 = make_seq(intermediate * hidden, 0.015f, 0.02f);
    const auto down_w0 = make_seq(hidden * intermediate, 0.02f, 0.02f);

    DummyWeightSource weights;
    weights.add("model.embed_tokens.weight", make_tensor(embed_weight, embed_shape));
    weights.add("model.layers[0].post_attention_layernorm.weight", make_tensor(post_norm_weight0, norm_shape));
    weights.add("model.layers[0].mlp.gate_proj.weight", make_tensor(gate_w0, mlp_up_shape));
    weights.add("model.layers[0].mlp.up_proj.weight", make_tensor(up_w0, mlp_up_shape));
    weights.add("model.layers[0].mlp.down_proj.weight", make_tensor(down_w0, mlp_down_shape));
    weights.add("model.norm.weight", make_tensor(norm_weight, norm_shape));

    DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::Qwen3DenseConfig cfg;
    cfg.architecture = "qwen3";
    cfg.hidden_size = static_cast<int32_t>(hidden);
    cfg.intermediate_size = static_cast<int32_t>(intermediate);
    cfg.num_hidden_layers = static_cast<int32_t>(num_layers);
    cfg.rms_norm_eps = 1e-6f;
    cfg.hidden_act = "silu";
    cfg.tie_word_embeddings = true;

    ov::genai::modeling::models::Qwen3ForCausalLM model(ctx, cfg);
    ov::genai::modeling::weights::load_model(model, weights, finalizer);

    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});

    auto logits = model.forward(input_ids, position_ids);
    auto ov_model = ctx.build_model({logits.output()});

    ov::serialize(ov_model, "qwen3_dummy_original.xml");

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    ov::serialize(compiled.get_runtime_model(), "qwen3_dummy_compiled.xml");

    const std::vector<int64_t> input_ids_data = {0, 3};
    const std::vector<int64_t> position_ids_data = {0, 1};

    ov::Tensor input_ids_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(input_ids_tensor.data(), input_ids_data.data(), input_ids_data.size() * sizeof(int64_t));
    request.set_input_tensor(0, input_ids_tensor);

    ov::Tensor position_ids_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(position_ids_tensor.data(), position_ids_data.data(), position_ids_data.size() * sizeof(int64_t));
    request.set_input_tensor(1, position_ids_tensor);

    request.infer();

    auto hidden0 = embedding_ref(input_ids_data, embed_weight, batch, seq_len, vocab, hidden);
    std::vector<float> residual;
    bool has_residual = false;

    std::vector<float> normed;
    std::vector<float> sum;

    if (has_residual) {
        sum = add_ref(hidden0, residual);
        normed = rmsnorm_ref(sum, post_norm_weight0, batch, seq_len, hidden, cfg.rms_norm_eps);
        residual = sum;
    } else {
        normed = rmsnorm_ref(hidden0, post_norm_weight0, batch, seq_len, hidden, cfg.rms_norm_eps);
        residual = hidden0;
        has_residual = true;
    }
    hidden0 = mlp_ref(normed, gate_w0, up_w0, down_w0, batch, seq_len, hidden, intermediate);

    if (has_residual) {
        sum = add_ref(hidden0, residual);
        hidden0 = rmsnorm_ref(sum, norm_weight, batch, seq_len, hidden, cfg.rms_norm_eps);
    } else {
        hidden0 = rmsnorm_ref(hidden0, norm_weight, batch, seq_len, hidden, cfg.rms_norm_eps);
    }
    auto expected = linear_ref_3d(hidden0, embed_weight, batch, seq_len, hidden, vocab);

    expect_tensor_near(request.get_output_tensor(), expected, 1e-3f);
}
