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
#include "modeling/weights/weight_finalizer.hpp"
#include "modeling/weights/weight_loader.hpp"
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

std::vector<float> merge_heads_ref(const std::vector<float>& x,
                                   size_t batch,
                                   size_t seq_len,
                                   size_t num_heads,
                                   size_t head_dim) {
    std::vector<float> out(batch * seq_len * num_heads * head_dim, 0.0f);
    const size_t hidden = num_heads * head_dim;
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const size_t out_base = (b * seq_len + s) * hidden;
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t in_base = ((b * num_heads + h) * seq_len + s) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    out[out_base + h * head_dim + d] = x[in_base + d];
                }
            }
        }
    }
    return out;
}

std::vector<float> apply_rope_ref(const std::vector<float>& x,
                                  const std::vector<int64_t>& positions,
                                  size_t batch,
                                  size_t seq_len,
                                  size_t num_heads,
                                  size_t head_dim,
                                  float rope_theta) {
    const size_t half_dim = head_dim / 2;
    std::vector<float> inv_freq(half_dim, 0.0f);
    for (size_t i = 0; i < half_dim; ++i) {
        float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
        inv_freq[i] = 1.0f / std::pow(rope_theta, exponent);
    }

    std::vector<float> out(x.size(), 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const float pos = static_cast<float>(positions[b * seq_len + s]);
            for (size_t i = 0; i < half_dim; ++i) {
                const float angle = pos * inv_freq[i];
                const float c = std::cos(angle);
                const float si = std::sin(angle);
                for (size_t h = 0; h < num_heads; ++h) {
                    const size_t base = ((b * num_heads + h) * seq_len + s) * head_dim;
                    const float x1 = x[base + i];
                    const float x2 = x[base + i + half_dim];
                    out[base + i] = x1 * c - x2 * si;
                    out[base + i + half_dim] = x1 * si + x2 * c;
                }
            }
        }
    }
    return out;
}

std::vector<float> attention_ref(const std::vector<float>& hidden,
                                 const std::vector<float>& q_w,
                                 const std::vector<float>& k_w,
                                 const std::vector<float>& v_w,
                                 const std::vector<float>& o_w,
                                 const std::vector<int64_t>& positions,
                                 size_t batch,
                                 size_t seq_len,
                                 size_t hidden_size,
                                 size_t num_heads,
                                 size_t head_dim,
                                 float rope_theta) {
    auto q = linear_ref_3d(hidden, q_w, batch, seq_len, hidden_size, hidden_size);
    auto k = linear_ref_3d(hidden, k_w, batch, seq_len, hidden_size, hidden_size);
    auto v = linear_ref_3d(hidden, v_w, batch, seq_len, hidden_size, hidden_size);

    auto qh = to_heads_ref(q, batch, seq_len, num_heads, head_dim);
    auto kh = to_heads_ref(k, batch, seq_len, num_heads, head_dim);
    auto vh = to_heads_ref(v, batch, seq_len, num_heads, head_dim);

    qh = apply_rope_ref(qh, positions, batch, seq_len, num_heads, head_dim, rope_theta);
    kh = apply_rope_ref(kh, positions, batch, seq_len, num_heads, head_dim, rope_theta);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> attn_probs(batch * num_heads * seq_len * seq_len, 0.0f);
    std::vector<float> scores(seq_len, 0.0f);

    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                float max_score = -1e30f;
                for (size_t j = 0; j < seq_len; ++j) {
                    float acc = 0.0f;
                    const size_t q_base = ((b * num_heads + h) * seq_len + i) * head_dim;
                    const size_t k_base = ((b * num_heads + h) * seq_len + j) * head_dim;
                    for (size_t d = 0; d < head_dim; ++d) {
                        acc += qh[q_base + d] * kh[k_base + d];
                    }
                    acc *= scale;
                    scores[j] = acc;
                    if (acc > max_score) {
                        max_score = acc;
                    }
                }
                float sum = 0.0f;
                for (size_t j = 0; j < seq_len; ++j) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum += scores[j];
                }
                const size_t prob_base = ((b * num_heads + h) * seq_len + i) * seq_len;
                for (size_t j = 0; j < seq_len; ++j) {
                    attn_probs[prob_base + j] = scores[j] / sum;
                }
            }
        }
    }

    std::vector<float> context(batch * num_heads * seq_len * head_dim, 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                const size_t prob_base = ((b * num_heads + h) * seq_len + i) * seq_len;
                const size_t ctx_base = ((b * num_heads + h) * seq_len + i) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    float acc = 0.0f;
                    for (size_t j = 0; j < seq_len; ++j) {
                        const size_t v_base = ((b * num_heads + h) * seq_len + j) * head_dim;
                        acc += attn_probs[prob_base + j] * vh[v_base + d];
                    }
                    context[ctx_base + d] = acc;
                }
            }
        }
    }

    auto merged = merge_heads_ref(context, batch, seq_len, num_heads, head_dim);
    return linear_ref_3d(merged, o_w, batch, seq_len, hidden_size, hidden_size);
}

void expect_tensor_near(const ov::Tensor& output, const std::vector<float>& expected, float tol) {
    ASSERT_EQ(output.get_size(), expected.size());
    const float* out_data = output.data<const float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(out_data[i], expected[i], tol);
    }
}

}  // namespace

TEST(Qwen3Attention, MatchesReference) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 2;
    const size_t hidden = 4;
    const size_t num_heads = 2;
    const size_t num_kv_heads = 2;
    const size_t head_dim = 2;
    const float rope_theta = 10000.0f;

    const ov::Shape attn_weight_shape{hidden, hidden};

    const auto q_w = make_seq(hidden * hidden, 0.01f, 0.01f);
    const auto k_w = make_seq(hidden * hidden, 0.02f, 0.01f);
    const auto v_w = make_seq(hidden * hidden, 0.03f, 0.01f);
    const auto o_w = make_seq(hidden * hidden, 0.04f, 0.01f);

    DummyWeightSource weights;
    weights.add("self_attn.q_proj.weight", make_tensor(q_w, attn_weight_shape));
    weights.add("self_attn.k_proj.weight", make_tensor(k_w, attn_weight_shape));
    weights.add("self_attn.v_proj.weight", make_tensor(v_w, attn_weight_shape));
    weights.add("self_attn.o_proj.weight", make_tensor(o_w, attn_weight_shape));

    DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::Qwen3DenseConfig cfg;
    cfg.hidden_size = static_cast<int32_t>(hidden);
    cfg.num_attention_heads = static_cast<int32_t>(num_heads);
    cfg.num_key_value_heads = static_cast<int32_t>(num_kv_heads);
    cfg.head_dim = static_cast<int32_t>(head_dim);
    cfg.rope_theta = rope_theta;
    cfg.attention_bias = true;

    ov::genai::modeling::models::Qwen3Attention attn(ctx, "self_attn", cfg);
    ov::genai::modeling::weights::load_model(attn, weights, finalizer);

    auto hidden_states = ctx.parameter("hidden_states", ov::element::f32, ov::PartialShape{batch, seq_len, hidden});
    auto positions = ctx.parameter("positions", ov::element::i64, ov::PartialShape{batch, seq_len});

    auto output = attn.forward(positions, hidden_states);
    auto ov_model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    const std::vector<float> hidden_data = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
    };
    const std::vector<int64_t> position_ids = {0, 1};

    ov::Tensor hidden_tensor(ov::element::f32, {batch, seq_len, hidden});
    std::memcpy(hidden_tensor.data(), hidden_data.data(), hidden_data.size() * sizeof(float));
    request.set_input_tensor(0, hidden_tensor);

    ov::Tensor pos_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(pos_tensor.data(), position_ids.data(), position_ids.size() * sizeof(int64_t));
    request.set_input_tensor(1, pos_tensor);

    request.infer();

    auto expected = attention_ref(hidden_data,
                                  q_w,
                                  k_w,
                                  v_w,
                                  o_w,
                                  position_ids,
                                  batch,
                                  seq_len,
                                  hidden,
                                  num_heads,
                                  head_dim,
                                  rope_theta);

    expect_tensor_near(request.get_output_tensor(), expected, 1e-3f);
}
