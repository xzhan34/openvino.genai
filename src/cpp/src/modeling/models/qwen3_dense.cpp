// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_dense.hpp"

#include <cmath>
#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/ops.hpp"

namespace {

ov::genai::modeling::Tensor silu(const ov::genai::modeling::Tensor& x) {
    auto node = std::make_shared<ov::op::v4::Swish>(x.output());
    return ov::genai::modeling::Tensor(node, x.context());
}

ov::Output<ov::Node> i64_const(int64_t value) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {value});
}

ov::Output<ov::Node> i64_vec(const std::vector<int64_t>& values) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

ov::Output<ov::Node> i32_vec(const std::vector<int32_t>& values) {
    return ov::op::v0::Constant::create(ov::element::i32, ov::Shape{values.size()}, values);
}

ov::genai::modeling::Tensor add_bias_if_present(const ov::genai::modeling::Tensor& x,
                                                const ov::genai::modeling::Tensor* bias) {
    if (!bias) {
        return x;
    }
    return x + *bias;
}

ov::genai::modeling::Tensor reshape_to_heads(const ov::genai::modeling::Tensor& x,
                                             int32_t num_heads,
                                             int32_t head_dim) {
    auto* ctx = x.context();
    auto shape = i64_vec({0, 0, num_heads, head_dim});
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(x.output(), shape, true);
    auto order = i32_vec({0, 2, 1, 3});
    auto transposed = std::make_shared<ov::op::v1::Transpose>(reshaped, order);
    return ov::genai::modeling::Tensor(transposed, ctx);
}

ov::genai::modeling::Tensor merge_heads(const ov::genai::modeling::Tensor& x, int32_t hidden_size) {
    auto* ctx = x.context();
    auto order = i32_vec({0, 2, 1, 3});
    auto transposed = std::make_shared<ov::op::v1::Transpose>(x.output(), order);
    auto shape = i64_vec({0, 0, hidden_size});
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(transposed, shape, true);
    return ov::genai::modeling::Tensor(reshaped, ctx);
}

std::pair<ov::genai::modeling::Tensor, ov::genai::modeling::Tensor> rope_cos_sin(
    const ov::genai::modeling::Tensor& positions,
    int32_t head_dim,
    float rope_theta) {
    auto* ctx = positions.context();
    const int32_t half_dim = head_dim / 2;
    std::vector<float> inv_freq(static_cast<size_t>(half_dim));
    for (int32_t i = 0; i < half_dim; ++i) {
        float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
        inv_freq[static_cast<size_t>(i)] = 1.0f / std::pow(rope_theta, exponent);
    }

    auto inv_freq_const =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{static_cast<size_t>(half_dim)}, inv_freq);
    auto inv_freq_shape = i64_vec({1, 1, half_dim});
    auto inv_freq_reshaped = std::make_shared<ov::op::v1::Reshape>(inv_freq_const, inv_freq_shape, false);

    auto pos_f = positions.to(ov::element::f32);
    auto pos_unsq = std::make_shared<ov::op::v0::Unsqueeze>(pos_f.output(), i64_const(2));
    auto freqs = std::make_shared<ov::op::v1::Multiply>(pos_unsq, inv_freq_reshaped, ov::op::AutoBroadcastType::NUMPY);
    auto cos_node = std::make_shared<ov::op::v0::Cos>(freqs);
    auto sin_node = std::make_shared<ov::op::v0::Sin>(freqs);
    return {ov::genai::modeling::Tensor(cos_node, ctx), ov::genai::modeling::Tensor(sin_node, ctx)};
}

ov::genai::modeling::Tensor apply_rope(const ov::genai::modeling::Tensor& x,
                                       const ov::genai::modeling::Tensor& cos,
                                       const ov::genai::modeling::Tensor& sin,
                                       int32_t head_dim) {
    auto* ctx = x.context();
    auto cos_unsq = std::make_shared<ov::op::v0::Unsqueeze>(cos.output(), i64_const(1));
    auto sin_unsq = std::make_shared<ov::op::v0::Unsqueeze>(sin.output(), i64_const(1));
    auto axis = i64_vec({3});
    int64_t half_dim = head_dim / 2;

    auto x1 = std::make_shared<ov::opset13::Slice>(x.output(), i64_vec({0}), i64_vec({half_dim}), i64_vec({1}), axis);
    auto x2 = std::make_shared<ov::opset13::Slice>(x.output(), i64_vec({half_dim}), i64_vec({head_dim}), i64_vec({1}), axis);

    auto x1_cos = std::make_shared<ov::op::v1::Multiply>(x1, cos_unsq, ov::op::AutoBroadcastType::NUMPY);
    auto x2_sin = std::make_shared<ov::op::v1::Multiply>(x2, sin_unsq, ov::op::AutoBroadcastType::NUMPY);
    auto x1_sin = std::make_shared<ov::op::v1::Multiply>(x1, sin_unsq, ov::op::AutoBroadcastType::NUMPY);
    auto x2_cos = std::make_shared<ov::op::v1::Multiply>(x2, cos_unsq, ov::op::AutoBroadcastType::NUMPY);

    auto rot1 = std::make_shared<ov::op::v1::Subtract>(x1_cos, x2_sin, ov::op::AutoBroadcastType::NUMPY);
    auto rot2 = std::make_shared<ov::op::v1::Add>(x1_sin, x2_cos, ov::op::AutoBroadcastType::NUMPY);
    auto rot = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{rot1, rot2}, 3);
    return ov::genai::modeling::Tensor(rot, ctx);
}

ov::genai::modeling::Tensor repeat_kv(const ov::genai::modeling::Tensor& x,
                                      int32_t num_heads,
                                      int32_t num_kv_heads,
                                      int32_t head_dim) {
    if (num_heads == num_kv_heads) {
        return x;
    }
    auto* ctx = x.context();
    const int32_t repeats = num_heads / num_kv_heads;
    auto unsq = std::make_shared<ov::op::v0::Unsqueeze>(x.output(), i64_const(2));

    auto shape = std::make_shared<ov::op::v3::ShapeOf>(x.output(), ov::element::i64);
    auto batch = std::make_shared<ov::op::v8::Gather>(shape, i64_vec({0}), i64_const(0));
    auto seq = std::make_shared<ov::op::v8::Gather>(shape, i64_vec({2}), i64_const(0));

    auto kv_heads = i64_vec({num_kv_heads});
    auto rep = i64_vec({repeats});
    auto hdim = i64_vec({head_dim});

    auto target = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{batch, kv_heads, rep, seq, hdim}, 0);
    auto broadcast = std::make_shared<ov::op::v3::Broadcast>(unsq, target, ov::op::BroadcastType::NUMPY);

    auto heads = i64_vec({num_heads});
    auto reshape_shape = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{batch, heads, seq, hdim}, 0);
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(broadcast, reshape_shape, false);
    return ov::genai::modeling::Tensor(reshaped, ctx);
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

Qwen3Attention::Qwen3Attention(BuilderContext& ctx,
                               const std::string& name,
                               const Qwen3DenseConfig& cfg,
                               Module* parent)
    : Module(name, ctx, parent),
      num_heads_(cfg.num_attention_heads),
      num_kv_heads_(cfg.num_key_value_heads > 0 ? cfg.num_key_value_heads : cfg.num_attention_heads),
      head_dim_(cfg.head_dim > 0 ? cfg.head_dim : (cfg.hidden_size / cfg.num_attention_heads)),
      hidden_size_(cfg.hidden_size),
      scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))),
      rope_theta_(cfg.rope_theta),
      use_qk_norm_(!cfg.attention_bias),
      q_norm_(ctx, "q_norm", cfg.rms_norm_eps, this),
      k_norm_(ctx, "k_norm", cfg.rms_norm_eps, this) {
    if (num_heads_ <= 0 || head_dim_ <= 0) {
        OPENVINO_THROW("Invalid attention head configuration");
    }
    if (hidden_size_ != num_heads_ * head_dim_) {
        OPENVINO_THROW("hidden_size must equal num_attention_heads * head_dim");
    }
    if (num_heads_ % num_kv_heads_ != 0) {
        OPENVINO_THROW("num_attention_heads must be divisible by num_key_value_heads");
    }
    q_proj_param_ = &register_parameter("q_proj.weight");
    k_proj_param_ = &register_parameter("k_proj.weight");
    v_proj_param_ = &register_parameter("v_proj.weight");
    o_proj_param_ = &register_parameter("o_proj.weight");

    q_bias_param_ = &register_parameter("q_proj.bias");
    k_bias_param_ = &register_parameter("k_proj.bias");
    v_bias_param_ = &register_parameter("v_proj.bias");
    o_bias_param_ = &register_parameter("o_proj.bias");
}

const Tensor& Qwen3Attention::q_proj_weight() const {
    if (!q_proj_param_) {
        OPENVINO_THROW("Qwen3Attention q_proj parameter not registered");
    }
    return q_proj_param_->value();
}

const Tensor& Qwen3Attention::k_proj_weight() const {
    if (!k_proj_param_) {
        OPENVINO_THROW("Qwen3Attention k_proj parameter not registered");
    }
    return k_proj_param_->value();
}

const Tensor& Qwen3Attention::v_proj_weight() const {
    if (!v_proj_param_) {
        OPENVINO_THROW("Qwen3Attention v_proj parameter not registered");
    }
    return v_proj_param_->value();
}

const Tensor& Qwen3Attention::o_proj_weight() const {
    if (!o_proj_param_) {
        OPENVINO_THROW("Qwen3Attention o_proj parameter not registered");
    }
    return o_proj_param_->value();
}

const Tensor* Qwen3Attention::q_proj_bias() const {
    return (q_bias_param_ && q_bias_param_->is_bound()) ? &q_bias_param_->value() : nullptr;
}

const Tensor* Qwen3Attention::k_proj_bias() const {
    return (k_bias_param_ && k_bias_param_->is_bound()) ? &k_bias_param_->value() : nullptr;
}

const Tensor* Qwen3Attention::v_proj_bias() const {
    return (v_bias_param_ && v_bias_param_->is_bound()) ? &v_bias_param_->value() : nullptr;
}

const Tensor* Qwen3Attention::o_proj_bias() const {
    return (o_bias_param_ && o_bias_param_->is_bound()) ? &o_bias_param_->value() : nullptr;
}

Tensor Qwen3Attention::forward(const Tensor& positions, const Tensor& hidden_states) const {
    auto q = add_bias_if_present(ops::linear(hidden_states, q_proj_weight()), q_proj_bias());
    auto k = add_bias_if_present(ops::linear(hidden_states, k_proj_weight()), k_proj_bias());
    auto v = add_bias_if_present(ops::linear(hidden_states, v_proj_weight()), v_proj_bias());

    auto q_heads = reshape_to_heads(q, num_heads_, head_dim_);
    auto k_heads = reshape_to_heads(k, num_kv_heads_, head_dim_);
    auto v_heads = reshape_to_heads(v, num_kv_heads_, head_dim_);

    if (use_qk_norm_ && q_norm_.weight_param().is_bound()) {
        q_heads = q_norm_.forward(q_heads);
    }
    if (use_qk_norm_ && k_norm_.weight_param().is_bound()) {
        k_heads = k_norm_.forward(k_heads);
    }

    auto cos_sin = rope_cos_sin(positions, head_dim_, rope_theta_);
    auto q_rot = apply_rope(q_heads, cos_sin.first, cos_sin.second, head_dim_);
    auto k_rot = apply_rope(k_heads, cos_sin.first, cos_sin.second, head_dim_);

    auto k_expanded = repeat_kv(k_rot, num_heads_, num_kv_heads_, head_dim_);
    auto v_expanded = repeat_kv(v_heads, num_heads_, num_kv_heads_, head_dim_);

    auto scale_node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {scaling_});
    auto scaled_q = q_rot * Tensor(scale_node, q_rot.context());

    auto scores = ops::matmul(scaled_q, k_expanded, false, true);
    auto scores_f32 = scores.to(ov::element::f32);
    auto softmax = std::make_shared<ov::op::v1::Softmax>(scores_f32.output(), 3);
    Tensor attn_probs(softmax, scores.context());

    auto context = ops::matmul(attn_probs, v_expanded, false, false);
    auto merged = merge_heads(context, hidden_size_);
    auto out = add_bias_if_present(ops::linear(merged, o_proj_weight()), o_proj_bias());
    return out;
}

Qwen3MLP::Qwen3MLP(BuilderContext& ctx, const std::string& name, const Qwen3DenseConfig& cfg, Module* parent)
    : Module(name, ctx, parent) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported Qwen3 MLP activation: ", cfg.hidden_act);
    }
    gate_proj_param_ = &register_parameter("gate_proj.weight");
    up_proj_param_ = &register_parameter("up_proj.weight");
    down_proj_param_ = &register_parameter("down_proj.weight");
}

const Tensor& Qwen3MLP::gate_proj_weight() const {
    if (!gate_proj_param_) {
        OPENVINO_THROW("Qwen3MLP gate projection parameter not registered");
    }
    return gate_proj_param_->value();
}

const Tensor& Qwen3MLP::up_proj_weight() const {
    if (!up_proj_param_) {
        OPENVINO_THROW("Qwen3MLP up projection parameter not registered");
    }
    return up_proj_param_->value();
}

const Tensor& Qwen3MLP::down_proj_weight() const {
    if (!down_proj_param_) {
        OPENVINO_THROW("Qwen3MLP down projection parameter not registered");
    }
    return down_proj_param_->value();
}

Tensor Qwen3MLP::forward(const Tensor& x) const {
    auto gate = ops::linear(x, gate_proj_weight());
    auto up = ops::linear(x, up_proj_weight());
    auto gated = silu(gate) * up;
    return ops::linear(gated, down_proj_weight());
}

Qwen3DecoderLayer::Qwen3DecoderLayer(BuilderContext& ctx,
                                     const std::string& name,
                                     const Qwen3DenseConfig& cfg,
                                     Module* parent)
    : Module(name, ctx, parent),
      self_attn_(ctx, "self_attn", cfg, this),
      mlp_(ctx, "mlp", cfg, this),
      input_layernorm_(ctx, "input_layernorm", cfg.rms_norm_eps, this),
      post_attention_layernorm_(ctx, "post_attention_layernorm", cfg.rms_norm_eps, this) {}

std::pair<Tensor, Tensor> Qwen3DecoderLayer::forward(const Tensor& positions,
                                                     const Tensor& hidden_states,
                                                     const std::optional<Tensor>& residual) const {
    Tensor normed;
    Tensor next_residual;
    if (residual) {
        auto norm_out = input_layernorm_.forward(hidden_states, *residual);
        normed = norm_out.first;
        next_residual = norm_out.second;
    } else {
        normed = input_layernorm_.forward(hidden_states);
        next_residual = hidden_states;
    }
    auto attn_out = self_attn_.forward(positions, normed);
    auto post_norm = post_attention_layernorm_.forward(attn_out, next_residual);
    auto mlp_out = mlp_.forward(post_norm.first);
    return {mlp_out, post_norm.second};
}

Qwen3Model::Qwen3Model(BuilderContext& ctx, const Qwen3DenseConfig& cfg, Module* parent)
    : Module("model", ctx, parent),
      embed_tokens_(ctx, "embed_tokens", this),
      layers_(),
      norm_(ctx, "norm", cfg.rms_norm_eps, this) {
    layers_.reserve(static_cast<size_t>(cfg.num_hidden_layers));
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        layers_.emplace_back(ctx, "layers[" + std::to_string(i) + "]", cfg, this);
    }
}

Tensor Qwen3Model::forward(const Tensor& input_ids, const Tensor& position_ids) {
    auto hidden_states = embed_tokens_.forward(input_ids);
    std::optional<Tensor> residual;
    for (auto& layer : layers_) {
        auto layer_out = layer.forward(position_ids, hidden_states, residual);
        hidden_states = layer_out.first;
        residual = layer_out.second;
    }
    if (residual) {
        return norm_.forward(hidden_states, *residual).first;
    }
    return norm_.forward(hidden_states);
}

VocabEmbedding& Qwen3Model::embed_tokens() {
    return embed_tokens_;
}

RMSNorm& Qwen3Model::norm() {
    return norm_;
}

Qwen3ForCausalLM::Qwen3ForCausalLM(BuilderContext& ctx, const Qwen3DenseConfig& cfg, Module* parent)
    : Module("", ctx, parent),
      cfg_(cfg),
      model_(ctx, cfg, this),
      lm_head_(ctx, "lm_head", this) {
    if (cfg_.tie_word_embeddings) {
        lm_head_.tie_to(model_.embed_tokens().weight_param());
    }
}

Tensor Qwen3ForCausalLM::forward(const Tensor& input_ids, const Tensor& position_ids) {
    auto hidden = model_.forward(input_ids, position_ids);
    return lm_head_.forward(hidden);
}

Qwen3Model& Qwen3ForCausalLM::model() {
    return model_;
}

LMHead& Qwen3ForCausalLM::lm_head() {
    return lm_head_;
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
