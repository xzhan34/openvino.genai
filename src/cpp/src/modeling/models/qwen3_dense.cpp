// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_dense.hpp"

#include <cmath>
#include <openvino/core/except.hpp>
#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/llm.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"

namespace {

ov::genai::modeling::Tensor add_bias_if_present(const ov::genai::modeling::Tensor& x,
                                                const ov::genai::modeling::Tensor* bias) {
    if (!bias) {
        return x;
    }
    return x + *bias;
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
      q_norm_(ctx, "q_norm", cfg.rms_norm_eps, this),
      k_norm_(ctx, "k_norm", cfg.rms_norm_eps, this) {
    if (num_heads_ <= 0 || head_dim_ <= 0) {
        OPENVINO_THROW("Invalid attention head configuration");
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

std::pair<Tensor, Tensor> Qwen3Attention::append_kv_cache(const Tensor& keys,
                                                          const Tensor& values,
                                                          const Tensor& beam_idx) const {
    auto* op_ctx = keys.context();
    // Build stateful KV cache (ReadValue/Assign) so incremental decoding keeps context.
    auto batch = shape::dim(keys, 0);
    auto kv_heads = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(num_kv_heads_)});
    auto zero_len = ops::const_vec(op_ctx, std::vector<int64_t>{0});
    auto head_dim = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(head_dim_)});
    auto cache_shape = shape::make({batch, kv_heads, zero_len, head_dim});

    auto zero = Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx).to(keys.dtype());
    auto k_init = shape::broadcast_to(zero, cache_shape);
    auto v_init = shape::broadcast_to(zero, cache_shape);

    const std::string cache_prefix = full_path().empty() ? name() : full_path();
    const std::string k_name = cache_prefix + ".key_cache";
    const std::string v_name = cache_prefix + ".value_cache";

    ov::op::util::VariableInfo k_info{ov::PartialShape{-1, num_kv_heads_, -1, head_dim_},
                                      keys.dtype(),
                                      k_name};
    auto k_var = std::make_shared<ov::op::util::Variable>(k_info);
    auto k_read = std::make_shared<ov::op::v6::ReadValue>(k_init.output(), k_var);

    ov::op::util::VariableInfo v_info{ov::PartialShape{-1, num_kv_heads_, -1, head_dim_},
                                      values.dtype(),
                                      v_name};
    auto v_var = std::make_shared<ov::op::util::Variable>(v_info);
    auto v_read = std::make_shared<ov::op::v6::ReadValue>(v_init.output(), v_var);

    auto k_cached = ops::gather(Tensor(k_read->output(0), op_ctx), beam_idx, 0);
    auto v_cached = ops::gather(Tensor(v_read->output(0), op_ctx), beam_idx, 0);

    auto k_combined = ops::concat({k_cached, keys}, 2);
    auto v_combined = ops::concat({v_cached, values}, 2);

    auto k_assign = std::make_shared<ov::opset13::Assign>(k_combined.output(), k_var);
    auto v_assign = std::make_shared<ov::opset13::Assign>(v_combined.output(), v_var);
    ctx().register_sink(k_assign);
    ctx().register_sink(v_assign);

    return {k_combined, v_combined};
}

Tensor Qwen3Attention::forward(const Tensor& positions, const Tensor& hidden_states, const Tensor& beam_idx) const {
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(positions, head_dim_, rope_theta_, policy);
    auto seq_len = Tensor(shape::dim(positions, 1), positions.context()).squeeze(0);
    auto causal_mask = ops::llm::causal_mask_from_seq_len(seq_len);
    return forward(hidden_states, beam_idx, cos_sin.first, cos_sin.second, causal_mask);
}

Tensor Qwen3Attention::forward(const Tensor& hidden_states,
                               const Tensor& beam_idx,
                               const Tensor& rope_cos,
                               const Tensor& rope_sin,
                               const Tensor& causal_mask) const {
    auto q = add_bias_if_present(ops::linear(hidden_states, q_proj_weight()), q_proj_bias());
    auto k = add_bias_if_present(ops::linear(hidden_states, k_proj_weight()), k_proj_bias());
    auto v = add_bias_if_present(ops::linear(hidden_states, v_proj_weight()), v_proj_bias());

    auto q_heads = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto k_heads = k.reshape({0, 0, num_kv_heads_, head_dim_}) .permute({0, 2, 1, 3});
    auto v_heads = v.reshape({0, 0, num_kv_heads_, head_dim_}) .permute({0, 2, 1, 3});

    if (q_norm_.weight_param().is_bound()) {
        q_heads = q_norm_.forward(q_heads);
    }
    if (k_norm_.weight_param().is_bound()) {
        k_heads = k_norm_.forward(k_heads);
    }

    auto* policy = &ctx().op_policy();
    auto q_rot = ops::llm::apply_rope(q_heads, rope_cos, rope_sin, head_dim_, policy);
    auto k_rot = ops::llm::apply_rope(k_heads, rope_cos, rope_sin, head_dim_, policy);

    auto cached = append_kv_cache(k_rot, v_heads, beam_idx);
    auto k_expanded = ops::llm::repeat_kv(cached.first, num_heads_, num_kv_heads_, head_dim_);
    auto v_expanded = ops::llm::repeat_kv(cached.second, num_heads_, num_kv_heads_, head_dim_);

    // Build causal mask that works with KV cache scenario
    auto mask = ops::llm::build_kv_causal_mask(q_rot, k_expanded);
    auto context = ops::llm::sdpa(q_rot, k_expanded, v_expanded, scaling_, 3, &mask, false, policy);
    const int64_t attn_out_dim = static_cast<int64_t>(num_heads_) * head_dim_;
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});
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
    auto gated = ops::silu(gate) * up;
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

std::pair<Tensor, Tensor> Qwen3DecoderLayer::forward(const Tensor& hidden_states,
                                                     const Tensor& beam_idx,
                                                     const Tensor& rope_cos,
                                                     const Tensor& rope_sin,
                                                     const Tensor& causal_mask,
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
    auto attn_out = self_attn_.forward(normed, beam_idx, rope_cos, rope_sin, causal_mask);
    auto post_norm = post_attention_layernorm_.forward(attn_out, next_residual);
    auto mlp_out = mlp_.forward(post_norm.first);
    return {mlp_out, post_norm.second};
}

Qwen3Model::Qwen3Model(BuilderContext& ctx, const Qwen3DenseConfig& cfg, Module* parent)
    : Module("model", ctx, parent),
      embed_tokens_(ctx, "embed_tokens", this),
      layers_(),
      norm_(ctx, "norm", cfg.rms_norm_eps, this),
      head_dim_(cfg.head_dim > 0
                    ? cfg.head_dim
                    : (cfg.num_attention_heads > 0 ? (cfg.hidden_size / cfg.num_attention_heads) : 0)),
      rope_theta_(cfg.rope_theta) {
    layers_.reserve(static_cast<size_t>(cfg.num_hidden_layers));
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        layers_.emplace_back(ctx, "layers[" + std::to_string(i) + "]", cfg, this);
    }
}

Tensor Qwen3Model::forward(const Tensor& input_ids, const Tensor& position_ids, const Tensor& beam_idx) {
    auto hidden_states = embed_tokens_.forward(input_ids);
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, head_dim_, rope_theta_, policy);
    auto seq_len = Tensor(shape::dim(position_ids, 1), position_ids.context()).squeeze(0);
    auto causal_mask = ops::llm::causal_mask_from_seq_len(seq_len);
    std::optional<Tensor> residual;
    for (auto& layer : layers_) {
        auto layer_out = layer.forward(hidden_states, beam_idx, cos_sin.first, cos_sin.second, causal_mask, residual);
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

Tensor Qwen3ForCausalLM::forward(const Tensor& input_ids,
                                 const Tensor& position_ids,
                                 const Tensor& beam_idx) {
    auto hidden = model_.forward(input_ids, position_ids, beam_idx);
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
