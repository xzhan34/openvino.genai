// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_dense.hpp"

#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/ops.hpp"

namespace {

ov::genai::modeling::Tensor silu(const ov::genai::modeling::Tensor& x) {
    auto node = std::make_shared<ov::op::v4::Swish>(x.output());
    return ov::genai::modeling::Tensor(node, x.context());
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

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
      mlp_(ctx, "mlp", cfg, this),
      input_layernorm_(ctx, "input_layernorm", cfg.rms_norm_eps, this),
      post_attention_layernorm_(ctx, "post_attention_layernorm", cfg.rms_norm_eps, this) {}

std::pair<Tensor, Tensor> Qwen3DecoderLayer::forward(const Tensor& positions,
                                                     const Tensor& hidden_states,
                                                     const std::optional<Tensor>& residual) const {
    (void)positions;
    Tensor normed;
    Tensor next_residual;
    if (residual) {
        auto norm_out = post_attention_layernorm_.forward(hidden_states, *residual);
        normed = norm_out.first;
        next_residual = norm_out.second;
    } else {
        normed = post_attention_layernorm_.forward(hidden_states);
        next_residual = hidden_states;
    }
    auto mlp_out = mlp_.forward(normed);
    return {mlp_out, next_residual};
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
