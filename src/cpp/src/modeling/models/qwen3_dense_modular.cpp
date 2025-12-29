// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_dense_modular.hpp"

#include <openvino/opsets/opset13.hpp>

namespace {

ov::genai::modeling::Tensor attach_pipeline_inputs(const ov::genai::modeling::Tensor& hidden,
                                                   const ov::genai::modeling::Tensor& attention_mask,
                                                   const ov::genai::modeling::Tensor& position_ids,
                                                   const ov::genai::modeling::Tensor& beam_idx) {
    using namespace ov::op;

    auto* ctx = hidden.context();

    // hidden *= unsqueeze(f32(attention_mask), -1)
    auto attn_f = std::make_shared<v0::Convert>(attention_mask.output(), ov::element::f32);
    auto attn_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto attn_unsq = std::make_shared<v0::Unsqueeze>(attn_f, attn_axis);
    ov::genai::modeling::Tensor attn_mask_t(attn_unsq, ctx);
    auto masked = hidden * attn_mask_t;

    // hidden += unsqueeze(f32(position_ids), -1)
    auto pos_f = std::make_shared<v0::Convert>(position_ids.output(), ov::element::f32);
    auto pos_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto pos_unsq = std::make_shared<v0::Unsqueeze>(pos_f, pos_axis);
    ov::genai::modeling::Tensor pos_t(pos_unsq, ctx);
    auto with_pos = masked + pos_t;

    // hidden += reduce_sum(f32(beam_idx))
    auto beam_f = std::make_shared<v0::Convert>(beam_idx.output(), ov::element::f32);
    auto beam_axes = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto beam_sum = std::make_shared<v1::ReduceSum>(beam_f, beam_axes, false);
    ov::genai::modeling::Tensor beam_sum_t(beam_sum, ctx);
    return with_pos + beam_sum_t;
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

Qwen3Model::Qwen3Model(BuilderContext& ctx, const Qwen3DenseConfig& cfg, Module* parent)
    : Module("model", ctx, parent),
      embed_tokens_(ctx, "embed_tokens", this),
      norm_(ctx, "norm", cfg.rms_norm_eps, this) {}

Tensor Qwen3Model::forward(const Tensor& input_ids,
                           const Tensor& attention_mask,
                           const Tensor& position_ids,
                           const Tensor& beam_idx) {
    auto hidden = embed_tokens_(input_ids);
    auto attached = attach_pipeline_inputs(hidden, attention_mask, position_ids, beam_idx);
    return norm_(attached);
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
                                 const Tensor& attention_mask,
                                 const Tensor& position_ids,
                                 const Tensor& beam_idx) {
    auto hidden = model_.forward(input_ids, attention_mask, position_ids, beam_idx);
    return lm_head_(hidden);
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
