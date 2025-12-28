// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_dense.hpp"

#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/models/qwen3_dense_weights.hpp"
#include "modeling/layer/lm_head.hpp"
#include "modeling/layer/rms_norm.hpp"
#include "modeling/layer/vocab_embedding.hpp"
#include "modeling/ops/tensor.hpp"

namespace {

void set_name(const std::shared_ptr<ov::Node>& node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
}

ov::genai::modeling::Tensor attach_pipeline_inputs(const ov::genai::modeling::Tensor& hidden,
                                                   const ov::Output<ov::Node>& attention_mask,
                                                   const ov::Output<ov::Node>& position_ids,
                                                   const ov::Output<ov::Node>& beam_idx,
                                                   ov::genai::modeling::OpContext* ctx) {
    using namespace ov::op;

    // NOTE: The OV GenAI pipeline expects these inputs to be present. To avoid them being pruned,
    // we introduce a minimal dependency on them.

    // hidden *= unsqueeze(f32(attention_mask), -1)
    auto attn_f = std::make_shared<v0::Convert>(attention_mask, ov::element::f32);
    auto attn_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto attn_unsq = std::make_shared<v0::Unsqueeze>(attn_f, attn_axis);
    ov::genai::modeling::Tensor attn_mask_t(attn_unsq, ctx);
    auto masked = hidden * attn_mask_t;

    // hidden += unsqueeze(f32(position_ids), -1)
    auto pos_f = std::make_shared<v0::Convert>(position_ids, ov::element::f32);
    auto pos_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto pos_unsq = std::make_shared<v0::Unsqueeze>(pos_f, pos_axis);
    ov::genai::modeling::Tensor pos_t(pos_unsq, ctx);
    auto with_pos = masked + pos_t;

    // hidden += reduce_sum(f32(beam_idx))
    auto beam_f = std::make_shared<v0::Convert>(beam_idx, ov::element::f32);
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

std::shared_ptr<ov::Model> build_qwen3_dense_dummy(const Qwen3DenseConfig& cfg,
                                                   weights::IWeightProvider& weights,
                                                   OpContext& ctx) {
    (void)cfg.hidden_size;

    auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    set_name(input_ids, "input_ids");

    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    set_name(attention_mask, "attention_mask");

    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    set_name(position_ids, "position_ids");

    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    set_name(beam_idx, "beam_idx");

    const auto w = load_qwen3_dense_weights(weights);

    Tensor input_ids_t(input_ids->output(0), &ctx);
    VocabEmbedding embed(w.embed_tokens);
    auto hidden_states = embed(input_ids_t);

    hidden_states = attach_pipeline_inputs(hidden_states,
                                           attention_mask->output(0),
                                           position_ids->output(0),
                                           beam_idx->output(0),
                                           &ctx);

    RMSNorm norm(w.final_norm, cfg.rms_norm_eps);
    auto final_norm = norm(hidden_states);

    LMHead lm_head(w.lm_head);
    auto logits = lm_head(final_norm);

    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, "logits");

    ov::ParameterVector inputs{input_ids, attention_mask, position_ids, beam_idx};
    return std::make_shared<ov::Model>(ov::OutputVector{result->output(0)}, ov::SinkVector{}, inputs);
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
