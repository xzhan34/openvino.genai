// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "modeling/models/qwen3_5/qwen3_5_weight_specs.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/synthetic_weight_source.hpp"

namespace {

ov::genai::modeling::models::Qwen3_5Config make_small_moe_cfg() {
    using namespace ov::genai::modeling::models;
    Qwen3_5Config cfg = Qwen3_5Config::make_dummy_moe35b_config();

    cfg.text.vocab_size = 64;
    cfg.text.hidden_size = 16;
    cfg.text.intermediate_size = 0;
    cfg.text.moe_intermediate_size = 8;
    cfg.text.shared_expert_intermediate_size = 8;
    cfg.text.num_experts = 4;
    cfg.text.num_experts_per_tok = 2;
    cfg.text.num_hidden_layers = 2;
    cfg.text.num_attention_heads = 4;
    cfg.text.num_key_value_heads = 2;
    cfg.text.head_dim = 4;
    cfg.text.max_position_embeddings = 256;
    cfg.text.partial_rotary_factor = 0.5f;
    cfg.text.layer_types = {"linear_attention", "full_attention"};
    cfg.text.linear_conv_kernel_dim = 2;
    cfg.text.linear_key_head_dim = 4;
    cfg.text.linear_value_head_dim = 4;
    cfg.text.linear_num_key_heads = 2;
    cfg.text.linear_num_value_heads = 2;
    cfg.text.rope.mrope_interleaved = true;
    cfg.text.rope.mrope_section = {1, 1, 0};

    cfg.vision.model_type = "qwen3_5_moe";
    cfg.vision.depth = 1;
    cfg.vision.hidden_size = 8;
    cfg.vision.intermediate_size = 16;
    cfg.vision.num_heads = 2;
    cfg.vision.in_channels = 3;
    cfg.vision.patch_size = 2;
    cfg.vision.temporal_patch_size = 1;
    cfg.vision.spatial_merge_size = 2;
    cfg.vision.out_hidden_size = cfg.text.hidden_size;
    cfg.vision.num_position_embeddings = 16;
    cfg.vision.deepstack_visual_indexes.clear();

    cfg.finalize();
    cfg.validate();
    return cfg;
}

ov::Tensor make_beam_idx() {
    ov::Tensor beam(ov::element::i32, {1});
    beam.data<int32_t>()[0] = 0;
    return beam;
}

}  // namespace

TEST(Qwen3_5MoeDummyE2E, TextPrefillAndDecodeSmoke) {
    namespace tests = ov::genai::modeling::tests;
    const auto cfg = make_small_moe_cfg();
    auto specs = ov::genai::modeling::models::build_qwen3_5_text_weight_specs(cfg.text);
    ov::genai::modeling::weights::SyntheticWeightSource source(std::move(specs), 2031u, -0.02f, 0.02f);
    tests::DummyWeightFinalizer finalizer;

    auto text_model = ov::genai::modeling::models::create_qwen3_5_text_model(cfg, source, finalizer, false, false);
    ov::Core core;
    auto compiled = core.compile_model(text_model, "GPU");

    ov::Tensor input_ids(ov::element::i64, {1, 3});
    auto* ids = input_ids.data<int64_t>();
    ids[0] = 11;
    ids[1] = 12;
    ids[2] = 13;

    ov::Tensor attention_mask(ov::element::i64, {1, 3});
    auto* attn = attention_mask.data<int64_t>();
    attn[0] = 1;
    attn[1] = 1;
    attn[2] = 1;

    ov::genai::modeling::models::Qwen3_5InputPlanner planner(cfg);
    auto plan = planner.build_plan(input_ids, &attention_mask, nullptr);

    auto beam_idx = make_beam_idx();
    auto request = compiled.create_infer_request();
    request.reset_state();
    request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kInputIds, input_ids);
    request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kAttentionMask, attention_mask);
    request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kPositionIds, plan.position_ids);
    request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kBeamIdx, beam_idx);
    request.infer();

    ov::Tensor logits = request.get_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kLogits);
    ASSERT_EQ(logits.get_shape(), (ov::Shape{1, 3, static_cast<size_t>(cfg.text.vocab_size)}));

    ov::Tensor step_ids(ov::element::i64, {1, 1});
    step_ids.data<int64_t>()[0] = 14;
    ov::Tensor step_mask(ov::element::i64, {1, 1});
    step_mask.data<int64_t>()[0] = 1;
    auto decode_pos =
        ov::genai::modeling::models::Qwen3_5InputPlanner::build_decode_position_ids(plan.rope_deltas, 3, 1);

    request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kInputIds, step_ids);
    request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kAttentionMask, step_mask);
    request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kPositionIds, decode_pos);
    request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kBeamIdx, beam_idx);
    request.infer();

    logits = request.get_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kLogits);
    ASSERT_EQ(logits.get_shape(), (ov::Shape{1, 1, static_cast<size_t>(cfg.text.vocab_size)}));
}

