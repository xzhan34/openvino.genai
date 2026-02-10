// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <openvino/core/except.hpp>

#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"

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

}  // namespace

TEST(Qwen3_5MoeConfig, DummyMoeFactoryProducesValidMoeConfig) {
    const auto cfg = ov::genai::modeling::models::Qwen3_5Config::make_dummy_moe35b_config();
    EXPECT_EQ(cfg.model_type, "qwen3_5_moe");
    EXPECT_EQ(cfg.text.model_type, "qwen3_5_moe_text");
    EXPECT_TRUE(cfg.text.is_moe_enabled());
    EXPECT_GT(cfg.text.num_experts, 0);
    EXPECT_GT(cfg.text.num_experts_per_tok, 0);
}

TEST(Qwen3_5MoeConfig, RejectsPartialMoeFields) {
    auto cfg = make_small_moe_cfg();
    cfg.text.moe_intermediate_size = 0;
    cfg.text.shared_expert_intermediate_size = 0;
    cfg.finalize();
    EXPECT_THROW(cfg.validate(), ov::Exception);
}

TEST(Qwen3_5MoeConfig, RejectsTopKGreaterThanExpertCount) {
    auto cfg = make_small_moe_cfg();
    cfg.text.num_experts_per_tok = cfg.text.num_experts + 1;
    cfg.finalize();
    EXPECT_THROW(cfg.validate(), ov::Exception);
}

