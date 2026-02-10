// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <string>

#include <gtest/gtest.h>

#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "modeling/models/qwen3_5/qwen3_5_weight_specs.hpp"

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

std::optional<ov::Shape> find_shape(
    const std::vector<ov::genai::modeling::weights::SyntheticWeightSpec>& specs,
    const std::string& name) {
    for (const auto& spec : specs) {
        if (spec.name == name) {
            return spec.shape;
        }
    }
    return std::nullopt;
}

}  // namespace

TEST(Qwen3_5MoeWeightSpecs, EmitsPackedMoeWeightsAndNoDenseMlpWeights) {
    const auto cfg = make_small_moe_cfg();
    const auto specs = ov::genai::modeling::models::build_qwen3_5_text_weight_specs(cfg.text);

    const auto gate_shape = find_shape(specs, "model.layers[0].mlp.gate.weight");
    const auto gate_up_shape = find_shape(specs, "model.layers[0].mlp.experts.gate_up_proj");
    const auto down_shape = find_shape(specs, "model.layers[0].mlp.experts.down_proj");
    const auto shared_gate_shape = find_shape(specs, "model.layers[0].mlp.shared_expert.gate_proj.weight");
    const auto shared_up_shape = find_shape(specs, "model.layers[0].mlp.shared_expert.up_proj.weight");
    const auto shared_down_shape = find_shape(specs, "model.layers[0].mlp.shared_expert.down_proj.weight");
    const auto shared_gate_scalar_shape = find_shape(specs, "model.layers[0].mlp.shared_expert_gate.weight");

    ASSERT_TRUE(gate_shape.has_value());
    ASSERT_TRUE(gate_up_shape.has_value());
    ASSERT_TRUE(down_shape.has_value());
    ASSERT_TRUE(shared_gate_shape.has_value());
    ASSERT_TRUE(shared_up_shape.has_value());
    ASSERT_TRUE(shared_down_shape.has_value());
    ASSERT_TRUE(shared_gate_scalar_shape.has_value());

    EXPECT_EQ(*gate_shape, (ov::Shape{4, 16}));
    EXPECT_EQ(*gate_up_shape, (ov::Shape{4, 16, 16}));
    EXPECT_EQ(*down_shape, (ov::Shape{4, 16, 8}));
    EXPECT_EQ(*shared_gate_shape, (ov::Shape{8, 16}));
    EXPECT_EQ(*shared_up_shape, (ov::Shape{8, 16}));
    EXPECT_EQ(*shared_down_shape, (ov::Shape{16, 8}));
    EXPECT_EQ(*shared_gate_scalar_shape, (ov::Shape{1, 16}));

    EXPECT_FALSE(find_shape(specs, "model.layers[0].mlp.gate_proj.weight").has_value());
    EXPECT_FALSE(find_shape(specs, "model.layers[0].mlp.up_proj.weight").has_value());
    EXPECT_FALSE(find_shape(specs, "model.layers[0].mlp.down_proj.weight").has_value());
}

