// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"

namespace {

ov::genai::modeling::models::Qwen3_5Config make_small_cfg() {
    using namespace ov::genai::modeling::models;
    Qwen3_5Config cfg;
    cfg.model_type = "qwen3_5";

    cfg.text.model_type = "qwen3_5_text";
    cfg.text.vocab_size = 64;
    cfg.text.hidden_size = 16;
    cfg.text.intermediate_size = 32;
    cfg.text.num_hidden_layers = 2;
    cfg.text.num_attention_heads = 4;
    cfg.text.num_key_value_heads = 2;
    cfg.text.head_dim = 4;
    cfg.text.partial_rotary_factor = 0.5f;
    cfg.text.max_position_embeddings = 256;
    cfg.text.layer_types = {"linear_attention", "full_attention"};
    cfg.text.linear_conv_kernel_dim = 2;
    cfg.text.linear_key_head_dim = 4;
    cfg.text.linear_value_head_dim = 4;
    cfg.text.linear_num_key_heads = 2;
    cfg.text.linear_num_value_heads = 2;
    cfg.text.rope.mrope_interleaved = true;
    cfg.text.rope.mrope_section = {1, 1, 0};

    cfg.vision.model_type = "qwen3_5";
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

    cfg.image_token_id = 7;
    cfg.video_token_id = 8;
    cfg.vision_start_token_id = 9;
    cfg.vision_end_token_id = 10;

    cfg.finalize();
    cfg.validate();
    return cfg;
}

}  // namespace

TEST(Qwen3_5Config, DummyDense9bUsesUpdatedSpecialTokenIds) {
    const auto cfg = ov::genai::modeling::models::Qwen3_5Config::make_dummy_dense9b_config();
    EXPECT_EQ(cfg.image_token_id, 248056);
    EXPECT_EQ(cfg.video_token_id, 248057);
    EXPECT_EQ(cfg.vision_start_token_id, 248053);
    EXPECT_EQ(cfg.vision_end_token_id, 248054);
}

TEST(Qwen3_5RopePlanner, BuildPlanProducesThreeChannelPositionIds) {
    const auto cfg = make_small_cfg();
    ov::genai::modeling::models::Qwen3_5InputPlanner planner(cfg);

    ov::Tensor input_ids(ov::element::i64, {1, 3});
    auto* ids = input_ids.data<int64_t>();
    ids[0] = 11;
    ids[1] = cfg.image_token_id;
    ids[2] = 12;

    ov::Tensor attention_mask(ov::element::i64, {1, 3});
    auto* mask = attention_mask.data<int64_t>();
    mask[0] = 1;
    mask[1] = 1;
    mask[2] = 1;

    ov::Tensor grid_thw(ov::element::i64, {1, 3});
    auto* grid = grid_thw.data<int64_t>();
    grid[0] = 1;
    grid[1] = 2;
    grid[2] = 2;

    auto plan = planner.build_plan(input_ids, &attention_mask, &grid_thw);
    const auto pos_shape = plan.position_ids.get_shape();
    ASSERT_EQ(pos_shape.size(), 3u);
    EXPECT_EQ(pos_shape[0], 3u);
    EXPECT_EQ(pos_shape[1], 1u);
    EXPECT_EQ(pos_shape[2], 3u);

    const auto visual_mask_shape = plan.visual_pos_mask.get_shape();
    ASSERT_EQ(visual_mask_shape.size(), 2u);
    EXPECT_EQ(visual_mask_shape[0], 1u);
    EXPECT_EQ(visual_mask_shape[1], 3u);
    EXPECT_EQ(plan.visual_pos_mask.data<char>()[1], 1);

    const auto rope_shape = plan.rope_deltas.get_shape();
    ASSERT_EQ(rope_shape.size(), 2u);
    EXPECT_EQ(rope_shape[0], 1u);
    EXPECT_EQ(rope_shape[1], 1u);
}

TEST(Qwen3_5RopePlanner, BuildPlanHandlesImageAndVideoPlaceholders) {
    const auto cfg = make_small_cfg();
    ov::genai::modeling::models::Qwen3_5InputPlanner planner(cfg);

    ov::Tensor input_ids(ov::element::i64, {1, 7});
    auto* ids = input_ids.data<int64_t>();
    ids[0] = 11;
    ids[1] = cfg.vision_start_token_id;
    ids[2] = cfg.image_token_id;
    ids[3] = 12;
    ids[4] = cfg.vision_start_token_id;
    ids[5] = cfg.video_token_id;
    ids[6] = 13;

    ov::Tensor attention_mask(ov::element::i64, {1, 7});
    auto* mask = attention_mask.data<int64_t>();
    for (size_t i = 0; i < 7; ++i) {
        mask[i] = 1;
    }

    ov::Tensor image_grid_thw(ov::element::i64, {1, 3});
    auto* image_grid = image_grid_thw.data<int64_t>();
    image_grid[0] = 1;
    image_grid[1] = 2;
    image_grid[2] = 2;

    ov::Tensor video_grid_thw(ov::element::i64, {1, 3});
    auto* video_grid = video_grid_thw.data<int64_t>();
    video_grid[0] = 1;
    video_grid[1] = 2;
    video_grid[2] = 2;

    auto plan = planner.build_plan(input_ids, &attention_mask, &image_grid_thw, &video_grid_thw);

    const char* visual_mask = plan.visual_pos_mask.data<const char>();
    EXPECT_EQ(visual_mask[2], 1);
    EXPECT_EQ(visual_mask[5], 1);

    const int64_t* pos = plan.position_ids.data<const int64_t>();
    for (int64_t i = 0; i < 7; ++i) {
        EXPECT_EQ(pos[i], i);
        EXPECT_EQ(pos[7 + i], i);
        EXPECT_EQ(pos[14 + i], i);
    }

    EXPECT_EQ(plan.rope_deltas.data<const int64_t>()[0], 0);
}

TEST(Qwen3_5RopePlanner, BuildPlanExpandsVideoGridByFrameCount) {
    const auto cfg = make_small_cfg();
    ov::genai::modeling::models::Qwen3_5InputPlanner planner(cfg);

    ov::Tensor input_ids(ov::element::i64, {1, 4});
    auto* ids = input_ids.data<int64_t>();
    ids[0] = cfg.vision_start_token_id;
    ids[1] = cfg.video_token_id;
    ids[2] = cfg.vision_start_token_id;
    ids[3] = cfg.video_token_id;

    ov::Tensor attention_mask(ov::element::i64, {1, 4});
    auto* mask = attention_mask.data<int64_t>();
    mask[0] = 1;
    mask[1] = 1;
    mask[2] = 1;
    mask[3] = 1;

    ov::Tensor video_grid_thw(ov::element::i64, {1, 3});
    auto* video_grid = video_grid_thw.data<int64_t>();
    video_grid[0] = 2;
    video_grid[1] = 2;
    video_grid[2] = 2;

    auto plan = planner.build_plan(input_ids, &attention_mask, nullptr, &video_grid_thw);

    const char* visual_mask = plan.visual_pos_mask.data<const char>();
    EXPECT_EQ(visual_mask[1], 1);
    EXPECT_EQ(visual_mask[3], 1);

    const int64_t* pos = plan.position_ids.data<const int64_t>();
    EXPECT_EQ(pos[0], 0);
    EXPECT_EQ(pos[1], 1);
    EXPECT_EQ(pos[2], 2);
    EXPECT_EQ(pos[3], 3);
    EXPECT_EQ(plan.rope_deltas.data<const int64_t>()[0], 0);
}

TEST(Qwen3_5RopePlanner, DecodePositionIdsApplyRopeDeltas) {
    ov::Tensor rope_deltas(ov::element::i64, {1, 1});
    rope_deltas.data<int64_t>()[0] = 3;

    auto position_ids =
        ov::genai::modeling::models::Qwen3_5InputPlanner::build_decode_position_ids(rope_deltas, 10, 2);
    const auto shape = position_ids.get_shape();
    ASSERT_EQ(shape.size(), 3u);
    EXPECT_EQ(shape[0], 3u);
    EXPECT_EQ(shape[1], 1u);
    EXPECT_EQ(shape[2], 2u);

    const auto* data = position_ids.data<const int64_t>();
    EXPECT_EQ(data[0], 13);
    EXPECT_EQ(data[1], 14);
    EXPECT_EQ(data[2], 13);
    EXPECT_EQ(data[3], 14);
    EXPECT_EQ(data[4], 13);
    EXPECT_EQ(data[5], 14);
}
