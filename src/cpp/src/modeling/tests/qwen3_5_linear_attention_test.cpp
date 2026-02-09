// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include <gtest/gtest.h>

#include <openvino/op/read_value.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

TEST(Qwen3_5LinearAttention, BuildsGraphAndRegistersLinearStates) {
    ov::genai::modeling::BuilderContext ctx;

    ov::genai::modeling::models::Qwen3_5TextModelConfig cfg;
    cfg.hidden_size = 16;
    cfg.linear_num_key_heads = 2;
    cfg.linear_num_value_heads = 4;
    cfg.linear_key_head_dim = 4;
    cfg.linear_value_head_dim = 4;
    cfg.linear_conv_kernel_dim = 4;
    cfg.rms_norm_eps = 1e-6f;
    cfg.hidden_act = "silu";

    ov::genai::modeling::models::Qwen3_5GatedDeltaNet linear_attn(ctx, "linear_attn", cfg, 0);

    test_utils::DummyWeightSource weights;
    test_utils::DummyWeightFinalizer finalizer;

    const int32_t key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    const int32_t value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    const int32_t conv_dim = key_dim * 2 + value_dim;
    const int32_t proj_qkv = key_dim * 2 + value_dim;

    weights.add(
        "linear_attn.in_proj_qkv.weight",
        test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(proj_qkv * cfg.hidden_size), 0.01f, 0.001f),
                                {static_cast<size_t>(proj_qkv), static_cast<size_t>(cfg.hidden_size)}));
    weights.add(
        "linear_attn.in_proj_z.weight",
        test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(value_dim * cfg.hidden_size), 0.015f, 0.001f),
                                {static_cast<size_t>(value_dim), static_cast<size_t>(cfg.hidden_size)}));
    weights.add(
        "linear_attn.in_proj_b.weight",
        test_utils::make_tensor(
            test_utils::make_seq(static_cast<size_t>(cfg.linear_num_value_heads * cfg.hidden_size), 0.02f, 0.001f),
            {static_cast<size_t>(cfg.linear_num_value_heads), static_cast<size_t>(cfg.hidden_size)}));
    weights.add(
        "linear_attn.in_proj_a.weight",
        test_utils::make_tensor(
            test_utils::make_seq(static_cast<size_t>(cfg.linear_num_value_heads * cfg.hidden_size), 0.03f, 0.001f),
            {static_cast<size_t>(cfg.linear_num_value_heads), static_cast<size_t>(cfg.hidden_size)}));
    weights.add(
        "linear_attn.conv1d.weight",
        test_utils::make_tensor(
            test_utils::make_seq(static_cast<size_t>(conv_dim * cfg.linear_conv_kernel_dim), 0.04f, 0.001f),
            {static_cast<size_t>(conv_dim), static_cast<size_t>(cfg.linear_conv_kernel_dim)}));
    weights.add("linear_attn.A_log",
                test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(cfg.linear_num_value_heads), 0.05f, 0.001f),
                                        {static_cast<size_t>(cfg.linear_num_value_heads)}));
    weights.add("linear_attn.dt_bias",
                test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(cfg.linear_num_value_heads), 0.06f, 0.001f),
                                        {static_cast<size_t>(cfg.linear_num_value_heads)}));
    weights.add("linear_attn.norm.weight",
                test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(cfg.linear_value_head_dim), 1.0f, 0.0f),
                                        {static_cast<size_t>(cfg.linear_value_head_dim)}));
    weights.add(
        "linear_attn.out_proj.weight",
        test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(cfg.hidden_size * value_dim), 0.07f, 0.001f),
                                {static_cast<size_t>(cfg.hidden_size), static_cast<size_t>(value_dim)}));

    ov::genai::modeling::weights::load_model(linear_attn, weights, finalizer);

    auto hidden_states = ctx.parameter("hidden_states", ov::element::f32, ov::PartialShape{1, 2, cfg.hidden_size});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{1, 2});

    auto out = linear_attn.forward(hidden_states, beam_idx, &attention_mask, nullptr);
    auto ov_model = ctx.build_model({out.output()});

    ASSERT_EQ(ov_model->outputs().size(), 1u);
    ASSERT_EQ(ov_model->get_output_partial_shape(0).rank().get_length(), 3);

    bool has_conv_state = false;
    bool has_recurrent_state = false;
    size_t read_value_count = 0;
    size_t assign_count = 0;
    for (const auto& op : ov_model->get_ops()) {
        if (auto read = ov::as_type_ptr<ov::op::v6::ReadValue>(op)) {
            read_value_count++;
            const auto id = read->get_variable_id();
            has_conv_state = has_conv_state || id.find("linear_states.0.conv") != std::string::npos;
            has_recurrent_state = has_recurrent_state || id.find("linear_states.0.recurrent") != std::string::npos;
        }
        if (ov::as_type_ptr<ov::op::v6::Assign>(op)) {
            assign_count++;
        }
    }

    EXPECT_GE(read_value_count, 2u);
    EXPECT_GE(assign_count, 2u);
    EXPECT_TRUE(has_conv_state);
    EXPECT_TRUE(has_recurrent_state);
}
