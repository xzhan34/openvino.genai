// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include <openvino/op/read_value.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_next/modeling_qwen3_next.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

TEST(Qwen3NextLinearAttention, BuildsGraphAndRegistersLinearStates) {
    ov::genai::modeling::BuilderContext ctx;

    ov::genai::modeling::models::Qwen3NextConfig cfg;
    cfg.hidden_size = 2048;
    cfg.linear_num_key_heads = 16;
    cfg.linear_num_value_heads = 32;
    cfg.linear_key_head_dim = 128;
    cfg.linear_value_head_dim = 128;
    cfg.linear_conv_kernel_dim = 4;
    cfg.rms_norm_eps = 1e-6f;

    ov::genai::modeling::models::Qwen3NextGatedDeltaNet linear_attn(ctx, "linear_attn", cfg, 0);

    test_utils::DummyWeightSource weights;
    test_utils::DummyWeightFinalizer finalizer;

    const int32_t key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    const int32_t value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    const int32_t conv_dim = key_dim * 2 + value_dim;
    const int32_t proj_qkvz = key_dim * 2 + value_dim * 2;
    const int32_t proj_ba = cfg.linear_num_value_heads * 2;

    weights.add("linear_attn.in_proj_qkvz.weight",
                test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(proj_qkvz * cfg.hidden_size), 0.01f, 0.001f),
                                        {static_cast<size_t>(proj_qkvz), static_cast<size_t>(cfg.hidden_size)}));
    weights.add("linear_attn.in_proj_ba.weight",
                test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(proj_ba * cfg.hidden_size), 0.02f, 0.001f),
                                        {static_cast<size_t>(proj_ba), static_cast<size_t>(cfg.hidden_size)}));
    weights.add("linear_attn.conv1d.weight",
                test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(conv_dim * cfg.linear_conv_kernel_dim), 0.03f, 0.001f),
                                        {static_cast<size_t>(conv_dim), static_cast<size_t>(cfg.linear_conv_kernel_dim)}));
    weights.add("linear_attn.A_log",
                test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(cfg.linear_num_value_heads), 0.04f, 0.001f),
                                        {static_cast<size_t>(cfg.linear_num_value_heads)}));
    weights.add("linear_attn.dt_bias",
                test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(cfg.linear_num_value_heads), 0.05f, 0.001f),
                                        {static_cast<size_t>(cfg.linear_num_value_heads)}));
    weights.add("linear_attn.norm.weight",
                test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(cfg.linear_value_head_dim), 1.0f, 0.0f),
                                        {static_cast<size_t>(cfg.linear_value_head_dim)}));
    weights.add("linear_attn.out_proj.weight",
                test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(cfg.hidden_size * value_dim), 0.06f, 0.001f),
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

TEST(Qwen3NextLinearAttention, CompilesAndInfersOnGPU) {
    ov::genai::modeling::BuilderContext ctx;

    ov::genai::modeling::models::Qwen3NextConfig cfg;
    cfg.hidden_size = 2048;
    cfg.linear_num_key_heads = 16;
    cfg.linear_num_value_heads = 32;
    cfg.linear_key_head_dim = 128;
    cfg.linear_value_head_dim = 128;
    cfg.linear_conv_kernel_dim = 4;
    cfg.rms_norm_eps = 1e-6f;

    ov::genai::modeling::models::Qwen3NextGatedDeltaNet linear_attn(ctx, "linear_attn", cfg, 0);

    test_utils::DummyWeightSource weights;
    test_utils::DummyWeightFinalizer finalizer;

    const int32_t key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    const int32_t value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    const int32_t conv_dim = key_dim * 2 + value_dim;
    const int32_t proj_qkvz = key_dim * 2 + value_dim * 2;
    const int32_t proj_ba = cfg.linear_num_value_heads * 2;

    weights.add("linear_attn.in_proj_qkvz.weight",
                test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(proj_qkvz * cfg.hidden_size), 0.01f, 0.001f),
                                        {static_cast<size_t>(proj_qkvz), static_cast<size_t>(cfg.hidden_size)}));
    weights.add("linear_attn.in_proj_ba.weight",
                test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(proj_ba * cfg.hidden_size), 0.02f, 0.001f),
                                        {static_cast<size_t>(proj_ba), static_cast<size_t>(cfg.hidden_size)}));
    weights.add("linear_attn.conv1d.weight",
                test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(conv_dim * cfg.linear_conv_kernel_dim), 0.03f, 0.001f),
                                        {static_cast<size_t>(conv_dim), static_cast<size_t>(cfg.linear_conv_kernel_dim)}));
    weights.add("linear_attn.A_log",
                test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(cfg.linear_num_value_heads), 0.04f, 0.001f),
                                        {static_cast<size_t>(cfg.linear_num_value_heads)}));
    weights.add("linear_attn.dt_bias",
                test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(cfg.linear_num_value_heads), 0.05f, 0.001f),
                                        {static_cast<size_t>(cfg.linear_num_value_heads)}));
    weights.add("linear_attn.norm.weight",
                test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(cfg.linear_value_head_dim), 1.0f, 0.0f),
                                        {static_cast<size_t>(cfg.linear_value_head_dim)}));
    weights.add("linear_attn.out_proj.weight",
                test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(cfg.hidden_size * value_dim), 0.06f, 0.001f),
                                        {static_cast<size_t>(cfg.hidden_size), static_cast<size_t>(value_dim)}));

    ov::genai::modeling::weights::load_model(linear_attn, weights, finalizer);

    auto hidden_states = ctx.parameter("hidden_states", ov::element::f32, ov::PartialShape{1, 2, cfg.hidden_size});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{1, 2});

    auto out = linear_attn.forward(hidden_states, beam_idx, &attention_mask, nullptr);
    auto ov_model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    ov::Tensor hidden_states_tensor(ov::element::f32, ov::Shape{1, 2, static_cast<size_t>(cfg.hidden_size)});
    ov::Tensor beam_idx_tensor(ov::element::i32, ov::Shape{1});
    ov::Tensor attention_mask_tensor(ov::element::i64, ov::Shape{1, 2});

    std::fill_n(hidden_states_tensor.data<float>(), hidden_states_tensor.get_size(), 0.1f);
    beam_idx_tensor.data<int32_t>()[0] = 0;
    auto* mask_data = attention_mask_tensor.data<int64_t>();
    mask_data[0] = 1;
    mask_data[1] = 1;

    request.set_tensor("hidden_states", hidden_states_tensor);
    request.set_tensor("beam_idx", beam_idx_tensor);
    request.set_tensor("attention_mask", attention_mask_tensor);
    request.infer();

    const auto output = request.get_output_tensor(0);
    EXPECT_EQ(output.get_shape(), (ov::Shape{1, 2, static_cast<size_t>(cfg.hidden_size)}));
}

