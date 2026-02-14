// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/op/read_value.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/quantization_config.hpp"
#include "modeling/weights/synthetic_weight_source.hpp"
#include "modeling/weights/weight_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

ov::genai::modeling::models::Qwen3_5TextModelConfig make_qwen3_5_9b_attention_cfg() {
    const auto dense_9b = ov::genai::modeling::models::Qwen3_5Config::make_dummy_dense9b_config();

    ov::genai::modeling::models::Qwen3_5TextModelConfig cfg;
    cfg.hidden_size = dense_9b.text.hidden_size;
    cfg.num_attention_heads = dense_9b.text.num_attention_heads;
    cfg.num_key_value_heads = dense_9b.text.num_key_value_heads;
    cfg.head_dim = dense_9b.text.head_dim;
    cfg.rms_norm_eps = dense_9b.text.rms_norm_eps;
    cfg.rope_theta = dense_9b.text.rope_theta;
    cfg.partial_rotary_factor = dense_9b.text.partial_rotary_factor;
    cfg.hidden_act = dense_9b.text.hidden_act;

    cfg.linear_conv_kernel_dim = dense_9b.text.linear_conv_kernel_dim;
    cfg.linear_key_head_dim = dense_9b.text.linear_key_head_dim;
    cfg.linear_value_head_dim = dense_9b.text.linear_value_head_dim;
    cfg.linear_num_key_heads = dense_9b.text.linear_num_key_heads;
    cfg.linear_num_value_heads = dense_9b.text.linear_num_value_heads;
    return cfg;
}

int32_t resolve_rotary_dim(const ov::genai::modeling::models::Qwen3_5TextModelConfig& cfg) {
    int32_t rotary_dim = static_cast<int32_t>(std::floor(static_cast<float>(cfg.head_dim) * cfg.partial_rotary_factor));
    rotary_dim = std::max<int32_t>(0, std::min<int32_t>(rotary_dim, cfg.head_dim));
    if ((rotary_dim % 2) != 0) {
        rotary_dim -= 1;
    }
    return rotary_dim;
}

std::vector<ov::genai::modeling::weights::SyntheticWeightSpec> make_attention_weight_specs(
    const ov::genai::modeling::models::Qwen3_5TextModelConfig& cfg,
    const std::string& prefix) {
    const size_t hidden = static_cast<size_t>(cfg.hidden_size);
    const size_t q_out = static_cast<size_t>(cfg.num_attention_heads) * static_cast<size_t>(cfg.head_dim) * 2ull;
    const size_t kv_out = static_cast<size_t>(cfg.num_key_value_heads) * static_cast<size_t>(cfg.head_dim);
    const size_t attn_hidden = static_cast<size_t>(cfg.num_attention_heads) * static_cast<size_t>(cfg.head_dim);

    return {
        {prefix + ".q_proj.weight", {q_out, hidden}, ov::element::f32},
        {prefix + ".k_proj.weight", {kv_out, hidden}, ov::element::f32},
        {prefix + ".v_proj.weight", {kv_out, hidden}, ov::element::f32},
        {prefix + ".o_proj.weight", {hidden, attn_hidden}, ov::element::f32},
        {prefix + ".q_norm.weight", {static_cast<size_t>(cfg.head_dim)}, ov::element::f32},
        {prefix + ".k_norm.weight", {static_cast<size_t>(cfg.head_dim)}, ov::element::f32},
    };
}

std::vector<ov::genai::modeling::weights::SyntheticWeightSpec> make_linear_attention_weight_specs(
    const ov::genai::modeling::models::Qwen3_5TextModelConfig& cfg,
    const std::string& prefix) {
    const size_t hidden = static_cast<size_t>(cfg.hidden_size);
    const size_t key_dim = static_cast<size_t>(cfg.linear_num_key_heads) * static_cast<size_t>(cfg.linear_key_head_dim);
    const size_t value_dim = static_cast<size_t>(cfg.linear_num_value_heads) * static_cast<size_t>(cfg.linear_value_head_dim);
    const size_t conv_dim = key_dim * 2ull + value_dim;
    const size_t proj_qkv = key_dim * 2ull + value_dim;

    return {
        {prefix + ".in_proj_qkv.weight", {proj_qkv, hidden}, ov::element::f32},
        {prefix + ".in_proj_z.weight", {value_dim, hidden}, ov::element::f32},
        {prefix + ".in_proj_b.weight", {static_cast<size_t>(cfg.linear_num_value_heads), hidden}, ov::element::f32},
        {prefix + ".in_proj_a.weight", {static_cast<size_t>(cfg.linear_num_value_heads), hidden}, ov::element::f32},
        {prefix + ".conv1d.weight", {conv_dim, static_cast<size_t>(cfg.linear_conv_kernel_dim)}, ov::element::f32},
        {prefix + ".A_log", {static_cast<size_t>(cfg.linear_num_value_heads)}, ov::element::f32},
        {prefix + ".dt_bias", {static_cast<size_t>(cfg.linear_num_value_heads)}, ov::element::f32},
        {prefix + ".norm.weight", {static_cast<size_t>(cfg.linear_value_head_dim)}, ov::element::f32},
        {prefix + ".out_proj.weight", {hidden, value_dim}, ov::element::f32},
    };
}

ov::genai::modeling::weights::QuantizationConfig make_int4_quant_config(const std::string& prefix) {
    ov::genai::modeling::weights::QuantizationConfig quant_config;
    quant_config.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_ASYM;
    quant_config.group_size = 128;
    quant_config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::NONE;
    quant_config.selection.only_2d_weights = true;
    quant_config.selection.include_weights = {
        prefix + ".q_proj.weight",
        prefix + ".k_proj.weight",
        prefix + ".v_proj.weight",
        prefix + ".o_proj.weight"};
    return quant_config;
}

ov::genai::modeling::weights::QuantizationConfig make_linear_int4_quant_config(const std::string& prefix) {
    ov::genai::modeling::weights::QuantizationConfig quant_config;
    quant_config.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_ASYM;
    quant_config.group_size = 128;
    quant_config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::NONE;
    quant_config.selection.only_2d_weights = true;
    quant_config.selection.include_weights = {
        prefix + ".in_proj_qkv.weight",
        prefix + ".in_proj_z.weight",
        prefix + ".in_proj_b.weight",
        prefix + ".in_proj_a.weight",
        prefix + ".conv1d.weight",
        prefix + ".out_proj.weight"};
    return quant_config;
}

}  // namespace

TEST(Qwen3_5GatedDeltaNet, BuildsGraphWithPartialRopeAndGate) {
    ov::genai::modeling::BuilderContext ctx;

    const auto cfg = make_qwen3_5_9b_attention_cfg();
    ov::genai::modeling::models::Qwen3_5GatedDeltaNet linear_attn(ctx, "linear_attn", cfg, 0);

    auto specs = make_linear_attention_weight_specs(cfg, "linear_attn");
    ov::genai::modeling::weights::SyntheticWeightSource weights(std::move(specs), 2025u, -0.02f, 0.02f);
    auto quant_config = make_linear_int4_quant_config("linear_attn");
    ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(quant_config);
    ov::genai::modeling::weights::load_model(linear_attn, weights, finalizer);

    auto hidden_states = ctx.parameter("hidden_states", ov::element::f32, ov::PartialShape{1, 2, cfg.hidden_size});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{1, 2});

    auto out = linear_attn.forward(hidden_states, beam_idx, &attention_mask, nullptr);
    auto ov_model = ctx.build_model({out.output()});

    const std::vector<std::string> expected_compressed_nodes = {
        "linear_attn.in_proj_qkv.weight_compressed",
        "linear_attn.in_proj_z.weight_compressed",
        "linear_attn.in_proj_b.weight_compressed",
        "linear_attn.in_proj_a.weight_compressed",
        "linear_attn.conv1d.weight_compressed",
        "linear_attn.out_proj.weight_compressed"};
    for (const auto& expected_name : expected_compressed_nodes) {
        bool found = false;
        for (const auto& op : ov_model->get_ops()) {
            if (op->get_friendly_name() == expected_name) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Missing quantized node: " << expected_name;
    }

    ov::serialize(ov_model, "qwen3_5_gated_deltanet_gpu_original.xml");

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();
    ov::serialize(compiled.get_runtime_model(), "qwen3_5_gated_deltanet_gpu_compiled.xml");

    ov::Tensor hidden_states_tensor(ov::element::f32, ov::Shape{1, 2, static_cast<size_t>(cfg.hidden_size)});
    ov::Tensor beam_idx_tensor(ov::element::i32, ov::Shape{1});
    ov::Tensor attention_mask_tensor(ov::element::i64, ov::Shape{1, 2});

    std::fill_n(hidden_states_tensor.data<float>(), hidden_states_tensor.get_size(), 0.1f);
    beam_idx_tensor.data<int32_t>()[0] = 0;
    std::fill_n(attention_mask_tensor.data<int64_t>(), attention_mask_tensor.get_size(), 1);

    request.set_tensor("hidden_states", hidden_states_tensor);
    request.set_tensor("beam_idx", beam_idx_tensor);
    request.set_tensor("attention_mask", attention_mask_tensor);
    request.infer();

    const auto output = request.get_output_tensor(0);
    EXPECT_EQ(output.get_shape(), (ov::Shape{1, 2, static_cast<size_t>(cfg.hidden_size)}));
}

TEST(Qwen3_5Attention, CompilesAndInfersOnGPU) {
    ov::genai::modeling::BuilderContext ctx;

    const auto cfg = make_qwen3_5_9b_attention_cfg();
    const int32_t rotary_dim = resolve_rotary_dim(cfg);
    ASSERT_GT(rotary_dim, 0);

    ov::genai::modeling::models::Qwen3_5Attention attn(ctx, "self_attn", cfg);

    auto specs = make_attention_weight_specs(cfg, "self_attn");
    ov::genai::modeling::weights::SyntheticWeightSource weights(std::move(specs), 2026u, -0.02f, 0.02f);
    auto quant_config = make_int4_quant_config("self_attn");
    ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(quant_config);
    ov::genai::modeling::weights::load_model(attn, weights, finalizer);

    auto hidden_states = ctx.parameter("hidden_states", ov::element::f32, ov::PartialShape{1, 2, cfg.hidden_size});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{1});
    auto rope_cos = ctx.parameter("rope_cos", ov::element::f32, ov::PartialShape{1, 2, rotary_dim / 2});
    auto rope_sin = ctx.parameter("rope_sin", ov::element::f32, ov::PartialShape{1, 2, rotary_dim / 2});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{1, 2});

    auto out = attn.forward(hidden_states, beam_idx, rope_cos, rope_sin, &attention_mask);
    auto ov_model = ctx.build_model({out.output()});

    const std::vector<std::string> expected_compressed_nodes = {
        "self_attn.q_proj.weight_compressed",
        "self_attn.k_proj.weight_compressed",
        "self_attn.v_proj.weight_compressed",
        "self_attn.o_proj.weight_compressed"};
    for (const auto& expected_name : expected_compressed_nodes) {
        bool found = false;
        for (const auto& op : ov_model->get_ops()) {
            if (op->get_friendly_name() == expected_name) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Missing quantized node: " << expected_name;
    }

    ov::serialize(ov_model, "qwen3_5_attention_gpu_original.xml");

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();
    ov::serialize(compiled.get_runtime_model(), "qwen3_5_attention_gpu_compiled.xml");

    ov::Tensor hidden_states_tensor(ov::element::f32, ov::Shape{1, 2, static_cast<size_t>(cfg.hidden_size)});
    ov::Tensor beam_idx_tensor(ov::element::i32, ov::Shape{1});
    ov::Tensor rope_cos_tensor(ov::element::f32, ov::Shape{1, 2, static_cast<size_t>(rotary_dim / 2)});
    ov::Tensor rope_sin_tensor(ov::element::f32, ov::Shape{1, 2, static_cast<size_t>(rotary_dim / 2)});
    ov::Tensor attention_mask_tensor(ov::element::i64, ov::Shape{1, 2});

    std::fill_n(hidden_states_tensor.data<float>(), hidden_states_tensor.get_size(), 0.1f);
    beam_idx_tensor.data<int32_t>()[0] = 0;
    std::fill_n(rope_cos_tensor.data<float>(), rope_cos_tensor.get_size(), 1.0f);
    std::fill_n(rope_sin_tensor.data<float>(), rope_sin_tensor.get_size(), 0.0f);
    std::fill_n(attention_mask_tensor.data<int64_t>(), attention_mask_tensor.get_size(), 1);

    request.set_tensor("hidden_states", hidden_states_tensor);
    request.set_tensor("beam_idx", beam_idx_tensor);
    request.set_tensor("rope_cos", rope_cos_tensor);
    request.set_tensor("rope_sin", rope_sin_tensor);
    request.set_tensor("attention_mask", attention_mask_tensor);
    request.infer();

    const auto output = request.get_output_tensor(0);
    EXPECT_EQ(output.get_shape(), (ov::Shape{1, 2, static_cast<size_t>(cfg.hidden_size)}));
}

TEST(Qwen3_5Attention, StatefulPrefillAndDecodeOnGPU) {
    ov::genai::modeling::BuilderContext ctx;

    const auto cfg = make_qwen3_5_9b_attention_cfg();
    const int32_t rotary_dim = resolve_rotary_dim(cfg);
    ASSERT_GT(rotary_dim, 0);

    ov::genai::modeling::models::Qwen3_5Attention attn(ctx, "self_attn", cfg);

    auto specs = make_attention_weight_specs(cfg, "self_attn");
    ov::genai::modeling::weights::SyntheticWeightSource weights(std::move(specs), 2027u, -0.02f, 0.02f);
    auto quant_config = make_int4_quant_config("self_attn");
    ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(quant_config);
    ov::genai::modeling::weights::load_model(attn, weights, finalizer);

    auto hidden_states = ctx.parameter("hidden_states", ov::element::f32, ov::PartialShape{1, -1, cfg.hidden_size});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{1});
    auto rope_cos = ctx.parameter("rope_cos", ov::element::f32, ov::PartialShape{1, -1, rotary_dim / 2});
    auto rope_sin = ctx.parameter("rope_sin", ov::element::f32, ov::PartialShape{1, -1, rotary_dim / 2});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{1, -1});

    auto out = attn.forward(hidden_states, beam_idx, rope_cos, rope_sin, &attention_mask);
    auto ov_model = ctx.build_model({out.output()});
    ov::serialize(ov_model, "qwen3_5_attention_stateful_original.xml");

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();
    ov::serialize(compiled.get_runtime_model(), "qwen3_5_attention_stateful_compiled.xml");

    auto states = request.query_state();
    ASSERT_EQ(states.size(), 2u);

    bool has_key_state = false;
    bool has_value_state = false;
    for (const auto& state : states) {
        has_key_state = has_key_state || state.get_name().find("key") != std::string::npos;
        has_value_state = has_value_state || state.get_name().find("value") != std::string::npos;
    }
    EXPECT_TRUE(has_key_state);
    EXPECT_TRUE(has_value_state);

    ov::Tensor prefill_hidden(ov::element::f32, ov::Shape{1, 3, static_cast<size_t>(cfg.hidden_size)});
    ov::Tensor prefill_beam(ov::element::i32, ov::Shape{1});
    ov::Tensor prefill_rope_cos(ov::element::f32, ov::Shape{1, 3, static_cast<size_t>(rotary_dim / 2)});
    ov::Tensor prefill_rope_sin(ov::element::f32, ov::Shape{1, 3, static_cast<size_t>(rotary_dim / 2)});
    ov::Tensor prefill_mask(ov::element::i64, ov::Shape{1, 3});
    std::fill_n(prefill_hidden.data<float>(), prefill_hidden.get_size(), 0.1f);
    prefill_beam.data<int32_t>()[0] = 0;
    std::fill_n(prefill_rope_cos.data<float>(), prefill_rope_cos.get_size(), 1.0f);
    std::fill_n(prefill_rope_sin.data<float>(), prefill_rope_sin.get_size(), 0.0f);
    std::fill_n(prefill_mask.data<int64_t>(), prefill_mask.get_size(), 1);

    request.set_tensor("hidden_states", prefill_hidden);
    request.set_tensor("beam_idx", prefill_beam);
    request.set_tensor("rope_cos", prefill_rope_cos);
    request.set_tensor("rope_sin", prefill_rope_sin);
    request.set_tensor("attention_mask", prefill_mask);
    request.infer();

    auto prefill_output = request.get_output_tensor(0);
    EXPECT_EQ(prefill_output.get_shape(), (ov::Shape{1, 3, static_cast<size_t>(cfg.hidden_size)}));

    ov::Tensor decode_hidden(ov::element::f32, ov::Shape{1, 1, static_cast<size_t>(cfg.hidden_size)});
    ov::Tensor decode_rope_cos(ov::element::f32, ov::Shape{1, 1, static_cast<size_t>(rotary_dim / 2)});
    ov::Tensor decode_rope_sin(ov::element::f32, ov::Shape{1, 1, static_cast<size_t>(rotary_dim / 2)});
    ov::Tensor decode_mask(ov::element::i64, ov::Shape{1, 1});
    std::fill_n(decode_hidden.data<float>(), decode_hidden.get_size(), 0.2f);
    std::fill_n(decode_rope_cos.data<float>(), decode_rope_cos.get_size(), 1.0f);
    std::fill_n(decode_rope_sin.data<float>(), decode_rope_sin.get_size(), 0.0f);
    decode_mask.data<int64_t>()[0] = 1;

    request.set_tensor("hidden_states", decode_hidden);
    request.set_tensor("beam_idx", prefill_beam);
    request.set_tensor("rope_cos", decode_rope_cos);
    request.set_tensor("rope_sin", decode_rope_sin);
    request.set_tensor("attention_mask", decode_mask);
    request.infer();

    auto decode_output = request.get_output_tensor(0);
    EXPECT_EQ(decode_output.get_shape(), (ov::Shape{1, 1, static_cast<size_t>(cfg.hidden_size)}));

    ov::Shape key_shape;
    ov::Shape value_shape;
    states = request.query_state();
    for (const auto& state : states) {
        if (state.get_name().find("key") != std::string::npos) {
            key_shape = state.get_state().get_shape();
        } else if (state.get_name().find("value") != std::string::npos) {
            value_shape = state.get_state().get_shape();
        }
    }

    ASSERT_EQ(key_shape.size(), 4u);
    ASSERT_EQ(value_shape.size(), 4u);
    EXPECT_EQ(key_shape[0], 1u);
    EXPECT_EQ(key_shape[1], static_cast<size_t>(cfg.num_key_value_heads));
    EXPECT_EQ(key_shape[2], 4u);
    EXPECT_EQ(key_shape[3], static_cast<size_t>(cfg.head_dim));
    EXPECT_EQ(value_shape[0], 1u);
    EXPECT_EQ(value_shape[1], static_cast<size_t>(cfg.num_key_value_heads));
    EXPECT_EQ(value_shape[2], 4u);
    EXPECT_EQ(value_shape[3], static_cast<size_t>(cfg.head_dim));
}

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
