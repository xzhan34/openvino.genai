// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <unordered_set>

#include "modeling/models/qwen3_omni/modeling_qwen3_omni.hpp"
#include "modeling/models/qwen3_omni/processing_qwen3_omni_audio.hpp"
#include "modeling/models/qwen3_omni/processing_qwen3_omni_vision.hpp"
#include "modeling/models/qwen3_omni/qwen3_omni_pipeline.hpp"
#include "modeling/tests/test_utils.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace tests {

using ov::genai::modeling::models::Qwen3OmniAudioProcess;
using ov::genai::modeling::models::Qwen3OmniConfig;
using ov::genai::modeling::models::Qwen3OmniVisionProcess;

namespace {

void add_zero_weight(DummyWeightSource& source, const std::string& name, const ov::Shape& shape) {
    source.add(name, ov::Tensor(ov::element::f32, shape));
}

}  // namespace

std::shared_ptr<ov::Model> create_dummy_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1});
    auto result = std::make_shared<ov::op::v0::Result>(input);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}

TEST(Qwen3OmniSpecTest, ResolveCodePredictorStepsRespectsConfigAndOptions) {
    Qwen3OmniConfig cfg;
    cfg.text.hidden_size = 8;
    cfg.text.num_hidden_layers = 1;
    cfg.text.num_attention_heads = 2;
    cfg.text.head_dim = 4;
    cfg.text.num_key_value_heads = 1;
    cfg.talker_config_raw = nlohmann::json::parse(R"({
        "hidden_size": 32,
        "code_predictor_config": {
            "num_code_groups": 6
        }
    })");

    ov::genai::modeling::models::Qwen3OmniPipelineBuildOptions options;
    EXPECT_EQ(ov::genai::modeling::models::resolve_qwen3_omni_code_predictor_steps(cfg, options), 5);

    options.code_predictor_steps = 3;
    EXPECT_EQ(ov::genai::modeling::models::resolve_qwen3_omni_code_predictor_steps(cfg, options), 3);

    options.code_predictor_steps = 20;
    EXPECT_EQ(ov::genai::modeling::models::resolve_qwen3_omni_code_predictor_steps(cfg, options), 5);
}

TEST(Qwen3OmniSpecTest, ValidatePipelineModelsChecksCompleteness) {
    ov::genai::modeling::models::Qwen3OmniPipelineModels models;
    EXPECT_THROW(ov::genai::modeling::models::validate_qwen3_omni_pipeline_models(models, 1), ov::Exception);

    models.text = create_dummy_model();
    models.vision = create_dummy_model();
    models.talker_embedding = create_dummy_model();
    models.talker_codec_embedding = create_dummy_model();
    models.talker = create_dummy_model();
    models.talker_prefill = create_dummy_model();
    models.talker_decode = create_dummy_model();
    models.code_predictor_codec_embedding = create_dummy_model();
    models.speech_decoder = create_dummy_model();

    models.code_predictor_ar = {create_dummy_model(), create_dummy_model()};
    models.code_predictor_single_codec_embedding = {create_dummy_model(), create_dummy_model()};
    EXPECT_NO_THROW(ov::genai::modeling::models::validate_qwen3_omni_pipeline_models(models, 2));

    models.code_predictor_single_codec_embedding.pop_back();
    EXPECT_THROW(ov::genai::modeling::models::validate_qwen3_omni_pipeline_models(models, 2), ov::Exception);

    EXPECT_THROW(ov::genai::modeling::models::validate_qwen3_omni_pipeline_models(models, 0), ov::Exception);
}

TEST(Qwen3OmniSpecTest, ParseNestedThinkerConfig) {
    const char* text = R"({
        "model_type": "qwen3_omni",
        "architectures": ["Qwen3OmniForConditionalGeneration"],
        "thinker_config": {
            "audio_token_id": 151646,
            "image_token_id": 151655,
            "video_token_id": 151656,
            "audio_start_token_id": 151647,
            "vision_start_token_id": 151652,
            "text_config": {
                "model_type": "qwen3_omni_text",
                "vocab_size": 151936,
                "hidden_size": 2560,
                "intermediate_size": 9728,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 32768,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000,
                "hidden_act": "silu",
                "attention_bias": false,
                "rope_scaling": {
                    "mrope_interleaved": true,
                    "mrope_section": [24, 20, 20],
                    "rope_type": "default"
                }
            },
            "vision_config": {
                "model_type": "qwen3_omni_vision_encoder",
                "depth": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_heads": 16,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "out_hidden_size": 2560,
                "num_position_embeddings": 2304,
                "deepstack_visual_indexes": [8, 16, 24]
            },
            "audio_config": {
                "model_type": "qwen3_omni_audio_encoder",
                "num_mel_bins": 128,
                "downsample_hidden_size": 480,
                "encoder_layers": 32,
                "encoder_attention_heads": 20,
                "encoder_ffn_dim": 5120,
                "d_model": 1280,
                "output_dim": 2560,
                "n_window": 25,
                "n_window_infer": 8,
                "layer_norm_eps": 1e-5,
                "activation_function": "gelu_new"
            }
        }
    })";

    auto json = nlohmann::json::parse(text);
    Qwen3OmniConfig cfg = Qwen3OmniConfig::from_json(json);

    EXPECT_EQ(cfg.model_type, "qwen3_omni");
    EXPECT_EQ(cfg.text.hidden_size, 2560);
    EXPECT_EQ(cfg.text.num_key_value_heads, 8);
    EXPECT_TRUE(cfg.text.rope.mrope_interleaved);
    EXPECT_EQ(cfg.vision.hidden_size, 1152);
    EXPECT_EQ(cfg.audio.d_model, 1280);
    EXPECT_EQ(cfg.audio.downsample_hidden_size, 480);
    EXPECT_EQ(cfg.audio.n_window, 25);
    EXPECT_EQ(cfg.audio.n_window_infer, 8);
    EXPECT_EQ(cfg.audio.activation_function, "gelu_new");
    EXPECT_EQ(cfg.image_token_id, 151655);
    EXPECT_EQ(cfg.audio_token_id, 151646);
}

TEST(Qwen3OmniSpecTest, VisionAndAudioProcessingHelpers) {
    auto resized = Qwen3OmniVisionProcess::smart_resize(1080, 1920);
    EXPECT_EQ(resized.first % 28, 0);
    EXPECT_EQ(resized.second % 28, 0);

    nlohmann::json conv = nlohmann::json::array({
        {
            {"role", "user"},
            {"content", nlohmann::json::array({
                {{"type", "image"}, {"image", "a.png"}},
                {{"type", "audio"}, {"audio", "file:///tmp/a.wav"}},
                {{"type", "video"}, {"video", "https://example.com/v.mp4"}}
            })}
        }
    });

    auto vision_infos = Qwen3OmniVisionProcess::extract_vision_info(conv);
    EXPECT_EQ(vision_infos.size(), 2u);

    auto audios = Qwen3OmniAudioProcess::extract_audio_entries(conv, true);
    ASSERT_EQ(audios.size(), 2u);
    EXPECT_TRUE(audios[1].from_video);
}

TEST(Qwen3OmniSpecTest, PythonBridgeProcessingSmoke) {
    nlohmann::json empty_conversations = nlohmann::json::array();

    auto audio_result = Qwen3OmniAudioProcess::process_audio_info_via_python(empty_conversations, false);
    EXPECT_TRUE(audio_result.is_null());

    auto vision_result = Qwen3OmniVisionProcess::process_vision_info_via_python(empty_conversations, true);
    ASSERT_TRUE(vision_result.is_array());
    ASSERT_EQ(vision_result.size(), 3u);
    EXPECT_TRUE(vision_result[0].is_null());
    EXPECT_TRUE(vision_result[1].is_null());
    EXPECT_TRUE(vision_result[2].is_object());
    EXPECT_TRUE(vision_result[2].contains("fps"));

    auto audio_feature_result = Qwen3OmniAudioProcess::process_audio_features_via_python(
        empty_conversations,
        "",
        false);
    ASSERT_TRUE(audio_feature_result.is_object());
    EXPECT_TRUE(audio_feature_result.contains("input_ids"));
    EXPECT_TRUE(audio_feature_result.contains("attention_mask"));
    EXPECT_TRUE(audio_feature_result.contains("position_ids"));
    EXPECT_TRUE(audio_feature_result.contains("visual_pos_mask"));
    EXPECT_TRUE(audio_feature_result.contains("rope_deltas"));
    EXPECT_TRUE(audio_feature_result.contains("visual_embeds_padded"));
    EXPECT_TRUE(audio_feature_result.contains("deepstack_padded"));
    EXPECT_TRUE(audio_feature_result.contains("audio_features"));
    EXPECT_TRUE(audio_feature_result.contains("audio_pos_mask"));
    EXPECT_TRUE(audio_feature_result.contains("feature_attention_mask"));
    EXPECT_TRUE(audio_feature_result.contains("audio_feature_lengths"));
    EXPECT_TRUE(audio_feature_result.contains("video_second_per_grid"));
    EXPECT_TRUE(audio_feature_result["input_ids"].is_null());
    EXPECT_TRUE(audio_feature_result["attention_mask"].is_null());
    EXPECT_TRUE(audio_feature_result["position_ids"].is_null());
    EXPECT_TRUE(audio_feature_result["visual_pos_mask"].is_null());
    EXPECT_TRUE(audio_feature_result["rope_deltas"].is_null());
    EXPECT_TRUE(audio_feature_result["visual_embeds_padded"].is_null());
    EXPECT_TRUE(audio_feature_result["deepstack_padded"].is_array());
    EXPECT_TRUE(audio_feature_result["deepstack_padded"].empty());
    EXPECT_TRUE(audio_feature_result["audio_features"].is_null());
    EXPECT_TRUE(audio_feature_result["audio_pos_mask"].is_null());
    EXPECT_TRUE(audio_feature_result["feature_attention_mask"].is_null());
    EXPECT_TRUE(audio_feature_result["audio_feature_lengths"].is_null());
    EXPECT_TRUE(audio_feature_result["video_second_per_grid"].is_null());
}

TEST(Qwen3OmniSpecTest, PipelineMultimodalProcessingIncludesAudioAndVision) {
    nlohmann::json conv = nlohmann::json::array({
        {
            {"role", "user"},
            {"content", nlohmann::json::array({
                {{"type", "image"}, {"image", "/home/wanglei/model/Qwen3_omni/demo/cars.jpg"}},
                {{"type", "audio"}, {"audio", "/home/wanglei/model/Qwen3_omni/demo/cough.wav"}},
                {{"type", "text"}, {"text", "describe"}}
            })}
        }
    });

    auto mm = ov::genai::modeling::models::process_qwen3_omni_multimodal_info(
        conv,
        false,
        true);

    ASSERT_TRUE(mm.audios.is_array());
    EXPECT_EQ(mm.audios.size(), 1u);
    ASSERT_TRUE(mm.images.is_array());
    EXPECT_EQ(mm.images.size(), 1u);
    EXPECT_TRUE(mm.videos.is_null());
    ASSERT_TRUE(mm.video_kwargs.is_object());
    EXPECT_TRUE(mm.video_kwargs.contains("fps"));
}

TEST(Qwen3OmniSpecTest, PipelineAudioFeatureInfoSmokeWithEmptyConversations) {
    nlohmann::json empty_conversations = nlohmann::json::array();
    auto info = ov::genai::modeling::models::process_qwen3_omni_audio_feature_info(
        empty_conversations,
        "",
        false);

    EXPECT_TRUE(info.audio_features.is_null());
    EXPECT_TRUE(info.audio_pos_mask.is_null());
    EXPECT_TRUE(info.input_ids.is_null());
    EXPECT_TRUE(info.attention_mask.is_null());
    EXPECT_TRUE(info.position_ids.is_null());
    EXPECT_TRUE(info.visual_pos_mask.is_null());
    EXPECT_TRUE(info.rope_deltas.is_null());
    EXPECT_TRUE(info.visual_embeds_padded.is_null());
    EXPECT_TRUE(info.deepstack_padded.is_array());
    EXPECT_TRUE(info.deepstack_padded.empty());
    EXPECT_TRUE(info.feature_attention_mask.is_null());
    EXPECT_TRUE(info.audio_feature_lengths.is_null());
    EXPECT_TRUE(info.video_second_per_grid.is_null());
}

TEST(Qwen3OmniSpecTest, BuildTextModelWithThinkerPrefixWeights) {
    Qwen3OmniConfig cfg;
    cfg.model_type = "qwen3_omni";
    cfg.text.model_type = "qwen3_omni_text";
    cfg.text.vocab_size = 32;
    cfg.text.hidden_size = 8;
    cfg.text.intermediate_size = 16;
    cfg.text.num_hidden_layers = 1;
    cfg.text.num_attention_heads = 2;
    cfg.text.num_key_value_heads = 1;
    cfg.text.head_dim = 4;
    cfg.text.max_position_embeddings = 64;
    cfg.text.attention_bias = false;
    cfg.text.rope.mrope_interleaved = true;
    cfg.text.rope.mrope_section = {1, 1, 2};

    cfg.vision.model_type = "qwen3_omni_vision_encoder";
    cfg.vision.hidden_size = 8;
    cfg.vision.out_hidden_size = 8;
    cfg.vision.deepstack_visual_indexes.clear();

    cfg.image_token_id = 7;
    cfg.video_token_id = 8;
    cfg.vision_start_token_id = 9;
    cfg.vision_end_token_id = 10;
    cfg.finalize();
    cfg.validate();

    DummyWeightSource source;
    DummyWeightFinalizer finalizer;

    add_zero_weight(source, "thinker.model.embed_tokens.weight", {32, 8});
    add_zero_weight(source, "thinker.model.layers.0.self_attn.q_proj.weight", {8, 8});
    add_zero_weight(source, "thinker.model.layers.0.self_attn.k_proj.weight", {4, 8});
    add_zero_weight(source, "thinker.model.layers.0.self_attn.v_proj.weight", {4, 8});
    add_zero_weight(source, "thinker.model.layers.0.self_attn.o_proj.weight", {8, 8});
    add_zero_weight(source, "thinker.model.layers.0.self_attn.q_norm.weight", {4});
    add_zero_weight(source, "thinker.model.layers.0.self_attn.k_norm.weight", {4});
    add_zero_weight(source, "thinker.model.layers.0.mlp.gate_proj.weight", {16, 8});
    add_zero_weight(source, "thinker.model.layers.0.mlp.up_proj.weight", {16, 8});
    add_zero_weight(source, "thinker.model.layers.0.mlp.down_proj.weight", {8, 16});
    add_zero_weight(source, "thinker.model.layers.0.input_layernorm.weight", {8});
    add_zero_weight(source, "thinker.model.layers.0.post_attention_layernorm.weight", {8});
    add_zero_weight(source, "thinker.model.norm.weight", {8});
    add_zero_weight(source, "thinker.lm_head.weight", {32, 8});

    auto model = ov::genai::modeling::models::create_qwen3_omni_text_model(
        cfg,
        source,
        finalizer,
        false,
        false);

    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->inputs().size(), 4u);
    EXPECT_EQ(model->outputs().size(), 1u);
}

TEST(Qwen3OmniSpecTest, BuildTextModelWithVisionTextMultimodalInputsEmbeds) {
    Qwen3OmniConfig cfg;
    cfg.model_type = "qwen3_omni";
    cfg.text.model_type = "qwen3_omni_text";
    cfg.text.vocab_size = 32;
    cfg.text.hidden_size = 8;
    cfg.text.intermediate_size = 16;
    cfg.text.num_hidden_layers = 1;
    cfg.text.num_attention_heads = 2;
    cfg.text.num_key_value_heads = 1;
    cfg.text.head_dim = 4;
    cfg.text.max_position_embeddings = 64;
    cfg.text.attention_bias = false;
    cfg.text.rope.mrope_interleaved = true;
    cfg.text.rope.mrope_section = {1, 1, 2};

    cfg.vision.model_type = "qwen3_omni_vision_encoder";
    cfg.vision.hidden_size = 8;
    cfg.vision.out_hidden_size = 8;
    cfg.vision.deepstack_visual_indexes = {0, 2};

    cfg.image_token_id = 7;
    cfg.video_token_id = 8;
    cfg.vision_start_token_id = 9;
    cfg.vision_end_token_id = 10;
    cfg.finalize();
    cfg.validate();

    DummyWeightSource source;
    DummyWeightFinalizer finalizer;

    add_zero_weight(source, "thinker.model.embed_tokens.weight", {32, 8});
    add_zero_weight(source, "thinker.model.layers.0.self_attn.q_proj.weight", {8, 8});
    add_zero_weight(source, "thinker.model.layers.0.self_attn.k_proj.weight", {4, 8});
    add_zero_weight(source, "thinker.model.layers.0.self_attn.v_proj.weight", {4, 8});
    add_zero_weight(source, "thinker.model.layers.0.self_attn.o_proj.weight", {8, 8});
    add_zero_weight(source, "thinker.model.layers.0.self_attn.q_norm.weight", {4});
    add_zero_weight(source, "thinker.model.layers.0.self_attn.k_norm.weight", {4});
    add_zero_weight(source, "thinker.model.layers.0.mlp.gate_proj.weight", {16, 8});
    add_zero_weight(source, "thinker.model.layers.0.mlp.up_proj.weight", {16, 8});
    add_zero_weight(source, "thinker.model.layers.0.mlp.down_proj.weight", {8, 16});
    add_zero_weight(source, "thinker.model.layers.0.input_layernorm.weight", {8});
    add_zero_weight(source, "thinker.model.layers.0.post_attention_layernorm.weight", {8});
    add_zero_weight(source, "thinker.model.norm.weight", {8});
    add_zero_weight(source, "thinker.lm_head.weight", {32, 8});

    auto model = ov::genai::modeling::models::create_qwen3_omni_text_model(
        cfg,
        source,
        finalizer,
        true,
        true);

    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->inputs().size(), 10u);
    EXPECT_EQ(model->outputs().size(), 1u);

    std::unordered_set<std::string> input_names;
    for (const auto& input : model->inputs()) {
        for (const auto& name : input.get_names()) {
            input_names.insert(name);
        }
    }

    EXPECT_TRUE(input_names.count("inputs_embeds") != 0);
    EXPECT_TRUE(input_names.count("visual_embeds") != 0);
    EXPECT_TRUE(input_names.count("visual_pos_mask") != 0);
    EXPECT_TRUE(input_names.count("audio_features") != 0);
    EXPECT_TRUE(input_names.count("audio_pos_mask") != 0);
    EXPECT_TRUE(input_names.count("deepstack_embeds.0") != 0);
    EXPECT_TRUE(input_names.count("deepstack_embeds.1") != 0);
    EXPECT_TRUE(input_names.count("input_ids") == 0);
}

TEST(Qwen3OmniSpecTest, ParseAndConvertTalkerAndCode2WavConfigs) {
    const char* text = R"({
        "model_type": "qwen3_omni",
        "thinker_config": {
            "text_config": {
                "model_type": "qwen3_omni_text",
                "vocab_size": 151936,
                "hidden_size": 2560,
                "intermediate_size": 9728,
                "num_hidden_layers": 2,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 80,
                "max_position_embeddings": 32768
            }
        },
        "talker_config": {
            "hidden_size": 1792,
            "num_attention_heads": 14,
            "num_key_value_heads": 7,
            "head_dim": 128,
            "intermediate_size": 5632,
            "num_hidden_layers": 20,
            "vocab_size": 4096,
            "text_vocab_size": 151936,
            "text_hidden_size": 1024,
            "mrope_interleaved": true,
            "mrope_section": [24, 20, 20],
            "codec_bos_token_id": 3001,
            "codec_eos_token_id": 3002,
            "codec_pad_token_id": 3000,
            "code_predictor_config": {
                "hidden_size": 896,
                "num_attention_heads": 14,
                "num_key_value_heads": 7,
                "head_dim": 64,
                "intermediate_size": 3584,
                "num_hidden_layers": 4,
                "vocab_size": 2048,
                "num_code_groups": 16,
                "rope_theta": 1000000
            }
        },
        "code2wav_config": {
            "num_quantizers": 16,
            "codebook_size": 2048,
            "codebook_dim": 32,
            "latent_dim": 640,
            "transformer_hidden": 1280,
            "transformer_heads": 10,
            "transformer_head_dim": 128,
            "transformer_layers": 6,
            "transformer_intermediate": 2560,
            "decoder_dim": 1600,
            "pre_upsample_ratios": [2, 2],
            "decoder_upsample_rates": [8, 5, 4, 3]
        }
    })";

    auto json = nlohmann::json::parse(text);
    Qwen3OmniConfig cfg = Qwen3OmniConfig::from_json(json);

    ASSERT_TRUE(cfg.talker_config_raw.is_object());
    ASSERT_TRUE(cfg.code2wav_config_raw.is_object());

    auto talker_cfg = ov::genai::modeling::models::to_qwen3_omni_talker_config(cfg);
    EXPECT_EQ(talker_cfg.hidden_size, 1792);
    EXPECT_EQ(talker_cfg.num_hidden_layers, 20);
    EXPECT_EQ(talker_cfg.codec_bos_token_id, 3001);

    auto cp_cfg = ov::genai::modeling::models::to_qwen3_omni_code_predictor_config(cfg);
    EXPECT_EQ(cp_cfg.hidden_size, 896);
    EXPECT_EQ(cp_cfg.num_hidden_layers, 4);
    EXPECT_EQ(cp_cfg.talker_hidden_size, 1792);

    auto decoder_cfg = ov::genai::modeling::models::to_qwen3_omni_speech_decoder_config(cfg);
    EXPECT_EQ(decoder_cfg.latent_dim, 640);
    EXPECT_EQ(decoder_cfg.transformer_hidden, 1280);
    EXPECT_EQ(decoder_cfg.decoder_dim, 1600);
    ASSERT_EQ(decoder_cfg.pre_upsample_ratios.size(), 2u);
    EXPECT_EQ(decoder_cfg.pre_upsample_ratios[0], 2);
}

TEST(Qwen3OmniSpecTest, BuildAudioEncoderModelSmoke) {
    Qwen3OmniConfig cfg;
    cfg.audio.num_mel_bins = 16;
    cfg.audio.downsample_hidden_size = 4;
    cfg.audio.d_model = 8;
    cfg.audio.output_dim = 8;
    DummyWeightSource source;
    DummyWeightFinalizer finalizer;

    const int64_t feat_after = (((cfg.audio.num_mel_bins + 1) / 2 + 1) / 2 + 1) / 2;
    const int64_t down_hidden = 4;

    add_zero_weight(source, "thinker.audio_tower.conv2d1.weight", {down_hidden, 1, 3, 3});
    add_zero_weight(source, "thinker.audio_tower.conv2d1.bias", {down_hidden});
    add_zero_weight(source, "thinker.audio_tower.conv2d2.weight", {down_hidden, down_hidden, 3, 3});
    add_zero_weight(source, "thinker.audio_tower.conv2d2.bias", {down_hidden});
    add_zero_weight(source, "thinker.audio_tower.conv2d3.weight", {down_hidden, down_hidden, 3, 3});
    add_zero_weight(source, "thinker.audio_tower.conv2d3.bias", {down_hidden});
    add_zero_weight(source, "thinker.audio_tower.conv_out.weight", {cfg.audio.d_model, down_hidden * feat_after});
    add_zero_weight(source, "thinker.audio_tower.ln_post.weight", {cfg.audio.d_model});
    add_zero_weight(source, "thinker.audio_tower.ln_post.bias", {cfg.audio.d_model});
    add_zero_weight(source, "thinker.audio_tower.proj1.weight", {cfg.audio.d_model, cfg.audio.d_model});
    add_zero_weight(source, "thinker.audio_tower.proj1.bias", {cfg.audio.d_model});
    add_zero_weight(source, "thinker.audio_tower.proj2.weight", {cfg.audio.output_dim, cfg.audio.d_model});
    add_zero_weight(source, "thinker.audio_tower.proj2.bias", {cfg.audio.output_dim});

    auto model = ov::genai::modeling::models::create_qwen3_omni_audio_encoder_model(
        cfg,
        source,
        finalizer);

    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->inputs().size(), 3u);
    EXPECT_EQ(model->outputs().size(), 1u);

    std::unordered_set<std::string> input_names;
    for (const auto& input : model->inputs()) {
        for (const auto& name : input.get_names()) {
            input_names.insert(name);
        }
    }
    EXPECT_TRUE(input_names.count("input_features") != 0);
    EXPECT_TRUE(input_names.count("feature_attention_mask") != 0);
    EXPECT_TRUE(input_names.count("audio_feature_lengths") != 0);

    std::unordered_set<std::string> output_names;
    for (const auto& output : model->outputs()) {
        for (const auto& name : output.get_names()) {
            output_names.insert(name);
        }
    }
    EXPECT_TRUE(output_names.count("audio_features") != 0);
}

}  // namespace tests
}  // namespace modeling
}  // namespace genai
}  // namespace ov
