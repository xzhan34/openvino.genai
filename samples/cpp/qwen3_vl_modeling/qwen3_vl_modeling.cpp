// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>

#include "load_image.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

#include "modeling/models/qwen3_vl_input_planner.hpp"
#include "modeling/models/qwen3_vl_spec.hpp"
#include "modeling/models/qwen3_vl_text.hpp"
#include "modeling/models/qwen3_vl_vision.hpp"
#include "modeling/models/qwen3_vl_vision_preprocess.hpp"

namespace {

std::string build_prompt(const std::string& user_prompt, int64_t image_tokens) {
    std::string prompt = "<|im_start|>user\n<|vision_start|>";
    prompt.reserve(prompt.size() + static_cast<size_t>(image_tokens) * 12 + user_prompt.size() + 64);
    for (int64_t i = 0; i < image_tokens; ++i) {
        prompt += "<|image_pad|>";
    }
    prompt += "<|vision_end|>\n";
    prompt += user_prompt;
    prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

int64_t argmax_last_token(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3) {
        throw std::runtime_error("logits must have shape [B, S, V]");
    }
    if (shape[0] != 1) {
        throw std::runtime_error("Only batch=1 is supported in this sample");
    }
    const size_t seq_len = shape[1];
    const size_t vocab = shape[2];
    const size_t offset = (seq_len - 1) * vocab;

    if (logits.get_element_type() == ov::element::f16) {
        const auto* data = logits.data<const ov::float16>() + offset;
        ov::float16 max_val = data[0];
        size_t max_idx = 0;
        for (size_t i = 1; i < vocab; ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }
        return static_cast<int64_t>(max_idx);
    }
    if (logits.get_element_type() == ov::element::bf16) {
        const auto* data = logits.data<const ov::bfloat16>() + offset;
        ov::bfloat16 max_val = data[0];
        size_t max_idx = 0;
        for (size_t i = 1; i < vocab; ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }
        return static_cast<int64_t>(max_idx);
    }
    if (logits.get_element_type() != ov::element::f32) {
        throw std::runtime_error("Unsupported logits dtype");
    }
    const auto* data = logits.data<const float>() + offset;
    float max_val = data[0];
    size_t max_idx = 0;
    for (size_t i = 1; i < vocab; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return static_cast<int64_t>(max_idx);
}

ov::Tensor make_beam_idx(size_t batch) {
    ov::Tensor beam_idx(ov::element::i32, {batch});
    auto* data = beam_idx.data<int32_t>();
    for (size_t i = 0; i < batch; ++i) {
        data[i] = static_cast<int32_t>(i);
    }
    return beam_idx;
}

ov::Tensor make_zero_tensor(const ov::element::Type& type, const ov::Shape& shape) {
    ov::Tensor tensor(type, shape);
    std::memset(tensor.data(), 0, tensor.get_byte_size());
    return tensor;
}

std::string resolve_pos_embed_name(ov::genai::modeling::weights::WeightSource& source) {
    const std::vector<std::string> candidates = {
        "model.visual.pos_embed.weight",
        "visual.pos_embed.weight",
        "pos_embed.weight"
    };
    for (const auto& name : candidates) {
        if (source.has(name)) {
            return name;
        }
    }
    for (const auto& name : source.keys()) {
        if (name.find("pos_embed.weight") != std::string::npos) {
            return name;
        }
    }
    throw std::runtime_error("Failed to locate visual.pos_embed.weight in safetensors");
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <MODEL_DIR> <IMAGE_PATH> [PROMPT] [DEVICE] [MAX_NEW_TOKENS]\n";
        return 1;
    }

    const std::filesystem::path model_dir = argv[1];
    const std::filesystem::path image_path = argv[2];
    const std::string user_prompt = (argc > 3) ? argv[3] : "Describe the image.";
    const std::string device = (argc > 4) ? argv[4] : "GPU";
    const int max_new_tokens = (argc > 5) ? std::stoi(argv[5]) : 64;

    auto cfg = ov::genai::modeling::models::Qwen3VLConfig::from_json_file(model_dir);
    ov::genai::modeling::models::Qwen3VLVisionPreprocessConfig pre_cfg;
    const auto pre_cfg_path = model_dir / "preprocessor_config.json";
    if (std::filesystem::exists(pre_cfg_path)) {
        pre_cfg = ov::genai::modeling::models::Qwen3VLVisionPreprocessConfig::from_json_file(pre_cfg_path);
    }

    auto data = ov::genai::safetensors::load_safetensors(model_dir);
    ov::genai::safetensors::SafetensorsWeightSource source(std::move(data));

    ov::genai::safetensors::SafetensorsWeightFinalizer vision_finalizer;
    auto vision_model = ov::genai::modeling::models::create_qwen3_vl_vision_model(cfg, source, vision_finalizer);

    ov::genai::safetensors::SafetensorsWeightFinalizer text_finalizer;
    auto text_model = ov::genai::modeling::models::create_qwen3_vl_text_model(cfg, source, text_finalizer, false, true);

    ov::Core core;
    auto compiled_vision = core.compile_model(vision_model, device);
    auto compiled_text = core.compile_model(text_model, device);

    auto image = utils::load_image(image_path);
    const std::string pos_embed_name = resolve_pos_embed_name(source);
    const ov::Tensor pos_embed_weight = source.get_tensor(pos_embed_name);

    ov::genai::modeling::models::Qwen3VLVisionPreprocessor preprocessor(cfg.vision, pre_cfg);
    auto vision_inputs = preprocessor.preprocess(image, pos_embed_weight);

    auto vision_request = compiled_vision.create_infer_request();
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kPixelValues, vision_inputs.pixel_values);
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kGridThw, vision_inputs.grid_thw);
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kPosEmbeds, vision_inputs.pos_embeds);
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kRotaryCos, vision_inputs.rotary_cos);
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kRotarySin, vision_inputs.rotary_sin);
    vision_request.infer();

    ov::Tensor visual_embeds =
        vision_request.get_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kVisualEmbeds);
    std::vector<ov::Tensor> deepstack_embeds;
    deepstack_embeds.reserve(cfg.vision.deepstack_visual_indexes.size());
    for (size_t i = 0; i < cfg.vision.deepstack_visual_indexes.size(); ++i) {
        std::string name =
            std::string(ov::genai::modeling::models::Qwen3VLVisionIO::kDeepstackEmbedsPrefix) + "." +
            std::to_string(i);
        deepstack_embeds.push_back(vision_request.get_tensor(name));
    }

    ov::genai::Tokenizer tokenizer(model_dir);
    const int64_t image_tokens =
        ov::genai::modeling::models::Qwen3VLVisionPreprocessor::count_visual_tokens(
            vision_inputs.grid_thw, cfg.vision.spatial_merge_size);
    const std::string prompt = build_prompt(user_prompt, image_tokens);
    auto tokenized = tokenizer.encode(prompt, ov::genai::add_special_tokens(false));

    auto input_ids = tokenized.input_ids;
    auto attention_mask = tokenized.attention_mask;
    const size_t batch = input_ids.get_shape().at(0);
    const int64_t prompt_len = static_cast<int64_t>(input_ids.get_shape().at(1));

    ov::genai::modeling::models::Qwen3VLInputPlanner planner(cfg);
    auto plan = planner.build_plan(input_ids, &attention_mask, &vision_inputs.grid_thw);

    auto visual_padded =
        ov::genai::modeling::models::Qwen3VLInputPlanner::scatter_visual_embeds(visual_embeds, plan.visual_pos_mask);
    auto deepstack_padded =
        ov::genai::modeling::models::Qwen3VLInputPlanner::scatter_deepstack_embeds(
            deepstack_embeds, plan.visual_pos_mask);

    auto beam_idx = make_beam_idx(batch);

    auto text_request = compiled_text.create_infer_request();
    text_request.reset_state();
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kInputIds, input_ids);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kAttentionMask, attention_mask);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kPositionIds, plan.position_ids);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kBeamIdx, beam_idx);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kVisualEmbeds, visual_padded);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kVisualPosMask, plan.visual_pos_mask);
    for (size_t i = 0; i < deepstack_padded.size(); ++i) {
        std::string name =
            std::string(ov::genai::modeling::models::Qwen3VLTextIO::kDeepstackEmbedsPrefix) + "." +
            std::to_string(i);
        text_request.set_tensor(name, deepstack_padded[i]);
    }
    text_request.infer();

    ov::Tensor logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kLogits);
    int64_t next_id = argmax_last_token(logits);
    std::vector<int64_t> generated;
    generated.reserve(static_cast<size_t>(max_new_tokens));
    generated.push_back(next_id);

    const int64_t eos_token_id = tokenizer.get_eos_token_id();
    ov::Tensor step_ids(ov::element::i64, {batch, 1});
    ov::Tensor step_mask(ov::element::i64, {batch, 1});
    auto* step_mask_data = step_mask.data<int64_t>();
    for (size_t b = 0; b < batch; ++b) {
        step_mask_data[b] = 1;
    }

    ov::Tensor decode_visual =
        make_zero_tensor(ov::element::f32, {batch, 1, static_cast<size_t>(cfg.text.hidden_size)});
    ov::Tensor decode_visual_mask = make_zero_tensor(ov::element::boolean, {batch, 1});
    std::vector<ov::Tensor> decode_deepstack;
    decode_deepstack.reserve(deepstack_padded.size());
    for (size_t i = 0; i < deepstack_padded.size(); ++i) {
        decode_deepstack.push_back(
            make_zero_tensor(ov::element::f32, {batch, 1, static_cast<size_t>(cfg.text.hidden_size)}));
    }

    int64_t past_len = prompt_len;
    for (int step = 1; step < max_new_tokens; ++step) {
        if (eos_token_id >= 0 && next_id == eos_token_id) {
            break;
        }
        auto* step_data = step_ids.data<int64_t>();
        for (size_t b = 0; b < batch; ++b) {
            step_data[b] = next_id;
        }

        auto position_ids =
            ov::genai::modeling::models::Qwen3VLInputPlanner::build_decode_position_ids(
                plan.rope_deltas, past_len, 1);

        text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kInputIds, step_ids);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kAttentionMask, step_mask);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kPositionIds, position_ids);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kBeamIdx, beam_idx);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kVisualEmbeds, decode_visual);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kVisualPosMask, decode_visual_mask);
        for (size_t i = 0; i < decode_deepstack.size(); ++i) {
            std::string name =
                std::string(ov::genai::modeling::models::Qwen3VLTextIO::kDeepstackEmbedsPrefix) + "." +
                std::to_string(i);
            text_request.set_tensor(name, decode_deepstack[i]);
        }

        text_request.infer();
        logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kLogits);
        next_id = argmax_last_token(logits);
        generated.push_back(next_id);
        past_len += 1;
    }

    std::string output = tokenizer.decode(generated, ov::genai::skip_special_tokens(true));
    std::cout << output << std::endl;
    return 0;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return 1;
}
