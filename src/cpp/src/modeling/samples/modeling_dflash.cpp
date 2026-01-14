// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>

#include "openvino/genai/tokenizer.hpp"
#include "loaders/model_config.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

#include "modeling/models/dflash_draft.hpp"
#include "modeling/models/qwen3_dense.hpp"

namespace {

std::vector<int64_t> tensor_to_ids(const ov::Tensor& ids_tensor) {
    const auto shape = ids_tensor.get_shape();
    if (shape.size() != 2 || shape[0] != 1) {
        throw std::runtime_error("input_ids tensor must have shape [1, S]");
    }
    const size_t seq_len = shape[1];
    const auto* data = ids_tensor.data<const int64_t>();
    return std::vector<int64_t>(data, data + seq_len);
}

ov::Tensor make_ids_tensor(const std::vector<int64_t>& ids) {
    ov::Tensor tensor(ov::element::i64, {1, ids.size()});
    std::memcpy(tensor.data(), ids.data(), ids.size() * sizeof(int64_t));
    return tensor;
}

ov::Tensor make_attention_mask(size_t seq_len) {
    ov::Tensor mask(ov::element::i64, {1, seq_len});
    auto* data = mask.data<int64_t>();
    for (size_t i = 0; i < seq_len; ++i) {
        data[i] = 1;
    }
    return mask;
}

ov::Tensor make_position_ids(size_t seq_len) {
    ov::Tensor ids(ov::element::i64, {1, seq_len});
    auto* data = ids.data<int64_t>();
    for (size_t i = 0; i < seq_len; ++i) {
        data[i] = static_cast<int64_t>(i);
    }
    return ids;
}

template <typename T>
int64_t argmax_row(const T* data, size_t vocab) {
    T max_val = data[0];
    size_t max_idx = 0;
    for (size_t i = 1; i < vocab; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return static_cast<int64_t>(max_idx);
}

std::vector<int64_t> argmax_logits_slice(const ov::Tensor& logits, size_t start, size_t count) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3 || shape[0] != 1) {
        throw std::runtime_error("logits tensor must have shape [1, S, V]");
    }
    const size_t seq_len = shape[1];
    const size_t vocab = shape[2];
    if (start + count > seq_len) {
        throw std::runtime_error("logits slice out of range");
    }

    std::vector<int64_t> tokens;
    tokens.reserve(count);

    if (logits.get_element_type() == ov::element::f16) {
        const auto* data = logits.data<const ov::float16>();
        for (size_t i = 0; i < count; ++i) {
            tokens.push_back(argmax_row(data + (start + i) * vocab, vocab));
        }
        return tokens;
    }
    if (logits.get_element_type() == ov::element::bf16) {
        const auto* data = logits.data<const ov::bfloat16>();
        for (size_t i = 0; i < count; ++i) {
            tokens.push_back(argmax_row(data + (start + i) * vocab, vocab));
        }
        return tokens;
    }
    if (logits.get_element_type() == ov::element::f32) {
        const auto* data = logits.data<const float>();
        for (size_t i = 0; i < count; ++i) {
            tokens.push_back(argmax_row(data + (start + i) * vocab, vocab));
        }
        return tokens;
    }
    throw std::runtime_error("Unsupported logits dtype");
}

int64_t argmax_last_token(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3 || shape[0] != 1) {
        throw std::runtime_error("logits tensor must have shape [1, S, V]");
    }
    const size_t seq_len = shape[1];
    return argmax_logits_slice(logits, seq_len - 1, 1).front();
}

int64_t resolve_mask_token_id(const ov::genai::Tokenizer& tokenizer) {
    const auto vocab = tokenizer.get_vocab();
    const auto it = vocab.find("<|MASK|>");
    if (it != vocab.end()) {
        return it->second;
    }
    // Fallback: use pad token if defined, otherwise eos as a safe placeholder.
    const int64_t pad_id = tokenizer.get_pad_token_id();
    if (pad_id != -1) {
        return pad_id;
    }
    const int64_t eos_id = tokenizer.get_eos_token_id();
    if (eos_id != -1) {
        return eos_id;
    }
    throw std::runtime_error("Tokenizer has no <|MASK|>, pad, or eos token to use as mask placeholder");
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <TARGET_MODEL_DIR> <DRAFT_MODEL_DIR> [PROMPT] [DEVICE] [MAX_NEW_TOKENS] [BLOCK_SIZE]\n";
        return 1;
    }

    const std::filesystem::path target_dir = argv[1];
    const std::filesystem::path draft_dir = argv[2];
    const std::string prompt = (argc > 3) ? argv[3] : "Tell me a short story about a robot.";
    const std::string device = (argc > 4) ? argv[4] : "CPU";
    const int max_new_tokens = (argc > 5) ? std::stoi(argv[5]) : 64;
    const int block_size_arg = (argc > 6) ? std::stoi(argv[6]) : 0;

    auto target_cfg = ov::genai::loaders::ModelConfig::from_hf_json(target_dir / "config.json");
    auto draft_cfg = ov::genai::loaders::ModelConfig::from_hf_json(draft_dir / "config.json");

    ov::genai::modeling::models::Qwen3DenseConfig qwen_cfg;
    qwen_cfg.architecture = target_cfg.architecture;
    qwen_cfg.hidden_size = target_cfg.hidden_size;
    qwen_cfg.intermediate_size = target_cfg.intermediate_size;
    qwen_cfg.num_hidden_layers = target_cfg.num_hidden_layers;
    qwen_cfg.num_attention_heads = target_cfg.num_attention_heads;
    qwen_cfg.num_key_value_heads = target_cfg.num_key_value_heads > 0 ? target_cfg.num_key_value_heads
                                                                      : target_cfg.num_attention_heads;
    qwen_cfg.head_dim = target_cfg.head_dim > 0 ? target_cfg.head_dim
                                                : (target_cfg.hidden_size / target_cfg.num_attention_heads);
    qwen_cfg.rope_theta = target_cfg.rope_theta;
    qwen_cfg.rms_norm_eps = target_cfg.rms_norm_eps;
    qwen_cfg.hidden_act = target_cfg.hidden_act;
    qwen_cfg.attention_bias = target_cfg.attention_bias;
    qwen_cfg.tie_word_embeddings = target_cfg.tie_word_embeddings;

    ov::genai::modeling::models::DFlashDraftConfig dflash_cfg;
    dflash_cfg.hidden_size = draft_cfg.hidden_size;
    dflash_cfg.intermediate_size = draft_cfg.intermediate_size;
    dflash_cfg.num_hidden_layers = draft_cfg.num_hidden_layers;
    dflash_cfg.num_target_layers = (draft_cfg.num_target_layers > 0)
                                       ? draft_cfg.num_target_layers
                                       : target_cfg.num_hidden_layers;
    dflash_cfg.num_attention_heads = draft_cfg.num_attention_heads;
    dflash_cfg.num_key_value_heads = draft_cfg.num_key_value_heads > 0
                                         ? draft_cfg.num_key_value_heads
                                         : draft_cfg.num_attention_heads;
    dflash_cfg.head_dim = draft_cfg.head_dim > 0
                              ? draft_cfg.head_dim
                              : (draft_cfg.hidden_size / draft_cfg.num_attention_heads);
    dflash_cfg.block_size = block_size_arg > 0 ? block_size_arg : draft_cfg.block_size;
    dflash_cfg.rms_norm_eps = draft_cfg.rms_norm_eps;
    dflash_cfg.rope_theta = draft_cfg.rope_theta;
    dflash_cfg.hidden_act = draft_cfg.hidden_act;
    dflash_cfg.attention_bias = draft_cfg.attention_bias;

    if (dflash_cfg.block_size <= 0) {
        dflash_cfg.block_size = 16;
    }
    if (dflash_cfg.block_size < 2) {
        throw std::runtime_error("block_size must be >= 2 for DFlash decoding");
    }

    const auto target_layer_ids = ov::genai::modeling::models::build_target_layer_ids(
        dflash_cfg.num_target_layers, dflash_cfg.num_hidden_layers);

    auto target_data = ov::genai::safetensors::load_safetensors(target_dir);
    ov::genai::safetensors::SafetensorsWeightSource target_source(std::move(target_data));
    ov::genai::safetensors::SafetensorsWeightFinalizer target_finalizer;

    auto draft_data = ov::genai::safetensors::load_safetensors(draft_dir);
    ov::genai::safetensors::SafetensorsWeightSource draft_source(std::move(draft_data));
    ov::genai::safetensors::SafetensorsWeightFinalizer draft_finalizer;

    auto target_model = ov::genai::modeling::models::create_qwen3_dflash_target_model_no_cache(
        qwen_cfg, target_layer_ids, target_source, target_finalizer);
    const auto target_hidden_type = target_model->output("target_hidden").get_element_type();
    auto draft_model = ov::genai::modeling::models::create_dflash_draft_model(
        dflash_cfg, draft_source, draft_finalizer, target_hidden_type);
    auto draft_hidden_type = draft_model->output("draft_hidden").get_element_type();
    auto embed_model = ov::genai::modeling::models::create_qwen3_embedding_model(
        qwen_cfg, target_source, target_finalizer);
    auto lm_head_model = ov::genai::modeling::models::create_qwen3_lm_head_model(
        qwen_cfg, target_source, target_finalizer, draft_hidden_type);

    ov::Core core;
    auto compiled_target = core.compile_model(target_model, device);
    auto compiled_embed = core.compile_model(embed_model, device);
    auto compiled_lm_head = core.compile_model(lm_head_model, device);
    auto compiled_draft = core.compile_model(draft_model, device);

    auto target_request = compiled_target.create_infer_request();
    auto embed_request = compiled_embed.create_infer_request();
    auto lm_head_request = compiled_lm_head.create_infer_request();
    auto draft_request = compiled_draft.create_infer_request();

    ov::genai::Tokenizer tokenizer(target_dir);
    const int64_t mask_token_id = resolve_mask_token_id(tokenizer);
    const int64_t eos_token_id = tokenizer.get_eos_token_id();

    auto encoded = tokenizer.encode(prompt, {ov::genai::add_special_tokens(false)});
    std::vector<int64_t> output_ids = tensor_to_ids(encoded.input_ids);
    const size_t prompt_len = output_ids.size();
    const size_t max_length = prompt_len + static_cast<size_t>(max_new_tokens);

    // Prefill: run target on prompt to get first token.
    target_request.set_tensor("input_ids", make_ids_tensor(output_ids));
    target_request.set_tensor("attention_mask", make_attention_mask(output_ids.size()));
    target_request.set_tensor("position_ids", make_position_ids(output_ids.size()));
    target_request.infer();
    auto logits = target_request.get_tensor("logits");
    int64_t next_token = argmax_last_token(logits);
    output_ids.push_back(next_token);

    while (output_ids.size() < max_length) {
        if (next_token == eos_token_id) {
            break;
        }

        // Build context features by running target on full context.
        target_request.set_tensor("input_ids", make_ids_tensor(output_ids));
        target_request.set_tensor("attention_mask", make_attention_mask(output_ids.size()));
        target_request.set_tensor("position_ids", make_position_ids(output_ids.size()));
        target_request.infer();
        auto target_hidden = target_request.get_tensor("target_hidden");

        // Build draft block inputs: [last_token, MASK, MASK, ...]
        std::vector<int64_t> block_ids(static_cast<size_t>(dflash_cfg.block_size), mask_token_id);
        block_ids[0] = output_ids.back();

        embed_request.set_tensor("input_ids", make_ids_tensor(block_ids));
        embed_request.infer();
        auto noise_embedding = embed_request.get_tensor("embeddings");

        draft_request.set_tensor("target_hidden", target_hidden);
        draft_request.set_tensor("noise_embedding", noise_embedding);
        draft_request.set_tensor("position_ids", make_position_ids(output_ids.size() + block_ids.size()));
        draft_request.infer();
        auto draft_hidden = draft_request.get_tensor("draft_hidden");

        // Align dtype for lm_head input if needed.
        const auto& lm_head_port = compiled_lm_head.input("hidden_states");
        if (draft_hidden.get_element_type() != lm_head_port.get_element_type()) {
            ov::Tensor converted(lm_head_port.get_element_type(), draft_hidden.get_shape());
            draft_hidden.copy_to(converted);
            draft_hidden = converted;
        }
        lm_head_request.set_tensor("hidden_states", draft_hidden);
        lm_head_request.infer();
        auto draft_logits = lm_head_request.get_tensor("logits");

        const size_t draft_len = block_ids.size() - 1;
        auto draft_tokens = argmax_logits_slice(draft_logits, 1, draft_len);

        // Verify with target model on full context + draft tokens.
        std::vector<int64_t> verify_ids = output_ids;
        verify_ids.insert(verify_ids.end(), draft_tokens.begin(), draft_tokens.end());
        target_request.set_tensor("input_ids", make_ids_tensor(verify_ids));
        target_request.set_tensor("attention_mask", make_attention_mask(verify_ids.size()));
        target_request.set_tensor("position_ids", make_position_ids(verify_ids.size()));
        target_request.infer();
        logits = target_request.get_tensor("logits");
        auto target_hidden_verify = target_request.get_tensor("target_hidden");

        const size_t start = output_ids.size() - 1;
        auto posterior = argmax_logits_slice(logits, start, block_ids.size());

        size_t accepted = 0;
        for (; accepted < draft_len; ++accepted) {
            if (draft_tokens[accepted] != posterior[accepted]) {
                break;
            }
        }

        for (size_t i = 0; i < accepted && output_ids.size() < max_length; ++i) {
            output_ids.push_back(draft_tokens[i]);
        }
        if (output_ids.size() >= max_length) {
            break;
        }
        next_token = posterior[accepted];
        output_ids.push_back(next_token);
    }

    std::vector<int64_t> generated(output_ids.begin() + static_cast<std::ptrdiff_t>(prompt_len), output_ids.end());
    auto text = tokenizer.decode(generated, {ov::genai::skip_special_tokens(true)});
    std::cout << text << std::endl;
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "DFlash sample failed: " << ex.what() << std::endl;
    return 1;
}
