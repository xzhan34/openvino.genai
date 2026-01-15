// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include <sstream>
#include <iomanip>
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

using Clock = std::chrono::steady_clock;

struct StageStats {
    double total_ms = 0.0;
    double max_ms = 0.0;
    size_t count = 0;

    void add(double ms) {
        total_ms += ms;
        max_ms = std::max(max_ms, ms);
        ++count;
    }

    double avg() const {
        return count ? total_ms / static_cast<double>(count) : 0.0;
    }
};

struct PerfStats {
    StageStats prefill_wall;
    StageStats target_ctx_wall;
    StageStats embed_wall;
    StageStats draft_wall;
    StageStats lm_head_wall;
    StageStats verify_wall;

    size_t draft_steps = 0;
    size_t accepted_tokens = 0;
    size_t generated_tokens = 0;
    double ttft_ms = 0.0;
    double total_generate_ms = 0.0;
};

double duration_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_stage_stats(const std::string& name, const StageStats& wall) {
    std::cout << "  " << name << ": wall " << wall.avg() << " ms (max " << wall.max_ms << ", "
              << wall.count << " runs)" << std::endl;
}

std::vector<int64_t> tensor_to_ids(const ov::Tensor& ids_tensor) {
    const auto shape = ids_tensor.get_shape();
    if (shape.size() != 2 || shape[0] != 1) {
        throw std::runtime_error("input_ids tensor must have shape [1, S]");
    }
    const size_t seq_len = shape[1];
    const auto* data = ids_tensor.data<const int64_t>();
    return std::vector<int64_t>(data, data + seq_len);
}

std::vector<int64_t> tensor_to_vec_i64(const ov::Tensor& t) {
    if (t.get_element_type() != ov::element::i64) {
        throw std::runtime_error("tensor_to_vec_i64 expects i64 tensor");
    }
    const auto shape = t.get_shape();
    const size_t total = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    const auto* data = t.data<const int64_t>();
    return std::vector<int64_t>(data, data + total);
}

std::string preview_vec(const std::vector<int64_t>& v, size_t n) {
    std::ostringstream oss;
    oss << "[";
    const size_t limit = std::min(n, v.size());
    for (size_t i = 0; i < limit; ++i) {
        if (i) {
            oss << ",";
        }
        oss << v[i];
    }
    if (v.size() > limit) {
        oss << "...";
    }
    oss << "]";
    return oss.str();
}

struct TensorStats {
    double min = 0;
    double max = 0;
    double mean = 0;
    size_t count = 0;
};

template <typename T>
TensorStats compute_stats(const T* data, size_t count) {
    TensorStats s;
    if (count == 0) {
        return s;
    }
    double sum = 0.0;
    s.min = static_cast<double>(data[0]);
    s.max = static_cast<double>(data[0]);
    for (size_t i = 0; i < count; ++i) {
        const double v = static_cast<double>(data[i]);
        sum += v;
        if (v < s.min) s.min = v;
        if (v > s.max) s.max = v;
    }
    s.count = count;
    s.mean = sum / static_cast<double>(count);
    return s;
}

TensorStats tensor_stats(const ov::Tensor& t) {
    const auto shape = t.get_shape();
    const size_t total = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    if (total == 0) {
        return {};
    }
    if (t.get_element_type() == ov::element::f16) {
        return compute_stats(t.data<const ov::float16>(), total);
    }
    if (t.get_element_type() == ov::element::bf16) {
        return compute_stats(t.data<const ov::bfloat16>(), total);
    }
    if (t.get_element_type() == ov::element::f32) {
        return compute_stats(t.data<const float>(), total);
    }
    return {};
}

std::string stats_to_str(const TensorStats& s) {
    if (s.count == 0) {
        return "{count=0}";
    }
    std::ostringstream oss;
    oss << "{count=" << s.count << ", mean=" << s.mean << ", min=" << s.min << ", max=" << s.max << "}";
    return oss.str();
}

std::vector<std::pair<int64_t, float>> topk_logits_row(const ov::Tensor& logits, size_t pos, size_t k) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3 || shape[0] != 1) {
        throw std::runtime_error("logits tensor must have shape [1, S, V]");
    }
    const size_t seq_len = shape[1];
    const size_t vocab = shape[2];
    if (pos >= seq_len) {
        throw std::runtime_error("topk position out of range");
    }
    std::vector<std::pair<int64_t, float>> pairs;
    pairs.reserve(vocab);
    if (logits.get_element_type() == ov::element::f16) {
        const auto* data = logits.data<const ov::float16>();
        const auto* row = data + pos * vocab;
        for (size_t i = 0; i < vocab; ++i) {
            pairs.emplace_back(static_cast<int64_t>(i), static_cast<float>(row[i]));
        }
    } else if (logits.get_element_type() == ov::element::bf16) {
        const auto* data = logits.data<const ov::bfloat16>();
        const auto* row = data + pos * vocab;
        for (size_t i = 0; i < vocab; ++i) {
            pairs.emplace_back(static_cast<int64_t>(i), static_cast<float>(row[i]));
        }
    } else if (logits.get_element_type() == ov::element::f32) {
        const auto* data = logits.data<const float>();
        const auto* row = data + pos * vocab;
        for (size_t i = 0; i < vocab; ++i) {
            pairs.emplace_back(static_cast<int64_t>(i), static_cast<float>(row[i]));
        }
    } else {
        return {};
    }
    std::partial_sort(pairs.begin(), pairs.begin() + static_cast<std::ptrdiff_t>(k), pairs.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
    pairs.resize(std::min(k, pairs.size()));
    return pairs;
}

std::string topk_to_str(const std::vector<std::pair<int64_t, float>>& kv) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < kv.size(); ++i) {
        if (i) oss << ",";
        oss << kv[i].first << ":" << kv[i].second;
    }
    oss << "]";
    return oss.str();
}

size_t count_nonfinite(const ov::Tensor& t) {
    const auto shape = t.get_shape();
    const size_t total = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    if (total == 0) {
        return 0;
    }
    size_t bad = 0;
    auto check = [&](auto ptr) {
        using U = std::decay_t<decltype(*ptr)>;
        for (size_t i = 0; i < total; ++i) {
            const double v = static_cast<double>(ptr[i]);
            if (!std::isfinite(v)) {
                ++bad;
            }
        }
    };
    if (t.get_element_type() == ov::element::f16) {
        check(t.data<const ov::float16>());
    } else if (t.get_element_type() == ov::element::bf16) {
        check(t.data<const ov::bfloat16>());
    } else if (t.get_element_type() == ov::element::f32) {
        check(t.data<const float>());
    }
    return bad;
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

ov::Tensor make_position_ids_with_overlap(size_t context_len, size_t draft_len) {
    const size_t total = context_len + draft_len;
    ov::Tensor ids(ov::element::i64, {1, total});
    auto* data = ids.data<int64_t>();
    for (size_t i = 0; i < context_len; ++i) {
        data[i] = static_cast<int64_t>(i);
    }
    for (size_t i = context_len; i < total; ++i) {
        // First draft token reuses the last context position (context_len - 1),
        // following tokens advance normally.
        data[i] = static_cast<int64_t>(context_len - 1 + (i - context_len));
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

int64_t resolve_mask_token_id(ov::genai::Tokenizer tokenizer) {
    const auto vocab = tokenizer.get_vocab();
    const auto it = vocab.find("<|MASK|>");
    if (it != vocab.end()) {
        return it->second;
    }
    // Try common Qwen padding/fim tokens that are in-vocab and don't require resizing.
    auto try_single_token = [&](const std::string& token) -> std::optional<int64_t> {
        try {
            auto encoded = tokenizer.encode(token, {ov::genai::add_special_tokens(false)}).input_ids;
            if (encoded.get_shape() == ov::Shape{1, 1}) {
                return encoded.data<const int64_t>()[0];
            }
        } catch (...) {
        }
        return std::nullopt;
    };
    if (auto fim_pad = try_single_token("<|fim_pad|>")) {
        return *fim_pad;
    }
    if (auto vision_pad = try_single_token("<|vision_pad|>")) {
        return *vision_pad;
    }
    // Fallback: use pad token if defined, otherwise eos.
    const int64_t pad_id = tokenizer.get_pad_token_id();
    if (pad_id != -1) {
        return pad_id;
    }
    const int64_t eos_id = tokenizer.get_eos_token_id();
    if (eos_id != -1) {
        return eos_id;
    }
    throw std::runtime_error("Tokenizer has no suitable mask/pad/eos token to use as mask placeholder");
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

    PerfStats perf;

    auto target_model = ov::genai::modeling::models::create_qwen3_dflash_target_model_no_cache(
        qwen_cfg, target_layer_ids, target_source, target_finalizer);
    const auto draft_compute_type = ov::element::f32;
    auto draft_model = ov::genai::modeling::models::create_dflash_draft_model(
        dflash_cfg, draft_source, draft_finalizer, draft_compute_type);
    auto draft_hidden_type = draft_model->output("draft_hidden").get_element_type();
    auto embed_model = ov::genai::modeling::models::create_qwen3_embedding_model(
        qwen_cfg, target_source, target_finalizer);
    auto lm_head_model = ov::genai::modeling::models::create_qwen3_lm_head_model(
        qwen_cfg, target_source, target_finalizer, draft_compute_type);

    ov::Core core;
    // Force all sub-models to run in FP32 to rule out BF16/F16 precision issues.
    ov::AnyMap compile_cfg = {
        {ov::hint::inference_precision.name(), ov::element::f32}
    };
    auto compiled_target = core.compile_model(target_model, device, compile_cfg);
    auto compiled_embed = core.compile_model(embed_model, device, compile_cfg);
    auto compiled_lm_head = core.compile_model(lm_head_model, device, compile_cfg);
    auto compiled_draft = core.compile_model(draft_model, device, compile_cfg);

    auto target_request = compiled_target.create_infer_request();
    auto embed_request = compiled_embed.create_infer_request();
    auto lm_head_request = compiled_lm_head.create_infer_request();
    auto draft_request = compiled_draft.create_infer_request();

    ov::genai::Tokenizer tokenizer(target_dir);
    const int64_t mask_token_id = resolve_mask_token_id(tokenizer);
    const int64_t eos_token_id = tokenizer.get_eos_token_id();
    std::cout << "[DEBUG] mask_token_id=" << mask_token_id << " eos_token_id=" << eos_token_id << std::endl;

    auto encoded = tokenizer.encode(prompt, {ov::genai::add_special_tokens(false)});
    std::vector<int64_t> output_ids = tensor_to_ids(encoded.input_ids);
    const size_t prompt_len = output_ids.size();
    const size_t max_length = prompt_len + static_cast<size_t>(max_new_tokens);

    const auto generation_start = Clock::now();

    // Prefill: run target on prompt to get first token.
    auto prefill_start = Clock::now();
    target_request.set_tensor("input_ids", make_ids_tensor(output_ids));
    target_request.set_tensor("attention_mask", make_attention_mask(output_ids.size()));
    target_request.set_tensor("position_ids", make_position_ids(output_ids.size()));
    target_request.infer();
    auto logits = target_request.get_tensor("logits");
    auto prefill_end = Clock::now();
    const double prefill_ms = duration_ms(prefill_start, prefill_end);
    perf.prefill_wall.add(prefill_ms);

    int64_t next_token = argmax_last_token(logits);
    perf.ttft_ms = duration_ms(generation_start, Clock::now());
    output_ids.push_back(next_token);

    while (output_ids.size() < max_length) {
        if (next_token == eos_token_id) {
            break;
        }

        // Build context features by running target on full context.
        auto target_ctx_start = Clock::now();
        target_request.set_tensor("input_ids", make_ids_tensor(output_ids));
        target_request.set_tensor("attention_mask", make_attention_mask(output_ids.size()));
        target_request.set_tensor("position_ids", make_position_ids(output_ids.size()));
        target_request.infer();
        auto target_ctx_end = Clock::now();
        perf.target_ctx_wall.add(duration_ms(target_ctx_start, target_ctx_end));
        auto target_hidden = target_request.get_tensor("target_hidden");
        if (target_hidden.get_element_type() != ov::element::f32) {
            ov::Tensor target_f32(ov::element::f32, target_hidden.get_shape());
            target_hidden.copy_to(target_f32);
            target_hidden = target_f32;
        }

        // Build draft block inputs: [last_token, MASK, MASK, ...]
        std::vector<int64_t> block_ids(static_cast<size_t>(dflash_cfg.block_size), mask_token_id);
        block_ids[0] = output_ids.back();

        auto embed_start = Clock::now();
        embed_request.set_tensor("input_ids", make_ids_tensor(block_ids));
        embed_request.infer();
        auto embed_end = Clock::now();
        perf.embed_wall.add(duration_ms(embed_start, embed_end));
        auto noise_embedding = embed_request.get_tensor("embeddings");

        const size_t pos_start = output_ids.size() - 1;  // last accepted token
        const size_t pos_end = pos_start + block_ids.size() - 1;
        const auto position_ids = make_position_ids_with_overlap(output_ids.size(), block_ids.size());
        auto draft_start = Clock::now();
        draft_request.set_tensor("target_hidden", target_hidden);
        draft_request.set_tensor("noise_embedding", noise_embedding);
        draft_request.set_tensor("position_ids", position_ids);
        draft_request.infer();
        auto draft_end = Clock::now();
        perf.draft_wall.add(duration_ms(draft_start, draft_end));
        auto draft_hidden = draft_request.get_tensor("draft_hidden");

        // Align dtype for lm_head input if needed.
        const auto& lm_head_port = compiled_lm_head.input("hidden_states");
        if (draft_hidden.get_element_type() != lm_head_port.get_element_type()) {
            ov::Tensor converted(lm_head_port.get_element_type(), draft_hidden.get_shape());
            draft_hidden.copy_to(converted);
            draft_hidden = converted;
        }
        auto lm_head_start = Clock::now();
        lm_head_request.set_tensor("hidden_states", draft_hidden);
        lm_head_request.infer();
        auto lm_head_end = Clock::now();
        perf.lm_head_wall.add(duration_ms(lm_head_start, lm_head_end));
        auto draft_logits = lm_head_request.get_tensor("logits");

        const size_t draft_len = block_ids.size() - 1;
        auto draft_tokens = argmax_logits_slice(draft_logits, 1, draft_len);

        // Verify with target model on full context + draft tokens.
        std::vector<int64_t> verify_ids = output_ids;
        verify_ids.insert(verify_ids.end(), draft_tokens.begin(), draft_tokens.end());
        auto verify_start = Clock::now();
        target_request.set_tensor("input_ids", make_ids_tensor(verify_ids));
        target_request.set_tensor("attention_mask", make_attention_mask(verify_ids.size()));
        target_request.set_tensor("position_ids", make_position_ids(verify_ids.size()));
        target_request.infer();
        auto verify_end = Clock::now();
        perf.verify_wall.add(duration_ms(verify_start, verify_end));
        logits = target_request.get_tensor("logits");

        const size_t start = output_ids.size() - 1;
        auto posterior = argmax_logits_slice(logits, start, block_ids.size());

        size_t accepted = 0;
        for (; accepted < draft_len; ++accepted) {
            if (draft_tokens[accepted] != posterior[accepted]) {
                break;
            }
        }

        // Debug: dump alignment for the first few draft steps.
        if (perf.draft_steps < 5) {
            const size_t preview_n = 5;
            const auto block_preview = preview_vec(block_ids, preview_n);
            const auto pos_preview = preview_vec(tensor_to_vec_i64(position_ids), block_ids.size());
            const auto target_hidden_shape = target_hidden.get_shape();
            const auto draft_hidden_shape = draft_hidden.get_shape();
            const auto noise_stats = stats_to_str(tensor_stats(noise_embedding));
            const auto draft_hidden_stats = stats_to_str(tensor_stats(draft_hidden));
            const auto target_hidden_stats = stats_to_str(tensor_stats(target_hidden));
            const auto target_hidden_bad = count_nonfinite(target_hidden);
            const auto draft_hidden_bad = count_nonfinite(draft_hidden);
            const auto logits_top0 = topk_to_str(topk_logits_row(draft_logits, 0, 5));
            const auto logits_top1 = topk_to_str(topk_logits_row(draft_logits, 1, 5));
            std::cout << "[DEBUG] step=" << (perf.draft_steps + 1) << " pos_range=[" << pos_start << "," << pos_end
                      << "] block_ids" << block_preview
                      << " pos_ids" << pos_preview
                      << " draft_tokens" << preview_vec(draft_tokens, preview_n)
                      << " posterior" << preview_vec(posterior, preview_n)
                      << " target_hidden_shape=[" << target_hidden_shape[0] << "," << target_hidden_shape[1] << ","
                      << target_hidden_shape[2] << "] draft_hidden_shape=[" << draft_hidden_shape[0] << ","
                      << draft_hidden_shape[1] << "," << draft_hidden_shape[2] << "]"
                      << " noise_stats=" << noise_stats
                      << " target_hidden_stats=" << target_hidden_stats << " target_hidden_bad=" << target_hidden_bad
                      << " draft_hidden_stats=" << draft_hidden_stats << " draft_hidden_bad=" << draft_hidden_bad
                      << " logits_top0=" << logits_top0
                      << " logits_top1=" << logits_top1
                      << " accepted=" << accepted << std::endl;
        }

        const size_t before_accept = output_ids.size();
        for (size_t i = 0; i < accepted && output_ids.size() < max_length; ++i) {
            output_ids.push_back(draft_tokens[i]);
        }
        const size_t accepted_pushed = output_ids.size() - before_accept;
        ++perf.draft_steps;
        perf.accepted_tokens += accepted_pushed;
        if (output_ids.size() >= max_length) {
            break;
        }
        next_token = posterior[accepted];
        output_ids.push_back(next_token);
    }

    const auto generation_end = Clock::now();
    perf.total_generate_ms = duration_ms(generation_start, generation_end);
    perf.generated_tokens = output_ids.size() - prompt_len;

    const size_t tokens_after_first = perf.generated_tokens > 0 ? perf.generated_tokens - 1 : 0;
    const double tpot_ms = tokens_after_first > 0
                               ? (perf.total_generate_ms - perf.ttft_ms) / static_cast<double>(tokens_after_first)
                               : 0.0;
    const double throughput = perf.total_generate_ms > 0
                                  ? (static_cast<double>(perf.generated_tokens) * 1000.0) / perf.total_generate_ms
                                  : 0.0;
    const double avg_accept = perf.draft_steps > 0
                                  ? static_cast<double>(perf.accepted_tokens) / static_cast<double>(perf.draft_steps)
                                  : 0.0;
    const size_t target_generated = perf.generated_tokens > perf.accepted_tokens
                                        ? perf.generated_tokens - perf.accepted_tokens
                                        : 0;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[Tokens] prompt=" << prompt_len << ", generated=" << perf.generated_tokens
              << ", draft_accepted=" << perf.accepted_tokens << ", target_only=" << target_generated
              << ", avg_accept_per_block=" << avg_accept << std::endl;
    std::cout << "[Latency] TTFT=" << perf.ttft_ms << " ms, TPOT=" << tpot_ms
              << " ms/token, total_generate=" << perf.total_generate_ms
              << " ms, throughput=" << throughput << " tokens/s" << std::endl;
    std::cout << "[Stage timings] wall-clock (ms):" << std::endl;
    print_stage_stats("prefill target", perf.prefill_wall);
    print_stage_stats("target ctx", perf.target_ctx_wall);
    print_stage_stats("embed", perf.embed_wall);
    print_stage_stats("draft", perf.draft_wall);
    print_stage_stats("lm_head", perf.lm_head_wall);
    print_stage_stats("target verify", perf.verify_wall);

    std::vector<int64_t> generated(output_ids.begin() + static_cast<std::ptrdiff_t>(prompt_len), output_ids.end());
    auto text = tokenizer.decode(generated, {ov::genai::skip_special_tokens(true)});
    std::cout << text << std::endl;
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "DFlash sample failed: " << ex.what() << std::endl;
    return 1;
}
