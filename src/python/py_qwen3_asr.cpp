// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <limits>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <openvino/openvino.hpp>
#include <openvino/runtime/properties.hpp>

#include "loaders/model_config.hpp"
#include "modeling/models/qwen3_asr/modeling_qwen3_asr.hpp"
#include "modeling/weights/quantization_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "py_utils.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"
#include "tokenizer/tokenizers_path.hpp"
#include "whisper/feature_extractor.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;

namespace {

struct ParsedASROutput {
    std::string language;
    std::string text;
};

struct Qwen3ASRDecodedResult {
    std::string raw_text;
    std::string language;
    std::string text;
};

bool has_ir_model_pair(const std::filesystem::path& xml_path, const std::filesystem::path& bin_path) {
    return std::filesystem::exists(xml_path) && std::filesystem::is_regular_file(xml_path) &&
           std::filesystem::exists(bin_path) && std::filesystem::is_regular_file(bin_path);
}

bool model_has_input_named(const std::shared_ptr<ov::Model>& model, const char* name) {
    for (const auto& input : model->inputs()) {
        for (const auto& input_name : input.get_names()) {
            if (input_name == name) {
                return true;
            }
        }
    }
    return false;
}

bool text_model_cache_supports_mode(const std::shared_ptr<ov::Model>& model, bool expect_audio_inputs) {
    const bool has_audio_embeds =
        model_has_input_named(model, ov::genai::modeling::models::Qwen3ASRTextIO::kAudioEmbeds);
    const bool has_audio_pos_mask =
        model_has_input_named(model, ov::genai::modeling::models::Qwen3ASRTextIO::kAudioPosMask);
    const bool has_audio_inputs = has_audio_embeds || has_audio_pos_mask;
    return has_audio_inputs == expect_audio_inputs;
}

double elapsed_ms(const std::chrono::steady_clock::time_point& start,
                  const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

ov::genai::modeling::models::Qwen3ASRTextConfig to_text_cfg(const ov::genai::loaders::ModelConfig& cfg) {
    using ov::genai::modeling::models::Qwen3ASRTextConfig;
    Qwen3ASRTextConfig out;
    out.architecture = "qwen3_asr";
    out.vocab_size = cfg.vocab_size;
    out.hidden_size = cfg.hidden_size;
    out.intermediate_size = cfg.intermediate_size;
    out.num_hidden_layers = cfg.num_hidden_layers;
    out.num_attention_heads = cfg.num_attention_heads;
    out.num_key_value_heads = cfg.num_key_value_heads > 0 ? cfg.num_key_value_heads : cfg.num_attention_heads;
    out.head_dim = cfg.head_dim > 0 ? cfg.head_dim : (cfg.hidden_size / std::max(1, cfg.num_attention_heads));
    out.max_position_embeddings = cfg.max_position_embeddings;
    out.rms_norm_eps = cfg.rms_norm_eps;
    out.rope_theta = cfg.rope_theta;
    out.hidden_act = cfg.hidden_act;
    out.attention_bias = cfg.attention_bias;
    out.tie_word_embeddings = cfg.tie_word_embeddings;

    if (out.hidden_size <= 0 || out.intermediate_size <= 0 || out.num_hidden_layers <= 0 ||
        out.num_attention_heads <= 0 || out.vocab_size <= 0) {
        throw std::runtime_error("Invalid text config parsed from config.json");
    }
    return out;
}

ov::genai::modeling::models::Qwen3ASRAudioConfig to_audio_cfg(const ov::genai::loaders::ModelConfig& cfg) {
    using ov::genai::modeling::models::Qwen3ASRAudioConfig;
    Qwen3ASRAudioConfig out;
    out.architecture = "qwen3_asr_audio_encoder";
    out.num_mel_bins = cfg.audio_num_mel_bins > 0 ? cfg.audio_num_mel_bins : out.num_mel_bins;
    out.d_model = cfg.audio_hidden_size > 0 ? cfg.audio_hidden_size : (cfg.hidden_size > 0 ? cfg.hidden_size : out.d_model);
    out.encoder_layers = cfg.audio_num_hidden_layers > 0 ? cfg.audio_num_hidden_layers
                                                          : (cfg.num_hidden_layers > 0 ? cfg.num_hidden_layers : out.encoder_layers);
    out.encoder_attention_heads = cfg.audio_num_attention_heads > 0 ? cfg.audio_num_attention_heads
                                                                     : (cfg.num_attention_heads > 0 ? cfg.num_attention_heads : out.encoder_attention_heads);
    out.encoder_ffn_dim = cfg.audio_intermediate_size > 0 ? cfg.audio_intermediate_size
                                                           : (cfg.intermediate_size > 0 ? cfg.intermediate_size : out.encoder_ffn_dim);
    out.max_source_positions = cfg.audio_max_position_embeddings > 0 ? cfg.audio_max_position_embeddings
                                                                      : (cfg.max_position_embeddings > 0 ? cfg.max_position_embeddings : out.max_source_positions);
    out.downsample_hidden_size = cfg.audio_downsample_hidden_size > 0 ? cfg.audio_downsample_hidden_size : out.downsample_hidden_size;
    out.output_dim = cfg.audio_output_dim > 0 ? cfg.audio_output_dim : out.output_dim;
    out.n_window = cfg.audio_n_window > 0 ? cfg.audio_n_window : out.n_window;
    out.n_window_infer = cfg.audio_n_window_infer > 0 ? cfg.audio_n_window_infer : out.n_window_infer;
    out.activation_function = !cfg.audio_hidden_act.empty() ? cfg.audio_hidden_act
                                                             : (cfg.hidden_act.empty() ? out.activation_function : cfg.hidden_act);

    if (out.d_model <= 0 || out.encoder_layers <= 0 || out.encoder_attention_heads <= 0 || out.encoder_ffn_dim <= 0) {
        throw std::runtime_error("Invalid audio config parsed from config.json");
    }
    return out;
}

ov::Tensor make_i64(const ov::Shape& shape, int64_t value = 0) {
    ov::Tensor t(ov::element::i64, shape);
    std::fill(t.data<int64_t>(), t.data<int64_t>() + t.get_size(), value);
    return t;
}

ov::Tensor make_i32(const ov::Shape& shape, int32_t value = 0) {
    ov::Tensor t(ov::element::i32, shape);
    std::fill(t.data<int32_t>(), t.data<int32_t>() + t.get_size(), value);
    return t;
}

ov::Tensor make_position_ids_3d(size_t batch, size_t seq_len) {
    ov::Tensor t(ov::element::i64, ov::Shape{3, batch, seq_len});
    int64_t* ptr = t.data<int64_t>();
    for (size_t plane = 0; plane < 3; ++plane) {
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                ptr[(plane * batch + b) * seq_len + s] = static_cast<int64_t>(s);
            }
        }
    }
    return t;
}

ov::Tensor make_i64_from_vector(const std::vector<int64_t>& values) {
    ov::Tensor t(ov::element::i64, ov::Shape{1, values.size()});
    std::copy(values.begin(), values.end(), t.data<int64_t>());
    return t;
}

std::vector<int64_t> tensor_row_to_i64_vector(const ov::Tensor& t) {
    const auto shape = t.get_shape();
    if (shape.size() != 2 || shape[0] != 1 || t.get_element_type() != ov::element::i64) {
        throw std::runtime_error("Expected tensor shape [1, N] with i64 type");
    }
    const int64_t* src = t.data<const int64_t>();
    return std::vector<int64_t>(src, src + shape[1]);
}

ov::Tensor make_audio_pos_mask_from_flags(const std::vector<char>& flags) {
    ov::Tensor t(ov::element::boolean, ov::Shape{1, flags.size()});
    char* ptr = t.data<char>();
    std::copy(flags.begin(), flags.end(), ptr);
    return t;
}

ov::Tensor make_audio_embeds_for_mask_positions(const ov::Tensor& audio_embeds, const std::vector<char>& audio_flags) {
    const auto src_shape = audio_embeds.get_shape();
    if (src_shape.size() != 3 || src_shape[0] != 1 || audio_embeds.get_element_type() != ov::element::f32) {
        throw std::runtime_error("Expected audio_embeds shape [1, T, H] with f32 type");
    }

    const size_t src_seq = src_shape[1];
    const size_t hidden = src_shape[2];
    size_t true_count = 0;
    for (char v : audio_flags) {
        true_count += (v != 0) ? 1 : 0;
    }
    if (true_count != src_seq) {
        throw std::runtime_error("Audio mask true-count must match audio_embeds sequence length");
    }

    ov::Tensor out(ov::element::f32, ov::Shape{1, audio_flags.size(), hidden});
    float* dst = out.data<float>();
    std::fill(dst, dst + out.get_size(), 0.0f);

    const float* src = audio_embeds.data<const float>();
    size_t src_idx = 0;
    for (size_t pos = 0; pos < audio_flags.size(); ++pos) {
        if (audio_flags[pos] == 0) {
            continue;
        }
        std::copy(src + src_idx * hidden,
                  src + (src_idx + 1) * hidden,
                  dst + pos * hidden);
        ++src_idx;
    }
    return out;
}

int64_t argmax_last_token_id(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0) {
        throw std::runtime_error("Unexpected logits shape for argmax");
    }

    const size_t vocab = shape[2];
    const size_t offset = (shape[0] - 1) * shape[1] * vocab + (shape[1] - 1) * vocab;
    const float* row = logits.data<const float>() + offset;

    size_t best_idx = 0;
    float best_val = row[0];
    for (size_t i = 1; i < vocab; ++i) {
        if (row[i] > best_val) {
            best_val = row[i];
            best_idx = i;
        }
    }
    return static_cast<int64_t>(best_idx);
}

int64_t argmax_last_token_id_excluding(const ov::Tensor& logits, const std::unordered_set<int64_t>& excluded_ids) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0) {
        throw std::runtime_error("Unexpected logits shape for argmax");
    }

    const size_t vocab = shape[2];
    const size_t offset = (shape[0] - 1) * shape[1] * vocab + (shape[1] - 1) * vocab;
    const float* row = logits.data<const float>() + offset;

    int64_t best_idx = -1;
    float best_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < vocab; ++i) {
        const int64_t tid = static_cast<int64_t>(i);
        if (excluded_ids.find(tid) != excluded_ids.end()) {
            continue;
        }
        if (row[i] > best_val) {
            best_val = row[i];
            best_idx = tid;
        }
    }

    if (best_idx < 0) {
        return argmax_last_token_id(logits);
    }
    return best_idx;
}

bool has_recent_ngram_loop(const std::vector<int64_t>& ids, size_t ngram, size_t repeats) {
    if (ngram == 0 || repeats < 2) {
        return false;
    }
    const size_t needed = ngram * repeats;
    if (ids.size() < needed) {
        return false;
    }

    const size_t base = ids.size() - needed;
    for (size_t r = 1; r < repeats; ++r) {
        for (size_t i = 0; i < ngram; ++i) {
            if (ids[base + i] != ids[base + r * ngram + i]) {
                return false;
            }
        }
    }
    return true;
}

bool trim_recent_duplicate_ngram(std::vector<int64_t>& ids, size_t min_ngram, size_t max_ngram) {
    if (min_ngram == 0 || max_ngram < min_ngram) {
        return false;
    }
    const size_t capped_max = std::min(max_ngram, ids.size() / 2);
    if (capped_max < min_ngram) {
        return false;
    }

    for (size_t n = capped_max; n >= min_ngram; --n) {
        const size_t off0 = ids.size() - 2 * n;
        const size_t off1 = ids.size() - n;
        bool same = true;
        for (size_t i = 0; i < n; ++i) {
            if (ids[off0 + i] != ids[off1 + i]) {
                same = false;
                break;
            }
        }
        if (same) {
            ids.resize(ids.size() - n);
            return true;
        }
        if (n == min_ngram) {
            break;
        }
    }
    return false;
}

bool is_language_tag_token(const std::string& token) {
    if (token.size() < 6) {
        return false;
    }
    if (token.rfind("<|", 0) != 0 || token.substr(token.size() - 2) != "|>") {
        return false;
    }
    const std::string tag = token.substr(2, token.size() - 4);
    if (tag.size() < 2 || tag.size() > 5) {
        return false;
    }
    for (char c : tag) {
        if (!std::isalpha(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

std::string trim_copy(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

std::string normalize_language_name(const std::string& raw) {
    const std::string s = trim_copy(raw);
    if (s.empty()) {
        return {};
    }

    std::string out;
    out.reserve(s.size());
    bool seen_lower = false;
    for (char c : s) {
        if (!std::isalpha(static_cast<unsigned char>(c))) {
            break;
        }
        if (!out.empty() && seen_lower && std::isupper(static_cast<unsigned char>(c))) {
            break;
        }
        if (std::islower(static_cast<unsigned char>(c))) {
            seen_lower = true;
        }
        out.push_back(c);
    }
    return out;
}

std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

void erase_all_tokens(std::string& text, const std::vector<std::string>& tokens) {
    for (const auto& token : tokens) {
        std::string::size_type pos = 0;
        while ((pos = text.find(token, pos)) != std::string::npos) {
            text.erase(pos, token.size());
        }
    }
}

std::string strip_after_markdown_fence(const std::string& text) {
    const std::string::size_type pos = text.find("```");
    if (pos == std::string::npos) {
        return text;
    }
    return text.substr(0, pos);
}

void strip_leading_asr_separators(std::string& text) {
    static const std::array<std::string, 5> prefixes = {"<asr_text>", "BitFields", "<LM>", "(Initialized)", "<|box_end|>"};
    bool changed = true;
    while (changed) {
        changed = false;
        text = trim_copy(text);
        for (const auto& prefix : prefixes) {
            if (text.rfind(prefix, 0) == 0) {
                text.erase(0, prefix.size());
                changed = true;
                break;
            }
        }
        while (!text.empty() &&
               (text.front() == '.' || text.front() == ',' || text.front() == ';' || text.front() == ':' ||
                text.front() == '!' || text.front() == '?' || std::isspace(static_cast<unsigned char>(text.front())))) {
            text.erase(text.begin());
            changed = true;
        }
    }
}

std::string clean_transcript_text(const std::string& raw_text) {
    std::string out = raw_text;
    erase_all_tokens(out, {"<asr_text>", "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<|im_start|>", "<|im_end|>", "<BR>", "<|box_end|>"});
    out = strip_after_markdown_fence(out);
    out = trim_copy(out);
    while (!out.empty() && (out.front() == '.' || out.front() == ',' || out.front() == ';' || out.front() == ':' ||
                             out.front() == '!' || out.front() == '?' || std::isspace(static_cast<unsigned char>(out.front())))) {
        out.erase(out.begin());
    }
    return trim_copy(out);
}

std::string extract_language_from_prefix(std::string& text) {
    static const std::regex lang_prefix(R"(^\s*language\s+([A-Za-z]+)\s*)", std::regex_constants::icase);
    std::smatch m;
    if (std::regex_search(text, m, lang_prefix) && m.size() >= 2) {
        const std::string lang = normalize_language_name(m[1].str());
        text.erase(0, static_cast<size_t>(m.position(0) + m.length(0)));
        text = trim_copy(text);
        return lang;
    }
    return {};
}

ParsedASROutput parse_asr_output(const std::string& raw_text) {
    ParsedASROutput result;
    std::string text = trim_copy(raw_text);
    if (text.empty()) {
        return result;
    }

    const std::string asr_tag = "<asr_text>";
    const std::string::size_type tag_pos = text.find(asr_tag);
    if (tag_pos != std::string::npos) {
        std::string meta_part = text.substr(0, tag_pos);
        std::string text_part = text.substr(tag_pos + asr_tag.size());
        const std::string meta_lower = to_lower_ascii(meta_part);
        if (meta_lower.find("language none") != std::string::npos) {
            strip_leading_asr_separators(text_part);
            result.text = clean_transcript_text(text_part);
            return result;
        }
        std::string parsed_lang = extract_language_from_prefix(meta_part);
        if (!parsed_lang.empty()) {
            result.language = parsed_lang;
        }
        strip_leading_asr_separators(text_part);
        result.text = clean_transcript_text(text_part);
        return result;
    }

    std::string text_part = text;
    const std::string parsed_lang = extract_language_from_prefix(text_part);
    if (!parsed_lang.empty()) {
        if (to_lower_ascii(parsed_lang) != "none") {
            result.language = parsed_lang;
        }
        strip_leading_asr_separators(text_part);
        result.text = clean_transcript_text(text_part);
        return result;
    }

    result.text = clean_transcript_text(text);
    return result;
}

bool is_metadata_only_asr_output(const std::string& raw_text) {
    std::string text = trim_copy(raw_text);
    if (text.empty()) {
        return false;
    }

    const std::string asr_tag = "<asr_text>";
    const std::string::size_type tag_pos = text.find(asr_tag);
    std::string text_part;
    if (tag_pos != std::string::npos) {
        std::string meta_part = text.substr(0, tag_pos);
        if (extract_language_from_prefix(meta_part).empty()) {
            return false;
        }
        text_part = text.substr(tag_pos + asr_tag.size());
    } else {
        text_part = text;
        if (extract_language_from_prefix(text_part).empty()) {
            return false;
        }
    }

    strip_leading_asr_separators(text_part);
    return clean_transcript_text(text_part).empty();
}

bool has_recent_metadata_prefix_loop(ov::genai::Tokenizer& tokenizer,
                                     const std::vector<int64_t>& ids,
                                     size_t min_ngram,
                                     size_t max_ngram) {
    if (min_ngram == 0 || max_ngram < min_ngram) {
        return false;
    }
    const size_t capped_max = std::min(max_ngram, ids.size() / 2);
    if (capped_max < min_ngram) {
        return false;
    }

    for (size_t n = capped_max; n >= min_ngram; --n) {
        const size_t off0 = ids.size() - 2 * n;
        const size_t off1 = ids.size() - n;
        bool same = true;
        for (size_t i = 0; i < n; ++i) {
            if (ids[off0 + i] != ids[off1 + i]) {
                same = false;
                break;
            }
        }
        if (same) {
            const std::vector<int64_t> tail(ids.begin() + static_cast<std::ptrdiff_t>(off1), ids.end());
            if (is_metadata_only_asr_output(tokenizer.decode(tail, {ov::genai::skip_special_tokens(false)}))) {
                return true;
            }
        }
        if (n == min_ngram) {
            break;
        }
    }
    return false;
}

bool trim_recent_metadata_prefix_ngram(ov::genai::Tokenizer& tokenizer,
                                       std::vector<int64_t>& ids,
                                       size_t min_ngram,
                                       size_t max_ngram) {
    if (min_ngram == 0 || max_ngram < min_ngram) {
        return false;
    }
    const size_t capped_max = std::min(max_ngram, ids.size() / 2);
    if (capped_max < min_ngram) {
        return false;
    }

    for (size_t n = capped_max; n >= min_ngram; --n) {
        const size_t off0 = ids.size() - 2 * n;
        const size_t off1 = ids.size() - n;
        bool same = true;
        for (size_t i = 0; i < n; ++i) {
            if (ids[off0 + i] != ids[off1 + i]) {
                same = false;
                break;
            }
        }
        if (same) {
            const std::vector<int64_t> tail(ids.begin() + static_cast<std::ptrdiff_t>(off1), ids.end());
            if (is_metadata_only_asr_output(tokenizer.decode(tail, {ov::genai::skip_special_tokens(false)}))) {
                ids.resize(ids.size() - 2 * n);
                return true;
            }
        }
        if (n == min_ngram) {
            break;
        }
    }
    return false;
}

std::string detect_language_from_tokens(ov::genai::Tokenizer& tokenizer, const std::vector<int64_t>& generated_ids) {
    static const std::unordered_map<std::string, std::string> language_tag_to_name = {
        {"ar", "Arabic"},       {"cs", "Czech"},      {"da", "Danish"},      {"de", "German"},
        {"el", "Greek"},        {"en", "English"},    {"es", "Spanish"},     {"fa", "Persian"},
        {"fi", "Finnish"},      {"fil", "Filipino"},  {"fr", "French"},      {"hi", "Hindi"},
        {"hu", "Hungarian"},    {"id", "Indonesian"}, {"it", "Italian"},     {"ja", "Japanese"},
        {"ko", "Korean"},       {"mk", "Macedonian"}, {"ms", "Malay"},       {"nl", "Dutch"},
        {"pl", "Polish"},       {"pt", "Portuguese"}, {"ro", "Romanian"},    {"ru", "Russian"},
        {"sv", "Swedish"},      {"th", "Thai"},       {"tr", "Turkish"},     {"vi", "Vietnamese"},
        {"yue", "Cantonese"},   {"zh", "Chinese"},
    };
    for (size_t i = 0; i < generated_ids.size() && i < 8; ++i) {
        const std::string token = tokenizer.decode({generated_ids[i]}, {ov::genai::skip_special_tokens(false)});
        if (is_language_tag_token(token)) {
            const std::string tag = to_lower_ascii(token.substr(2, token.size() - 4));
            const auto it = language_tag_to_name.find(tag);
            if (it != language_tag_to_name.end()) {
                return it->second;
            }
        }
    }
    return "unknown";
}

std::string detect_language_from_language_prefix_tokens(ov::genai::Tokenizer& tokenizer,
                                                        const std::vector<int64_t>& generated_ids) {
    if (generated_ids.size() < 2) {
        return {};
    }

    std::string first = trim_copy(tokenizer.decode({generated_ids[0]}, {ov::genai::skip_special_tokens(false)}));
    std::string first_lower = first;
    std::transform(first_lower.begin(), first_lower.end(), first_lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (first_lower != "language") {
        return {};
    }

    const std::string second = trim_copy(tokenizer.decode({generated_ids[1]}, {ov::genai::skip_special_tokens(false)}));
    if (second.empty()) {
        return {};
    }
    return normalize_language_name(second);
}

int64_t find_token_id(const std::unordered_map<std::string, int64_t>& vocab,
                      const std::initializer_list<std::string>& candidates,
                      int64_t fallback = -1) {
    for (const auto& tok : candidates) {
        const auto it = vocab.find(tok);
        if (it != vocab.end()) {
            return it->second;
        }
    }
    return fallback;
}

void add_token_if_found(const std::unordered_map<std::string, int64_t>& vocab,
                        const std::string& token,
                        std::unordered_set<int64_t>& out) {
    const auto it = vocab.find(token);
    if (it != vocab.end()) {
        out.insert(it->second);
    }
}

std::string build_python_style_asr_prompt(ov::genai::Tokenizer& tokenizer,
                                          const std::string& context,
                                          const std::optional<std::string>& forced_language) {
    const std::string effective_context = context.empty() ? "Transcribe this audio." : context;
    (void)tokenizer;

    std::string prompt = "<|im_start|>system\n" + effective_context + "<|im_end|>\n";
    prompt +=
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n";

    if (forced_language.has_value() && !forced_language->empty()) {
        prompt += "language " + *forced_language + "<asr_text>";
    }

    return prompt;
}

ov::Tensor make_audio_features_from_pcm(const std::vector<float>& raw,
                                        const std::filesystem::path& preprocessor_json,
                                        size_t expected_mel_bins) {
    ov::genai::WhisperFeatureExtractor extractor(preprocessor_json);
    ov::genai::WhisperFeatures features = extractor.extract(raw);
    if (features.n_frames == 0 || features.feature_size == 0) {
        throw std::runtime_error("Failed to extract audio features from PCM input");
    }

    const size_t actual_mel_bins = features.feature_size;
    const size_t frames = features.n_frames;
    const size_t mel_bins = expected_mel_bins > 0 ? expected_mel_bins : actual_mel_bins;

    ov::Tensor tensor(ov::element::f32, ov::Shape{1, mel_bins, frames});
    float* dst = tensor.data<float>();
    const float* src = features.data.data();

    for (size_t m = 0; m < mel_bins; ++m) {
        const float* src_row = (m < actual_mel_bins) ? (src + m * frames) : nullptr;
        float* dst_row = dst + m * frames;
        if (src_row != nullptr) {
            std::copy(src_row, src_row + frames, dst_row);
        } else {
            std::fill(dst_row, dst_row + frames, 0.0f);
        }
    }

    return tensor;
}

class Qwen3ASRInferenceEngine {
public:
    Qwen3ASRInferenceEngine(const std::filesystem::path& text_model_dir,
                           const std::optional<std::filesystem::path>& audio_model_dir,
                           const std::string& device,
                           int32_t max_new_tokens,
                           bool cache_model,
                           const ov::AnyMap& compile_properties)
        : text_model_dir_(text_model_dir),
          audio_model_dir_(audio_model_dir.value_or(text_model_dir)),
          device_(device),
          max_new_tokens_(max_new_tokens),
                      compile_properties_(compile_properties) {
        if (max_new_tokens_ <= 0) {
            throw std::runtime_error("max_new_tokens must be > 0");
        }

        text_cfg_ = to_text_cfg(ov::genai::loaders::ModelConfig::from_hf_json(text_model_dir_ / "config.json"));
        audio_cfg_ = to_audio_cfg(ov::genai::loaders::ModelConfig::from_hf_json(audio_model_dir_ / "config.json"));

        const std::filesystem::path text_xml = text_model_dir_ / "modeling_qwen3_asr_text_with_audio.xml";
        const std::filesystem::path text_bin = text_model_dir_ / "modeling_qwen3_asr_text_with_audio.bin";
        const std::filesystem::path audio_xml = audio_model_dir_ / "modeling_qwen3_asr_audio.xml";
        const std::filesystem::path audio_bin = audio_model_dir_ / "modeling_qwen3_asr_audio.bin";

        auto quant_cfg = ov::genai::modeling::weights::parse_quantization_config_from_env();
        if (quant_cfg.enabled() && quant_cfg.group_size <= 0) {
            throw std::runtime_error("OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE must be > 0 when quantization is enabled");
        }

        ov::Core core;
        std::shared_ptr<ov::Model> text_model;
        std::shared_ptr<ov::Model> audio_model;
        if (cache_model && has_ir_model_pair(text_xml, text_bin)) {
            auto candidate = core.read_model(text_xml.string(), text_bin.string());
            if (text_model_cache_supports_mode(candidate, true)) {
                text_model = std::move(candidate);
            }
        }
        if (cache_model && has_ir_model_pair(audio_xml, audio_bin)) {
            audio_model = core.read_model(audio_xml.string(), audio_bin.string());
        }

        if (!text_model) {
            auto text_data = ov::genai::safetensors::load_safetensors(text_model_dir_);
            ov::genai::safetensors::SafetensorsWeightSource text_source(std::move(text_data));
            ov::genai::safetensors::SafetensorsWeightFinalizer text_finalizer(quant_cfg);
            text_model = ov::genai::modeling::models::create_qwen3_asr_text_model(
                text_cfg_, text_source, text_finalizer, false, true);
            if (cache_model) {
                ov::serialize(text_model, text_xml.string(), text_bin.string());
            }
        }

        if (!audio_model) {
            auto audio_data = ov::genai::safetensors::load_safetensors(audio_model_dir_);
            ov::genai::safetensors::SafetensorsWeightSource audio_source(std::move(audio_data));
            ov::genai::safetensors::SafetensorsWeightFinalizer audio_finalizer;
            audio_model = ov::genai::modeling::models::create_qwen3_asr_audio_encoder_model(
                audio_cfg_, audio_source, audio_finalizer);
            if (cache_model) {
                ov::serialize(audio_model, audio_xml.string(), audio_bin.string());
            }
        }

        compiled_text_ = core.compile_model(text_model, device_, compile_properties_);
        compiled_audio_ = core.compile_model(audio_model, device_, compile_properties_);

        ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
        tokenizer_.emplace(text_model_dir_, ov::AnyMap{{"fix_mistral_regex", true}});
        vocab_ = tokenizer_->get_vocab();
        audio_pad_token_id_ = find_token_id(vocab_, {"<|audio_pad|>"}, -1);
        if (audio_pad_token_id_ < 0) {
            throw std::runtime_error("Failed to find <|audio_pad|> token id in tokenizer vocab");
        }
        bos_token_id_ = tokenizer_->get_bos_token_id();
        eos_token_id_ = tokenizer_->get_eos_token_id();
        if (bos_token_id_ < 0) {
            bos_token_id_ = eos_token_id_ >= 0 ? eos_token_id_ : 1;
        }

        add_token_if_found(vocab_, "<|audio_pad|>", excluded_decode_ids_);
        add_token_if_found(vocab_, "<|audio_start|>", excluded_decode_ids_);
        add_token_if_found(vocab_, "<|audio_end|>", excluded_decode_ids_);
        add_token_if_found(vocab_, "<|im_start|>", excluded_decode_ids_);
        add_token_if_found(vocab_, "<|im_end|>", excluded_decode_ids_);
        add_token_if_found(vocab_, "<non_speech>", excluded_decode_ids_);
        add_token_if_found(vocab_, "<asr_text>", excluded_decode_ids_);
        add_token_if_found(vocab_, "<|asr_text|>", excluded_decode_ids_);
        add_token_if_found(vocab_, "<|box_end|>", excluded_decode_ids_);
        for (int i = 1; i <= 27; ++i) {
            add_token_if_found(vocab_, "<blank" + std::to_string(i) + ">", excluded_decode_ids_);
        }

        add_token_if_found(vocab_, "<|endoftext|>", stop_token_ids_);
        add_token_if_found(vocab_, "<|im_end|>", stop_token_ids_);
        add_token_if_found(vocab_, "<|eot_id|>", stop_token_ids_);

        dot_token_id_ = find_token_id(vocab_, {"."}, -1);
        excl_token_id_ = find_token_id(vocab_, {"!"}, -1);
        qmark_token_id_ = find_token_id(vocab_, {"?"}, -1);

        preprocessor_json_ = audio_model_dir_ / "preprocessor_config.json";
        if (!std::filesystem::exists(preprocessor_json_)) {
            preprocessor_json_ = text_model_dir_ / "preprocessor_config.json";
        }
    }

    Qwen3ASRDecodedResult generate(const std::vector<float>& pcm16k,
                                   const std::string& context,
                                   const std::optional<std::string>& forced_language) {
        if (pcm16k.empty()) {
            return {};
        }

        ov::Tensor input_audio_features = make_audio_features_from_pcm(
            pcm16k, preprocessor_json_, static_cast<size_t>(std::max(1, audio_cfg_.num_mel_bins)));

        auto audio_request = compiled_audio_.create_infer_request();
        const size_t batch = 1;
        const size_t audio_frames = input_audio_features.get_shape()[2];

        audio_request.set_tensor(ov::genai::modeling::models::Qwen3ASRAudioIO::kInputAudioFeatures, input_audio_features);
        auto input_lengths = make_i64({batch});
        input_lengths.data<int64_t>()[0] = static_cast<int64_t>(audio_frames);
        audio_request.set_tensor(ov::genai::modeling::models::Qwen3ASRAudioIO::kAudioFeatureLengths, input_lengths);
        audio_request.infer();

        ov::Tensor audio_embeds = audio_request.get_tensor(ov::genai::modeling::models::Qwen3ASRAudioIO::kAudioEmbeds);
        const auto embeds_shape = audio_embeds.get_shape();
        if (embeds_shape.size() != 3 || embeds_shape[2] != static_cast<size_t>(text_cfg_.hidden_size)) {
            throw std::runtime_error("Unexpected audio_embeds shape from audio encoder");
        }
        const size_t audio_seq_len = embeds_shape[1];

        const std::optional<std::string> normalized_language =
            (forced_language.has_value() && !forced_language->empty())
                ? std::optional<std::string>(normalize_language_name(*forced_language))
                : std::nullopt;
        const std::string instruction_prompt = build_python_style_asr_prompt(*tokenizer_, context, normalized_language);
        auto encoded_prompt = tokenizer_->encode(instruction_prompt, {ov::genai::add_special_tokens(false)});
        std::vector<int64_t> prompt_ids = tensor_row_to_i64_vector(encoded_prompt.input_ids);

        std::vector<int64_t> input_ids;
        input_ids.reserve(prompt_ids.size() + audio_seq_len + static_cast<size_t>(max_new_tokens_));
        bool replaced_audio_placeholder = false;
        for (int64_t tid : prompt_ids) {
            if (!replaced_audio_placeholder && tid == audio_pad_token_id_) {
                input_ids.insert(input_ids.end(), audio_seq_len, audio_pad_token_id_);
                replaced_audio_placeholder = true;
            } else {
                input_ids.push_back(tid);
            }
        }
        if (!replaced_audio_placeholder) {
            input_ids.insert(input_ids.begin(), audio_seq_len, audio_pad_token_id_);
            input_ids.push_back(bos_token_id_);
        }

        std::vector<char> base_audio_pos_flags(input_ids.size(), 0);
        for (size_t i = 0; i < input_ids.size(); ++i) {
            if (input_ids[i] == audio_pad_token_id_) {
                base_audio_pos_flags[i] = 1;
            }
        }

        ov::Tensor prompt_audio_embeds = make_audio_embeds_for_mask_positions(audio_embeds, base_audio_pos_flags);
        ov::Tensor prompt_audio_pos_mask = make_audio_pos_mask_from_flags(base_audio_pos_flags);

        std::vector<char> no_audio_flags(1, 0);
        ov::Tensor empty_audio_embeds(ov::element::f32, ov::Shape{1, 0, embeds_shape[2]});
        ov::Tensor step_audio_embeds = make_audio_embeds_for_mask_positions(empty_audio_embeds, no_audio_flags);
        ov::Tensor step_audio_pos_mask = make_audio_pos_mask_from_flags(no_audio_flags);

        auto text_request = compiled_text_.create_infer_request();
        text_request.reset_state();
        text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kBeamIdx, make_i32({batch}, 0));

        const size_t prompt_seq_len = input_ids.size();
        const size_t max_total_seq_len = prompt_seq_len + static_cast<size_t>(max_new_tokens_);
        std::vector<int64_t> attention_mask_storage(max_total_seq_len, 1);
        auto make_attention_mask_view = [&](size_t seq_len) -> ov::Tensor {
            return ov::Tensor(ov::element::i64, ov::Shape{batch, seq_len}, attention_mask_storage.data());
        };

        text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kInputIds, make_i64_from_vector(input_ids));
        text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kAttentionMask, make_attention_mask_view(prompt_seq_len));
        text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kPositionIds, make_position_ids_3d(batch, prompt_seq_len));
        text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kAudioEmbeds, prompt_audio_embeds);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kAudioPosMask, prompt_audio_pos_mask);

        text_request.infer();

        std::vector<int64_t> generated_ids;
        generated_ids.reserve(static_cast<size_t>(max_new_tokens_));
        int64_t prev_token_id = std::numeric_limits<int64_t>::min();
        int32_t same_token_run = 0;

        auto process_logits_and_append = [&](const ov::Tensor& logits) -> bool {
            const int64_t next_token_id = argmax_last_token_id(logits);
            if ((eos_token_id_ >= 0 && next_token_id == eos_token_id_) ||
                (stop_token_ids_.find(next_token_id) != stop_token_ids_.end())) {
                return true;
            }
            generated_ids.push_back(next_token_id);
            return false;
        };

        bool stop_generation = false;
        {
            ov::Tensor logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kLogits);
            stop_generation = process_logits_and_append(logits);
        }

        ov::Tensor one_token_ids(ov::element::i64, ov::Shape{1, 1});
        ov::Tensor one_token_pos_ids(ov::element::i64, ov::Shape{3, 1, 1});
        int64_t* one_token_pos_ptr = one_token_pos_ids.data<int64_t>();

        size_t total_seq_len = prompt_seq_len;
        while (!stop_generation && generated_ids.size() < static_cast<size_t>(max_new_tokens_)) {
            const int64_t token_to_feed = generated_ids.back();
            one_token_ids.data<int64_t>()[0] = token_to_feed;

            total_seq_len += 1;
            const int64_t pos = static_cast<int64_t>(total_seq_len - 1);
            one_token_pos_ptr[0] = pos;
            one_token_pos_ptr[1] = pos;
            one_token_pos_ptr[2] = pos;

            text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kInputIds, one_token_ids);
            text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kAttentionMask, make_attention_mask_view(total_seq_len));
            text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kPositionIds, one_token_pos_ids);
            text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kAudioEmbeds, step_audio_embeds);
            text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kAudioPosMask, step_audio_pos_mask);

            text_request.infer();
            ov::Tensor logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kLogits);
            if (process_logits_and_append(logits)) {
                break;
            }
        }

        Qwen3ASRDecodedResult result;
        if (!generated_ids.empty()) {
            result.raw_text = tokenizer_->decode(generated_ids, {ov::genai::skip_special_tokens(false)});
        }
        if (is_metadata_only_asr_output(result.raw_text)) {
            result.raw_text.clear();
            generated_ids.clear();
        }
        ParsedASROutput parsed = parse_asr_output(result.raw_text);
        result.text = parsed.text;
        result.language = parsed.language;
        if (result.language.empty()) {
            result.language = detect_language_from_tokens(*tokenizer_, generated_ids);
        }
        if (result.language.empty() || result.language == "unknown") {
            const std::string language_from_tokens = detect_language_from_language_prefix_tokens(*tokenizer_, generated_ids);
            if (!language_from_tokens.empty()) {
                result.language = language_from_tokens;
            }
        }
        if (result.language.empty()) {
            result.language = "unknown";
        }
        return result;
    }

private:
    std::filesystem::path text_model_dir_;
    std::filesystem::path audio_model_dir_;
    std::filesystem::path preprocessor_json_;
    std::string device_;
    int32_t max_new_tokens_ = 512;
    ov::AnyMap compile_properties_;
    ov::genai::modeling::models::Qwen3ASRTextConfig text_cfg_;
    ov::genai::modeling::models::Qwen3ASRAudioConfig audio_cfg_;
    ov::CompiledModel compiled_text_;
    ov::CompiledModel compiled_audio_;
    std::optional<ov::genai::Tokenizer> tokenizer_;
    std::unordered_map<std::string, int64_t> vocab_;
    std::unordered_set<int64_t> excluded_decode_ids_;
    std::unordered_set<int64_t> stop_token_ids_;
    int64_t audio_pad_token_id_ = -1;
    int64_t bos_token_id_ = -1;
    int64_t eos_token_id_ = -1;
    int64_t dot_token_id_ = -1;
    int64_t excl_token_id_ = -1;
    int64_t qmark_token_id_ = -1;
};

}  // namespace

void init_qwen3_asr(py::module_& m) {
    py::class_<Qwen3ASRDecodedResult>(m, "Qwen3ASRDecodedResult", "Decoded output from the low-level Qwen3 ASR engine")
        .def_readonly("raw_text", &Qwen3ASRDecodedResult::raw_text)
        .def_readonly("language", &Qwen3ASRDecodedResult::language)
        .def_readonly("text", &Qwen3ASRDecodedResult::text);

    py::class_<Qwen3ASRInferenceEngine>(m, "Qwen3ASRInferenceEngine", "Low-level OpenVINO GenAI-backed Qwen3 ASR inference engine")
        .def(
            py::init([](const std::filesystem::path& models_path,
                        const std::string& device,
                        int32_t max_new_tokens,
                        const std::optional<std::filesystem::path>& audio_model_path,
                        bool cache_model,
                        const std::optional<std::string>& openvino_cache_dir,
                        const py::kwargs& kwargs) {
                ov::AnyMap properties = pyutils::kwargs_to_any_map(kwargs);
                if (openvino_cache_dir.has_value() && !openvino_cache_dir->empty()) {
                    properties.insert({ov::cache_dir(*openvino_cache_dir)});
                }
                return std::make_unique<Qwen3ASRInferenceEngine>(
                    models_path, audio_model_path, device, max_new_tokens, cache_model, properties);
            }),
            py::arg("models_path"),
            py::arg("device") = "GPU",
            py::arg("max_new_tokens") = 512,
            py::arg("audio_model_path") = py::none(),
            py::arg("cache_model") = false,
            py::arg("openvino_cache_dir") = py::none(),
            R"(
            Construct a low-level OpenVINO GenAI-backed Qwen3 ASR engine.

            models_path (os.PathLike): Directory with the text model and tokenizer assets.
            device (str): Device to run inference on.
            max_new_tokens (int): Maximum generated tokens per chunk.
            audio_model_path (os.PathLike | None): Optional separate audio-model directory.
            cache_model (bool): If true, cache generated IR files next to the model.
            openvino_cache_dir (str | None): Optional OpenVINO compile cache directory.
            kwargs: Additional OpenVINO compile properties.
        )")
        .def(
            "generate",
            [](Qwen3ASRInferenceEngine& engine,
               py::array_t<float, py::array::c_style | py::array::forcecast> audio,
               const std::string& context,
               const std::optional<std::string>& language) {
                std::vector<float> pcm(audio.size());
                std::copy_n(audio.data(), audio.size(), pcm.begin());
                py::gil_scoped_release rel;
                return engine.generate(pcm, context, language);
            },
            py::arg("audio"),
            py::arg("context") = "",
            py::arg("language") = py::none(),
            R"(
            Run one ASR decode on a single 16kHz mono PCM float waveform.

            audio (numpy.ndarray): 1D waveform array.
            context (str): Optional system/context prompt.
            language (str | None): Optional forced language.

            Returns:
                Qwen3ASRDecodedResult
        )");
}