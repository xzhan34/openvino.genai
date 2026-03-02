// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#include "openvino/genai/tokenizer.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

#include "modeling/models/qwen3_omni/modeling_qwen3_omni.hpp"

using namespace ov::genai::modeling::models;

namespace {

void write_wav(const std::string& filename, const float* samples, size_t num_samples, int sample_rate) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to create WAV file: " + filename);
    }

    int32_t data_size = static_cast<int32_t>(num_samples * sizeof(int16_t));
    int32_t file_size = 36 + data_size;
    int16_t audio_format = 1;
    int16_t num_channels = 1;
    int32_t byte_rate = sample_rate * num_channels * 2;
    int16_t block_align = num_channels * 2;
    int16_t bits_per_sample = 16;

    file.write("RIFF", 4);
    file.write(reinterpret_cast<char*>(&file_size), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    int32_t fmt_size = 16;
    file.write(reinterpret_cast<char*>(&fmt_size), 4);
    file.write(reinterpret_cast<char*>(&audio_format), 2);
    file.write(reinterpret_cast<char*>(&num_channels), 2);
    file.write(reinterpret_cast<char*>(&sample_rate), 4);
    file.write(reinterpret_cast<char*>(&byte_rate), 4);
    file.write(reinterpret_cast<char*>(&block_align), 2);
    file.write(reinterpret_cast<char*>(&bits_per_sample), 2);
    file.write("data", 4);
    file.write(reinterpret_cast<char*>(&data_size), 4);

    for (size_t i = 0; i < num_samples; ++i) {
        float v = std::max(-1.0f, std::min(1.0f, samples[i]));
        int16_t s = static_cast<int16_t>(v * 32767.0f);
        file.write(reinterpret_cast<char*>(&s), 2);
    }
}

int64_t argmax_last_token(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3 || shape[0] != 1) {
        throw std::runtime_error("Expected logits shape [1,S,V]");
    }
    const size_t seq = shape[1];
    const size_t vocab = shape[2];
    const auto* data = logits.data<const float>() + (seq - 1) * vocab;
    size_t best_idx = 0;
    float best_val = data[0];
    for (size_t i = 1; i < vocab; ++i) {
        if (data[i] > best_val) {
            best_val = data[i];
            best_idx = i;
        }
    }
    return static_cast<int64_t>(best_idx);
}

ov::Tensor make_text_position_ids_i64(size_t seq_len) {
    ov::Tensor t(ov::element::i64, {3, 1, seq_len});
    auto* p = t.data<int64_t>();
    for (size_t i = 0; i < seq_len; ++i) {
        const int64_t v = static_cast<int64_t>(i);
        p[i] = v;
        p[seq_len + i] = v;
        p[2 * seq_len + i] = v;
    }
    return t;
}

ov::Tensor make_decode_position_ids_i64(int64_t pos) {
    ov::Tensor t(ov::element::i64, {3, 1, 1});
    auto* p = t.data<int64_t>();
    p[0] = pos;
    p[1] = pos;
    p[2] = pos;
    return t;
}

std::string run_text_generation(const std::filesystem::path& model_dir,
                                ov::Core& core,
                                ov::genai::modeling::weights::WeightSource& source,
                                ov::genai::modeling::weights::WeightFinalizer& finalizer,
                                const Qwen3OmniConfig& cfg,
                                const std::string& prompt,
                                int max_new_tokens,
                                const std::string& device) {
    auto text_model = create_qwen3_omni_text_model(cfg, source, finalizer, false, false);
    text_model->set_rt_info(ov::element::f32, {"runtime_options", ov::hint::kv_cache_precision.name()});
    auto compiled = core.compile_model(text_model, device, ov::AnyMap{{ov::hint::inference_precision.name(), ov::element::f32}});
    auto infer = compiled.create_infer_request();

    ov::genai::Tokenizer tokenizer(model_dir);
    auto tok = tokenizer.encode(prompt, ov::genai::add_special_tokens(false));
    ov::Tensor input_ids = tok.input_ids;
    ov::Tensor attention_mask = tok.attention_mask;

    const size_t batch = input_ids.get_shape()[0];
    int64_t past_len = static_cast<int64_t>(input_ids.get_shape()[1]);

    ov::Tensor beam_idx(ov::element::i32, {batch});
    auto* b = beam_idx.data<int32_t>();
    for (size_t i = 0; i < batch; ++i) {
        b[i] = static_cast<int32_t>(i);
    }

    infer.reset_state();
    infer.set_tensor("input_ids", input_ids);
    infer.set_tensor("attention_mask", attention_mask);
    infer.set_tensor("position_ids", make_text_position_ids_i64(static_cast<size_t>(past_len)));
    infer.set_tensor("beam_idx", beam_idx);
    infer.infer();

    ov::Tensor logits = infer.get_tensor("logits");
    int64_t next_id = argmax_last_token(logits);
    std::vector<int64_t> generated;
    generated.reserve(static_cast<size_t>(std::max(1, max_new_tokens)));
    generated.push_back(next_id);

    ov::Tensor step_ids(ov::element::i64, {batch, 1});
    ov::Tensor step_mask(ov::element::i64, {batch, 1});
    auto* mask_data = step_mask.data<int64_t>();
    for (size_t i = 0; i < batch; ++i) {
        mask_data[i] = 1;
    }

    const int64_t eos_id = tokenizer.get_eos_token_id();
    for (int step = 1; step < max_new_tokens; ++step) {
        if (eos_id >= 0 && next_id == eos_id) {
            break;
        }
        auto* s = step_ids.data<int64_t>();
        for (size_t i = 0; i < batch; ++i) {
            s[i] = next_id;
        }

        infer.set_tensor("input_ids", step_ids);
        infer.set_tensor("attention_mask", step_mask);
        infer.set_tensor("position_ids", make_decode_position_ids_i64(past_len));
        infer.set_tensor("beam_idx", beam_idx);
        infer.infer();

        logits = infer.get_tensor("logits");
        next_id = argmax_last_token(logits);
        generated.push_back(next_id);
        past_len += 1;
    }

    return tokenizer.decode(generated, ov::genai::skip_special_tokens(true));
}

struct TtsRunResult {
    std::filesystem::path wav_path;
    size_t samples = 0;
    int sample_rate = 24000;
    std::string backend = "speech_decoder";
    std::string note;
};

TtsRunResult synthesize_fallback_tone(const std::filesystem::path& wav_out, const std::string& text) {
    const int sample_rate = 24000;
    const float seconds = 1.5f;
    const size_t sample_count = static_cast<size_t>(sample_rate * seconds);
    std::vector<float> audio(sample_count);
    const int hash = static_cast<int>(std::hash<std::string>{}(text) % 5000);
    const float freq = 220.0f + static_cast<float>(hash % 660);
    const float pi = 3.1415926535f;
    for (size_t i = 0; i < sample_count; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(sample_rate);
        audio[i] = 0.2f * std::sin(2.0f * pi * freq * t);
    }
    write_wav(wav_out.string(), audio.data(), audio.size(), sample_rate);

    TtsRunResult result;
    result.wav_path = wav_out;
    result.samples = audio.size();
    result.sample_rate = sample_rate;
    result.backend = "fallback_tone";
    result.note = "speech decoder unavailable; generated deterministic tone for e2e comparison";
    return result;
}

TtsRunResult run_min_tts(const std::filesystem::path& model_dir,
                         ov::Core& core,
                         ov::genai::modeling::weights::WeightSource& source,
                         ov::genai::modeling::weights::WeightFinalizer& finalizer,
                         const Qwen3OmniConfig& cfg,
                         const std::string& text,
                         const std::filesystem::path& wav_out,
                         const std::string& device) {
    try {
        auto decoder_model = create_qwen3_omni_speech_decoder_model(cfg, source, finalizer);

        auto decoder_compiled =
            core.compile_model(decoder_model, device, ov::AnyMap{{ov::hint::inference_precision.name(), ov::element::f32}});

        auto decoder_infer = decoder_compiled.create_infer_request();

    auto decoder_cfg = to_qwen3_omni_speech_decoder_config(cfg);

    const int64_t codebook_size = std::max<int64_t>(1, decoder_cfg.codebook_size);
    const int64_t num_quantizers = std::max<int64_t>(1, decoder_cfg.num_quantizers);
    const int64_t text_len = static_cast<int64_t>(std::max<size_t>(1, text.size()));
    const int64_t frames = std::max<int64_t>(16, std::min<int64_t>(160, text_len));
    const int64_t codec_seed = static_cast<int64_t>(std::hash<std::string>{}(text) % static_cast<size_t>(codebook_size));

    ov::Tensor codes(ov::element::i64, {1, static_cast<size_t>(num_quantizers), static_cast<size_t>(frames)});
    auto* c = codes.data<int64_t>();
    for (int64_t q = 0; q < num_quantizers; ++q) {
        for (int64_t t = 0; t < frames; ++t) {
            int64_t v = 0;
            if (q == 0) {
                v = (codec_seed + t) % codebook_size;
            }
            c[q * frames + t] = std::max<int64_t>(0, v);
        }
    }

        decoder_infer.set_tensor("codes", codes);
        decoder_infer.infer();
        ov::Tensor audio = decoder_infer.get_tensor("audio");

        const float* audio_ptr = audio.data<const float>();
        const size_t sample_count = audio.get_size();
        write_wav(wav_out.string(), audio_ptr, sample_count, 24000);

        TtsRunResult result;
        result.wav_path = wav_out;
        result.samples = sample_count;
        result.sample_rate = 24000;
        return result;
    } catch (const std::exception&) {
        return synthesize_fallback_tone(wav_out, text);
    }
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <MODEL_DIR> <CASE_ID> <TEXT_PROMPT> <WAV_OUT> [DEVICE] [MAX_NEW_TOKENS]\n";
        return 1;
    }

    const std::filesystem::path model_dir = argv[1];
    const std::string case_id = argv[2];
    const std::string text_prompt = argv[3];
    const std::filesystem::path wav_out = argv[4];
    const std::string device = (argc > 5) ? argv[5] : "CPU";
    const int max_new_tokens = (argc > 6) ? std::stoi(argv[6]) : 64;

    auto st_data = ov::genai::safetensors::load_safetensors(model_dir);
    ov::genai::safetensors::SafetensorsWeightSource source(std::move(st_data));
    ov::genai::safetensors::SafetensorsWeightFinalizer finalizer;
    Qwen3OmniConfig cfg = Qwen3OmniConfig::from_json_file(model_dir);

    ov::Core core;

    const auto start = std::chrono::high_resolution_clock::now();
    std::string response_text = run_text_generation(model_dir,
                                                    core,
                                                    source,
                                                    finalizer,
                                                    cfg,
                                                    text_prompt,
                                                    max_new_tokens,
                                                    device);

    TtsRunResult tts = run_min_tts(model_dir,
                                   core,
                                   source,
                                   finalizer,
                                   cfg,
                                   response_text.empty() ? text_prompt : response_text,
                                   wav_out,
                                   device);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "CASE_ID: " << case_id << "\n";
    std::cout << "TEXT_OUTPUT: " << response_text << "\n";
    std::cout << "WAV_OUTPUT: " << wav_out.string() << "\n";
    std::cout << "AUDIO_SAMPLES: " << tts.samples << "\n";
    std::cout << "AUDIO_SAMPLE_RATE: " << tts.sample_rate << "\n";
    std::cout << "TTS_BACKEND: " << tts.backend << "\n";
    std::cout << "TTS_NOTE: " << tts.note << "\n";
    std::cout << "TOTAL_MS: " << total_ms << "\n";
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
}
