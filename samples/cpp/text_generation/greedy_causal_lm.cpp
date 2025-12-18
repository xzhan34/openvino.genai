// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (3 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\" [DEVICE] [NUM_WARMUP] [NUM_ITER] [MAX_NEW_TOKENS]");

    std::string models_path = argv[1];
    std::string prompt = argv[2];
    std::string device = argc > 3 ? argv[3] : "GPU";  // GPU can be used as well
    std::size_t num_warmup = argc > 4 ? std::stoul(argv[4]) : 1;
    std::size_t num_iter = argc > 5 ? std::stoul(argv[5]) : 3;
    std::size_t max_new_tokens = argc > 6 ? std::stoul(argv[6]) : 100;
    num_iter = std::max<std::size_t>(num_iter, 1);

    ov::genai::LLMPipeline pipe(models_path, device);
    ov::genai::GenerationConfig config;
    config.max_new_tokens = max_new_tokens;
    config.apply_chat_template = false;

    auto input_data = pipe.get_tokenizer().encode(prompt);
    std::size_t prompt_token_size = input_data.input_ids.get_shape()[1];
    std::cout << "Prompt token size: " << prompt_token_size << std::endl;

    for (std::size_t i = 0; i < num_warmup; ++i)
        pipe.generate(prompt, config);

    ov::genai::DecodedResults res = pipe.generate(prompt, config);
    ov::genai::PerfMetrics metrics = res.perf_metrics;
    for (std::size_t i = 0; i + 1 < num_iter; ++i) {
        res = pipe.generate(prompt, config);
        metrics = metrics + res.perf_metrics;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Output token size: " << res.perf_metrics.get_num_generated_tokens() << std::endl;
    std::cout << "Load time: " << metrics.get_load_time() << " ms" << std::endl;
    std::cout << "Generate time: " << metrics.get_generate_duration().mean << " ± " << metrics.get_generate_duration().std << " ms" << std::endl;
    std::cout << "Tokenization time: " << metrics.get_tokenization_duration().mean << " ± " << metrics.get_tokenization_duration().std << " ms" << std::endl;
    std::cout << "Detokenization time: " << metrics.get_detokenization_duration().mean << " ± " << metrics.get_detokenization_duration().std << " ms" << std::endl;
    std::cout << "TTFT: " << metrics.get_ttft().mean  << " ± " << metrics.get_ttft().std << " ms" << std::endl;
    std::cout << "TPOT: " << metrics.get_tpot().mean  << " ± " << metrics.get_tpot().std << " ms/token " << std::endl;
    std::cout << "Throughput: " << metrics.get_throughput().mean  << " ± " << metrics.get_throughput().std << " tokens/s" << std::endl;
    std::cout << "Generated text: " << res << std::endl;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
