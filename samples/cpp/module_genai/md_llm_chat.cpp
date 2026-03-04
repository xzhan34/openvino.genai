// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <openvino/genai/module_genai/pipeline.hpp>

#include <stdexcept>

#include "yaml-cpp/yaml.h"
#include "utils/utils.hpp"

inline ov::AnyMap parse_inputs_from_yaml_cfg(const std::filesystem::path& cfg_yaml_path,
                                             const std::string& prompt) {
    ov::AnyMap inputs;
    YAML::Node input_params = utils::find_param_module_in_yaml(cfg_yaml_path);

    for (const auto& entry : input_params) {
        if (!entry["name"] || !entry["type"]) {
            continue;
        }

        const std::string param_name = entry["name"].as<std::string>();
        const std::string param_type = entry["type"].as<std::string>();

        if (param_type == "String" && utils::contains_key(param_name, {"prompt"})) {
            if (prompt.empty()) {
                throw std::runtime_error("Prompt string is empty.");
            }
            inputs[param_name] = prompt;
        }
    }
    return inputs;
}

int main(int argc, char* argv[]) {
    try {
        if (argc <= 1) {
            throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                     "\n"
                                     "  -cfg config.yaml\n"
                                     "  -prompt: input prompt\n");
        }

        std::filesystem::path config_path = utils::get_input_arg(argc, argv, "-cfg", std::string{});
        std::string prompt = utils::get_input_arg(argc, argv, "-prompt", std::string{});

        ov::AnyMap inputs = parse_inputs_from_yaml_cfg(config_path, prompt);

        for (const auto& [key, value] : inputs) {
            std::cout << "[Input] " << key << ": " << value.as<std::string>() << std::endl;
        }

        ov::genai::module::ModulePipeline pipe(config_path);

        pipe.generate(inputs);

        std::cout << "Generation Result: " << pipe.get_output("generated_text").as<std::string>() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
