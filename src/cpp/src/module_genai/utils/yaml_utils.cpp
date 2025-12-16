// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/utils/yaml_utils.hpp"

#include <yaml-cpp/yaml.h>

namespace ov {
namespace genai {

namespace module {
namespace utils {

InputPort parse_input_port(const YAML::Node& node, bool is_input) {
    InputPort port;
    if (node["name"]) {
        port.name = node["name"].as<std::string>();
    }
    if (node["type"]) {
        port.type = node["type"].as<std::string>();
    }
    if (is_input && node["source"]) {
        port.source = node["source"].as<std::string>();
    }
    return port;
}

OutputPort parse_output_port(const YAML::Node& node, bool is_input) {
    OutputPort port;
    if (node["name"]) {
        port.name = node["name"].as<std::string>();
    }
    if (node["type"]) {
        port.type = node["type"].as<std::string>();
    }
    return port;
}

ModuleDesc parse_module(const YAML::Node& node) {
    ModuleDesc module;

    if (node["type"]) {
        std::string md_type = node["type"].as<std::string>();
        module.type = ModuleTypeConverter::fromString(md_type);
        OPENVINO_ASSERT(module.type != ModuleType::Unknown, "Unknown ModuleType string: " + md_type);
    }

    if (node["device"])
        module.device = node["device"].as<std::string>();
    if (node["description"])
        module.description = node["description"].as<std::string>();

    if (node["inputs"] && node["inputs"].IsSequence()) {
        for (const auto& input_node : node["inputs"]) {
            module.inputs.push_back(parse_input_port(input_node, true));
        }
    }

    if (node["outputs"] && node["outputs"].IsSequence()) {
        for (const auto& output_node : node["outputs"]) {
            module.outputs.push_back(parse_output_port(output_node, false));
        }
    }

    if (node["params"] && node["params"].IsMap()) {
        for (YAML::const_iterator it = node["params"].begin(); it != node["params"].end(); ++it) {
            module.params[it->first.as<std::string>()] =
                it->second.IsScalar() ? it->second.as<std::string>() : "[Complex Value]";
        }
    }

    return module;
}

std::pair<std::string, std::string> parse_source(const std::string& source) {
    size_t dot_pos = source.find('.');
    OPENVINO_ASSERT(dot_pos != std::string::npos, "Source string doesn't contain '.'");

    std::string part1 = source.substr(0, dot_pos);
    std::string part2 = source.substr(dot_pos + 1);
    return {part1, part2};
}

PipelineModuleDesc load_config(const std::string& cfg_path) {
    PipelineModuleDesc pipeline_desc;
    try {
        YAML::Node config = YAML::LoadFile(cfg_path);

        const YAML::Node& global = config["global_context"];
        if (global) {
            std::string device = global["default_device"] ? global["default_device"].as<std::string>() : "N/A";
            bool shared_mem = global["enable_shared_memory"] ? global["enable_shared_memory"].as<bool>() : false;

            std::cout << "  Default Device: " << device << std::endl;
            std::cout << "  Enable Shared Memory: " << (shared_mem ? "True" : "False") << std::endl;
        }

        std::cout << "\n" << std::endl;
        std::cout << "#### 🧩 Pipeline Modules #####################" << std::endl;

        const YAML::Node& modules_node = config["pipeline_modules"];
        if (modules_node && modules_node.IsMap()) {
            for (YAML::const_iterator it = modules_node.begin(); it != modules_node.end(); ++it) {
                std::string module_name = it->first.as<std::string>();
                const YAML::Node& module_config = it->second;

                ModuleDesc module = parse_module(module_config);
                module.name = module_name;
                pipeline_desc[module_name] = module;

                std::cout << module << std::endl;
            }
        } else {
            std::cout << "Error: 'pipeline_modules' key not found or is not a map." << std::endl;
        }

    } catch (const YAML::BadFile& e) {
        std::cerr << "Error: Could not find or open 'config.yaml'. Please make sure the file exists." << std::endl;
    } catch (const YAML::Exception& e) {
        std::cerr << "Error parsing YAML: " << e.what() << std::endl;
    }
    return pipeline_desc;
}
}  // namespace utils
}  // namespace module
}  // namespace genai
}  // namespace ov