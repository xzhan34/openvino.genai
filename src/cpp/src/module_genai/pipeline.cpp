// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/module_genai/pipeline.hpp"

#include <optional>

#include "module.hpp"
#include "utils/yaml_utils.hpp"
#include "modules/md_io.hpp"

namespace ov {
namespace genai {

namespace module {

// config_path: yaml file.
ModulePipeline::ModulePipeline(const std::filesystem::path& config_path) {
    auto pipeline_desc = utils::load_config(config_path);

    // Construct pipeline
    construct_pipeline(pipeline_desc, m_modules);

    // Sort pipeline
    m_modules = sort_pipeline(m_modules);
}

ModulePipeline::~ModulePipeline() {}

// input all parameters in config.yaml
// "prompt": string
// "image": image ov::Tensor or std::vector<ov::Tensor>
// "video": video ov::Tensor
void ModulePipeline::generate(ov::AnyMap& inputs, StreamerVariant streamer) {
    for (auto& module : m_modules) {
        if (module->is_input_module) {
            std::dynamic_pointer_cast<ParameterModule>(module)->run(inputs);
        } else {
            module->run();
        }
    }
}

ov::Any ModulePipeline::get_output(const std::string& output_name) {
    return ov::Any();
}

void ModulePipeline::start_chat(const std::string& system_message) {}

void ModulePipeline::finish_chat() {}

}  // namespace module
}  // namespace genai
}  // namespace ov
