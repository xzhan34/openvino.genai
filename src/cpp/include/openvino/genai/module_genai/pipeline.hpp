// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/module_genai/module_base.hpp"

namespace ov {
namespace genai {

namespace module {

using PipelineModuleInstance = std::vector<IBaseModule::PTR>;

class OPENVINO_GENAI_EXPORTS ModulePipeline {
private:
    PipelineModuleInstance m_modules;

public:
    // config_path: yaml file.
    ModulePipeline(const std::filesystem::path& config_path);

    ~ModulePipeline();

    // input all parameters in config.yaml
    // "prompt": string
    // "image": image ov::Tensor or std::vector<ov::Tensor>
    // "video": video ov::Tensor
    void generate(ov::AnyMap& inputs, StreamerVariant streamer = std::monostate());

    ov::Any get_output(const std::string& output_name);

    void start_chat(const std::string& system_message = {});

    void finish_chat();

private:
    std::map<std::string, ov::Any> outputs;
};

}  // namespace module
}  // namespace genai
}  // namespace ov
