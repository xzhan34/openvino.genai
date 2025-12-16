// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "module_desc.hpp"
#include "module_type.hpp"
#include "openvino/genai/module_genai/pipeline.hpp"

namespace ov {
namespace genai {

namespace module {


class IBaseModuleCom : public IBaseModule {
protected:
    ModuleDesc m_desc;
    using ModuleIOMap = std::pair<IBaseModuleCom::PTR, const std::string>;
    std::vector<ModuleIOMap> inputs;
    std::vector<ModuleIOMap> outputs;

public:
    IBaseModuleCom(const ModuleDesc& desc) : m_desc(desc) {
        std::cout << "IBaseModuleCom:" << m_desc << std::endl;
    }

    virtual bool initialize() = 0;

    virtual void run() = 0;

    virtual std::string get_name() override {
        return m_desc.name;
    }

    const ModuleDesc& get_module_desc() const {
        return m_desc;
    }

    void push_input(IBaseModuleCom::PTR md_ptr, const std::string output_name) {
        inputs.emplace_back(md_ptr, output_name);
    }

    void push_output(IBaseModuleCom::PTR& md_ptr, const std::string& input_name) {
        outputs.emplace_back(md_ptr, input_name);
    }
};
#define PRINT_POS() std::cout << "Run: " << m_desc.type << "[" << m_desc.name << "]" << std::endl

using PipelineModuleDesc = std::unordered_map<std::string, ModuleDesc>;

void construct_pipeline(const PipelineModuleDesc& pipeline_desc, PipelineModuleInstance& pipeline_instance);

PipelineModuleInstance sort_pipeline(PipelineModuleInstance& pipeline_instrance);

}  // namespace module
}  // namespace genai
}  // namespace ov