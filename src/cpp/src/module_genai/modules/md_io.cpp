// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_io.hpp"

namespace ov {
namespace genai {
namespace module {
ParameterModule::ParameterModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {
    is_input_module = true;
    // std::cout << "ParameterModule:" << m_desc << std::endl;
}

void ParameterModule::run(ov::AnyMap& inputs) {
    std::cout << "Run: " << ModuleTypeConverter::toString(static_cast<ModuleType>(module_desc->type)) << "["
              << module_desc->name << "]" << std::endl;

    for (auto& output : this->outputs) {
        OPENVINO_ASSERT(inputs.find(output.first) != inputs.end(),
                        "Can't find input data:" + output.first);
        output.second.data = inputs[output.first];
        std::cout << "    Pass " << output.first << " to output port" << std::endl;
    }
}

ResultModule::ResultModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {}

void ResultModule::run() {
    std::cout << "Run: " << ModuleTypeConverter::toString(static_cast<ModuleType>(module_desc->type)) << "["
              << module_desc->name << "]" << std::endl;
}

}  // namespace module
}  // namespace genai
}  // namespace ov
