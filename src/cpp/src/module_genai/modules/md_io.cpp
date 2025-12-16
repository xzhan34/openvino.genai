// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_io.hpp"

namespace ov {
namespace genai {
namespace module {
ParameterModule::ParameterModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {
    // std::cout << "ParameterModule:" << m_desc << std::endl;
}

void ParameterModule::run() {
    std::cout << "Run: " << ModuleTypeConverter::toString(static_cast<ModuleType>(module_desc->type)) << "["
              << module_desc->name << "]" << std::endl;
}

bool ParameterModule::initialize() {
    return true;
}

ResultModule::ResultModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {}

void ResultModule::run() {
    std::cout << "Run: " << ModuleTypeConverter::toString(static_cast<ModuleType>(module_desc->type)) << "["
              << module_desc->name << "]" << std::endl;
}

bool ResultModule::initialize() {
    return true;
}

}  // namespace module
}  // namespace genai
}  // namespace ov
