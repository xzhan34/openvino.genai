// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_io.hpp"

namespace ov {
namespace genai {
namespace module {
ParameterModule::ParameterModule(const ModuleDesc& desc) : IBaseModuleCom(desc) {
    std::cout << "ParameterModule:" << m_desc << std::endl;
}

void ParameterModule::run() {
    PRINT_POS();
}

bool ParameterModule::initialize() {
    return true;
}

ResultModule::ResultModule(const ModuleDesc& desc) : IBaseModuleCom(desc) {}

void ResultModule::run() {
    PRINT_POS();
}

bool ResultModule::initialize() {
    return true;
}

}  // namespace module
}  // namespace genai
}  // namespace ov
