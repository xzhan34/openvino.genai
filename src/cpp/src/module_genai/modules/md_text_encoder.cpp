// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_text_encoder.hpp"

#include <chrono>
#include <thread>

namespace ov {
namespace genai {
namespace module {

TextEncoderModule::TextEncoderModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {
    
}

bool TextEncoderModule::initialize() {
    return true;
}

void TextEncoderModule::run() {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::cout << "Run: " << ModuleTypeConverter::toString(static_cast<ModuleType>(module_desc->type)) << "["
              << module_desc->name << "]" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
}

}  // namespace module
}  // namespace genai
}  // namespace ov
