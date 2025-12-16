// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_text_encoder.hpp"

#include <chrono>
#include <thread>

namespace ov {
namespace genai {
namespace module {

TextEncodeModule::TextEncodeModule(const ModuleDesc::PTR& desc) : IBaseModuleCom(desc) {}

bool TextEncodeModule::initialize() {
    return true;
}

void TextEncodeModule::run() {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    PRINT_POS();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
}

}  // namespace module
}  // namespace genai
}  // namespace ov
