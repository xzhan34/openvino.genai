// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <thread>

#include "md_img_preprocess.hpp"

namespace ov {
namespace genai {
namespace module {

    ImagePreprocesModule::ImagePreprocesModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {

    }

    bool ImagePreprocesModule::initialize() {
        return true;
    }

    void ImagePreprocesModule::run() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::cout << "Run: " << ModuleTypeConverter::toString(static_cast<ModuleType>(module_desc->type)) << "["
                  << module_desc->name << "]" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(700));
    }

}  // namespace module
}  // namespace genai
}  // namespace ov
