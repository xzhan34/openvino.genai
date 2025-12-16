// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <thread>

#include "md_img_preprocess.hpp"

namespace ov {
namespace genai {
namespace module {

    ImagePreprocesModule::ImagePreprocesModule(const ModuleDesc::PTR& desc) : IBaseModuleCom(desc) {

    }

    bool ImagePreprocesModule::initialize() {
        return true;
    }

    void ImagePreprocesModule::run() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        PRINT_POS();
        std::this_thread::sleep_for(std::chrono::milliseconds(700));
    }

}  // namespace module
}  // namespace genai
}  // namespace ov
