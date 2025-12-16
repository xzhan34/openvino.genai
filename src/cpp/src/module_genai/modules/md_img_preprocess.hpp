// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "module_genai/module.hpp"

namespace ov {
namespace genai {
namespace module {

class ImagePreprocesModule : public IBaseModuleCom {
protected:
    ImagePreprocesModule() = delete;
    ImagePreprocesModule(const ModuleDesc::PTR& desc);

public:
    ~ImagePreprocesModule() {
        std::cout << "~ImagePreprocesModule is called." << std::endl;
    }
    bool initialize() override;

    void run() override;

    using PTR = std::shared_ptr<ImagePreprocesModule>;
    static PTR create(const ModuleDesc::PTR& desc) {
        return PTR(new ImagePreprocesModule(desc));
    }
};

}  // namespace module
}  // namespace genai
}  // namespace ov
