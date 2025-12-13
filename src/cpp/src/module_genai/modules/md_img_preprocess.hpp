// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "module_genai/module.hpp"
#include "openvino/genai/module_genai/module_base.hpp"

namespace ov {
namespace genai {
namespace module {

class ImagePreprocesModule : public IBaseModule {
protected:
    ImagePreprocesModule() = delete;
    ImagePreprocesModule(const ModuleDesc& desc, const std::string& name);

public:
    bool initialize() override;

    void run() override;

    using PTR = std::shared_ptr<ImagePreprocesModule>;
    static PTR create(const ModuleDesc& desc, const std::string& name) {
        return PTR(new ImagePreprocesModule(desc, name));
    }
};

}  // namespace module
}  // namespace genai
}  // namespace ov
