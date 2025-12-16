// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"

namespace ov {
namespace genai {
namespace module {

class TextEncoderModule : public IBaseModule {
protected:
    TextEncoderModule() = delete;
    TextEncoderModule(const IBaseModuleDesc::PTR& desc);

public:
    ~TextEncoderModule() {
        std::cout << "~TextEncoderModule is called." << std::endl;
    }
    bool initialize() override;

    void run() override;

    using PTR = std::shared_ptr<TextEncoderModule>;
    static PTR create(const IBaseModuleDesc::PTR& desc) {
        return PTR(new TextEncoderModule(desc));
    }
};

}  // namespace module
}  // namespace genai
}  // namespace ov
