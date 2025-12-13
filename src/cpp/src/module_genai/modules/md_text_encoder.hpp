// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "module_genai/module.hpp"
#include "openvino/genai/module_genai/module_base.hpp"

namespace ov {
namespace genai {
namespace module {

class TextEncodeModule : public IBaseModule {
protected:
    TextEncodeModule() = delete;
    TextEncodeModule(const ModuleDesc& desc, const std::string& name);

public:
    ~TextEncodeModule() {
        std::cout << "~TextEncodeModule is called." << std::endl;
    }
    bool initialize() override;

    void run() override;

    using PTR = std::shared_ptr<TextEncodeModule>;
    static PTR create(const ModuleDesc& desc, const std::string& name) {
        return PTR(new TextEncodeModule(desc, name));
    }
};

}  // namespace module
}  // namespace genai
}  // namespace ov
