// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "module_genai/module.hpp"

namespace ov {
namespace genai {
namespace module {

class TextEncodeModule : public IBaseModuleCom {
protected:
    TextEncodeModule() = delete;
    TextEncodeModule(const ModuleDesc& desc);

public:
    ~TextEncodeModule() {
        std::cout << "~TextEncodeModule is called." << std::endl;
    }
    bool initialize() override;

    void run() override;

    using PTR = std::shared_ptr<TextEncodeModule>;
    static PTR create(const ModuleDesc& desc) {
        return PTR(new TextEncodeModule(desc));
    }
};

}  // namespace module
}  // namespace genai
}  // namespace ov
