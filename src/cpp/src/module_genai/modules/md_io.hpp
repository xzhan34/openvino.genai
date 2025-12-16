// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "module_genai/module.hpp"

namespace ov {
namespace genai {
namespace module {

class ParameterModule : public IBaseModuleCom {
protected:
    ParameterModule() = delete;
    ParameterModule(const ModuleDesc::PTR& desc);

public:
    ~ParameterModule() {
        std::cout << "~ParameterModule is called." << std::endl;
    }
    bool initialize() override;

    void run() override;

    using PTR = std::shared_ptr<ParameterModule>;
    static PTR create(const ModuleDesc::PTR& desc) {
        return PTR(new ParameterModule(desc));
    }

};

class ResultModule : public IBaseModuleCom {
protected:
    ResultModule() = delete;
    ResultModule(const ModuleDesc::PTR& desc);

public:
    ~ResultModule() {
        std::cout << "~ResultModule is called." << std::endl;
    }
    bool initialize() override;

    void run() override;

    using PTR = std::shared_ptr<ResultModule>;
    static PTR create(const ModuleDesc::PTR& desc) {
        return PTR(new ResultModule(desc));
    }
};

}  // namespace module
}  // namespace genai
}  // namespace ov
