// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "module_genai/module.hpp"
#include "openvino/genai/module_genai/module_base.hpp"

namespace ov {
namespace genai {
namespace module {

class ParameterModule : public IBaseModule {
protected:
    ParameterModule() = delete;
    ParameterModule(const ModuleDesc& desc, const std::string& name);

public:
    ~ParameterModule() {
        std::cout << "~ParameterModule is called." << std::endl;
    }
    bool initialize() override;

    void run() override;

    using PTR = std::shared_ptr<ParameterModule>;
    static PTR create(const ModuleDesc& desc, const std::string& name) {
        return PTR(new ParameterModule(desc, name));
    }
};

class ResultModule : public IBaseModule {
protected:
    ResultModule() = delete;
    ResultModule(const ModuleDesc& desc, const std::string& name);

public:
    ~ResultModule() {
        std::cout << "~ResultModule is called." << std::endl;
    }
    bool initialize() override;

    void run() override;

    using PTR = std::shared_ptr<ResultModule>;
    static PTR create(const ModuleDesc& desc, const std::string& name) {
        return PTR(new ResultModule(desc, name));
    }
};

}  // namespace module
}  // namespace genai
}  // namespace ov
