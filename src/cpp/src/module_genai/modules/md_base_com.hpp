// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "module_genai/module.hpp"
#include "openvino/genai/module_genai/module_base.hpp"
#include "module_genai/module_desc.hpp"

namespace ov {
namespace genai {
namespace module {

class IBaseModuleCom : public IBaseModule {
protected:
    ModuleDesc m_desc;

public:
    IBaseModuleCom(const ModuleDesc& desc) : m_desc(desc) {
        std::cout << "IBaseModuleCom:" << m_desc << std::endl;
    }

    virtual bool initialize() = 0;

    virtual void run() = 0;
};

#define PRINT_POS() std::cout << "Run: " << m_desc.type << "[" << m_desc.name << "]" << std::endl

}  // namespace module
}  // namespace genai
}  // namespace ov
