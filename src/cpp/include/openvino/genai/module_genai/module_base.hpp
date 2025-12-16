// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/visibility.hpp"
#include "openvino/core/any.hpp"

namespace ov {
namespace genai {
namespace module {

struct OutputPort {
    std::string name;
    std::string type;
};

struct InputPort {
    std::string name;
    std::string type;
    std::string source_module_name;
    std::string source_module_out_name;
};

class OPENVINO_GENAI_EXPORTS IBaseModuleDesc {
public:
    virtual ~IBaseModuleDesc() = default;

    std::string name = "Unknown";
    int type = 0;
    std::vector<InputPort> inputs;
    std::vector<OutputPort> outputs;
    std::string device;
    std::string description;
    std::unordered_map<std::string, std::string> params;

    using PTR = std::shared_ptr<IBaseModuleDesc>;
    static PTR create() {
        return std::make_shared<IBaseModuleDesc>();
    }
};

class OPENVINO_GENAI_EXPORTS IBaseModule {
public:
    ~IBaseModule() = default;
    using PTR = std::shared_ptr<IBaseModule>;
    struct OPENVINO_GENAI_EXPORTS InputModule {
        IBaseModule::PTR module_ptr;
        std::string out_port_name;
    };
    struct OPENVINO_GENAI_EXPORTS OutputModule {
        IBaseModule::PTR module_ptr;
        std::string in_port_name;
    };

    IBaseModule() = delete;
    IBaseModule(const IBaseModuleDesc::PTR& desc) : module_desc(desc) {
        std::cout << "Init IBaseModule with module name : " << module_desc->name << std::endl;
    }

    virtual bool initialize() = 0;

    virtual void run() = 0;

    const std::string& get_module_name() const {
        return module_desc->name;
    }

    std::vector<InputModule> inputs;
    std::vector<OutputModule> outputs;
    IBaseModuleDesc::PTR module_desc;
};

}  // namespace module
}  // namespace genai
}  // namespace ov
