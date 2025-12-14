// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/genai/module_genai/pipeline.hpp"
#include "module_type.hpp"

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
    std::string source;
};

struct ModuleDesc {
    ModuleType type = ModuleType::Unknown;
    std::string name = "Unknown";
    std::string device;
    std::string description;
    std::vector<InputPort> inputs;
    std::vector<OutputPort> outputs;
    std::unordered_map<std::string, std::string> params;
};
std::ostream& operator<<(std::ostream& os, const ModuleDesc& desc);

}  // namespace module
}  // namespace genai
}  // namespace ov