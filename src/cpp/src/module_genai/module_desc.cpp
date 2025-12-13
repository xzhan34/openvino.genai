// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_desc.hpp"

namespace ov {
namespace genai {

namespace module {

// The full implementation of the stream insertion operator.
std::ostream& operator<<(std::ostream& os, const ModuleDesc& desc) {
    // 1. Output the ModuleType
    os << "-- Module : " << ModuleTypeConverter::toString(desc.type) << " [" << desc.name << "]\n";

    // 2. Output other key fields
    os << "    Device: " << desc.device << "\n";
    os << "    Description: " << desc.description << "\n";

    // 3. Output Inputs and Outputs count
    os << "    Inputs (" << desc.inputs.size() << "):\n";
    for (const auto& input : desc.inputs) {
        // Use std::quoted for safety if values might contain spaces/special chars
        os << "      - name: " << input.name << "\n";
        os << "      - type: " << input.type << "\n";
        os << "      - source: " << input.source << "\n";
    }
    os << "    Onputs (" << desc.outputs.size() << "):\n";
    for (const auto& output : desc.outputs) {
        // Use std::quoted for safety if values might contain spaces/special chars
        os << "      - name: " << output.name << "\n";
        os << "      - type: " << output.type << "\n";
    }

    // 4. Output Parameters
    os << "    Params (" << desc.params.size() << "):\n";
    for (const auto& pair : desc.params) {
        os << "      - " << pair.first << ": " << pair.second << "\n";
    }

    return os;
}

}  // namespace module
}  // namespace genai
}  // namespace ov