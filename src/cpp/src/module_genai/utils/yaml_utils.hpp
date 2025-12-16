// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "module_genai/module.hpp"

namespace ov {
namespace genai {

namespace module {
namespace utils {

std::pair<std::string, std::string> parse_source(const std::string& source);

PipelineModuleDesc load_config(const std::string& cfg_path);

}  // namespace utils
}  // namespace module
}  // namespace genai
}  // namespace ov