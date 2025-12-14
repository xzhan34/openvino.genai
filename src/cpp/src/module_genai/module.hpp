// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "module_desc.hpp"
#include "module_type.hpp"
#include "openvino/genai/module_genai/pipeline.hpp"

namespace ov {
namespace genai {

namespace module {

using PipelineModuleDesc = std::unordered_map<std::string, ModuleDesc>;

void construct_pipeline(const PipelineModuleDesc& pipeline_desc, PipelineModuleInstance& pipeline_instance);

PipelineModuleInstance sort_pipeline(PipelineModuleInstance& pipeline_instrance);

}  // namespace module
}  // namespace genai
}  // namespace ov