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

using PipelineModuleInstance = std::vector<IBaseModule::PTR>;

PipelineModuleInstance construct_pipeline(const PipelineModuleDesc& pipeline_desc);

void build_pipeline(PipelineModuleInstance& pipeline_instrance);

PipelineModuleInstance sort_pipeline(PipelineModuleInstance& pipeline_instrance);

}  // namespace module
}  // namespace genai
}  // namespace ov