// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modules/md_img_preprocess.hpp"
#include "modules/md_text_encoder.hpp"
#include "module.hpp"

namespace ov {
namespace genai {
namespace module {

void build_pipeline(PipelineModuleInstance& pipeline_instrance) {
    // Build relationship of all modules.
    for (auto& module : pipeline_instrance) {
        
    }
}

}  // namespace module
}  // namespace genai
}  // namespace ov