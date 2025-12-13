// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modules/md_img_preprocess.hpp"
#include "modules/md_text_encoder.hpp"
#include "module.hpp"

namespace ov {
namespace genai {
namespace module {

PipelineModuleInstance construct_pipeline(const PipelineModuleDesc& pipeline_desc) {
    PipelineModuleInstance pipeline_instance;

    for (auto& module_desc : pipeline_desc) {
        IBaseModule::PTR module_ptr = nullptr;
        switch (module_desc.second.type) {
        case ModuleType::ImagePreprocessModule:
            module_ptr = ImagePreprocesModule::create(module_desc.second, module_desc.first);
            break;
        case ModuleType::TextEncoderModule:
            module_ptr = TextEncodeModule::create(module_desc.second, module_desc.first);
            break;
        default:
            break;
        }
        OPENVINO_ASSERT(module_ptr, "No implementation for type: " + ModuleTypeConverter::toString(module_desc.second.type));
        pipeline_instance.push_back(module_ptr);
    }

    return pipeline_instance;
}

}  // namespace module
}  // namespace genai
}  // namespace ov