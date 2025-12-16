// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modules/md_img_preprocess.hpp"
#include "modules/md_text_encoder.hpp"
#include "modules/md_io.hpp"
#include "module.hpp"
#include "utils/yaml_utils.hpp"

namespace ov {
namespace genai {
namespace module {

void module_connect(PipelineModuleInstance& pipeline_instance) {
    std::unordered_map<std::string, IBaseModule::PTR> module_map;
    for (const auto& module_ptr : pipeline_instance) {
        IBaseModuleCom* md_ptr = dynamic_cast<IBaseModuleCom*>(module_ptr.get());
        
        // Process inputs
        for(auto& input : md_ptr->get_module_desc().inputs) {
            auto md_map = utils::parse_source(input.source);
            auto it = std::find_if(std::begin(pipeline_instance),
                                   std::end(pipeline_instance),
                                   [&](const IBaseModule::PTR& ptr) {
                                       return ptr->get_name() == md_map.first;
                                   });
            OPENVINO_ASSERT(it != std::end(pipeline_instance), "Can't find module, please check config yaml.");
            md_ptr->push_input(*it, md_map.second);
        }
    }
}

void construct_pipeline(const PipelineModuleDesc& pipeline_desc, PipelineModuleInstance& pipeline_instance) {
    for (auto& module_desc : pipeline_desc) {
        IBaseModule::PTR module_ptr = nullptr;
        switch (module_desc.second.type) {
        case ModuleType::ParameterModule:
            module_ptr = ParameterModule::create(module_desc.second);
            break;
        case ModuleType::ResultModule:
            module_ptr = ResultModule::create(module_desc.second);
            break;
        case ModuleType::ImagePreprocessModule:
            module_ptr = ImagePreprocesModule::create(module_desc.second);
            break;
        case ModuleType::TextEncoderModule:
            module_ptr = TextEncodeModule::create(module_desc.second);
            break;
        default:
            break;
        }
        OPENVINO_ASSERT(module_ptr, "No implementation for type: " + ModuleTypeConverter::toString(module_desc.second.type));
        pipeline_instance.push_back(module_ptr);
    }
    module_connect(pipeline_instance);
}

}  // namespace module
}  // namespace genai
}  // namespace ov