// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include "openvino/genai/image_generation/text2image_pipeline.hpp"
#include "openvino/genai/image_generation/image2image_pipeline.hpp"
#include "openvino/genai/image_generation/inpainting_pipeline.hpp"

namespace ov {
namespace genai {
namespace module {

/// @brief Unified generation module supporting multiple modalities
/// Supports: LLM, VLM, Text2Image, Image2Image, Inpainting, Text2Speech
class LLMInferenceModule : public IBaseModule {
protected:
    LLMInferenceModule() = delete;
    LLMInferenceModule(const IBaseModuleDesc::PTR& desc);

public:
    ~LLMInferenceModule() {
        std::cout << "~LLMInferenceModule is called." << std::endl;
    }

    void run() override;

    using PTR = std::shared_ptr<LLMInferenceModule>;
    static PTR create(const IBaseModuleDesc::PTR& desc) {
        return PTR(new LLMInferenceModule(desc));
    }
    static void print_static_config();

private:
    enum class PipelineType {
        LLM,              // Text generation
        VLM,              // Visual language model
        TEXT2IMAGE,       // Text to image generation
        IMAGE2IMAGE,      // Image to image generation  
        INPAINTING,       // Image inpainting
        TEXT2SPEECH       // Text to speech generation
    };
    
    bool initialize();
    bool load_generation_config(const std::string& config_path);
    PipelineType detect_pipeline_type(const std::filesystem::path& models_path);
    
    // Pipeline instances (only one will be initialized based on model type)
    std::shared_ptr<ov::genai::VLMPipeline> m_vlm_pipeline;
    std::shared_ptr<ov::genai::LLMPipeline> m_llm_pipeline;
    std::shared_ptr<ov::genai::Text2SpeechPipeline> m_text2speech_pipeline;
    std::shared_ptr<ov::genai::Text2ImagePipeline> m_text2image_pipeline;
    std::shared_ptr<ov::genai::Image2ImagePipeline> m_image2image_pipeline;
    std::shared_ptr<ov::genai::InpaintingPipeline> m_inpainting_pipeline;
    
    // Generation configurations
    ov::genai::GenerationConfig m_generation_config;
    
    // Pipeline type
    PipelineType m_pipeline_type = PipelineType::LLM;
};

REGISTER_MODULE_CONFIG(LLMInferenceModule);

}  // namespace module
}  // namespace genai
}  // namespace ov
