// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_llm_inference.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

namespace ov {
namespace genai {
namespace module {

void LLMInferenceModule::print_static_config() {
    std::cout << R"(
  llm_inference:                      # Module Name
    type: "LLMInferenceModule"
    description: "unified generation module supporting multiple modalities: LLM, VLM, Text2Image, Image2Image, Inpainting, Text2Speech"
    device: "GPU"                     # Target device: CPU, GPU, NPU
    inputs:
      # Text Generation (LLM/VLM)
      - name: "prompt"
        type: "string"                # Text prompt
        source: "user_input.prompt"
      - name: "images"
        type: "OVTensorVector"        # Optional: images for VLM/Image2Image/Inpainting
        source: "image_processor.images"
      - name: "videos"
        type: "OVTensorVector"        # Optional: videos for VLM
        source: "video_processor.videos"
      # Image Generation specific
      - name: "image_prompt"
        type: "string"                # Text prompt for image generation
        source: "user_input.image_prompt"
      - name: "init_image"
        type: "OVTensor"              # Initial image for Image2Image/Inpainting
        source: "image_loader.image"
      - name: "mask_image"
        type: "OVTensor"              # Mask for Inpainting
        source: "mask_generator.mask"
      # Speech Generation specific
      - name: "text"
        type: "string"                # Text for speech generation
        source: "user_input.text"
      - name: "speaker_embedding"
        type: "OVTensor"              # Optional: speaker embedding
        source: "speaker_encoder.embedding"
      # Common
      - name: "generation_config"
        type: "GenerationConfig/ImageGenerationConfig/SpeechGenerationConfig"
        source: "config_loader.generation_config"
    outputs:
      # Text outputs
      - name: "generated_text"
        type: "string"                # Generated text (LLM/VLM)
      - name: "decoded_results"
        type: "DecodedResults"        # Full text results
      # Image outputs
      - name: "generated_images"
        type: "OVTensorVector"        # Generated images (Text2Image/Image2Image/Inpainting)
      # Audio outputs
      - name: "generated_speeches"
        type: "OVTensorVector"        # Generated audio (Text2Speech)
      # Metrics
      - name: "perf_metrics"
        type: "PerfMetrics"           # Performance metrics
    params:
      models_path: "path/to/model_dir"                    # Model directory (required)
      pipeline_type: "auto"                               # "llm", "vlm", "text2image", "image2image", "inpainting", "text2speech", or "auto" (auto-detect)
      generation_config_path: ""                          # Optional: external config file
      # Text generation parameters
      max_new_tokens: "256"
      do_sample: "false"
      top_p: "1.0"
      top_k: "50"
      temperature: "1.0"
      repetition_penalty: "1.0"
      # Image generation parameters
      num_inference_steps: "50"                           # Denoising steps for diffusion models
      guidance_scale: "7.5"                               # CFG scale
      height: "512"
      width: "512"
      # Chat mode (LLM/VLM)
      enable_chat_mode: "false"
      system_message: ""
      chat_template: ""
      apply_chat_template: "true"
    )" << std::endl;
}

LLMInferenceModule::LLMInferenceModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {
    if (!initialize()) {
        std::cerr << "Failed to initialize LLMInferenceModule" << std::endl;
    }
}

LLMInferenceModule::PipelineType LLMInferenceModule::detect_pipeline_type(const std::filesystem::path& models_path) {
    // Check for image generation models (diffusion models)
    if (std::filesystem::exists(models_path / "unet" / "openvino_model.xml") ||
        std::filesystem::exists(models_path / "unet.xml")) {
        // Check for inpainting-specific files
        if (std::filesystem::exists(models_path / "config.json")) {
            std::ifstream config_file(models_path / "config.json");
            if (config_file.is_open()) {
                nlohmann::json config = nlohmann::json::parse(config_file);
                if (config.contains("_class_name")) {
                    std::string class_name = config["_class_name"];
                    if (class_name.find("Inpaint") != std::string::npos) {
                        return PipelineType::INPAINTING;
                    }
                    if (class_name.find("Img2Img") != std::string::npos || 
                        class_name.find("Image2Image") != std::string::npos) {
                        return PipelineType::IMAGE2IMAGE;
                    }
                }
            }
        }
        return PipelineType::TEXT2IMAGE;
    }
    
    // Check for speech generation models
    if (std::filesystem::exists(models_path / "openvino_encoder_model.xml") &&
        std::filesystem::exists(models_path / "openvino_decoder_model.xml") &&
        std::filesystem::exists(models_path / "openvino_vocoder.xml")) {
        return PipelineType::TEXT2SPEECH;
    }
    
    // Check for visual language models
    if (std::filesystem::exists(models_path / "openvino_vision_embeddings_model.xml")) {
        return PipelineType::VLM;
    }
    
    // Default to LLM
    return PipelineType::LLM;
}

bool LLMInferenceModule::load_generation_config(const std::string& config_path) {
    try {
        std::ifstream f(config_path);
        if (!f.is_open()) {
            std::cerr << "Failed to open generation config file: " << config_path << std::endl;
            return false;
        }
        nlohmann::json config_json = nlohmann::json::parse(f);
        
        // Load common generation parameters
        if (config_json.contains("max_new_tokens")) {
            m_generation_config.max_new_tokens = config_json["max_new_tokens"].get<size_t>();
        }
        if (config_json.contains("do_sample")) {
            m_generation_config.do_sample = config_json["do_sample"].get<bool>();
        }
        if (config_json.contains("top_p")) {
            m_generation_config.top_p = config_json["top_p"].get<float>();
        }
        if (config_json.contains("top_k")) {
            m_generation_config.top_k = config_json["top_k"].get<size_t>();
        }
        if (config_json.contains("temperature")) {
            m_generation_config.temperature = config_json["temperature"].get<float>();
        }
        if (config_json.contains("repetition_penalty")) {
            m_generation_config.repetition_penalty = config_json["repetition_penalty"].get<float>();
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading generation config: " << e.what() << std::endl;
        return false;
    }
}

bool LLMInferenceModule::initialize() {
    const auto& params = module_desc->params;
    
    // Get model path (required)
    auto it_models_path = params.find("models_path");
    if (it_models_path == params.end()) {
        std::cerr << "LLMInferenceModule[" << module_desc->name << "]: 'models_path' not found in params" << std::endl;
        return false;
    }
    std::filesystem::path models_path = it_models_path->second;
    
    // Get device
    std::string device = module_desc->device.empty() ? "GPU" : module_desc->device;
    
    // Prepare properties
    ov::AnyMap properties{};
    
    // Load generation config from file if specified
    auto it_config_path = params.find("generation_config_path");
    if (it_config_path != params.end() && !it_config_path->second.empty()) {
        load_generation_config(it_config_path->second);
    }
    
    // Override with parameters from module config
    auto apply_param = [&](const std::string& key, auto& target, auto converter) {
        auto it = params.find(key);
        if (it != params.end() && !it->second.empty()) {
            try {
                target = converter(it->second);
            } catch (...) {
                std::cerr << "Failed to parse parameter: " << key << std::endl;
            }
        }
    };
    
    apply_param("max_new_tokens", m_generation_config.max_new_tokens, 
                [](const std::string& s) { return std::stoull(s); });
    apply_param("do_sample", m_generation_config.do_sample,
                [](const std::string& s) { return s == "true" || s == "1"; });
    apply_param("top_p", m_generation_config.top_p,
                [](const std::string& s) { return std::stof(s); });
    apply_param("top_k", m_generation_config.top_k,
                [](const std::string& s) { return std::stoull(s); });
    apply_param("temperature", m_generation_config.temperature,
                [](const std::string& s) { return std::stof(s); });
    apply_param("repetition_penalty", m_generation_config.repetition_penalty,
                [](const std::string& s) { return std::stof(s); });
    apply_param("apply_chat_template", m_generation_config.apply_chat_template,
                [](const std::string& s) { return s == "true" || s == "1"; });
    
    // Determine pipeline type
    auto it_pipeline_type = params.find("pipeline_type");
    if (it_pipeline_type != params.end() && it_pipeline_type->second != "auto") {
        std::string type_str = it_pipeline_type->second;
        if (type_str == "llm") m_pipeline_type = PipelineType::LLM;
        else if (type_str == "vlm") m_pipeline_type = PipelineType::VLM;
        else if (type_str == "text2image") m_pipeline_type = PipelineType::TEXT2IMAGE;
        else if (type_str == "image2image") m_pipeline_type = PipelineType::IMAGE2IMAGE;
        else if (type_str == "inpainting") m_pipeline_type = PipelineType::INPAINTING;
        else if (type_str == "text2speech") m_pipeline_type = PipelineType::TEXT2SPEECH;
        else {
            std::cerr << "Unknown pipeline_type: " << type_str << ", falling back to auto-detect" << std::endl;
            m_pipeline_type = detect_pipeline_type(models_path);
        }
    } else {
        // Auto-detect pipeline type
        m_pipeline_type = detect_pipeline_type(models_path);
    }
    
    try {
        switch (m_pipeline_type) {
        case PipelineType::VLM:
        {
            std::cout << "Initializing VLM Pipeline..." << std::endl;
            m_vlm_pipeline = std::make_shared<ov::genai::VLMPipeline>(models_path, device, properties);
            m_vlm_pipeline->set_generation_config(m_generation_config);
            
            // Apply chat template if specified
            auto it_chat_template = params.find("chat_template");
            if (it_chat_template != params.end() && !it_chat_template->second.empty()) {
                m_vlm_pipeline->set_chat_template(it_chat_template->second);
            }
            
            // Start chat mode if enabled
            auto it_enable_chat = params.find("enable_chat_mode");
            if (it_enable_chat != params.end() && (it_enable_chat->second == "true" || it_enable_chat->second == "1")) {
                std::string system_message = "";
                auto it_system_msg = params.find("system_message");
                if (it_system_msg != params.end()) {
                    system_message = it_system_msg->second;
                }
                m_vlm_pipeline->start_chat(system_message);
            }
            break;
        }
            
        case PipelineType::LLM:
        {
            std::cout << "Initializing LLM Pipeline..." << std::endl;
            m_llm_pipeline = std::make_shared<ov::genai::LLMPipeline>(models_path, device, properties);
            m_llm_pipeline->set_generation_config(m_generation_config);
            
            // Apply chat template if specified (LLMPipeline uses tokenizer.set_chat_template)
            auto it_chat_template = params.find("chat_template");
            if (it_chat_template != params.end() && !it_chat_template->second.empty()) {
                m_llm_pipeline->get_tokenizer().set_chat_template(it_chat_template->second);
            }
            
            // Start chat mode if enabled
            auto it_enable_chat = params.find("enable_chat_mode");
            if (it_enable_chat != params.end() && (it_enable_chat->second == "true" || it_enable_chat->second == "1")) {
                std::string system_message = "";
                auto it_system_msg = params.find("system_message");
                if (it_system_msg != params.end()) {
                    system_message = it_system_msg->second;
                }
                m_llm_pipeline->start_chat(system_message);
            }
            break;
        }
            
        case PipelineType::TEXT2IMAGE:
        {
            std::cout << "Initializing Text2Image Pipeline..." << std::endl;
            m_text2image_pipeline = std::make_shared<ov::genai::Text2ImagePipeline>(models_path, device, properties);
            break;
        }
            
        case PipelineType::IMAGE2IMAGE:
        {
            std::cout << "Initializing Image2Image Pipeline..." << std::endl;
            m_image2image_pipeline = std::make_shared<ov::genai::Image2ImagePipeline>(models_path, device, properties);
            break;
        }
            
        case PipelineType::INPAINTING:
        {
            std::cout << "Initializing Inpainting Pipeline..." << std::endl;
            m_inpainting_pipeline = std::make_shared<ov::genai::InpaintingPipeline>(models_path, device, properties);
            break;
        }
            
        case PipelineType::TEXT2SPEECH:
        {
            std::cout << "Initializing Text2Speech Pipeline..." << std::endl;
            m_text2speech_pipeline = std::make_shared<ov::genai::Text2SpeechPipeline>(models_path, device, properties);
            break;
        }
        }
        
        const char* type_names[] = {"LLM", "VLM", "Text2Image", "Image2Image", "Inpainting", "Text2Speech"};
        std::cout << "LLMInferenceModule[" << module_desc->name << "] initialized successfully (" 
                  << type_names[static_cast<int>(m_pipeline_type)] << " mode)" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "LLMInferenceModule[" << module_desc->name << "]: Failed to create pipeline: " 
                  << e.what() << std::endl;
        return false;
    }
}

void LLMInferenceModule::run() {
    std::cout << "Run: " << ModuleTypeConverter::toString(static_cast<ModuleType>(module_desc->type)) << "["
              << module_desc->name << "]" << std::endl;

    prepare_inputs();
    
    switch (m_pipeline_type) {
    case PipelineType::VLM:
    case PipelineType::LLM:
    {
        OPENVINO_ASSERT(m_vlm_pipeline || m_llm_pipeline, 
                        "Text generation pipeline is not initialized.");
        
        // Get prompt (required)
        OPENVINO_ASSERT(this->inputs.count("prompt") != 0,
                        "LLMInferenceModule expects 'prompt' in its inputs.");
        std::string prompt = this->inputs["prompt"].data.as<std::string>();
        
        // Get optional generation config override
        ov::genai::GenerationConfig runtime_config = m_generation_config;
        if (this->inputs.count("generation_config") != 0) {
            runtime_config = this->inputs["generation_config"].data.as<ov::genai::GenerationConfig>();
        }
        
        if (m_pipeline_type == PipelineType::VLM && m_vlm_pipeline) {
        // VLM mode: handle images and videos
        std::vector<ov::Tensor> images;
        std::vector<ov::Tensor> videos;
        
        if (this->inputs.count("images") != 0) {
            images = this->inputs["images"].data.as<std::vector<ov::Tensor>>();
        }
        
        if (this->inputs.count("videos") != 0) {
            videos = this->inputs["videos"].data.as<std::vector<ov::Tensor>>();
        }
        
        // Generate
        ov::genai::VLMDecodedResults results;
        if (!images.empty() && !videos.empty()) {
            results = m_vlm_pipeline->generate(prompt, images, videos, runtime_config, std::monostate{});
        } else if (!images.empty()) {
            results = m_vlm_pipeline->generate(prompt, images, runtime_config, std::monostate{});
        } else {
            // Text only
            results = m_vlm_pipeline->generate(prompt, std::vector<ov::Tensor>{}, runtime_config, std::monostate{});
        }
        
        // Set outputs
        this->outputs["generated_text"].data = results.texts.empty() ? std::string("") : results.texts[0];
        this->outputs["decoded_results"].data = results;
        this->outputs["perf_metrics"].data = results.perf_metrics;
        
        } else if (m_llm_pipeline) {
            // LLM mode: text only
            ov::genai::DecodedResults results = m_llm_pipeline->generate(prompt, runtime_config, std::monostate{});
            
            // Set outputs
            this->outputs["generated_text"].data = results.texts.empty() ? std::string("") : results.texts[0];
            this->outputs["decoded_results"].data = results;
            this->outputs["perf_metrics"].data = results.perf_metrics;
        }
        break;
    }
    
    case PipelineType::TEXT2IMAGE:
    {
        OPENVINO_ASSERT(m_text2image_pipeline, "Text2Image pipeline is not initialized.");
        OPENVINO_ASSERT(this->inputs.count("image_prompt") != 0,
                        "Text2Image expects 'image_prompt' input.");
        
        std::string prompt = this->inputs["image_prompt"].data.as<std::string>();
        ov::Tensor generated_image = m_text2image_pipeline->generate(prompt);
        
        // Wrap single tensor in vector for consistency
        std::vector<ov::Tensor> images_vector = {generated_image};
        this->outputs["generated_images"].data = images_vector;
        this->outputs["perf_metrics"].data = m_text2image_pipeline->get_performance_metrics();
        break;
    }
    
    case PipelineType::IMAGE2IMAGE:
    {
        OPENVINO_ASSERT(m_image2image_pipeline, "Image2Image pipeline is not initialized.");
        OPENVINO_ASSERT(this->inputs.count("image_prompt") != 0 && this->inputs.count("init_image") != 0,
                        "Image2Image expects 'image_prompt' and 'init_image' inputs.");
        
        std::string prompt = this->inputs["image_prompt"].data.as<std::string>();
        ov::Tensor init_image = this->inputs["init_image"].data.as<ov::Tensor>();
        
        ov::Tensor generated_image = m_image2image_pipeline->generate(prompt, init_image);
        
        std::vector<ov::Tensor> images_vector = {generated_image};
        this->outputs["generated_images"].data = images_vector;
        this->outputs["perf_metrics"].data = m_image2image_pipeline->get_performance_metrics();
        break;
    }
    
    case PipelineType::INPAINTING:
    {
        OPENVINO_ASSERT(m_inpainting_pipeline, "Inpainting pipeline is not initialized.");
        OPENVINO_ASSERT(this->inputs.count("image_prompt") != 0 && 
                       this->inputs.count("init_image") != 0 &&
                       this->inputs.count("mask_image") != 0,
                        "Inpainting expects 'image_prompt', 'init_image', and 'mask_image' inputs.");
        
        std::string prompt = this->inputs["image_prompt"].data.as<std::string>();
        ov::Tensor init_image = this->inputs["init_image"].data.as<ov::Tensor>();
        ov::Tensor mask_image = this->inputs["mask_image"].data.as<ov::Tensor>();
        
        ov::Tensor generated_image = m_inpainting_pipeline->generate(prompt, init_image, mask_image);
        
        std::vector<ov::Tensor> images_vector = {generated_image};
        this->outputs["generated_images"].data = images_vector;
        this->outputs["perf_metrics"].data = m_inpainting_pipeline->get_performance_metrics();
        break;
    }
    
    case PipelineType::TEXT2SPEECH:
    {
        OPENVINO_ASSERT(m_text2speech_pipeline, "Text2Speech pipeline is not initialized.");
        OPENVINO_ASSERT(this->inputs.count("text") != 0,
                        "Text2Speech expects 'text' input.");
        
        std::string text = this->inputs["text"].data.as<std::string>();
        ov::Tensor speaker_embedding;
        
        if (this->inputs.count("speaker_embedding") != 0) {
            speaker_embedding = this->inputs["speaker_embedding"].data.as<ov::Tensor>();
        }
        
        auto results = m_text2speech_pipeline->generate(text, speaker_embedding);
        
        this->outputs["generated_speeches"].data = results.speeches;
        this->outputs["perf_metrics"].data = results.perf_metrics;
        break;
    }
    }
    
    std::cout << "LLMInferenceModule[" << module_desc->name << "] generation completed." << std::endl;
}

}  // namespace module
}  // namespace genai
}  // namespace ov
