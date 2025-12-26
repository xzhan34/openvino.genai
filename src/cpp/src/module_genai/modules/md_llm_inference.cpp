// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_llm_inference.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

// ============ To Be Removed: laod position_ids_list from files (debug purpose) ============
std::vector<std::pair<ov::Tensor, std::optional<int64_t>>> load_test_data_position_ids_list() {
    std::vector<std::pair<ov::Tensor, std::optional<int64_t>>> position_ids_list;
    
    std::ifstream meta_file("ut_test_data/position_ids_metadata.txt");
    if (!meta_file.is_open()) {
        std::cerr << "[Deserialize] Failed to open: " << "/position_ids_metadata.txt" << std::endl;
        return position_ids_list;
    }
    
    std::string line;
    size_t pair_count = 0;
    
    while (std::getline(meta_file, line)) {
        if (line.find("pair_count:") != std::string::npos) {
            pair_count = std::stoul(line.substr(line.find(":") + 2));
            break;
        }
    }
    
    for (size_t i = 0; i < pair_count; ++i) {
        std::string element_type_str;
        std::vector<size_t> shape;
        size_t byte_size = 0;
        bool has_rope_delta = false;
        int64_t rope_delta_value = 0;
        
        while (std::getline(meta_file, line)) {
            std::string prefix = "pair_" + std::to_string(i) + "_";
            
            if (line.find(prefix + "element_type:") != std::string::npos) {
                element_type_str = line.substr(line.find(":") + 2);
            } else if (line.find(prefix + "shape:") != std::string::npos) {
                std::string shape_str = line.substr(line.find(":") + 2);
                std::stringstream ss(shape_str);
                std::string dim;
                while (std::getline(ss, dim, ',')) {
                    shape.push_back(std::stoul(dim));
                }
            } else if (line.find(prefix + "byte_size:") != std::string::npos) {
                byte_size = std::stoul(line.substr(line.find(":") + 2));
            } else if (line.find(prefix + "has_rope_delta:") != std::string::npos) {
                has_rope_delta = (line.find("true") != std::string::npos);
                if (!has_rope_delta) break;
            } else if (line.find(prefix + "rope_delta:") != std::string::npos) {
                rope_delta_value = std::stoll(line.substr(line.find(":") + 2));
                break;
            }
        }
        
        ov::element::Type element_type = ov::element::i64;
        if (element_type_str == "f32") element_type = ov::element::f32;
        else if (element_type_str == "f16") element_type = ov::element::f16;
        else if (element_type_str == "i32") element_type = ov::element::i32;
        else if (element_type_str == "i64") element_type = ov::element::i64;
        
        ov::Tensor tensor(element_type, ov::Shape(shape));
        std::string bin_path = "ut_test_data/position_ids_" + std::to_string(i) + ".bin";
        std::ifstream bin_file(bin_path, std::ios::binary);
        if (bin_file.is_open()) {
            bin_file.read(reinterpret_cast<char*>(tensor.data()), byte_size);
            bin_file.close();
        }
        
        std::optional<int64_t> rope_delta = has_rope_delta ? std::optional<int64_t>(rope_delta_value) : std::nullopt;
        position_ids_list.emplace_back(std::move(tensor), rope_delta);
    }
    meta_file.close();
    
    return position_ids_list;
}
// ============ To Be Removed: laod position_ids_list from files (debug purpose) ============

namespace ov {
namespace genai {
namespace module {

void LLMInferenceModule::print_static_config() {
    std::cout << R"(
global_context:
  model_type: "qwen2_5_vl"
pipeline_modules:
  pipeline_params:
    type: "ParameterModule"
    outputs:
      - name: "embeds_list"
        type: "VecOVTensor"
    outputs:
      - name: "position_ids_list"
        type: "VecOVTensor"
  llm_inference:
    type: "LLMInferenceModule"
    description: "LLM module for Continuous Batch pipeline"
    device: "GPU"
    inputs:
      - name: "embeds_list"
        type: "VecOVTensor"
        source: "pipeline_params.embeds_list"
    inputs:
      - name: "position_ids_list"
        type: "VecOVTensor"
        source: "pipeline_params.position_ids_list"
    outputs:
      - name: "generated_text"
        type: "String"
    params:
      model_path: "./ut_pipelines/Qwen2.5-VL-3B-Instruct/INT4/"
      attention_backend: "SDPA"
      is_vlm: "0"
      generation_config_path: "./ut_pipelines/Qwen2.5-VL-3B-Instruct/INT4/generation_config.json"
      max_new_tokens: "256"
      do_sample: "false"
      top_p: "1.0"
      top_k: "50"
      temperature: "1.0"
      repetition_penalty: "1.0"
  pipeline_results:
    type: "ResultModule"
    device: "CPU"
    inputs:
      - name: "generated_text"
        type: "String"
        source: "llm_inference.generated_text"
    )" << std::endl;
}

LLMInferenceModule::LLMInferenceModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {
    if (!initialize()) {
        std::cerr << "Failed to initialize LLMInferenceModule" << std::endl;
    }
}


bool LLMInferenceModule::load_generation_config(const std::string& config_path) {
    try {
        std::ifstream f(config_path);
        if (!f.is_open()) {
        	GENAI_ERR("Failed to open generation config file: " + config_path);
            return false;
        }
        m_generation_config = ov::genai::GenerationConfig(config_path);
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
    	GENAI_ERR(std::string("Error loading generation config: ") + e.what());
        return false;
    }
}

bool LLMInferenceModule::initialize() {
    const auto& params = module_desc->params;

    auto it_models_path = params.find("model_path");
    if (it_models_path == params.end()) {
    	GENAI_ERR("LLMInferenceModule[" + module_desc->name + "]: 'models_path' not found in params");
        return false;
    }
    std::filesystem::path models_path = it_models_path->second;

    // Get device
    std::string device = module_desc->device.empty() ? "GPU" : module_desc->device;
    
    // Prepare properties to use continous batching pipeline
    auto it_backend = params.find("attention_backend");

    ov::AnyMap cfg{};
    cfg["ATTENTION_BACKEND"] = "PA";

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
    m_generation_config.num_assistant_tokens = 5;

    try {
		auto [properties, attention_backend] = utils::extract_attention_backend(cfg);
		auto [plugin_properties, scheduler_config] = utils::extract_scheduler_config(properties, utils::get_latency_oriented_scheduler_config());
		m_cb_pipeline = std::make_unique<ov::genai::VLMPipeline::VLMContinuousBatchingAdapter>(models_path, scheduler_config, device, plugin_properties);
    	return true;
    } catch (const std::exception& e) {
    	GENAI_ERR("LLMInferenceModule[" + module_desc->name + "]: Failed to create pipeline: " + e.what());
        return false;
    }
}

void LLMInferenceModule::run() {
    prepare_inputs();
    std::vector<ov::Tensor> input_embeds_list;
    std::vector<ov::Tensor> tmp_position_ids_list;
    std::vector<std::pair<ov::Tensor, std::optional<int64_t>>> position_ids_list;

    input_embeds_list = this->inputs["embeds_list"].data.as<std::vector<ov::Tensor>>();
#if 0 // to do debug how to pass position_ids_list from upstream module
    tmp_position_ids_list = this->inputs["position_ids_list"].data.as<std::vector<ov::Tensor>>();
    for (const auto& tensor : tmp_position_ids_list) {
    	position_ids_list.push_back({tensor, std::nullopt});
    }
#endif

    ov::genai::VLMDecodedResults vlm_results;

    {
   		std::vector<ov::genai::GenerationConfig> configs_vec = {m_generation_config};
   		std::vector<ov::genai::EncodedGenerationResult> results_vec;
    	// Convert to optional types for the public API
    	std::optional<std::vector<ov::Tensor>> opt_token_type_ids = std::nullopt;
    	std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>> opt_position_ids = std::nullopt;
		auto loaded_position_ids_list = load_test_data_position_ids_list();

    	results_vec = m_cb_pipeline->generate(input_embeds_list, configs_vec, std::monostate{}, opt_token_type_ids, loaded_position_ids_list);
		std::string generated_text = "";

    	if (results_vec.size()) {
    		// Set outputs - extract first result
    		auto& results = results_vec[0];
    		// Decode the generated token IDs to text
    		if (!results.m_generation_ids.empty() && !results.m_generation_ids[0].empty()) {
    			generated_text = m_cb_pipeline->get_tokenizer().decode(results.m_generation_ids[0]);
    			GENAI_INFO("LLM output: " + generated_text);
    		}
            this->outputs["generated_text"].data = generated_text;
    	}
    }
    GENAI_INFO("LLMInferenceModule[" + module_desc->name + "] generation completed.");
}

}  // namespace module
}  // namespace genai
}  // namespace ov
