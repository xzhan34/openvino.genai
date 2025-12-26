// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ut_modules_base.hpp"

class LlmInferenceModuleTest : public ModuleTestBase {
public:
    DEFINE_MODULE_TEST_CONSTRUCTOR(LlmInferenceModuleTest)

protected:
    std::string get_yaml_content() override {
        return R"(
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
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;

        std::vector<ov::Tensor> input_embeds_list;
        load_test_data_input_embeds_list(input_embeds_list);
        CHECK(input_embeds_list.size(), "Failed to load input embeds list data");
        inputs["embeds_list"] = input_embeds_list;

        std::vector<std::pair<ov::Tensor, std::optional<int64_t>>> position_ids_list;
        std::vector<ov::Tensor> loaded_position_ids_list;
        load_test_data_position_ids_list(position_ids_list);
        CHECK(position_ids_list.size(), "Failed to load position ids list data");
        for (auto id: position_ids_list) {
         	loaded_position_ids_list.push_back(id.first);
        }
        inputs["position_ids_list"] = loaded_position_ids_list;

        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto generated_text = pipe.get_output("generated_text").as<std::string>();

        bool contains_white_cat = generated_text.find("white cat") != std::string::npos;
        CHECK(contains_white_cat, "llm inference module does not work as expected");
    }

    bool load_test_data_position_ids_list(
    		std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>& position_ids_list) {
        std::ifstream meta_file("ut_test_data/position_ids_metadata.txt");
        if (!meta_file.is_open()) {
            std::cerr << "Failed to open: " << "ut_test_data/position_ids_metadata.txt" << std::endl;
            return false;
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

        return true;
    }

    bool load_test_data_input_embeds_list(std::vector<ov::Tensor>& input_embeds_list) {
    	std::ifstream meta_file("ut_test_data/embeds_metadata.txt");
    	if (!meta_file.is_open()) {
    		return false;
    	}

    	input_embeds_list.clear();

    	std::string line;
    	size_t tensor_count = 0;

    	// Parse tensor count
    	while (std::getline(meta_file, line)) {
    		if (line.find("tensor_count:") != std::string::npos) {
    			tensor_count = std::stoul(line.substr(line.find(":") + 2));
    			break;
    		}
    	}

    	// Parse each tensor's metadata and load binary data
    	for (size_t i = 0; i < tensor_count; ++i) {
    		std::string element_type_str;
    		std::vector<size_t> shape;
    		size_t byte_size = 0;

    		while (std::getline(meta_file, line)) {
    			std::string prefix = "tensor_" + std::to_string(i) + "_";
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
    				break;  // Got all info for this tensor
    			}
    		}

    		// Determine element type
    		ov::element::Type element_type = ov::element::f32;  // default
    		if (element_type_str == "f16")
    			element_type = ov::element::f16;
    		else if (element_type_str == "bf16")
    			element_type = ov::element::bf16;
    		else if (element_type_str == "f64")
    			element_type = ov::element::f64;
    		else if (element_type_str == "i32")
    			element_type = ov::element::i32;
    		else if (element_type_str == "i64")
    			element_type = ov::element::i64;

    		// Create tensor and load binary data
    		ov::Tensor tensor(element_type, ov::Shape(shape));
    		std::string bin_path = "ut_test_data/embeds_tensor_" + std::to_string(i) + ".bin";
    		std::ifstream bin_file(bin_path, std::ios::binary);
    		if (bin_file.is_open()) {
    			bin_file.read(reinterpret_cast<char*>(tensor.data()), byte_size);
    			bin_file.close();
    		}
    		input_embeds_list.push_back(std::move(tensor));
    	}
    	return true;
    }

};

REGISTER_MODULE_TEST(LlmInferenceModuleTest);
