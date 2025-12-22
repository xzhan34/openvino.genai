// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <filesystem>
#include <unordered_map>
#include <vector>
#include <memory>

#include "openvino/openvino.hpp"

namespace ov {
namespace genai {
namespace safetensors {

/**
 * @brief Information about a loaded tensor
 */
struct TensorInfo {
    std::string name;
    ov::element::Type dtype;
    ov::Shape shape;
    size_t data_offset_start;
    size_t data_offset_end;
};

/**
 * @brief Result of loading safetensors files
 */
struct SafetensorsData {
    std::unordered_map<std::string, ov::Tensor> tensors;
    std::unordered_map<std::string, TensorInfo> tensor_infos;
    std::map<std::string, std::string> metadata;
};

/**
 * @brief Check if a directory contains safetensors model files
 * 
 * @param model_dir Path to model directory
 * @return true if safetensors files are found
 */
bool is_safetensors_model(const std::filesystem::path& model_dir);

/**
 * @brief Load all safetensors files from a HuggingFace model directory
 * 
 * This function:
 * 1. Reads model.safetensors.index.json to find all shard files
 * 2. Loads each shard file using mmap for efficiency
 * 3. Converts tensors to ov::Tensor format
 * 
 * @param model_dir Path to the model directory
 * @return SafetensorsData containing all loaded tensors
 * @throws std::runtime_error if loading fails
 */
SafetensorsData load_safetensors(const std::filesystem::path& model_dir);

/**
 * @brief Load a single safetensors file
 * 
 * @param file_path Path to the .safetensors file
 * @return SafetensorsData containing loaded tensors
 */
SafetensorsData load_safetensors_file(const std::filesystem::path& file_path);

/**
 * @brief Map HuggingFace weight names to internal names
 * 
 * HuggingFace naming: model.layers.0.self_attn.q_proj.weight
 * Internal naming may differ based on the target format
 */
std::string map_hf_weight_name(const std::string& hf_name);

/**
 * @brief Convert safetensors dtype to OpenVINO element type
 */
ov::element::Type convert_dtype(int safetensors_dtype);

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
