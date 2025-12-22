// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <filesystem>
#include <memory>

#include "openvino/openvino.hpp"

namespace ov {
namespace genai {
namespace safetensors {

/**
 * @brief Create an OpenVINO model from HuggingFace safetensors format
 * 
 * This function:
 * 1. Loads config.json to get model configuration
 * 2. Loads all safetensors weight files
 * 3. Constructs the ov::Model using building blocks
 * 
 * @param model_dir Path to the HuggingFace model directory
 * @param enable_save_ov_model If true, save the generated model to disk
 * @return std::shared_ptr<ov::Model> The constructed OpenVINO model
 * @throws std::runtime_error if model creation fails
 */
std::shared_ptr<ov::Model> create_from_safetensors(
    const std::filesystem::path& model_dir,
    bool enable_save_ov_model = false);

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
