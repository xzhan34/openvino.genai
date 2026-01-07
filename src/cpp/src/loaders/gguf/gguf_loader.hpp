// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file gguf_loader.hpp
 * @brief GGUF format model loader implementation.
 * 
 * This loader handles .gguf files from llama.cpp ecosystem.
 * It integrates with the existing gguf_utils for file parsing.
 */

#include "loaders/model_loader.hpp"
#include "loaders/model_config.hpp"

namespace ov {
namespace genai {
namespace loaders {

/**
 * @brief Model loader for GGUF format files.
 * 
 * The GGUFLoader provides:
 * - Reading GGUF metadata to create ModelConfig
 * - Creating WeightSource that reads quantized tensors from GGUF
 * - Creating WeightFinalizer for dequantization (if needed)
 * - Extracting embedded tokenizer vocabulary
 * 
 * Weight naming convention:
 * - GGUF uses: blk.N.xxx format for layer weights
 * - Converted to canonical: model.layers[N].xxx
 * 
 * @see IModelLoader for interface documentation
 */
class GGUFLoader : public IModelLoader {
public:
    GGUFLoader() = default;
    ~GGUFLoader() override = default;

    // Non-copyable, non-movable
    GGUFLoader(const GGUFLoader&) = delete;
    GGUFLoader& operator=(const GGUFLoader&) = delete;
    GGUFLoader(GGUFLoader&&) = delete;
    GGUFLoader& operator=(GGUFLoader&&) = delete;

    /**
     * @brief Check if this loader supports the given path.
     * @param path Path to file or directory
     * @return true if path is a .gguf file
     */
    bool supports(const std::string& path) const override;

    /**
     * @brief Load model configuration from GGUF file.
     * @param path Path to .gguf file
     * @return ModelConfig populated from GGUF metadata
     */
    ModelConfig load_config(const std::string& path) const override;

    /**
     * @brief Create weight source for GGUF tensors.
     * @param path Path to .gguf file
     * @return WeightSource that provides tensor data
     */
    std::shared_ptr<ov::genai::WeightSource> create_weight_source(
        const std::string& path) const override;

    /**
     * @brief Create weight finalizer for dequantization.
     * @param config Model configuration
     * @return WeightFinalizer for post-processing tensors
     */
    std::shared_ptr<ov::genai::WeightFinalizer> create_weight_finalizer(
        const ModelConfig& config) const override;

    /**
     * @brief Load tokenizer from GGUF vocabulary.
     * @param path Path to .gguf file
     * @return Tokenizer/detokenizer pair
     */
    TokenizerPair load_tokenizer(const std::string& path) const override;
};

}  // namespace loaders
}  // namespace genai
}  // namespace ov
