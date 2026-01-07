// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file model_builder.hpp
 * @brief Factory for building models from configuration and weights
 */

#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>

#include <openvino/openvino.hpp>

#include "loaders/model_config.hpp"
#include "modeling/builder_context.hpp"
#include "modeling/weights/weight_source.hpp"
#include "modeling/weights/weight_finalizer.hpp"

namespace ov {
namespace genai {
namespace loaders {

/**
 * @brief Model architecture builder function type
 * 
 * @param ctx Builder context for creating the model graph
 * @param config Model configuration
 * @param source Weight source providing tensor data
 * @param finalizer Weight finalizer for post-processing
 * @return Constructed OpenVINO model
 */
using ArchitectureBuilder = std::function<std::shared_ptr<ov::Model>(
    modeling::BuilderContext& ctx,
    const ModelConfig& config,
    modeling::weights::WeightSource& source,
    modeling::weights::WeightFinalizer& finalizer)>;

/**
 * @brief Factory for building models from different architectures
 * 
 * ModelBuilder manages a registry of architecture-specific builders and
 * provides a unified interface for model construction.
 * 
 * Usage:
 * @code
 * auto model = ModelBuilder::instance().build(config, *source, *finalizer);
 * @endcode
 */
class ModelBuilder {
public:
    /**
     * @brief Get singleton instance
     */
    static ModelBuilder& instance();

    /**
     * @brief Register an architecture builder
     * 
     * @param architecture Architecture name (e.g., "qwen3", "llama")
     * @param builder Builder function
     * @return true if registration was successful
     */
    bool register_architecture(const std::string& architecture, ArchitectureBuilder builder);

    /**
     * @brief Build a model from configuration
     * 
     * @param config Model configuration
     * @param source Weight source
     * @param finalizer Weight finalizer
     * @return Constructed OpenVINO model
     * @throws std::runtime_error if architecture is not supported
     */
    std::shared_ptr<ov::Model> build(
        const ModelConfig& config,
        modeling::weights::WeightSource& source,
        modeling::weights::WeightFinalizer& finalizer) const;

    /**
     * @brief Check if an architecture is supported
     */
    bool has_architecture(const std::string& architecture) const;

    /**
     * @brief Get list of supported architectures
     */
    std::vector<std::string> supported_architectures() const;

private:
    ModelBuilder();
    ~ModelBuilder() = default;
    ModelBuilder(const ModelBuilder&) = delete;
    ModelBuilder& operator=(const ModelBuilder&) = delete;

    // Register built-in architectures
    void register_builtin_architectures();

    std::map<std::string, ArchitectureBuilder> builders_;
};

/**
 * @brief Helper macro for registering architecture builders
 * 
 * Usage:
 * @code
 * REGISTER_ARCHITECTURE("qwen3", build_qwen3_model);
 * @endcode
 */
#define REGISTER_ARCHITECTURE(arch, builder_fn) \
    static bool _registered_arch_##arch = \
        ov::genai::loaders::ModelBuilder::instance().register_architecture(arch, builder_fn)

}  // namespace loaders
}  // namespace genai
}  // namespace ov
