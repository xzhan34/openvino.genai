// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "loaders/model_builder.hpp"

#include <sstream>
#include <stdexcept>

namespace ov {
namespace genai {
namespace loaders {

ModelBuilder& ModelBuilder::instance() {
    static ModelBuilder instance;
    return instance;
}

void ModelBuilder::register_architecture(const std::string& arch_name, ArchitectureBuilder builder) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_builders.find(arch_name) != m_builders.end()) {
        // Allow re-registration for testing purposes, but log a warning
        // In production, this might indicate a problem
    }
    
    m_builders[arch_name] = std::move(builder);
    m_registered_archs.push_back(arch_name);
}

std::shared_ptr<ov::genai::Module> ModelBuilder::build(
    const ModelConfig& config,
    ov::genai::WeightSource& weight_source,
    const ov::AnyMap& properties) const {
    
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Normalize architecture name
    std::string normalized_arch = normalize_arch_name(config.architecture);
    
    auto it = m_builders.find(normalized_arch);
    if (it == m_builders.end()) {
        // Try case-insensitive search
        for (const auto& [name, builder] : m_builders) {
            if (normalize_arch_name(name) == normalized_arch) {
                return builder(config, weight_source, properties);
            }
        }
        
        std::stringstream ss;
        ss << "Unsupported architecture: " << config.architecture
           << " (normalized: " << normalized_arch << "). "
           << "Registered architectures: ";
        for (size_t i = 0; i < m_registered_archs.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << m_registered_archs[i];
        }
        throw std::runtime_error(ss.str());
    }
    
    return it->second(config, weight_source, properties);
}

bool ModelBuilder::has_architecture(const std::string& arch_name) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::string normalized = normalize_arch_name(arch_name);
    
    if (m_builders.find(normalized) != m_builders.end()) {
        return true;
    }
    
    // Case-insensitive search
    for (const auto& [name, _] : m_builders) {
        if (normalize_arch_name(name) == normalized) {
            return true;
        }
    }
    
    return false;
}

std::vector<std::string> ModelBuilder::registered_architectures() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_registered_archs;
}

std::string ModelBuilder::normalize_arch_name(const std::string& name) {
    std::string result;
    result.reserve(name.size());
    
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            result += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    
    return result;
}

}  // namespace loaders
}  // namespace genai
}  // namespace ov
