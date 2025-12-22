// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Define implementation before including header
#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors_utils/safetensors.hh"

#include "safetensors_utils/safetensors_loader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <set>

namespace ov {
namespace genai {
namespace safetensors {

namespace {

// Parse model.safetensors.index.json to get shard file list
std::vector<std::string> parse_index_json(const std::filesystem::path& index_path) {
    std::ifstream file(index_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open " + index_path.string());
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();
    
    // Find weight_map section
    size_t weight_map_pos = json.find("\"weight_map\"");
    if (weight_map_pos == std::string::npos) {
        throw std::runtime_error("weight_map not found in index.json");
    }
    
    // Extract unique file names from weight_map values
    std::set<std::string> files;
    size_t pos = weight_map_pos;
    
    // Simple extraction: find all .safetensors file references
    while ((pos = json.find(".safetensors", pos)) != std::string::npos) {
        // Find the start of the filename (after last quote before .safetensors)
        size_t end = pos + 12;  // length of ".safetensors"
        size_t start = json.rfind('"', pos);
        if (start != std::string::npos && start < pos) {
            start++;  // Skip the quote
            std::string filename = json.substr(start, end - start);
            files.insert(filename);
        }
        pos = end;
    }
    
    return std::vector<std::string>(files.begin(), files.end());
}

}  // anonymous namespace

ov::element::Type convert_dtype(int safetensors_dtype) {
    using namespace ::safetensors;
    
    switch (safetensors_dtype) {
        case kBOOL:     return ov::element::boolean;
        case kUINT8:    return ov::element::u8;
        case kINT8:     return ov::element::i8;
        case kINT16:    return ov::element::i16;
        case kUINT16:   return ov::element::u16;
        case kFLOAT16:  return ov::element::f16;
        case kBFLOAT16: return ov::element::bf16;
        case kINT32:    return ov::element::i32;
        case kUINT32:   return ov::element::u32;
        case kFLOAT32:  return ov::element::f32;
        case kFLOAT64:  return ov::element::f64;
        case kINT64:    return ov::element::i64;
        case kUINT64:   return ov::element::u64;
        default:
            throw std::runtime_error("Unknown safetensors dtype: " + std::to_string(safetensors_dtype));
    }
}

bool is_safetensors_model(const std::filesystem::path& model_dir) {
    // Check for single file
    if (std::filesystem::exists(model_dir / "model.safetensors")) {
        return true;
    }
    
    // Check for sharded files
    if (std::filesystem::exists(model_dir / "model.safetensors.index.json")) {
        return true;
    }
    
    // Check for any .safetensors file
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
        if (entry.path().extension() == ".safetensors") {
            return true;
        }
    }
    
    return false;
}

SafetensorsData load_safetensors_file(const std::filesystem::path& file_path) {
    SafetensorsData result;
    
    ::safetensors::safetensors_t st;
    std::string warn, err;
    
    // Use mmap for efficient loading
    bool success = ::safetensors::mmap_from_file(file_path.string(), &st, &warn, &err);
    
    if (!success) {
        throw std::runtime_error("Failed to load safetensors file: " + file_path.string() + 
                                 ", error: " + err);
    }
    
    if (!warn.empty()) {
        std::cerr << "Warning loading " << file_path << ": " << warn << std::endl;
    }
    
    // Validate data offsets
    if (!::safetensors::validate_data_offsets(st, err)) {
        throw std::runtime_error("Invalid data offsets in " + file_path.string() + ": " + err);
    }
    
    // Convert tensors to ov::Tensor
    const auto& keys = st.tensors.keys();
    for (size_t i = 0; i < keys.size(); i++) {
        const std::string& name = keys[i];
        ::safetensors::tensor_t tensor;
        
        if (!st.tensors.at(name, &tensor)) {
            continue;
        }
        
        // Convert dtype
        ov::element::Type ov_dtype = convert_dtype(static_cast<int>(tensor.dtype));
        
        // Convert shape
        ov::Shape ov_shape(tensor.shape.begin(), tensor.shape.end());
        
        // Calculate data pointer
        const uint8_t* data_ptr = st.databuffer_addr + tensor.data_offsets[0];
        size_t data_size = tensor.data_offsets[1] - tensor.data_offsets[0];
        
        // Create ov::Tensor
        // Note: We need to copy data since mmap will be released
        ov::Tensor ov_tensor(ov_dtype, ov_shape);
        std::memcpy(ov_tensor.data(), data_ptr, data_size);
        
        result.tensors[name] = std::move(ov_tensor);
        
        // Store tensor info
        TensorInfo info;
        info.name = name;
        info.dtype = ov_dtype;
        info.shape = ov_shape;
        info.data_offset_start = tensor.data_offsets[0];
        info.data_offset_end = tensor.data_offsets[1];
        result.tensor_infos[name] = info;
    }
    
    // Copy metadata
    const auto& meta_keys = st.metadata.keys();
    for (size_t i = 0; i < meta_keys.size(); i++) {
        const std::string& key = meta_keys[i];
        std::string value;
        if (st.metadata.at(key, &value)) {
            result.metadata[key] = value;
        }
    }
    
    return result;
}

SafetensorsData load_safetensors(const std::filesystem::path& model_dir) {
    SafetensorsData result;
    
    std::vector<std::filesystem::path> files_to_load;
    
    // Check for single file model
    std::filesystem::path single_file = model_dir / "model.safetensors";
    if (std::filesystem::exists(single_file)) {
        files_to_load.push_back(single_file);
    } else {
        // Check for sharded model
        std::filesystem::path index_file = model_dir / "model.safetensors.index.json";
        if (std::filesystem::exists(index_file)) {
            auto shard_names = parse_index_json(index_file);
            for (const auto& name : shard_names) {
                files_to_load.push_back(model_dir / name);
            }
        } else {
            // Load all .safetensors files in directory
            for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
                if (entry.path().extension() == ".safetensors") {
                    files_to_load.push_back(entry.path());
                }
            }
        }
    }
    
    if (files_to_load.empty()) {
        throw std::runtime_error("No safetensors files found in " + model_dir.string());
    }
    
    // Sort files for deterministic loading order
    std::sort(files_to_load.begin(), files_to_load.end());
    
    std::cout << "[Safetensors] Loading " << files_to_load.size() << " file(s)..." << std::endl;
    
    // Load each file
    for (const auto& file_path : files_to_load) {
        std::cout << "[Safetensors] Loading " << file_path.filename() << "..." << std::endl;
        
        SafetensorsData file_data = load_safetensors_file(file_path);
        
        // Merge tensors
        for (auto& [name, tensor] : file_data.tensors) {
            if (result.tensors.count(name)) {
                std::cerr << "Warning: Duplicate tensor name: " << name << std::endl;
            }
            result.tensors[name] = std::move(tensor);
        }
        
        // Merge tensor infos
        for (auto& [name, info] : file_data.tensor_infos) {
            result.tensor_infos[name] = std::move(info);
        }
        
        // Merge metadata (first file wins for duplicates)
        for (auto& [key, value] : file_data.metadata) {
            if (!result.metadata.count(key)) {
                result.metadata[key] = std::move(value);
            }
        }
    }
    
    std::cout << "[Safetensors] Loaded " << result.tensors.size() << " tensors" << std::endl;
    
    return result;
}

std::string map_hf_weight_name(const std::string& hf_name) {
    // For now, return the same name
    // HuggingFace naming is already quite standardized
    // We may need to add mapping later if building_blocks expects different names
    return hf_name;
}

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
