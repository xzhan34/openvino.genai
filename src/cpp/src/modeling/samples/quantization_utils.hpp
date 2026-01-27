// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <string>
#include <iostream>
#include "safetensors_utils/safetensors_weight_finalizer.hpp"

inline ov::genai::modeling::weights::QuantizationConfig create_quantization_config(
    std::string quant_mode, 
    int group_size, 
    std::string backup_mode) {

    ov::genai::modeling::weights::QuantizationConfig config;
    
    // Normalize to lowercase
    std::transform(quant_mode.begin(), quant_mode.end(), quant_mode.begin(), ::tolower);
    std::transform(backup_mode.begin(), backup_mode.end(), backup_mode.begin(), ::tolower);

    if (quant_mode.find("int4") != std::string::npos) {
        config.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_ASYM;
        if (quant_mode.find("sym") != std::string::npos) {
            config.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_SYM;
        }
    } else if (quant_mode.find("int8") != std::string::npos) {
        config.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_ASYM; // Default to ASYM for INT8 if unspecified? Or use creating logic
         if (quant_mode.find("sym") != std::string::npos) {
            config.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_SYM;
        }
    } else {
        // config.enabled = false; // "enabled" is a method, not a field. Setting mode to NONE disables it.
        config.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::NONE;
        return config;
    }

    config.group_size = group_size;

    // Backup mode
    if (backup_mode.find("int8") != std::string::npos) {
        config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_ASYM;
        if (backup_mode.find("sym") != std::string::npos) {
            config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_SYM;
        }
    } else if (backup_mode.empty() || backup_mode == "none") {
        config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::NONE;
    }
    
    return config;
}
