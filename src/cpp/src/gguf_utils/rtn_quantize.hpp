// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file rtn_quantize.hpp
 * @brief Round-To-Nearest (RTN) weight quantization for in-flight compression
 * 
 * This implements a simple but effective RTN algorithm that matches NNCF's
 * weight compression. It quantizes FP16/BF16 weights to INT4 or INT8.
 * 
 * NNCF Algorithm Reference (weight_lowering.py):
 *   - _calculate_signed_scale(): scale = max(|min|, -max) / 2^(bits-1)
 *   - _calculate_integer_quantized_weight(): round(weight / scale), clip to range
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "openvino/openvino.hpp"

namespace ov {
namespace genai {
namespace rtn {

/**
 * @brief Result of RTN quantization
 */
struct QuantizedWeight {
    ov::Tensor compressed;      ///< Quantized weights (INT4 or INT8)
    ov::Tensor scale;           ///< Scale tensor for dequantization
    ov::Tensor zero_point;      ///< Zero point (for asymmetric quantization)
    bool has_zero_point;        ///< True if asymmetric quantization was used
};

/**
 * @brief Convert BF16 to float
 */
inline float bf16_to_float(uint16_t bf16_val) {
    uint32_t fp32_bits = static_cast<uint32_t>(bf16_val) << 16;
    float result;
    std::memcpy(&result, &fp32_bits, sizeof(float));
    return result;
}

/**
 * @brief Convert FP16 to float
 */
inline float fp16_to_float(uint16_t fp16_val) {
    // Extract components
    uint32_t sign = (fp16_val >> 15) & 0x1;
    uint32_t exp = (fp16_val >> 10) & 0x1F;
    uint32_t mant = fp16_val & 0x3FF;
    
    uint32_t fp32_bits;
    if (exp == 0) {
        if (mant == 0) {
            // Zero
            fp32_bits = sign << 31;
        } else {
            // Subnormal
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            fp32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        // Inf or NaN
        fp32_bits = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        // Normal
        fp32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    
    float result;
    std::memcpy(&result, &fp32_bits, sizeof(float));
    return result;
}

/**
 * @brief Get float value from tensor at given index
 */
inline float get_float_value(const ov::Tensor& tensor, size_t idx) {
    auto dtype = tensor.get_element_type();
    if (dtype == ov::element::f32) {
        return static_cast<const float*>(tensor.data())[idx];
    } else if (dtype == ov::element::f16) {
        return fp16_to_float(static_cast<const uint16_t*>(tensor.data())[idx]);
    } else if (dtype == ov::element::bf16) {
        return bf16_to_float(static_cast<const uint16_t*>(tensor.data())[idx]);
    }
    throw std::runtime_error("Unsupported tensor dtype for quantization");
}

/**
 * @brief Quantize weights using symmetric INT4 RTN algorithm (NNCF-compatible)
 * 
 * NNCF Algorithm:
 *   factor = 2^(num_bits - 1) = 8 for INT4
 *   scale = max(|min|, |max|) / factor
 *   quantized = round(weight / scale)
 *   quantized = clamp(quantized, -8, 7)
 * 
 * @param weight Input weight tensor (FP16 or BF16), shape [out_features, in_features]
 * @param group_size Number of elements per quantization group (default: 128)
 * @return QuantizedWeight with INT4 weights packed as U8 and FP16 scales
 */
inline QuantizedWeight quantize_int4_sym(const ov::Tensor& weight, int group_size = 128) {
    auto shape = weight.get_shape();
    if (shape.size() != 2) {
        throw std::runtime_error("Weight tensor must be 2D for quantization");
    }
    
    size_t out_features = shape[0];
    size_t in_features = shape[1];
    size_t total_elements = out_features * in_features;
    
    // Calculate number of groups
    size_t num_groups = (in_features + group_size - 1) / group_size;
    
    // Allocate output tensors
    // INT4 weights: pack 2 values per byte
    size_t packed_in_features = (in_features + 1) / 2;
    ov::Tensor compressed(ov::element::u8, {out_features, packed_in_features});
    ov::Tensor scale(ov::element::f16, {out_features, num_groups});
    
    auto* compressed_data = static_cast<uint8_t*>(compressed.data());
    auto* scale_data = static_cast<uint16_t*>(scale.data());
    
    // NNCF constants for INT4 symmetric quantization
    constexpr int num_bits = 4;
    constexpr float factor = static_cast<float>(1 << (num_bits - 1));  // 2^3 = 8
    constexpr int8_t level_low = -(1 << (num_bits - 1));   // -8
    constexpr int8_t level_high = (1 << (num_bits - 1)) - 1;  // 7
    constexpr float eps = 1.1920929e-7f;  // FP32 machine epsilon
    
    // Quantize each row
    for (size_t row = 0; row < out_features; row++) {
        for (size_t g = 0; g < num_groups; g++) {
            size_t group_start = g * group_size;
            size_t group_end = std::min(group_start + group_size, in_features);
            
            // Find min and max in this group (NNCF style)
            float w_min = std::numeric_limits<float>::max();
            float w_max = std::numeric_limits<float>::lowest();
            for (size_t col = group_start; col < group_end; col++) {
                float val = get_float_value(weight, row * in_features + col);
                w_min = std::min(w_min, val);
                w_max = std::max(w_max, val);
            }
            
            // NNCF: scale = max(|min|, |max|) / factor
            // Using absolute values to match NNCF's signed scale calculation
            float w_abs_min = std::abs(w_min);
            float w_abs_max = std::abs(w_max);
            float max_abs = std::max(w_abs_min, w_abs_max);
            float scale_val = max_abs / factor;
            
            // Avoid division by zero (NNCF uses machine epsilon)
            if (scale_val < eps) {
                scale_val = eps;
            }
            
            // Store scale as FP16
            // Simple FP32 to FP16 conversion
            uint32_t fp32_bits;
            std::memcpy(&fp32_bits, &scale_val, sizeof(float));
            uint32_t sign = (fp32_bits >> 31) & 0x1;
            int32_t exp = ((fp32_bits >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = (fp32_bits >> 13) & 0x3FF;
            
            uint16_t fp16_bits;
            if (exp <= 0) {
                fp16_bits = static_cast<uint16_t>(sign << 15);  // Underflow to zero
            } else if (exp >= 31) {
                fp16_bits = static_cast<uint16_t>((sign << 15) | 0x7C00);  // Overflow to inf
            } else {
                fp16_bits = static_cast<uint16_t>((sign << 15) | (exp << 10) | mant);
            }
            scale_data[row * num_groups + g] = fp16_bits;
            
            // Quantize each element in the group
            for (size_t col = group_start; col < group_end; col++) {
                float val = get_float_value(weight, row * in_features + col);
                
                // NNCF: quantized = round(weight / scale), clip to [level_low, level_high]
                float scaled = val / scale_val;
                int8_t quantized = static_cast<int8_t>(std::round(scaled));
                quantized = std::max(level_low, std::min(level_high, quantized));
                
                // Pack into 4 bits as signed i4 (range [-8, 7])
                // Store as signed 4-bit value in the low 4 bits of a byte
                // Two's complement for negative values
                uint8_t packed = static_cast<uint8_t>(quantized) & 0x0F;
                
                // Pack two 4-bit values per byte
                size_t byte_idx = row * packed_in_features + col / 2;
                if (col % 2 == 0) {
                    compressed_data[byte_idx] = packed;  // Low nibble
                } else {
                    compressed_data[byte_idx] |= (packed << 4);  // High nibble
                }
            }
        }
    }
    
    QuantizedWeight result;
    result.compressed = compressed;
    result.scale = scale;
    result.has_zero_point = false;
    return result;
}

/**
 * @brief Quantize weights using symmetric INT8 RTN algorithm (NNCF-compatible)
 * 
 * NNCF Algorithm:
 *   factor = 2^(num_bits - 1) = 128 for INT8
 *   scale = max(|min|, |max|) / factor
 *   quantized = round(weight / scale)
 *   quantized = clamp(quantized, -128, 127)
 * 
 * @param weight Input weight tensor (FP16 or BF16), shape [out_features, in_features]
 * @param group_size Number of elements per quantization group (default: 128)
 * @return QuantizedWeight with INT8 weights and FP16 scales
 */
inline QuantizedWeight quantize_int8_sym(const ov::Tensor& weight, int group_size = 128) {
    auto shape = weight.get_shape();
    if (shape.size() != 2) {
        throw std::runtime_error("Weight tensor must be 2D for quantization");
    }
    
    size_t out_features = shape[0];
    size_t in_features = shape[1];
    
    // Calculate number of groups
    size_t num_groups = (in_features + group_size - 1) / group_size;
    
    // Allocate output tensors
    ov::Tensor compressed(ov::element::i8, {out_features, in_features});
    ov::Tensor scale(ov::element::f16, {out_features, num_groups});
    
    auto* compressed_data = static_cast<int8_t*>(compressed.data());
    auto* scale_data = static_cast<uint16_t*>(scale.data());
    
    // NNCF constants for INT8 symmetric quantization
    constexpr int num_bits = 8;
    constexpr float factor = static_cast<float>(1 << (num_bits - 1));  // 2^7 = 128
    constexpr int32_t level_low = -(1 << (num_bits - 1));   // -128
    constexpr int32_t level_high = (1 << (num_bits - 1)) - 1;  // 127
    constexpr float eps = 1.1920929e-7f;  // FP32 machine epsilon
    
    // Quantize each row
    for (size_t row = 0; row < out_features; row++) {
        for (size_t g = 0; g < num_groups; g++) {
            size_t group_start = g * group_size;
            size_t group_end = std::min(group_start + group_size, in_features);
            
            // Find min and max in this group (NNCF style)
            float w_min = std::numeric_limits<float>::max();
            float w_max = std::numeric_limits<float>::lowest();
            for (size_t col = group_start; col < group_end; col++) {
                float val = get_float_value(weight, row * in_features + col);
                w_min = std::min(w_min, val);
                w_max = std::max(w_max, val);
            }
            
            // NNCF: scale = max(|min|, |max|) / factor
            float w_abs_min = std::abs(w_min);
            float w_abs_max = std::abs(w_max);
            float max_abs = std::max(w_abs_min, w_abs_max);
            float scale_val = max_abs / factor;
            
            // Avoid division by zero (NNCF uses machine epsilon)
            if (scale_val < eps) {
                scale_val = eps;
            }
            
            // Store scale as FP16
            uint32_t fp32_bits;
            std::memcpy(&fp32_bits, &scale_val, sizeof(float));
            uint32_t sign = (fp32_bits >> 31) & 0x1;
            int32_t exp = ((fp32_bits >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = (fp32_bits >> 13) & 0x3FF;
            
            uint16_t fp16_bits;
            if (exp <= 0) {
                fp16_bits = static_cast<uint16_t>(sign << 15);
            } else if (exp >= 31) {
                fp16_bits = static_cast<uint16_t>((sign << 15) | 0x7C00);
            } else {
                fp16_bits = static_cast<uint16_t>((sign << 15) | (exp << 10) | mant);
            }
            scale_data[row * num_groups + g] = fp16_bits;
            
            // Quantize each element in the group
            for (size_t col = group_start; col < group_end; col++) {
                float val = get_float_value(weight, row * in_features + col);
                
                // NNCF: quantized = round(weight / scale), clip to [level_low, level_high]
                float scaled = val / scale_val;
                int32_t quantized = static_cast<int32_t>(std::round(scaled));
                quantized = std::max(level_low, std::min(level_high, quantized));
                
                compressed_data[row * in_features + col] = static_cast<int8_t>(quantized);
            }
        }
    }
    
    QuantizedWeight result;
    result.compressed = compressed;
    result.scale = scale;
    result.has_zero_point = false;
    return result;
}

}  // namespace rtn
}  // namespace genai
}  // namespace ov
