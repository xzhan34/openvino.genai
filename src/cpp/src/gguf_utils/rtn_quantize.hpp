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
 * 
 * Performance optimizations:
 *   - Type dispatch outside hot loops
 *   - Precomputed inverse scale (multiply instead of divide)
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
 * @brief Convert float to FP16
 */
inline uint16_t float_to_fp16(float value) {
    uint32_t fp32_bits;
    std::memcpy(&fp32_bits, &value, sizeof(float));
    uint32_t sign = (fp32_bits >> 31) & 0x1;
    int32_t exp = ((fp32_bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (fp32_bits >> 13) & 0x3FF;
    
    if (exp <= 0) {
        return static_cast<uint16_t>(sign << 15);  // Underflow to zero
    } else if (exp >= 31) {
        return static_cast<uint16_t>((sign << 15) | 0x7C00);  // Overflow to inf
    } else {
        return static_cast<uint16_t>((sign << 15) | (exp << 10) | mant);
    }
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
 * @brief Templated quantization for INT4 symmetric - optimized inner loop
 */
template<typename DataType, float (*ConvertFunc)(DataType)>
void quantize_int4_sym_typed(
    const DataType* src_data,
    uint8_t* compressed_data,
    uint16_t* scale_data,
    size_t out_features,
    size_t in_features,
    size_t packed_in_features,
    size_t num_groups,
    int group_size
) {
    constexpr float factor = 8.0f;  // 2^(4-1) = 8
    constexpr int8_t level_low = -8;
    constexpr int8_t level_high = 7;
    constexpr float eps = 1.1920929e-7f;
    
    for (size_t row = 0; row < out_features; row++) {
        const DataType* row_data = src_data + row * in_features;
        uint8_t* row_compressed = compressed_data + row * packed_in_features;
        uint16_t* row_scale = scale_data + row * num_groups;
        
        for (size_t g = 0; g < num_groups; g++) {
            size_t group_start = g * group_size;
            size_t group_end = std::min(group_start + static_cast<size_t>(group_size), in_features);
            
            // Single pass: find min/max
            float w_min = std::numeric_limits<float>::max();
            float w_max = std::numeric_limits<float>::lowest();
            
            for (size_t col = group_start; col < group_end; col++) {
                float val = ConvertFunc(row_data[col]);
                w_min = std::min(w_min, val);
                w_max = std::max(w_max, val);
            }
            
            // Compute scale
            float max_abs = std::max(std::abs(w_min), std::abs(w_max));
            float scale_val = max_abs / factor;
            if (scale_val < eps) scale_val = eps;
            float inv_scale = 1.0f / scale_val;
            
            // Store scale
            row_scale[g] = float_to_fp16(scale_val);
            
            // Quantize elements
            for (size_t col = group_start; col < group_end; col++) {
                float val = ConvertFunc(row_data[col]);
                int8_t quantized = static_cast<int8_t>(std::round(val * inv_scale));
                quantized = std::max(level_low, std::min(level_high, quantized));
                
                uint8_t packed = static_cast<uint8_t>(quantized) & 0x0F;
                size_t byte_idx = col / 2;
                if (col % 2 == 0) {
                    row_compressed[byte_idx] = packed;
                } else {
                    row_compressed[byte_idx] |= (packed << 4);
                }
            }
        }
    }
}

// Conversion function wrappers for template
inline float convert_f32(float val) { return val; }
inline float convert_f16(uint16_t val) { return fp16_to_float(val); }
inline float convert_bf16(uint16_t val) { return bf16_to_float(val); }

/**
 * @brief Templated quantization for INT4 asymmetric - optimized inner loop
 */
template<typename DataType, float (*ConvertFunc)(DataType)>
void quantize_int4_asym_typed(
    const DataType* src_data,
    uint8_t* compressed_data,
    uint16_t* scale_data,
    uint8_t* zero_point_data,
    size_t out_features,
    size_t in_features,
    size_t packed_in_features,
    size_t num_groups,
    int group_size
) {
    constexpr float levels = 15.0f;  // 2^4 - 1
    constexpr int32_t level_low = 0;
    constexpr int32_t level_high = 15;
    constexpr float eps = 1.1920929e-7f;
    
    for (size_t row = 0; row < out_features; row++) {
        const DataType* row_data = src_data + row * in_features;
        uint8_t* row_compressed = compressed_data + row * packed_in_features;
        uint16_t* row_scale = scale_data + row * num_groups;
        uint8_t* row_zp = zero_point_data + row * num_groups;
        
        for (size_t g = 0; g < num_groups; g++) {
            size_t group_start = g * group_size;
            size_t group_end = std::min(group_start + static_cast<size_t>(group_size), in_features);
            
            float w_min = std::numeric_limits<float>::max();
            float w_max = std::numeric_limits<float>::lowest();
            
            for (size_t col = group_start; col < group_end; col++) {
                float val = ConvertFunc(row_data[col]);
                w_min = std::min(w_min, val);
                w_max = std::max(w_max, val);
            }
            
            float scale_val = (w_max - w_min) / levels;
            if (scale_val < eps) scale_val = eps;
            float inv_scale = 1.0f / scale_val;
            
            int32_t zp = static_cast<int32_t>(std::round(-w_min * inv_scale));
            zp = std::max(level_low, std::min(level_high, zp));
            
            row_scale[g] = float_to_fp16(scale_val);
            row_zp[g] = static_cast<uint8_t>(zp);
            
            for (size_t col = group_start; col < group_end; col++) {
                float val = ConvertFunc(row_data[col]);
                int32_t quantized = static_cast<int32_t>(std::round(val * inv_scale)) + zp;
                quantized = std::max(level_low, std::min(level_high, quantized));
                
                uint8_t packed = static_cast<uint8_t>(quantized) & 0x0F;
                size_t byte_idx = col / 2;
                if (col % 2 == 0) {
                    row_compressed[byte_idx] = packed;
                } else {
                    row_compressed[byte_idx] |= (packed << 4);
                }
            }
        }
    }
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
 * Optimized version with:
 *   - Type dispatch outside hot loops
 *   - Precomputed inverse scale (multiply instead of divide)
 * 
 * @param weight Input weight tensor (FP16 or BF16), shape [out_features, in_features] or [batch, out_features, in_features]
 * @param group_size Number of elements per quantization group (default: 128)
 * @return QuantizedWeight with INT4 weights packed as U8 and FP16 scales
 */
inline QuantizedWeight quantize_int4_sym(const ov::Tensor& weight, int group_size = 128) {
    auto shape = weight.get_shape();
    if (shape.size() != 2 && shape.size() != 3) {
        throw std::runtime_error("Weight tensor must be 2D or 3D for quantization");
    }
    
    // Support both 2D [out_features, in_features] and 3D [batch, out_features, in_features]
    // For 3D, treat as [batch * out_features, in_features]
    size_t out_features = (shape.size() == 3) ? (shape[0] * shape[1]) : shape[0];
    size_t in_features = (shape.size() == 3) ? shape[2] : shape[1];
    ov::Shape output_shape = (shape.size() == 3) ? ov::Shape{shape[0], shape[1]} : ov::Shape{shape[0]};
    size_t num_groups = (in_features + group_size - 1) / group_size;
    size_t packed_in_features = (in_features + 1) / 2;
    
    // Output shapes: append packed_in_features or num_groups to output_shape
    ov::Shape compressed_shape = output_shape;
    compressed_shape.push_back(packed_in_features);
    ov::Shape scale_shape = output_shape;
    scale_shape.push_back(num_groups);
    
    ov::Tensor compressed(ov::element::u8, compressed_shape);
    ov::Tensor scale(ov::element::f16, scale_shape);
    
    auto* compressed_data = static_cast<uint8_t*>(compressed.data());
    auto* scale_data = static_cast<uint16_t*>(scale.data());
    
    // Initialize compressed data to zero (important for proper nibble packing)
    std::memset(compressed_data, 0, out_features * packed_in_features);
    
    // Dispatch based on dtype - type check happens ONCE, not per element
    auto dtype = weight.get_element_type();
    if (dtype == ov::element::f32) {
        quantize_int4_sym_typed<float, convert_f32>(
            static_cast<const float*>(weight.data()),
            compressed_data, scale_data,
            out_features, in_features, packed_in_features, num_groups, group_size);
    } else if (dtype == ov::element::f16) {
        quantize_int4_sym_typed<uint16_t, convert_f16>(
            static_cast<const uint16_t*>(weight.data()),
            compressed_data, scale_data,
            out_features, in_features, packed_in_features, num_groups, group_size);
    } else if (dtype == ov::element::bf16) {
        quantize_int4_sym_typed<uint16_t, convert_bf16>(
            static_cast<const uint16_t*>(weight.data()),
            compressed_data, scale_data,
            out_features, in_features, packed_in_features, num_groups, group_size);
    } else {
        throw std::runtime_error("Unsupported tensor dtype for quantization");
    }
    
    QuantizedWeight result;
    result.compressed = compressed;
    result.scale = scale;
    result.has_zero_point = false;
    return result;
}

/**
 * @brief Quantize weights using asymmetric INT4 RTN algorithm (NNCF-compatible)
 * 
 * NNCF Algorithm:
 *   levels = 2^num_bits - 1 = 15 for INT4
 *   scale = (max - min) / levels
 *   zero_point = round(-min / scale)
 *   quantized = round(weight / scale) + zero_point
 *   quantized = clamp(quantized, 0, 15)
 * 
 * Optimized version with:
 *   - Type dispatch outside hot loops
 *   - Precomputed inverse scale (multiply instead of divide)
 * 
 * @param weight Input weight tensor (FP16 or BF16), shape [out_features, in_features] or [batch, out_features, in_features]
 * @param group_size Number of elements per quantization group (default: 128)
 * @return QuantizedWeight with INT4 weights packed as U8, FP16 scales, and U8 zero points
 */
inline QuantizedWeight quantize_int4_asym(const ov::Tensor& weight, int group_size = 128) {
    auto shape = weight.get_shape();
    if (shape.size() != 2 && shape.size() != 3) {
        throw std::runtime_error("Weight tensor must be 2D or 3D for quantization");
    }
    
    // Support both 2D [out_features, in_features] and 3D [batch, out_features, in_features]
    // For 3D, treat as [batch * out_features, in_features]
    size_t out_features = (shape.size() == 3) ? (shape[0] * shape[1]) : shape[0];
    size_t in_features = (shape.size() == 3) ? shape[2] : shape[1];
    ov::Shape output_shape = (shape.size() == 3) ? ov::Shape{shape[0], shape[1]} : ov::Shape{shape[0]};
    size_t num_groups = (in_features + group_size - 1) / group_size;
    size_t packed_in_features = (in_features + 1) / 2;
    
    // Output shapes: append packed_in_features or num_groups to output_shape
    ov::Shape compressed_shape = output_shape;
    compressed_shape.push_back(packed_in_features);
    ov::Shape scale_shape = output_shape;
    scale_shape.push_back(num_groups);
    
    ov::Tensor compressed(ov::element::u8, compressed_shape);
    ov::Tensor scale(ov::element::f16, scale_shape);
    ov::Tensor zero_point(ov::element::u8, scale_shape);
    
    auto* compressed_data = static_cast<uint8_t*>(compressed.data());
    auto* scale_data = static_cast<uint16_t*>(scale.data());
    auto* zero_point_data = static_cast<uint8_t*>(zero_point.data());
    
    // Initialize compressed data to zero (for nibble packing)
    std::memset(compressed_data, 0, out_features * packed_in_features);
    
    auto dtype = weight.get_element_type();
    if (dtype == ov::element::f32) {
        quantize_int4_asym_typed<float, convert_f32>(
            static_cast<const float*>(weight.data()),
            compressed_data, scale_data, zero_point_data,
            out_features, in_features, packed_in_features, num_groups, group_size);
    } else if (dtype == ov::element::f16) {
        quantize_int4_asym_typed<uint16_t, convert_f16>(
            static_cast<const uint16_t*>(weight.data()),
            compressed_data, scale_data, zero_point_data,
            out_features, in_features, packed_in_features, num_groups, group_size);
    } else if (dtype == ov::element::bf16) {
        quantize_int4_asym_typed<uint16_t, convert_bf16>(
            static_cast<const uint16_t*>(weight.data()),
            compressed_data, scale_data, zero_point_data,
            out_features, in_features, packed_in_features, num_groups, group_size);
    } else {
        throw std::runtime_error("Unsupported tensor dtype for quantization");
    }
    
    QuantizedWeight result;
    result.compressed = compressed;
    result.scale = scale;
    result.zero_point = zero_point;
    result.has_zero_point = true;
    return result;
}

/**
 * @brief Templated quantization for INT8 symmetric - optimized inner loop
 */
template<typename DataType, float (*ConvertFunc)(DataType)>
void quantize_int8_sym_typed(
    const DataType* src_data,
    int8_t* compressed_data,
    uint16_t* scale_data,
    size_t out_features,
    size_t in_features,
    size_t num_groups,
    int group_size
) {
    constexpr float factor = 128.0f;  // 2^(8-1) = 128
    constexpr int32_t level_low = -128;
    constexpr int32_t level_high = 127;
    constexpr float eps = 1.1920929e-7f;
    
    for (size_t row = 0; row < out_features; row++) {
        const DataType* row_data = src_data + row * in_features;
        int8_t* row_compressed = compressed_data + row * in_features;
        uint16_t* row_scale = scale_data + row * num_groups;
        
        for (size_t g = 0; g < num_groups; g++) {
            size_t group_start = g * group_size;
            size_t group_end = std::min(group_start + static_cast<size_t>(group_size), in_features);
            
            // Single pass: find min/max
            float w_min = std::numeric_limits<float>::max();
            float w_max = std::numeric_limits<float>::lowest();
            
            for (size_t col = group_start; col < group_end; col++) {
                float val = ConvertFunc(row_data[col]);
                w_min = std::min(w_min, val);
                w_max = std::max(w_max, val);
            }
            
            // Compute scale
            float max_abs = std::max(std::abs(w_min), std::abs(w_max));
            float scale_val = max_abs / factor;
            if (scale_val < eps) scale_val = eps;
            float inv_scale = 1.0f / scale_val;
            
            // Store scale
            row_scale[g] = float_to_fp16(scale_val);
            
            // Quantize elements
            for (size_t col = group_start; col < group_end; col++) {
                float val = ConvertFunc(row_data[col]);
                int32_t quantized = static_cast<int32_t>(std::round(val * inv_scale));
                quantized = std::max(level_low, std::min(level_high, quantized));
                row_compressed[col] = static_cast<int8_t>(quantized);
            }
        }
    }
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
 * Optimized version with:
 *   - Type dispatch outside hot loops
 *   - Precomputed inverse scale (multiply instead of divide)
 * 
 * @param weight Input weight tensor (FP16 or BF16), shape [out_features, in_features] or [batch, out_features, in_features]
 * @param group_size Number of elements per quantization group (default: 128)
 * @return QuantizedWeight with INT8 weights and FP16 scales
 */
inline QuantizedWeight quantize_int8_sym(const ov::Tensor& weight, int group_size = 128) {
    auto shape = weight.get_shape();
    if (shape.size() != 2 && shape.size() != 3) {
        throw std::runtime_error("Weight tensor must be 2D or 3D for quantization");
    }
    
    // Support both 2D [out_features, in_features] and 3D [batch, out_features, in_features]
    // For 3D, treat as [batch * out_features, in_features]
    size_t out_features = (shape.size() == 3) ? (shape[0] * shape[1]) : shape[0];
    size_t in_features = (shape.size() == 3) ? shape[2] : shape[1];
    ov::Shape output_shape = (shape.size() == 3) ? ov::Shape{shape[0], shape[1]} : ov::Shape{shape[0]};
    size_t num_groups = (in_features + group_size - 1) / group_size;
    
    // Output shapes: append in_features or num_groups to output_shape
    ov::Shape compressed_shape = output_shape;
    compressed_shape.push_back(in_features);
    ov::Shape scale_shape = output_shape;
    scale_shape.push_back(num_groups);
    
    ov::Tensor compressed(ov::element::i8, compressed_shape);
    ov::Tensor scale(ov::element::f16, scale_shape);
    
    auto* compressed_data = static_cast<int8_t*>(compressed.data());
    auto* scale_data = static_cast<uint16_t*>(scale.data());
    
    // Dispatch based on dtype - type check happens ONCE, not per element
    auto dtype = weight.get_element_type();
    if (dtype == ov::element::f32) {
        quantize_int8_sym_typed<float, convert_f32>(
            static_cast<const float*>(weight.data()),
            compressed_data, scale_data,
            out_features, in_features, num_groups, group_size);
    } else if (dtype == ov::element::f16) {
        quantize_int8_sym_typed<uint16_t, convert_f16>(
            static_cast<const uint16_t*>(weight.data()),
            compressed_data, scale_data,
            out_features, in_features, num_groups, group_size);
    } else if (dtype == ov::element::bf16) {
        quantize_int8_sym_typed<uint16_t, convert_bf16>(
            static_cast<const uint16_t*>(weight.data()),
            compressed_data, scale_data,
            out_features, in_features, num_groups, group_size);
    } else {
        throw std::runtime_error("Unsupported tensor dtype for quantization");
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
