// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"
#include "gguf_utils/rtn_quantize.hpp"

#include <openvino/core/except.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/convert.hpp>

#include <iostream>
#include <chrono>
#include <iomanip>

namespace ov {
namespace genai {
namespace safetensors {

using namespace ov::genai::modeling::weights;

SafetensorsWeightFinalizer::SafetensorsWeightFinalizer() = default;

SafetensorsWeightFinalizer::SafetensorsWeightFinalizer(const QuantizationConfig& config)
    : selector_(config) {
    if (config.enabled()) {
        std::cout << "[SafetensorsWeightFinalizer] In-flight quantization enabled (NNCF-compatible):" << std::endl;
        
        // Helper lambda to print mode name
        auto mode_name = [](QuantizationConfig::Mode m) -> std::string {
            switch (m) {
                case QuantizationConfig::Mode::INT4_SYM: return "INT4_SYM";
                case QuantizationConfig::Mode::INT4_ASYM: return "INT4_ASYM";
                case QuantizationConfig::Mode::INT8_SYM: return "INT8_SYM";
                case QuantizationConfig::Mode::INT8_ASYM: return "INT8_ASYM";
                case QuantizationConfig::Mode::NONE: return "NONE";
                default: return "UNKNOWN";
            }
        };
        
        std::cout << "  Primary mode: " << mode_name(config.mode) 
                  << " (group_size=" << config.group_size << ")" << std::endl;
        std::cout << "  Backup mode: " << mode_name(config.backup_mode) 
                  << " (per-channel)" << std::endl;
        
        if (config.backup_mode == config.mode) {
            std::cout << "  -> All layers use same mode (" 
                      << mode_name(config.mode) << ", group_size=" << config.group_size << ")" << std::endl;
        } else if (config.backup_mode != QuantizationConfig::Mode::NONE) {
            std::cout << "  -> lm_head, embeddings will use backup mode (" 
                      << mode_name(config.backup_mode) << ", per-channel)" << std::endl;
        } else {
            std::cout << "  -> lm_head, embeddings will NOT be quantized (backup_mode=NONE)" << std::endl;
        }
        
        const auto& sel = config.selection;
        if (!sel.include_patterns.empty()) {
            std::cout << "  Include patterns: ";
            for (size_t i = 0; i < sel.include_patterns.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << sel.include_patterns[i];
            }
            std::cout << std::endl;
        }
        if (sel.layer_range.has_value()) {
            std::cout << "  Layer range: [" << sel.layer_range->first 
                      << ", " << sel.layer_range->second << "]" << std::endl;
        }
        if (sel.verbose) {
            std::cout << "  Verbose mode enabled" << std::endl;
        }
    }
}

SafetensorsWeightFinalizer::~SafetensorsWeightFinalizer() {
    if (total_weights_ > 0) {
        std::cout << "[SafetensorsWeightFinalizer] Statistics:" << std::endl;
        std::cout << "  Total weights processed: " << total_weights_ << std::endl;
        std::cout << "  Quantized weights: " << quantized_weights_ << std::endl;
        if (total_weights_ > 0) {
             float ratio = 100.0f * static_cast<float>(quantized_weights_) / static_cast<float>(total_weights_);
             std::cout << "  Quantization coverage: " << ratio << "%" << std::endl;
        }
        std::cout << "  Timing (ms):" << std::endl;
        std::cout << "    Fetch: " << std::fixed << std::setprecision(2) << total_fetch_time_ms_ << std::endl;
        std::cout << "    Quant: " << std::fixed << std::setprecision(2) << total_quant_time_ms_ << std::endl;
        std::cout << "    Graph: " << std::fixed << std::setprecision(2) << total_graph_time_ms_ << std::endl;
        std::cout << "    Total: " << std::fixed << std::setprecision(2) << (total_fetch_time_ms_ + total_quant_time_ms_ + total_graph_time_ms_) << std::endl;
    }
}

ov::genai::modeling::weights::FinalizedWeight SafetensorsWeightFinalizer::finalize(
    const std::string& name,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::OpContext& ctx) {
    if (!source.has(name)) {
        OPENVINO_THROW("Missing safetensors tensor: ", name);
    }

    // Check cache first
    const auto it_cached = cache_.find(name);
    if (it_cached != cache_.end()) {
        std::unordered_map<std::string, ov::genai::modeling::Tensor> empty_auxiliary;
        return ov::genai::modeling::weights::FinalizedWeight(
            ov::genai::modeling::Tensor(it_cached->second, &ctx),
            empty_auxiliary);
    }

    // Get tensor info for quantization check
    // Fetch tensor only once to avoid unnecessary caching
    auto start_fetch = std::chrono::high_resolution_clock::now();
    const ov::Tensor& tensor = source.get_tensor(name);
    auto end_fetch = std::chrono::high_resolution_clock::now();

    double fetch_ms = std::chrono::duration<double, std::milli>(end_fetch - start_fetch).count();
    total_fetch_time_ms_ += fetch_ms;

    ov::element::Type element_type = tensor.get_element_type();
    const ov::Shape& shape = tensor.get_shape();

    total_weights_++;

    // Check if in-flight quantization should be applied
    ov::Output<ov::Node> output;
    std::unordered_map<std::string, ov::genai::modeling::Tensor> auxiliary;
    
    double quant_ms = 0;
    double graph_ms = 0;
    bool logged = false;

    if (element_type == ov::element::f8e4m3 && has_fp8_scale_inv(source, name)) {
        auto start_graph = std::chrono::high_resolution_clock::now();
        const ov::Tensor& scale_inv = source.get_tensor(name + "_scale_inv");
        output = create_fp8_dequant_subgraph(name, tensor, scale_inv, ctx);
        auto end_graph = std::chrono::high_resolution_clock::now();
        graph_ms = std::chrono::duration<double, std::milli>(end_graph - start_graph).count();
        total_graph_time_ms_ += graph_ms;
    } else if (selector_.enabled() && selector_.should_quantize(name, shape, element_type)) {
        quantized_weights_++;
        // Quantize the weight during finalization
        // Tensor reference is already fetched above, no need to fetch again
        auto start_quant = std::chrono::high_resolution_clock::now();
        auto quant_result = quantize_weight(name, tensor, source, ctx);
        auto end_quant = std::chrono::high_resolution_clock::now();
        quant_ms = std::chrono::duration<double, std::milli>(end_quant - start_quant).count();
        total_quant_time_ms_ += quant_ms;

        auto start_graph = std::chrono::high_resolution_clock::now();
        if (is_moe_weight(name)) {
            // For MoE weights, return FinalizedWeight with scales and zps in auxiliary
            auto res = create_moe_subgraph(name, quant_result, shape, ctx);
            auto end_graph = std::chrono::high_resolution_clock::now();
            graph_ms = std::chrono::duration<double, std::milli>(end_graph - start_graph).count();
            total_graph_time_ms_ += graph_ms;
            if (logged) {
                std::cout << "[Weight: " << name << "] Fetch=" << std::fixed << std::setprecision(2) << fetch_ms 
                      << "ms, Quant=" << quant_ms << "ms, Graph=" << graph_ms << "ms" << std::endl;
            }
            return res;
        } else {
            output = create_dequant_subgraph(name, quant_result, shape);
        }
        auto end_graph = std::chrono::high_resolution_clock::now();
        graph_ms = std::chrono::duration<double, std::milli>(end_graph - start_graph).count();
        total_graph_time_ms_ += graph_ms;
        logged = false;
    } else {
        auto start_graph = std::chrono::high_resolution_clock::now();
        // Only create Constant when NOT quantizing
        // This avoids keeping the original tensor in memory when we compress it
        std::shared_ptr<ov::op::v0::Constant> constant = std::make_shared<ov::op::v0::Constant>(tensor);
        constant->set_friendly_name(name);
        constant->output(0).set_names({name});

        // Convert to F32 if needed (matching GGUF behavior)
        // This ensures consistency with how GGUF weights are processed
        if (element_type == ov::element::bf16 || 
            element_type == ov::element::f16 ||
            element_type == ov::element::f8e4m3) {
            auto converted = std::make_shared<ov::op::v0::Convert>(constant, ov::element::f32);
            output = converted->output(0);
        } else {
            output = constant->output(0);
        }
        auto end_graph = std::chrono::high_resolution_clock::now();
        graph_ms = std::chrono::duration<double, std::milli>(end_graph - start_graph).count();
        total_graph_time_ms_ += graph_ms;
    }
    
    if (logged) {
        std::cout << "[Weight: " << name << "] Fetch=" << std::fixed << std::setprecision(2) << fetch_ms 
                  << "ms, Quant=" << quant_ms << "ms, Graph=" << graph_ms << "ms" << std::endl;
    }
    
    cache_.emplace(name, output);

    return ov::genai::modeling::weights::FinalizedWeight(
        ov::genai::modeling::Tensor(output, &ctx),
        auxiliary);
}

bool SafetensorsWeightFinalizer::has_fp8_scale_inv(ov::genai::modeling::weights::WeightSource& source,
                                                    const std::string& name) const {
    static const std::string weight_suffix = ".weight";
    if (name.size() < weight_suffix.size() ||
        name.compare(name.size() - weight_suffix.size(), weight_suffix.size(), weight_suffix) != 0) {
        return false;
    }
    return source.has(name + "_scale_inv");
}

ov::Output<ov::Node> SafetensorsWeightFinalizer::create_fp8_dequant_subgraph(
    const std::string& name,
    const ov::Tensor& weight_fp8,
    const ov::Tensor& scale_inv,
    ov::genai::modeling::OpContext& ctx) {
    auto weight_const = std::make_shared<ov::op::v0::Constant>(weight_fp8);
    weight_const->set_friendly_name(name);
    weight_const->output(0).set_names({name});

    auto weight_f32 = std::make_shared<ov::op::v0::Convert>(weight_const, ov::element::f32);
    auto expanded_scale = expand_block_scale_to_weight(scale_inv, weight_fp8.get_shape(), ctx);
    auto scale_f32 = std::make_shared<ov::op::v0::Convert>(expanded_scale, ov::element::f32);
    auto dequant = std::make_shared<ov::op::v1::Multiply>(
        weight_f32->output(0),
        scale_f32->output(0),
        ov::op::AutoBroadcastType::NUMPY);
    return dequant->output(0);
}

ov::Output<ov::Node> SafetensorsWeightFinalizer::expand_block_scale_to_weight(
    const ov::Tensor& scale_inv,
    const ov::Shape& weight_shape,
    ov::genai::modeling::OpContext& ctx) const {
    auto scale_const = std::make_shared<ov::op::v0::Constant>(scale_inv);
    const ov::Shape scale_shape = scale_inv.get_shape();

    if (scale_shape == weight_shape) {
        return scale_const->output(0);
    }

    OPENVINO_ASSERT(scale_shape.size() == weight_shape.size(),
                    "FP8 scale rank must match weight rank. scale_shape=",
                    scale_shape,
                    ", weight_shape=",
                    weight_shape);

    auto interleaved_scale_shape = make_interleaved_scale_shape(scale_shape);
    auto interleaved_repeats = make_interleaved_tile_repeats(scale_shape, weight_shape);

    auto reshape_scale = std::make_shared<ov::op::v1::Reshape>(
        scale_const->output(0),
        ctx.const_i64_vec(interleaved_scale_shape),
        false);
    auto tiled = std::make_shared<ov::op::v0::Tile>(
        reshape_scale->output(0),
        ctx.const_i64_vec(interleaved_repeats));

    std::vector<int64_t> weight_shape_i64(weight_shape.begin(), weight_shape.end());
    auto final_reshape = std::make_shared<ov::op::v1::Reshape>(
        tiled->output(0),
        ctx.const_i64_vec(weight_shape_i64),
        false);
    return final_reshape->output(0);
}

std::vector<int64_t> SafetensorsWeightFinalizer::make_interleaved_scale_shape(const ov::Shape& scale_shape) const {
    std::vector<int64_t> result;
    result.reserve(scale_shape.size() * 2);
    for (size_t i = 0; i < scale_shape.size(); ++i) {
        result.push_back(static_cast<int64_t>(scale_shape[i]));
        result.push_back(1);
    }
    return result;
}

std::vector<int64_t> SafetensorsWeightFinalizer::make_interleaved_tile_repeats(const ov::Shape& scale_shape,
                                                                                const ov::Shape& weight_shape) const {
    OPENVINO_ASSERT(scale_shape.size() == weight_shape.size(),
                    "Scale and weight rank mismatch while expanding FP8 scales.");
    std::vector<int64_t> repeats;
    repeats.reserve(scale_shape.size() * 2);
    for (size_t i = 0; i < scale_shape.size(); ++i) {
        const auto s = scale_shape[i];
        const auto w = weight_shape[i];
        OPENVINO_ASSERT(s > 0 && w > 0, "Scale/weight dimensions must be > 0 for FP8 block expansion.");
        OPENVINO_ASSERT(w % s == 0,
                        "Weight dimension must be divisible by scale dimension for FP8 block expansion. dim ",
                        i,
                        ": weight=",
                        w,
                        ", scale=",
                        s);
        repeats.push_back(1);
        repeats.push_back(static_cast<int64_t>(w / s));
    }
    return repeats;
}

rtn::QuantizedWeight SafetensorsWeightFinalizer::quantize_weight(
    const std::string& name,
    const ov::Tensor& tensor,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::OpContext& ctx) {
    
    using namespace ov::genai::rtn;
    using Mode = QuantizationConfig::Mode;
    
    // NNCF-style: Get the quantization mode for this specific weight
    // This may return different modes for different layers (e.g., INT8 for lm_head, INT4 for others)
    Mode quant_mode = selector_.get_quantization_mode(name, tensor.get_shape(), tensor.get_element_type());
    int group_size = selector_.get_group_size(name);
    
    // RTN quantization functions expect 2D tensors
    OPENVINO_ASSERT(tensor.get_shape().size() == 2 || tensor.get_shape().size() == 3 || tensor.get_shape().size() == 1, 
                    "Only 1D, 2D and 3D tensors can be quantized, got shape: ", tensor.get_shape());
    
    // Quantize using RTN algorithms
    rtn::QuantizedWeight quant_result;
    
    switch (quant_mode) {
        case Mode::INT4_SYM:
            quant_result = rtn::quantize_int4_sym(tensor, group_size);
            break;
            
        case Mode::INT4_ASYM:
            quant_result = rtn::quantize_int4_asym(tensor, group_size);
            break;
            
        case Mode::INT8_SYM:
            quant_result = rtn::quantize_int8_sym(tensor, group_size);
            break;
            
        case Mode::INT8_ASYM:
            quant_result = rtn::quantize_int8_asym(tensor, group_size);
            break;
            
        default:
            OPENVINO_THROW("Unsupported quantization mode");
    }
    
    return quant_result;
}

ov::Output<ov::Node> SafetensorsWeightFinalizer::create_dequant_subgraph(
    const std::string& name,
    const rtn::QuantizedWeight& quant_result,
    const ov::Shape& original_shape) {
    
    // Support 1D [in], 2D [out, in] and 3D [batch, out, in] weights
    OPENVINO_ASSERT(original_shape.size() == 2 || original_shape.size() == 3 || original_shape.size() == 1, 
                    "Original shape must be 1D, 2D or 3D for dequantization");
    
    size_t out_features = 1;
    size_t in_features = original_shape[0];

    if (original_shape.size() == 2) {
        out_features = original_shape[0];
        in_features = original_shape[1];
    } else if (original_shape.size() == 3) {
        out_features = original_shape[0] * original_shape[1];
        in_features = original_shape[2];
    }
    
    // Create scale constant
    auto scale_const = std::make_shared<ov::op::v0::Constant>(quant_result.scale);
    scale_const->set_friendly_name(name + "_scale");
    
    const ov::Shape& scale_shape = scale_const->get_shape();
    // For 3D, scale_shape will be [batch, out, num_groups]
    // For 2D, scale_shape will be [out, num_groups]
    size_t num_groups = scale_shape.back();
    size_t group_size = in_features / num_groups;
    
    // Determine element type based on compressed type from result
    ov::element::Type elem_type = quant_result.compressed_type;
    bool has_zero_point = quant_result.has_zero_point && quant_result.zero_point.get_size() > 0;
    
    size_t num_elements = out_features * in_features;
    
    // Validation check
    if (elem_type == ov::element::u4 || elem_type == ov::element::i4) {
        // Check if size matches packed 4-bit expectation
        size_t expected_size = (out_features * ((in_features + 1) / 2)); 
        // Note: For grouped quantization, shapes might slightly differ due to padding, 
        // but generally packed size should be roughly half.
        // We skip strict size check here relying on the quantization function correctness.
    } else if (elem_type == ov::element::u8 || elem_type == ov::element::i8) {
        OPENVINO_ASSERT(quant_result.compressed.get_size() == num_elements, 
                        "INT8 compressed size mismatch");
    } else {
        OPENVINO_THROW("Unsupported compressed type for dequantization: ", elem_type);
    }
    
    // Step 1: Create grouped constant with appropriate element type
    // We treat everything as [out_features, num_groups, group_size] for logic simplicity
    ov::Shape grouped_shape = {out_features, num_groups, group_size};
    const void* data_ptr = quant_result.compressed.data();
    auto grouped_const = std::make_shared<ov::op::v0::Constant>(elem_type, grouped_shape, data_ptr);
    grouped_const->set_friendly_name(name + "_compressed");
    
    // Step 2: Convert to F16 for arithmetic operations
    auto weights_f16 = std::make_shared<ov::op::v0::Convert>(grouped_const, ov::element::f16);
    
    // Step 3: Reshape scale to [out_features, num_groups, 1] for broadcasting
    ov::Shape scale_broadcast_shape = {out_features, num_groups, 1};
    auto scale_reshape = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64,
        ov::Shape{scale_broadcast_shape.size()},
        std::vector<int64_t>(scale_broadcast_shape.begin(), scale_broadcast_shape.end())
    );
    auto scale_reshaped = std::make_shared<ov::op::v1::Reshape>(scale_const, scale_reshape, false);
    
    // Step 4: Apply dequantization formula
    std::shared_ptr<ov::Node> dequantized;
    if (has_zero_point) {
        // Asymmetric: weight_fp = (weight - zero_point) * scale
        auto zp_const = std::make_shared<ov::op::v0::Constant>(quant_result.zero_point);
        zp_const->set_friendly_name(name + "_zp");
        auto zp_reshaped = std::make_shared<ov::op::v1::Reshape>(zp_const, scale_reshape, false);
        auto zp_f16 = std::make_shared<ov::op::v0::Convert>(zp_reshaped, ov::element::f16);
        auto shifted = std::make_shared<ov::op::v1::Subtract>(
            weights_f16, zp_f16, ov::op::AutoBroadcastType::NUMPY);
        dequantized = std::make_shared<ov::op::v1::Multiply>(
            shifted, scale_reshaped, ov::op::AutoBroadcastType::NUMPY);
    } else {
        // Symmetric: weight_fp = weight * scale
        dequantized = std::make_shared<ov::op::v1::Multiply>(
            weights_f16, scale_reshaped, ov::op::AutoBroadcastType::NUMPY);
    }
    
    // Step 5: Reshape back to original shape [out_features, in_features] or [batch, out, in]
    auto final_shape = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64,
        ov::Shape{original_shape.size()},
        std::vector<int64_t>(original_shape.begin(), original_shape.end())
    );
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(dequantized, final_shape, false);
    
    // Step 6: Convert to F32
    return std::make_shared<ov::op::v0::Convert>(reshaped, ov::element::f32);
}

bool SafetensorsWeightFinalizer::is_moe_weight(const std::string& name) const {
    // Check for MoE weight patterns:
    // - model.layers[X].mlp.experts.X.gate_proj.weight
    // - model.layers[X].mlp.experts.X.up_proj.weight
    // - model.layers[X].mlp.experts.X.down_proj.weight
    return name.find("experts") != std::string::npos && 
           (name.find("gate_proj") != std::string::npos ||
            name.find("up_proj") != std::string::npos ||
            name.find("down_proj") != std::string::npos);
}

ov::genai::modeling::weights::FinalizedWeight SafetensorsWeightFinalizer::create_moe_subgraph(
    const std::string& name,
    const rtn::QuantizedWeight& quant_result,
    const ov::Shape& original_shape,
    ov::genai::modeling::OpContext& ctx) {
    
    // MoE weights: [out_features, in_features]
    OPENVINO_ASSERT(original_shape.size() == 2, "MoE original shape must be 2D");
    
    // Determine element type based on compressed type
    bool has_zero_point = quant_result.has_zero_point && quant_result.zero_point.get_size() > 0;
    ov::element::Type elem_type = quant_result.compressed_type;
    
    ov::Shape scale_shape = quant_result.scale.get_shape();
    ov::Shape ZERO_SHAPE = quant_result.zero_point.get_shape();
    size_t num_groups = scale_shape[1];
    size_t group_size = original_shape[1] / num_groups;

   // Create INT4 weight tensor with proper shape
    // [out_features, group_num, group_size]
    ov::Shape packed_shape{original_shape[0],
                           scale_shape[1],     // group_num
                           group_size};
    ;

    const void* data_ptr = quant_result.compressed.data();
    auto weight_compressed = std::make_shared<ov::op::v0::Constant>(elem_type, packed_shape, data_ptr);
    weight_compressed->set_friendly_name(name + "_compressed");

    // Create scale constant 
    auto scale_const = std::make_shared<ov::op::v0::Constant>(quant_result.scale);
    scale_const->set_friendly_name(name + "_scale");

    std::shared_ptr<ov::op::v0::Constant> zp_const;
    ov::Tensor zero_point_tensor(ov::element::u4, scale_shape);
    uint8_t* zero_point_data = static_cast<uint8_t*>(zero_point_tensor.data());
    if (has_zero_point) {
        // Handle zero-point packing if needed
        ov::element::Type zp_type = quant_result.zero_point.get_element_type();
        if (zp_type == ov::element::u8 && elem_type == ov::element::u4) {
            // Zero-points need to be packed from u8 to u4 nibbles
            const uint8_t* zp_data = quant_result.zero_point.data<uint8_t>();
            for (size_t i = 0; i < zero_point_tensor.get_byte_size(); ++i) {
                uint8_t bias1 = zp_data[i * 2];
                uint8_t bias2 = zp_data[i * 2 + 1];
                zero_point_data[i] = (bias2 << 4) | (bias1 & 0x0F);
            }
        } else {
           OPENVINO_THROW("Unsupported zero-point type in MoE weight finalization");
        }
    } else {
        OPENVINO_THROW("Unsupported: no zps");
    }

    zp_const = std::make_shared<ov::op::v0::Constant>(zero_point_tensor);
    zp_const->set_friendly_name(name + "_zp");
    // Transpose from [I, H] to [H, I]
    auto transpose_order = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0});
    
    auto scales_transposed = std::make_shared<ov::op::v1::Transpose>(scale_const, transpose_order);
    scales_transposed->set_friendly_name(name + "_scales_transposed");

    // Build auxiliary map with transposed scales and zero-points
    std::unordered_map<std::string, ov::genai::modeling::Tensor> auxiliary;
    auxiliary["scales"] = ov::genai::modeling::Tensor(scales_transposed, &ctx);

    if (has_zero_point && zp_const) {
        auto zero_points_transposed = std::make_shared<ov::op::v1::Transpose>(zp_const, transpose_order);
        zero_points_transposed->set_friendly_name(name + "_zps_transposed");
        auxiliary["zps"] = ov::genai::modeling::Tensor(zero_points_transposed, &ctx);
    }

    // Cache the weight
    cache_.emplace(name, weight_compressed);

    return ov::genai::modeling::weights::FinalizedWeight(
        ov::genai::modeling::Tensor(weight_compressed, &ctx),
        auxiliary);
}

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
