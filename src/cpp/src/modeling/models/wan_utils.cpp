// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/wan_utils.hpp"

#include <fstream>

#include <openvino/core/except.hpp>

#include "json_utils.hpp"

namespace {

void read_config_json_file(const std::filesystem::path& path, nlohmann::json& data) {
    std::ifstream file(path);
    if (!file.is_open()) {
        OPENVINO_THROW("Failed to open config file: ", path.string());
    }
    file >> data;
}

std::filesystem::path resolve_config_path(const std::filesystem::path& path) {
    if (std::filesystem::is_directory(path)) {
        return path / "config.json";
    }
    return path;
}

void parse_transformer_config(const nlohmann::json& data, ov::genai::modeling::models::WanTransformer3DConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "_class_name", cfg.class_name);
    read_json_param(data, "architectures", cfg.architectures);
    read_json_param(data, "patch_size", cfg.patch_size);
    read_json_param(data, "num_attention_heads", cfg.num_attention_heads);
    read_json_param(data, "attention_head_dim", cfg.attention_head_dim);
    read_json_param(data, "in_channels", cfg.in_channels);
    read_json_param(data, "out_channels", cfg.out_channels);
    read_json_param(data, "text_dim", cfg.text_dim);
    read_json_param(data, "freq_dim", cfg.freq_dim);
    read_json_param(data, "ffn_dim", cfg.ffn_dim);
    read_json_param(data, "num_layers", cfg.num_layers);
    read_json_param(data, "cross_attn_norm", cfg.cross_attn_norm);
    read_json_param(data, "qk_norm", cfg.qk_norm);
    read_json_param(data, "eps", cfg.eps);
    read_json_param(data, "image_dim", cfg.image_dim);
    read_json_param(data, "added_kv_proj_dim", cfg.added_kv_proj_dim);
    read_json_param(data, "rope_max_seq_len", cfg.rope_max_seq_len);
    read_json_param(data, "pos_embed_seq_len", cfg.pos_embed_seq_len);

    cfg.finalize();
    cfg.validate();
}

void parse_vae_config(const nlohmann::json& data, ov::genai::modeling::models::WanVAEConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "_class_name", cfg.class_name);
    read_json_param(data, "architectures", cfg.architectures);
    read_json_param(data, "base_dim", cfg.base_dim);
    read_json_param(data, "decoder_base_dim", cfg.decoder_base_dim);
    read_json_param(data, "z_dim", cfg.z_dim);
    read_json_param(data, "dim_mult", cfg.dim_mult);
    read_json_param(data, "num_res_blocks", cfg.num_res_blocks);
    read_json_param(data, "attn_scales", cfg.attn_scales);
    read_json_param(data, "temperal_downsample", cfg.temperal_downsample);
    read_json_param(data, "dropout", cfg.dropout);
    read_json_param(data, "latents_mean", cfg.latents_mean);
    read_json_param(data, "latents_std", cfg.latents_std);
    read_json_param(data, "is_residual", cfg.is_residual);
    read_json_param(data, "in_channels", cfg.in_channels);
    read_json_param(data, "out_channels", cfg.out_channels);
    read_json_param(data, "patch_size", cfg.patch_size);
    read_json_param(data, "scale_factor_temporal", cfg.scale_factor_temporal);
    read_json_param(data, "scale_factor_spatial", cfg.scale_factor_spatial);

    cfg.finalize();
    cfg.validate();
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

int32_t WanTransformer3DConfig::inner_dim() const {
    return num_attention_heads > 0 ? num_attention_heads * attention_head_dim : 0;
}

int32_t WanTransformer3DConfig::patch_volume() const {
    if (patch_size.empty()) {
        return 0;
    }
    int32_t volume = 1;
    for (int32_t v : patch_size) {
        volume *= v;
    }
    return volume;
}

bool WanTransformer3DConfig::use_image_condition() const {
    return image_dim.has_value();
}

void WanTransformer3DConfig::finalize() {
    if (out_channels <= 0) {
        out_channels = in_channels;
    }
    if (patch_size.empty()) {
        patch_size = {1, 2, 2};
    }
}

void WanTransformer3DConfig::validate() const {
    if (patch_size.size() != 3) {
        OPENVINO_THROW("WanTransformer3DConfig.patch_size must have 3 elements");
    }
    for (int32_t v : patch_size) {
        if (v <= 0) {
            OPENVINO_THROW("WanTransformer3DConfig.patch_size values must be > 0");
        }
    }
    if (num_attention_heads <= 0) {
        OPENVINO_THROW("WanTransformer3DConfig.num_attention_heads must be > 0");
    }
    if (attention_head_dim <= 0) {
        OPENVINO_THROW("WanTransformer3DConfig.attention_head_dim must be > 0");
    }
    if (num_layers <= 0) {
        OPENVINO_THROW("WanTransformer3DConfig.num_layers must be > 0");
    }
    if (ffn_dim <= 0) {
        OPENVINO_THROW("WanTransformer3DConfig.ffn_dim must be > 0");
    }
    if (in_channels <= 0 || out_channels <= 0) {
        OPENVINO_THROW("WanTransformer3DConfig.in_channels/out_channels must be > 0");
    }
    if (text_dim <= 0) {
        OPENVINO_THROW("WanTransformer3DConfig.text_dim must be > 0");
    }
    if (freq_dim <= 0) {
        OPENVINO_THROW("WanTransformer3DConfig.freq_dim must be > 0");
    }
    if (rope_max_seq_len <= 0) {
        OPENVINO_THROW("WanTransformer3DConfig.rope_max_seq_len must be > 0");
    }
    if (inner_dim() <= 0) {
        OPENVINO_THROW("WanTransformer3DConfig.inner_dim must be > 0");
    }
}

WanTransformer3DConfig WanTransformer3DConfig::from_json(const nlohmann::json& data) {
    WanTransformer3DConfig cfg;
    parse_transformer_config(data, cfg);
    return cfg;
}

WanTransformer3DConfig WanTransformer3DConfig::from_json_file(const std::filesystem::path& config_path) {
    nlohmann::json data;
    read_config_json_file(resolve_config_path(config_path), data);
    return from_json(data);
}

void WanVAEConfig::finalize() {
    if (decoder_base_dim <= 0) {
        decoder_base_dim = base_dim;
    }
}

void WanVAEConfig::validate() const {
    if (base_dim <= 0 || decoder_base_dim <= 0) {
        OPENVINO_THROW("WanVAEConfig.base_dim/decoder_base_dim must be > 0");
    }
    if (z_dim <= 0) {
        OPENVINO_THROW("WanVAEConfig.z_dim must be > 0");
    }
    if (dim_mult.empty()) {
        OPENVINO_THROW("WanVAEConfig.dim_mult must not be empty");
    }
    if (num_res_blocks <= 0) {
        OPENVINO_THROW("WanVAEConfig.num_res_blocks must be > 0");
    }
    if (!latents_mean.empty() && static_cast<int32_t>(latents_mean.size()) != z_dim) {
        OPENVINO_THROW("WanVAEConfig.latents_mean size must match z_dim");
    }
    if (!latents_std.empty() && static_cast<int32_t>(latents_std.size()) != z_dim) {
        OPENVINO_THROW("WanVAEConfig.latents_std size must match z_dim");
    }
    if (!temperal_downsample.empty() && dim_mult.size() > 1 &&
        temperal_downsample.size() != dim_mult.size() - 1) {
        OPENVINO_THROW("WanVAEConfig.temperal_downsample size must match dim_mult.size() - 1");
    }
    if (in_channels <= 0 || out_channels <= 0) {
        OPENVINO_THROW("WanVAEConfig.in_channels/out_channels must be > 0");
    }
    if (patch_size.has_value() && patch_size.value() <= 0) {
        OPENVINO_THROW("WanVAEConfig.patch_size must be > 0");
    }
    if (scale_factor_temporal <= 0 || scale_factor_spatial <= 0) {
        OPENVINO_THROW("WanVAEConfig.scale_factor_temporal/spatial must be > 0");
    }
}

WanVAEConfig WanVAEConfig::from_json(const nlohmann::json& data) {
    WanVAEConfig cfg;
    parse_vae_config(data, cfg);
    return cfg;
}

WanVAEConfig WanVAEConfig::from_json_file(const std::filesystem::path& config_path) {
    nlohmann::json data;
    read_config_json_file(resolve_config_path(config_path), data);
    return from_json(data);
}

void WanWeightMapping::apply_transformer_packed_mapping(ov::genai::modeling::Module& model) {
    auto& rules = model.packed_mapping().rules;
    rules.push_back({"transformer.", "", 0});
    rules.push_back({"model.", "", 0});
}

void WanWeightMapping::apply_vae_packed_mapping(ov::genai::modeling::Module& model) {
    auto& rules = model.packed_mapping().rules;
    rules.push_back({"vae.", "", 0});
    rules.push_back({"model.", "", 0});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
