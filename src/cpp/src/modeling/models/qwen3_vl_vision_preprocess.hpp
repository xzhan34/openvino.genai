// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "openvino/genai/visibility.hpp"
#include <openvino/openvino.hpp>

#include "modeling/models/qwen3_vl_spec.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct OPENVINO_GENAI_EXPORTS Qwen3VLVisionPreprocessConfig {
    int64_t min_pixels = 56 * 56;
    int64_t max_pixels = 28 * 28 * 1280;
    int32_t patch_size = 16;
    int32_t temporal_patch_size = 2;
    int32_t merge_size = 2;
    std::array<float, 3> image_mean = {0.5f, 0.5f, 0.5f};
    std::array<float, 3> image_std = {0.5f, 0.5f, 0.5f};
    bool do_resize = true;

    static Qwen3VLVisionPreprocessConfig from_json_file(const std::filesystem::path& path);
};

struct OPENVINO_GENAI_EXPORTS Qwen3VLVisionInputs {
    ov::Tensor pixel_values;
    ov::Tensor grid_thw;
    ov::Tensor pos_embeds;
    ov::Tensor rotary_cos;
    ov::Tensor rotary_sin;
};

class OPENVINO_GENAI_EXPORTS Qwen3VLVisionPreprocessor {
public:
    Qwen3VLVisionPreprocessor(const Qwen3VLVisionConfig& vision_cfg,
                              const Qwen3VLVisionPreprocessConfig& preprocess_cfg);

    Qwen3VLVisionInputs preprocess(const ov::Tensor& images,
                                   const ov::Tensor& pos_embed_weight) const;

    static int64_t count_visual_tokens(const ov::Tensor& grid_thw,
                                       int32_t spatial_merge_size);

private:
    Qwen3VLVisionConfig vision_cfg_;
    Qwen3VLVisionPreprocessConfig preprocess_cfg_;
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
