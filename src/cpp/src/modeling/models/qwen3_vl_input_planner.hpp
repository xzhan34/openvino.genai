// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "openvino/genai/visibility.hpp"
#include <openvino/openvino.hpp>

#include "modeling/models/qwen3_vl_spec.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct OPENVINO_GENAI_EXPORTS Qwen3VLInputPlan {
    ov::Tensor position_ids;
    ov::Tensor visual_pos_mask;
    ov::Tensor rope_deltas;
};

class OPENVINO_GENAI_EXPORTS Qwen3VLInputPlanner {
public:
    explicit Qwen3VLInputPlanner(const Qwen3VLConfig& cfg);

    Qwen3VLInputPlan build_plan(const ov::Tensor& input_ids,
                                const ov::Tensor* attention_mask = nullptr,
                                const ov::Tensor* image_grid_thw = nullptr) const;

    ov::Tensor build_visual_pos_mask(const ov::Tensor& input_ids,
                                     const ov::Tensor* attention_mask = nullptr) const;

    static ov::Tensor scatter_visual_embeds(const ov::Tensor& visual_embeds,
                                            const ov::Tensor& visual_pos_mask);

    static std::vector<ov::Tensor> scatter_deepstack_embeds(const std::vector<ov::Tensor>& deepstack_embeds,
                                                            const ov::Tensor& visual_pos_mask);

    static ov::Tensor build_decode_position_ids(const ov::Tensor& rope_deltas,
                                                int64_t past_length,
                                                int64_t seq_len);

private:
    int64_t image_token_id_ = 0;
    int64_t vision_start_token_id_ = 0;
    int32_t spatial_merge_size_ = 1;
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
