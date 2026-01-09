// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_vl_input_planner.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>

#include <openvino/core/except.hpp>

namespace {

template <typename T>
bool mask_value_at(const ov::Tensor& mask, size_t index) {
    const T* data = mask.data<const T>();
    return data[index] != static_cast<T>(0);
}

bool mask_value(const ov::Tensor& mask, size_t index) {
    switch (mask.get_element_type()) {
        case ov::element::boolean:
            return mask_value_at<char>(mask, index);
        case ov::element::i32:
            return mask_value_at<int32_t>(mask, index);
        case ov::element::i64:
            return mask_value_at<int64_t>(mask, index);
        case ov::element::u8:
            return mask_value_at<uint8_t>(mask, index);
        default:
            OPENVINO_THROW("Unsupported attention_mask dtype");
    }
}

void set_bool(ov::Tensor& mask, size_t index, bool value) {
    auto* data = mask.data<char>();
    data[index] = value ? 1 : 0;
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

Qwen3VLInputPlanner::Qwen3VLInputPlanner(const Qwen3VLConfig& cfg)
    : image_token_id_(cfg.image_token_id),
      vision_start_token_id_(cfg.vision_start_token_id),
      spatial_merge_size_(cfg.vision.spatial_merge_size) {}

ov::Tensor Qwen3VLInputPlanner::build_visual_pos_mask(const ov::Tensor& input_ids,
                                                      const ov::Tensor* attention_mask) const {
    if (input_ids.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("input_ids must be i64 for Qwen3VLInputPlanner");
    }
    const auto shape = input_ids.get_shape();
    if (shape.size() != 2) {
        OPENVINO_THROW("input_ids must have shape [B, S]");
    }
    if (attention_mask && attention_mask->get_shape() != shape) {
        OPENVINO_THROW("attention_mask must have the same shape as input_ids");
    }
    ov::Tensor mask(ov::element::boolean, shape);
    const int64_t* ids = input_ids.data<const int64_t>();
    const size_t total = input_ids.get_size();

    for (size_t idx = 0; idx < total; ++idx) {
        bool active = ids[idx] == image_token_id_;
        if (attention_mask && !mask_value(*attention_mask, idx)) {
            active = false;
        }
        set_bool(mask, idx, active);
    }
    return mask;
}

Qwen3VLInputPlan Qwen3VLInputPlanner::build_plan(const ov::Tensor& input_ids,
                                                 const ov::Tensor* attention_mask,
                                                 const ov::Tensor* image_grid_thw) const {
    if (input_ids.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("input_ids must be i64 for Qwen3VLInputPlanner");
    }
    const auto shape = input_ids.get_shape();
    if (shape.size() != 2) {
        OPENVINO_THROW("input_ids must have shape [B, S]");
    }
    if (attention_mask && attention_mask->get_shape() != shape) {
        OPENVINO_THROW("attention_mask must have the same shape as input_ids");
    }
    if (image_grid_thw) {
        const auto grid_shape = image_grid_thw->get_shape();
        if (image_grid_thw->get_element_type() != ov::element::i64) {
            OPENVINO_THROW("image_grid_thw must be i64");
        }
        if (grid_shape.size() != 2 || grid_shape[1] != 3) {
            OPENVINO_THROW("image_grid_thw must have shape [N, 3]");
        }
    }
    if (spatial_merge_size_ <= 0) {
        OPENVINO_THROW("spatial_merge_size must be > 0");
    }
    const size_t batch = shape[0];
    const size_t seq_len = shape[1];

    ov::Tensor position_ids(ov::element::i64, {3, batch, seq_len});
    std::memset(position_ids.data(), 0, position_ids.get_byte_size());

    ov::Tensor rope_deltas(ov::element::i64, {batch, 1});
    std::memset(rope_deltas.data(), 0, rope_deltas.get_byte_size());

    auto visual_pos_mask = build_visual_pos_mask(input_ids, attention_mask);

    const int64_t* ids = input_ids.data<const int64_t>();
    const int64_t* grid = image_grid_thw ? image_grid_thw->data<const int64_t>() : nullptr;
    const size_t grid_rows = image_grid_thw ? image_grid_thw->get_shape().at(0) : 0;
    size_t grid_index = 0;

    auto pos_data = position_ids.data<int64_t>();
    auto delta_data = rope_deltas.data<int64_t>();

    for (size_t b = 0; b < batch; ++b) {
        std::vector<int64_t> tokens;
        std::vector<size_t> active_indices;
        tokens.reserve(seq_len);
        active_indices.reserve(seq_len);

        for (size_t s = 0; s < seq_len; ++s) {
            const size_t idx = b * seq_len + s;
            if (attention_mask && !mask_value(*attention_mask, idx)) {
                continue;
            }
            tokens.push_back(ids[idx]);
            active_indices.push_back(s);
        }

        if (tokens.empty()) {
            delta_data[b] = 0;
            continue;
        }

        std::vector<int64_t> pos_t;
        std::vector<int64_t> pos_h;
        std::vector<int64_t> pos_w;
        pos_t.reserve(tokens.size());
        pos_h.reserve(tokens.size());
        pos_w.reserve(tokens.size());

        int64_t last_max = -1;
        size_t st = 0;
        size_t local_grid_index = grid_index;

        auto append_text = [&](size_t length) {
            if (length == 0) {
                return;
            }
            const int64_t base = last_max + 1;
            for (size_t i = 0; i < length; ++i) {
                const int64_t value = base + static_cast<int64_t>(i);
                pos_t.push_back(value);
                pos_h.push_back(value);
                pos_w.push_back(value);
            }
            last_max = base + static_cast<int64_t>(length) - 1;
        };

        auto append_visual = [&](int64_t t, int64_t h, int64_t w) {
            if (t <= 0 || h <= 0 || w <= 0) {
                OPENVINO_THROW("Invalid grid_thw values in Qwen3VLInputPlanner");
            }
            const int64_t llm_grid_t = t;
            const int64_t llm_grid_h = h / spatial_merge_size_;
            const int64_t llm_grid_w = w / spatial_merge_size_;
            if (llm_grid_h <= 0 || llm_grid_w <= 0) {
                OPENVINO_THROW("Invalid spatial_merge_size for grid_thw");
            }
            const int64_t base = last_max + 1;
            int64_t max_dim = 0;
            for (int64_t tt = 0; tt < llm_grid_t; ++tt) {
                for (int64_t hh = 0; hh < llm_grid_h; ++hh) {
                    for (int64_t ww = 0; ww < llm_grid_w; ++ww) {
                        pos_t.push_back(base + tt);
                        pos_h.push_back(base + hh);
                        pos_w.push_back(base + ww);
                        max_dim = std::max(max_dim, std::max(tt, std::max(hh, ww)));
                    }
                }
            }
            last_max = base + max_dim;
        };

        if (image_grid_thw) {
            while (true) {
                auto start_it = tokens.begin() + static_cast<std::vector<int64_t>::difference_type>(st);
                auto it = std::find(start_it, tokens.end(), image_token_id_);
                if (it == tokens.end()) {
                    break;
                }
                const size_t ed = static_cast<size_t>(std::distance(tokens.begin(), it));
                if (local_grid_index >= grid_rows) {
                    OPENVINO_THROW("image_grid_thw entries are fewer than image tokens");
                }
                append_text(ed - st);
                const int64_t t = grid[local_grid_index * 3 + 0];
                const int64_t h = grid[local_grid_index * 3 + 1];
                const int64_t w = grid[local_grid_index * 3 + 2];
                append_visual(t, h, w);

                const int64_t llm_grid_h = h / spatial_merge_size_;
                const int64_t llm_grid_w = w / spatial_merge_size_;
                const int64_t visual_len = t * llm_grid_h * llm_grid_w;
                if (ed + static_cast<size_t>(visual_len) > tokens.size()) {
                    OPENVINO_THROW("Image tokens length does not match grid_thw");
                }
                st = ed + static_cast<size_t>(visual_len);
                local_grid_index += 1;
            }
        }

        if (st < tokens.size()) {
            append_text(tokens.size() - st);
        }

        if (pos_t.size() != tokens.size()) {
            OPENVINO_THROW("Position ids length mismatch");
        }

        int64_t max_pos = pos_t.empty() ? 0 : pos_t.front();
        for (size_t i = 0; i < pos_t.size(); ++i) {
            max_pos = std::max(max_pos, pos_t[i]);
            max_pos = std::max(max_pos, pos_h[i]);
            max_pos = std::max(max_pos, pos_w[i]);
        }

        for (size_t i = 0; i < tokens.size(); ++i) {
            const size_t s = active_indices[i];
            const size_t base = b * seq_len + s;
            pos_data[0 * batch * seq_len + base] = pos_t[i];
            pos_data[1 * batch * seq_len + base] = pos_h[i];
            pos_data[2 * batch * seq_len + base] = pos_w[i];
        }

        if (attention_mask) {
            for (size_t s = 0; s < seq_len; ++s) {
                const size_t idx = b * seq_len + s;
                if (mask_value(*attention_mask, idx)) {
                    continue;
                }
                pos_data[0 * batch * seq_len + idx] = 1;
                pos_data[1 * batch * seq_len + idx] = 1;
                pos_data[2 * batch * seq_len + idx] = 1;
            }
        }

        delta_data[b] = max_pos + 1 - static_cast<int64_t>(seq_len);
        grid_index = local_grid_index;
    }

    return {position_ids, visual_pos_mask, rope_deltas};
}

ov::Tensor Qwen3VLInputPlanner::scatter_visual_embeds(const ov::Tensor& visual_embeds,
                                                      const ov::Tensor& visual_pos_mask) {
    const auto mask_shape = visual_pos_mask.get_shape();
    if (mask_shape.size() != 2) {
        OPENVINO_THROW("visual_pos_mask must have shape [B, S]");
    }
    const auto embeds_shape = visual_embeds.get_shape();
    if (embeds_shape.size() != 2) {
        OPENVINO_THROW("visual_embeds must have shape [V, H]");
    }
    const size_t batch = mask_shape[0];
    const size_t seq_len = mask_shape[1];
    const size_t hidden = embeds_shape[1];

    ov::Tensor out(visual_embeds.get_element_type(), {batch, seq_len, hidden});
    std::memset(out.data(), 0, out.get_byte_size());

    const size_t elem_size = visual_embeds.get_element_type().size();
    const size_t row_bytes = hidden * elem_size;

    const char* src = static_cast<const char*>(visual_embeds.data());
    char* dst = static_cast<char*>(out.data());

    size_t visual_idx = 0;
    const size_t total = batch * seq_len;
    for (size_t idx = 0; idx < total; ++idx) {
        if (!mask_value(visual_pos_mask, idx)) {
            continue;
        }
        if (visual_idx >= embeds_shape[0]) {
            OPENVINO_THROW("visual_embeds shorter than visual_pos_mask");
        }
        std::memcpy(dst + idx * row_bytes, src + visual_idx * row_bytes, row_bytes);
        visual_idx++;
    }
    if (visual_idx != embeds_shape[0]) {
        OPENVINO_THROW("visual_embeds length does not match visual_pos_mask");
    }
    return out;
}

std::vector<ov::Tensor> Qwen3VLInputPlanner::scatter_deepstack_embeds(
    const std::vector<ov::Tensor>& deepstack_embeds,
    const ov::Tensor& visual_pos_mask) {
    std::vector<ov::Tensor> out;
    out.reserve(deepstack_embeds.size());
    for (const auto& embed : deepstack_embeds) {
        out.push_back(scatter_visual_embeds(embed, visual_pos_mask));
    }
    return out;
}

ov::Tensor Qwen3VLInputPlanner::build_decode_position_ids(const ov::Tensor& rope_deltas,
                                                          int64_t past_length,
                                                          int64_t seq_len) {
    if (rope_deltas.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("rope_deltas must be i64");
    }
    if (past_length < 0 || seq_len <= 0) {
        OPENVINO_THROW("Invalid past_length or seq_len for decode position ids");
    }
    const auto shape = rope_deltas.get_shape();
    size_t batch = 0;
    if (shape.size() == 1) {
        batch = shape[0];
    } else if (shape.size() == 2) {
        if (shape[1] != 1) {
            OPENVINO_THROW("rope_deltas must have shape [B] or [B, 1]");
        }
        batch = shape[0];
    } else {
        OPENVINO_THROW("rope_deltas must have shape [B] or [B, 1]");
    }

    ov::Tensor position_ids(ov::element::i64, {3, batch, static_cast<size_t>(seq_len)});
    auto* out = position_ids.data<int64_t>();
    const int64_t* deltas = rope_deltas.data<const int64_t>();
    const size_t plane_stride = batch * static_cast<size_t>(seq_len);

    for (size_t b = 0; b < batch; ++b) {
        const int64_t base = past_length + deltas[b];
        for (int64_t s = 0; s < seq_len; ++s) {
            const int64_t value = base + s;
            const size_t idx = b * static_cast<size_t>(seq_len) + static_cast<size_t>(s);
            out[idx] = value;
            out[plane_stride + idx] = value;
            out[2 * plane_stride + idx] = value;
        }
    }

    return position_ids;
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
