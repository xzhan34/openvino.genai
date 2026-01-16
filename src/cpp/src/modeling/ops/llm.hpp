// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "modeling/ops/op_policy.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace ops {
namespace llm {

std::pair<Tensor, Tensor> rope_cos_sin(const Tensor& positions,
                                       int32_t head_dim,
                                       float rope_theta,
                                       const OpPolicy* policy = nullptr);
Tensor apply_rope(const Tensor& x,
                  const Tensor& cos,
                  const Tensor& sin,
                  int32_t head_dim,
                  const OpPolicy* policy = nullptr);
Tensor apply_rope_interleave(const Tensor& x,
                             const Tensor& cos,
                             const Tensor& sin,
                             int32_t head_dim,
                             const OpPolicy* policy = nullptr);
Tensor repeat_kv(const Tensor& x, int32_t num_heads, int32_t num_kv_heads, int32_t head_dim);
Tensor causal_mask_from_seq_len(const Tensor& seq_len);
Tensor causal_mask(const Tensor& scores);
// Build causal mask that works with KV cache: Q shape [batch, heads, q_len, head_dim],
// K shape [batch, heads, kv_len, head_dim]. Returns mask [batch, 1, q_len, kv_len].
// For decode step: q_len=1, mask allows attending to all cached + current positions.
Tensor build_kv_causal_mask(const Tensor& q, const Tensor& k);
// Helpers for handling qk_head_dim != v_head_dim in SDPA.
Tensor pad_to_head_dim(const Tensor& x, int32_t head_dim, int32_t target_head_dim);
Tensor slice_to_head_dim(const Tensor& x, int32_t head_dim, int32_t target_head_dim);
Tensor sdpa(const Tensor& q,
            const Tensor& k,
            const Tensor& v,
            float scale,
            int64_t softmax_axis,
            const Tensor* mask = nullptr,
            bool causal = false,
            const OpPolicy* policy = nullptr);

}  // namespace llm
}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
