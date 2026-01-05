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
Tensor repeat_kv(const Tensor& x, int32_t num_heads, int32_t num_kv_heads, int32_t head_dim);
Tensor causal_mask_from_seq_len(const Tensor& seq_len);
Tensor causal_mask(const Tensor& scores);
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
