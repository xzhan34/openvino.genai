// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {

class LMHead {
public:
    explicit LMHead(const Tensor& weight);

    // Decode path: compute logits for all tokens in `x`.
    Tensor operator()(const Tensor& x) const;

    // Prefill path: `x` is expected to be packed as [total_tokens, hidden],
    // `cu_seqlens_q` is expected to be [batch + 1] (prefix sums).
    // The layer selects the last token per sequence, then computes logits.
    Tensor operator()(const Tensor& x, const Tensor& cu_seqlens_q) const;

private:
    Tensor weight_;
};

}  // namespace modeling
}  // namespace genai
}  // namespace ov

