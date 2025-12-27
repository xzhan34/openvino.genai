// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {

class VocabEmbedding {
public:
    explicit VocabEmbedding(const Tensor& weight);

    // Equivalent to torch.nn.functional.embedding(ids, weight).
    Tensor operator()(const Tensor& ids) const;

private:
    Tensor weight_;
};

}  // namespace modeling
}  // namespace genai
}  // namespace ov

