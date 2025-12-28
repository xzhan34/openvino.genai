// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace weights {

class IWeightProvider {
public:
    virtual ~IWeightProvider() = default;

    // Checks whether the underlying weight source contains a tensor with the exact key,
    // e.g. "lm_head.weight", "model.embed_tokens.scales".
    virtual bool has(const std::string& key) const = 0;

    // Returns a weight tensor for `base_key` (without ".weight"), e.g. "model.embed_tokens".
    // The returned tensor is expected to be compatible with Modeling layer blocks.
    virtual Tensor get(const std::string& base_key) = 0;
};

}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov

