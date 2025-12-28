// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "modeling/weights/weights.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct Qwen3DenseWeights {
    Tensor embed_tokens;
    Tensor final_norm;
    Tensor lm_head;
};

inline Qwen3DenseWeights load_qwen3_dense_weights(weights::IWeightProvider& weights) {
    Qwen3DenseWeights out;
    out.embed_tokens = weights.get("model.embed_tokens");
    out.final_norm = weights.get("model.norm");

    // Keep tie embedding logic out of the model graph expression.
    if (weights.has("lm_head.weight")) {
        out.lm_head = weights.get("lm_head");
    } else {
        out.lm_head = out.embed_tokens;
    }

    return out;
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

