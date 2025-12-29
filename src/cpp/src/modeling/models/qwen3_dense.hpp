// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct Qwen3DenseConfig {
    std::string architecture = "qwen3";
    int32_t hidden_size = 0;
    float rms_norm_eps = 1e-6f;
    bool tie_word_embeddings = false;
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
