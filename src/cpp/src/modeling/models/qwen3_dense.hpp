// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <openvino/openvino.hpp>

#include "modeling/ops/context.hpp"
#include "modeling/weights/weights.hpp"

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

std::shared_ptr<ov::Model> build_qwen3_dense_dummy(const Qwen3DenseConfig& cfg,
                                                   weights::IWeightProvider& weights,
                                                   OpContext& ctx);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
