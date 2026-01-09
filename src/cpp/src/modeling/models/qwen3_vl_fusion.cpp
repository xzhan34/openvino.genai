// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_vl_fusion.hpp"

#include "modeling/ops/tensor_ops.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

EmbeddingInjector::EmbeddingInjector(BuilderContext& ctx, const std::string& name, Module* parent)
    : Module(name, ctx, parent) {}

Tensor EmbeddingInjector::forward(const Tensor& inputs_embeds,
                                  const Tensor& visual_embeds,
                                  const Tensor& visual_pos_mask) const {
    auto mask = visual_pos_mask.unsqueeze(2);
    auto updates = visual_embeds.to(inputs_embeds.dtype());
    return ops::tensor::masked_scatter(inputs_embeds, mask, updates);
}

DeepstackInjector::DeepstackInjector(BuilderContext& ctx, const std::string& name, Module* parent)
    : Module(name, ctx, parent) {}

Tensor DeepstackInjector::forward(const Tensor& hidden_states,
                                  const Tensor& visual_pos_mask,
                                  const Tensor& deepstack_embeds) const {
    auto mask = visual_pos_mask.unsqueeze(2);
    auto updates = deepstack_embeds.to(hidden_states.dtype());
    return ops::tensor::masked_add(hidden_states, mask, updates);
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
