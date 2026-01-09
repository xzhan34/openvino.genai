// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "modeling/builder_context.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

class EmbeddingInjector : public Module {
public:
    EmbeddingInjector(BuilderContext& ctx, const std::string& name, Module* parent = nullptr);

    // visual_embeds is expected to be aligned with inputs_embeds (e.g., [B, S, H]).
    Tensor forward(const Tensor& inputs_embeds,
                   const Tensor& visual_embeds,
                   const Tensor& visual_pos_mask) const;
};

class DeepstackInjector : public Module {
public:
    DeepstackInjector(BuilderContext& ctx, const std::string& name, Module* parent = nullptr);

    // deepstack_embeds is expected to be aligned with hidden_states (e.g., [B, S, H]).
    Tensor forward(const Tensor& hidden_states,
                   const Tensor& visual_pos_mask,
                   const Tensor& deepstack_embeds) const;
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
