// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {

class VocabEmbedding : public Module {
public:
    explicit VocabEmbedding(const Tensor& weight);
    VocabEmbedding(BuilderContext& ctx, const std::string& name, Module* parent = nullptr);

    // Equivalent to torch.nn.functional.embedding(ids, weight).
    Tensor operator()(const Tensor& ids) const;
    Parameter& weight_param();
    const Parameter& weight_param() const;

private:
    const Tensor& weight() const;

    Tensor weight_;
    Parameter* weight_param_ = nullptr;
};

}  // namespace modeling
}  // namespace genai
}  // namespace ov
