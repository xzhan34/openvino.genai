// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include "modeling/builder_context.hpp"
#include "modeling/layers/lm_head.hpp"
#include "modeling/layers/rms_norm.hpp"
#include "modeling/layers/vocab_embedding.hpp"
#include "modeling/ops/tensor.hpp"

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

class Qwen3Model : public Module {
public:
    Qwen3Model(BuilderContext& ctx, const Qwen3DenseConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& attention_mask,
                   const Tensor& position_ids,
                   const Tensor& beam_idx);

    VocabEmbedding& embed_tokens();
    RMSNorm& norm();

private:
    VocabEmbedding embed_tokens_;
    RMSNorm norm_;
};

class Qwen3ForCausalLM : public Module {
public:
    Qwen3ForCausalLM(BuilderContext& ctx, const Qwen3DenseConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& attention_mask,
                   const Tensor& position_ids,
                   const Tensor& beam_idx);

    Qwen3Model& model();
    LMHead& lm_head();

private:
    Qwen3DenseConfig cfg_;
    Qwen3Model model_;
    LMHead lm_head_;
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

