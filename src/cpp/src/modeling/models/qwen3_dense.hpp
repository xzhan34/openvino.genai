// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    float rms_norm_eps = 1e-6f;
    std::string hidden_act = "silu";
    bool tie_word_embeddings = false;
};

class Qwen3MLP : public Module {
public:
    Qwen3MLP(BuilderContext& ctx, const std::string& name, const Qwen3DenseConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

private:
    const Tensor& gate_proj_weight() const;
    const Tensor& up_proj_weight() const;
    const Tensor& down_proj_weight() const;

    WeightParameter* gate_proj_param_ = nullptr;
    WeightParameter* up_proj_param_ = nullptr;
    WeightParameter* down_proj_param_ = nullptr;
};

class Qwen3DecoderLayer : public Module {
public:
    Qwen3DecoderLayer(BuilderContext& ctx, const std::string& name, const Qwen3DenseConfig& cfg, Module* parent = nullptr);

    std::pair<Tensor, Tensor> forward(const Tensor& positions,
                                      const Tensor& hidden_states,
                                      const std::optional<Tensor>& residual) const;

private:
    Qwen3MLP mlp_;
    RMSNorm input_layernorm_;
    RMSNorm post_attention_layernorm_;
};

class Qwen3Model : public Module {
public:
    Qwen3Model(BuilderContext& ctx, const Qwen3DenseConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids);

    VocabEmbedding& embed_tokens();
    RMSNorm& norm();

private:
    VocabEmbedding embed_tokens_;
    std::vector<Qwen3DecoderLayer> layers_;
    RMSNorm norm_;
};

class Qwen3ForCausalLM : public Module {
public:
    Qwen3ForCausalLM(BuilderContext& ctx, const Qwen3DenseConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids);

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

