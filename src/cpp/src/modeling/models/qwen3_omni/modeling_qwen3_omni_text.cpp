// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_omni/modeling_qwen3_omni_internal.hpp"

#include <cmath>

#include <openvino/openvino.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/layers/lm_head.hpp"
#include "modeling/layers/rms_norm.hpp"
#include "modeling/layers/vocab_embedding.hpp"
#include "modeling/ops/llm.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/rope.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/weights/weight_loader.hpp"
#include "modeling/models/dflash_draft/dflash_draft.hpp"
#include "modeling/models/qwen3_omni/processing_qwen3_omni_vl.hpp"

namespace {

ov::genai::modeling::Tensor add_bias_if_present(const ov::genai::modeling::Tensor& x,
                                                const ov::genai::modeling::Tensor* bias) {
    if (!bias) {
        return x;
    }
    return x + *bias;
}

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

class Qwen3OmniEmbeddingInjector : public Module {
public:
    Qwen3OmniEmbeddingInjector(BuilderContext& ctx, const std::string& name, Module* parent = nullptr)
        : Module(name, ctx, parent) {}

    Tensor forward(const Tensor& inputs_embeds,
                   const Tensor& visual_embeds,
                   const Tensor& visual_pos_mask) const {
        auto mask = visual_pos_mask.unsqueeze(2);
        auto updates = visual_embeds.to(inputs_embeds.dtype());
        return ops::tensor::masked_scatter(inputs_embeds, mask, updates);
    }
};

class Qwen3OmniDeepstackInjector : public Module {
public:
    Qwen3OmniDeepstackInjector(BuilderContext& ctx, const std::string& name, Module* parent = nullptr)
        : Module(name, ctx, parent) {}

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& visual_pos_mask,
                   const Tensor& deepstack_embeds) const {
        auto mask = visual_pos_mask.unsqueeze(2);
        auto updates = deepstack_embeds.to(hidden_states.dtype());
        return ops::tensor::masked_add(hidden_states, mask, updates);
    }
};

class Qwen3OmniTextAttention : public Module {
public:
    Qwen3OmniTextAttention(BuilderContext& ctx,
                           const std::string& name,
                           const Qwen3OmniTextConfig& cfg,
                           Module* parent = nullptr)
        : Module(name, ctx, parent),
          num_heads_(cfg.num_attention_heads),
          num_kv_heads_(cfg.kv_heads()),
          head_dim_(cfg.resolved_head_dim()),
          hidden_size_(cfg.hidden_size),
          scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))),
          q_norm_(ctx, "q_norm", cfg.rms_norm_eps, this),
          k_norm_(ctx, "k_norm", cfg.rms_norm_eps, this) {
        if (num_heads_ <= 0 || head_dim_ <= 0) {
            OPENVINO_THROW("Invalid Qwen3OmniTextAttention head configuration");
        }
        if (num_heads_ % num_kv_heads_ != 0) {
            OPENVINO_THROW("num_attention_heads must be divisible by num_key_value_heads");
        }

        register_module("q_norm", &q_norm_);
        register_module("k_norm", &k_norm_);

        q_proj_param_ = &register_parameter("q_proj.weight");
        k_proj_param_ = &register_parameter("k_proj.weight");
        v_proj_param_ = &register_parameter("v_proj.weight");
        o_proj_param_ = &register_parameter("o_proj.weight");

        q_bias_param_ = &register_parameter("q_proj.bias");
        k_bias_param_ = &register_parameter("k_proj.bias");
        v_bias_param_ = &register_parameter("v_proj.bias");
        o_bias_param_ = &register_parameter("o_proj.bias");

        if (!cfg.attention_bias) {
            q_bias_param_->set_optional(true);
            k_bias_param_->set_optional(true);
            v_bias_param_->set_optional(true);
            o_bias_param_->set_optional(true);
        }
    }

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& beam_idx,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin) const {
        auto q = add_bias_if_present(ops::linear(hidden_states, q_proj_weight()), q_proj_bias());
        auto k = add_bias_if_present(ops::linear(hidden_states, k_proj_weight()), k_proj_bias());
        auto v = add_bias_if_present(ops::linear(hidden_states, v_proj_weight()), v_proj_bias());

        auto q_heads = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
        auto k_heads = k.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});
        auto v_heads = v.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});

        if (q_norm_.weight_param().is_bound()) {
            q_heads = q_norm_.forward(q_heads);
        }
        if (k_norm_.weight_param().is_bound()) {
            k_heads = k_norm_.forward(k_heads);
        }

        auto* policy = &ctx().op_policy();
        auto q_rot = ops::llm::apply_rope(q_heads, rope_cos, rope_sin, head_dim_, policy);
        auto k_rot = ops::llm::apply_rope(k_heads, rope_cos, rope_sin, head_dim_, policy);

        auto cached = append_kv_cache(k_rot, v_heads, beam_idx);
        auto k_expanded = ops::llm::repeat_kv(cached.first, num_heads_, num_kv_heads_, head_dim_);
        auto v_expanded = ops::llm::repeat_kv(cached.second, num_heads_, num_kv_heads_, head_dim_);

        auto mask = ops::llm::build_kv_causal_mask(q_rot, k_expanded);
        auto context = ops::llm::sdpa(q_rot, k_expanded, v_expanded, scaling_, 3, &mask, false, policy);
        const int64_t attn_out_dim = static_cast<int64_t>(num_heads_) * head_dim_;
        auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});
        auto out = add_bias_if_present(ops::linear(merged, o_proj_weight()), o_proj_bias());
        return out;
    }

private:
    const Tensor& q_proj_weight() const {
        if (!q_proj_param_) {
            OPENVINO_THROW("Qwen3OmniTextAttention q_proj parameter not registered");
        }
        return q_proj_param_->value();
    }

    const Tensor& k_proj_weight() const {
        if (!k_proj_param_) {
            OPENVINO_THROW("Qwen3OmniTextAttention k_proj parameter not registered");
        }
        return k_proj_param_->value();
    }

    const Tensor& v_proj_weight() const {
        if (!v_proj_param_) {
            OPENVINO_THROW("Qwen3OmniTextAttention v_proj parameter not registered");
        }
        return v_proj_param_->value();
    }

    const Tensor& o_proj_weight() const {
        if (!o_proj_param_) {
            OPENVINO_THROW("Qwen3OmniTextAttention o_proj parameter not registered");
        }
        return o_proj_param_->value();
    }

    const Tensor* q_proj_bias() const {
        return (q_bias_param_ && q_bias_param_->is_bound()) ? &q_bias_param_->value() : nullptr;
    }

    const Tensor* k_proj_bias() const {
        return (k_bias_param_ && k_bias_param_->is_bound()) ? &k_bias_param_->value() : nullptr;
    }

    const Tensor* v_proj_bias() const {
        return (v_bias_param_ && v_bias_param_->is_bound()) ? &v_bias_param_->value() : nullptr;
    }

    const Tensor* o_proj_bias() const {
        return (o_bias_param_ && o_bias_param_->is_bound()) ? &o_bias_param_->value() : nullptr;
    }

    std::pair<Tensor, Tensor> append_kv_cache(const Tensor& keys,
                                              const Tensor& values,
                                              const Tensor& beam_idx) const {
        auto* op_ctx = keys.context();
        auto batch = shape::dim(keys, 0);
        auto kv_heads = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(num_kv_heads_)});
        auto zero_len = ops::const_vec(op_ctx, std::vector<int64_t>{0});
        auto head_dim = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(head_dim_)});
        auto cache_shape = shape::make({batch, kv_heads, zero_len, head_dim});

        auto zero = Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx).to(keys.dtype());
        auto k_init = shape::broadcast_to(zero, cache_shape);
        auto v_init = shape::broadcast_to(zero, cache_shape);

        const std::string cache_prefix = full_path().empty() ? name() : full_path();
        const std::string k_name = cache_prefix + ".key_cache";
        const std::string v_name = cache_prefix + ".value_cache";

        ov::op::util::VariableInfo k_info{ov::PartialShape{-1, num_kv_heads_, -1, head_dim_},
                                          keys.dtype(),
                                          k_name};
        auto k_var = std::make_shared<ov::op::util::Variable>(k_info);
        auto k_read = std::make_shared<ov::op::v6::ReadValue>(k_init.output(), k_var);

        ov::op::util::VariableInfo v_info{ov::PartialShape{-1, num_kv_heads_, -1, head_dim_},
                                          values.dtype(),
                                          v_name};
        auto v_var = std::make_shared<ov::op::util::Variable>(v_info);
        auto v_read = std::make_shared<ov::op::v6::ReadValue>(v_init.output(), v_var);

        auto k_cached = ops::gather(Tensor(k_read->output(0), op_ctx), beam_idx, 0);
        auto v_cached = ops::gather(Tensor(v_read->output(0), op_ctx), beam_idx, 0);

        auto k_combined = ops::concat({k_cached, keys}, 2);
        auto v_combined = ops::concat({v_cached, values}, 2);

        auto k_assign = std::make_shared<ov::opset13::Assign>(k_combined.output(), k_var);
        auto v_assign = std::make_shared<ov::opset13::Assign>(v_combined.output(), v_var);
        ctx().register_sink(k_assign);
        ctx().register_sink(v_assign);

        return {k_combined, v_combined};
    }

    int32_t num_heads_ = 0;
    int32_t num_kv_heads_ = 0;
    int32_t head_dim_ = 0;
    int32_t hidden_size_ = 0;
    float scaling_ = 1.0f;

    RMSNorm q_norm_;
    RMSNorm k_norm_;

    WeightParameter* q_proj_param_ = nullptr;
    WeightParameter* k_proj_param_ = nullptr;
    WeightParameter* v_proj_param_ = nullptr;
    WeightParameter* o_proj_param_ = nullptr;

    WeightParameter* q_bias_param_ = nullptr;
    WeightParameter* k_bias_param_ = nullptr;
    WeightParameter* v_bias_param_ = nullptr;
    WeightParameter* o_bias_param_ = nullptr;
};

class Qwen3OmniTextMLP : public Module {
public:
    Qwen3OmniTextMLP(BuilderContext& ctx,
                     const std::string& name,
                     const Qwen3OmniTextConfig& cfg,
                     Module* parent = nullptr)
        : Module(name, ctx, parent) {
        if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
            OPENVINO_THROW("Unsupported Qwen3OmniText MLP activation: ", cfg.hidden_act);
        }
        gate_proj_param_ = &register_parameter("gate_proj.weight");
        up_proj_param_ = &register_parameter("up_proj.weight");
        down_proj_param_ = &register_parameter("down_proj.weight");
    }

    Tensor forward(const Tensor& x) const {
        auto gate = ops::linear(x, gate_proj_weight());
        auto up = ops::linear(x, up_proj_weight());
        auto gated = ops::silu(gate) * up;
        return ops::linear(gated, down_proj_weight());
    }

private:
    const Tensor& gate_proj_weight() const {
        if (!gate_proj_param_) {
            OPENVINO_THROW("Qwen3OmniTextMLP gate projection parameter not registered");
        }
        return gate_proj_param_->value();
    }

    const Tensor& up_proj_weight() const {
        if (!up_proj_param_) {
            OPENVINO_THROW("Qwen3OmniTextMLP up projection parameter not registered");
        }
        return up_proj_param_->value();
    }

    const Tensor& down_proj_weight() const {
        if (!down_proj_param_) {
            OPENVINO_THROW("Qwen3OmniTextMLP down projection parameter not registered");
        }
        return down_proj_param_->value();
    }

    WeightParameter* gate_proj_param_ = nullptr;
    WeightParameter* up_proj_param_ = nullptr;
    WeightParameter* down_proj_param_ = nullptr;
};

class Qwen3OmniTextDecoderLayer : public Module {
public:
    Qwen3OmniTextDecoderLayer(BuilderContext& ctx,
                              const std::string& name,
                              const Qwen3OmniTextConfig& cfg,
                              Module* parent = nullptr)
        : Module(name, ctx, parent),
          self_attn_(ctx, "self_attn", cfg, this),
          mlp_(ctx, "mlp", cfg, this),
          input_layernorm_(ctx, "input_layernorm", cfg.rms_norm_eps, this),
          post_attention_layernorm_(ctx, "post_attention_layernorm", cfg.rms_norm_eps, this) {
        register_module("self_attn", &self_attn_);
        register_module("mlp", &mlp_);
        register_module("input_layernorm", &input_layernorm_);
        register_module("post_attention_layernorm", &post_attention_layernorm_);
    }

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& beam_idx,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin) const {
        auto normed = input_layernorm_.forward(hidden_states);
        auto attn_out = self_attn_.forward(normed, beam_idx, rope_cos, rope_sin);
        auto residual = hidden_states + attn_out;
        auto post_norm = post_attention_layernorm_.forward(residual);
        auto mlp_out = mlp_.forward(post_norm);
        return residual + mlp_out;
    }

private:
    Qwen3OmniTextAttention self_attn_;
    Qwen3OmniTextMLP mlp_;
    RMSNorm input_layernorm_;
    RMSNorm post_attention_layernorm_;
};

class Qwen3OmniTextModel : public Module {
public:
    Qwen3OmniTextModel(BuilderContext& ctx, const Qwen3OmniTextConfig& cfg, Module* parent = nullptr)
        : Module("language_model", ctx, parent),
          cfg_(cfg),
          embed_tokens_(ctx, "embed_tokens", this),
          embedding_injector_(ctx, "embedding_injector", this),
          deepstack_injector_(ctx, "deepstack_injector", this),
          layers_(),
          norm_(ctx, "norm", cfg.rms_norm_eps, this),
          head_dim_(cfg.resolved_head_dim()) {
        register_module("embed_tokens", &embed_tokens_);
        register_module("embedding_injector", &embedding_injector_);
        register_module("deepstack_injector", &deepstack_injector_);
        register_module("norm", &norm_);

        if (!cfg_.rope.rope_type.empty() && cfg_.rope.rope_type != "default") {
            OPENVINO_THROW("Unsupported Qwen3Omni rope_type: ", cfg_.rope.rope_type);
        }

        layers_.reserve(static_cast<size_t>(cfg.num_hidden_layers));
        for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
            const std::string layer_name = std::string("layers.") + std::to_string(i);
            layers_.emplace_back(ctx, layer_name, cfg, this);
            register_module(layer_name, &layers_.back());
        }
    }

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx,
                   const Tensor* visual_embeds = nullptr,
                   const Tensor* visual_pos_mask = nullptr,
                   const Tensor* audio_features = nullptr,
                   const Tensor* audio_pos_mask = nullptr,
                   const std::vector<Tensor>* deepstack_embeds = nullptr) {
        auto hidden_states = embed_tokens_.forward(input_ids);
        return forward_embeds(hidden_states,
                              position_ids,
                              beam_idx,
                              visual_embeds,
                              visual_pos_mask,
                              audio_features,
                              audio_pos_mask,
                              deepstack_embeds);
    }

    Tensor forward_embeds(const Tensor& inputs_embeds,
                          const Tensor& position_ids,
                          const Tensor& beam_idx,
                          const Tensor* visual_embeds = nullptr,
                          const Tensor* visual_pos_mask = nullptr,
                          const Tensor* audio_features = nullptr,
                          const Tensor* audio_pos_mask = nullptr,
                          const std::vector<Tensor>* deepstack_embeds = nullptr) {
        auto cos_sin = build_mrope_cos_sin(position_ids);
        Tensor hidden_states = inputs_embeds;
        if (visual_embeds && visual_pos_mask) {
            hidden_states = embedding_injector_.forward(hidden_states, *visual_embeds, *visual_pos_mask);
        }
        if (audio_features && audio_pos_mask) {
            hidden_states = embedding_injector_.forward(hidden_states, *audio_features, *audio_pos_mask);
        }
        // Sort capture IDs for efficient lookup during the layer loop
        auto sorted_capture_ids = capture_layer_ids_;
        std::sort(sorted_capture_ids.begin(), sorted_capture_ids.end());
        size_t capture_idx = 0;

        for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
            hidden_states = layers_[layer_idx].forward(hidden_states, beam_idx, cos_sin.first, cos_sin.second);
            if (deepstack_embeds && visual_pos_mask && layer_idx < deepstack_embeds->size()) {
                hidden_states = deepstack_injector_.forward(hidden_states,
                                                            *visual_pos_mask,
                                                            (*deepstack_embeds)[layer_idx]);
            }
            // Capture intermediate hidden state at selected layer indices
            if (capture_idx < sorted_capture_ids.size() &&
                static_cast<int32_t>(layer_idx) == sorted_capture_ids[capture_idx]) {
                captured_hidden_.push_back(hidden_states);
                ++capture_idx;
            }
        }
        return norm_.forward(hidden_states);
    }

    VocabEmbedding& embed_tokens() {
        return embed_tokens_;
    }

    /// Run the same forward as forward_embeds, but also capture hidden states
    /// at the specified layer indices and return them concatenated along dim=-1.
    std::pair<Tensor, Tensor> forward_with_selected_layers(
        const Tensor& inputs_embeds,
        const Tensor& position_ids,
        const Tensor& beam_idx,
        const std::vector<int32_t>& layer_ids,
        const Tensor* visual_embeds = nullptr,
        const Tensor* visual_pos_mask = nullptr,
        const Tensor* audio_features = nullptr,
        const Tensor* audio_pos_mask = nullptr,
        const std::vector<Tensor>* deepstack_embeds = nullptr) {
        capture_layer_ids_ = layer_ids;
        captured_hidden_.clear();

        auto final_out = forward_embeds(inputs_embeds,
                                        position_ids,
                                        beam_idx,
                                        visual_embeds,
                                        visual_pos_mask,
                                        audio_features,
                                        audio_pos_mask,
                                        deepstack_embeds);

        capture_layer_ids_.clear();

        if (captured_hidden_.empty()) {
            return {final_out, final_out};
        }
        auto concat_hidden = ops::concat(captured_hidden_, 2);
        captured_hidden_.clear();
        return {final_out, concat_hidden};
    }

private:
    std::pair<Tensor, Tensor> build_mrope_cos_sin(const Tensor& position_ids) const {
        auto* ctx = position_ids.context();
        const int32_t half_dim = head_dim_ / 2;
        std::vector<float> inv_freq(static_cast<size_t>(half_dim));
        for (int32_t i = 0; i < half_dim; ++i) {
            float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim_);
            inv_freq[static_cast<size_t>(i)] = 1.0f / std::pow(cfg_.rope_theta, exponent);
        }

        auto inv_freq_const = ops::const_vec(ctx, inv_freq);
        Tensor inv_freq_tensor(inv_freq_const, ctx);
        auto inv_freq_reshaped = inv_freq_tensor.reshape({1, 1, static_cast<int64_t>(half_dim)}, false);

        auto pos_t = ops::slice(position_ids, 0, 1, 1, 0).squeeze(0).to(ov::element::f32);
        auto pos_h = ops::slice(position_ids, 1, 2, 1, 0).squeeze(0).to(ov::element::f32);
        auto pos_w = ops::slice(position_ids, 2, 3, 1, 0).squeeze(0).to(ov::element::f32);

        auto freqs_t = pos_t.unsqueeze(2) * inv_freq_reshaped;
        if (!cfg_.rope.mrope_interleaved) {
            return {freqs_t.cos(), freqs_t.sin()};
        }

        auto freqs_h = pos_h.unsqueeze(2) * inv_freq_reshaped;
        auto freqs_w = pos_w.unsqueeze(2) * inv_freq_reshaped;
        auto freqs_all = ops::tensor::stack({freqs_t, freqs_h, freqs_w}, 0);
        auto freqs = ops::rope::mrope_interleaved(freqs_all, cfg_.rope.mrope_section);
        return {freqs.cos(), freqs.sin()};
    }

    Qwen3OmniTextConfig cfg_;
    VocabEmbedding embed_tokens_;
    Qwen3OmniEmbeddingInjector embedding_injector_;
    Qwen3OmniDeepstackInjector deepstack_injector_;
    std::vector<Qwen3OmniTextDecoderLayer> layers_;
    RMSNorm norm_;
    int32_t head_dim_ = 0;

    // DFlash layer capture support — set by forward_with_selected_layers
    std::vector<int32_t> capture_layer_ids_;
    std::vector<Tensor> captured_hidden_;
};

class Qwen3OmniTextForCausalLM : public Module {
public:
    Qwen3OmniTextForCausalLM(BuilderContext& ctx,
                             const Qwen3OmniTextConfig& cfg,
                             Module* parent = nullptr)
        : Module("", ctx, parent),
          cfg_(cfg),
          model_(ctx, cfg, this),
          lm_head_(ctx, "lm_head", this) {
        register_module("language_model", &model_);
        register_module("lm_head", &lm_head_);

        if (cfg_.tie_word_embeddings) {
            lm_head_.tie_to(model_.embed_tokens().weight_param());
        }
    }

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx,
                   const Tensor* visual_embeds = nullptr,
                   const Tensor* visual_pos_mask = nullptr,
                   const Tensor* audio_features = nullptr,
                   const Tensor* audio_pos_mask = nullptr,
                   const std::vector<Tensor>* deepstack_embeds = nullptr) {
        auto hidden = model_.forward(input_ids,
                                     position_ids,
                                     beam_idx,
                                     visual_embeds,
                                     visual_pos_mask,
                                     audio_features,
                                     audio_pos_mask,
                                     deepstack_embeds);
        return lm_head_.forward(hidden);
    }

    Tensor forward_embeds(const Tensor& inputs_embeds,
                          const Tensor& position_ids,
                          const Tensor& beam_idx,
                          const Tensor* visual_embeds = nullptr,
                          const Tensor* visual_pos_mask = nullptr,
                          const Tensor* audio_features = nullptr,
                          const Tensor* audio_pos_mask = nullptr,
                          const std::vector<Tensor>* deepstack_embeds = nullptr) {
        auto hidden = model_.forward_embeds(inputs_embeds,
                                            position_ids,
                                            beam_idx,
                                            visual_embeds,
                                            visual_pos_mask,
                                            audio_features,
                                            audio_pos_mask,
                                            deepstack_embeds);
        return lm_head_.forward(hidden);
    }

    /// DFlash: run forward and capture hidden states at selected layers.
    /// Returns {logits, target_hidden} where target_hidden is [B, S, hidden*num_layers].
    std::pair<Tensor, Tensor> forward_with_selected_layers(
        const Tensor& input_ids,
        const Tensor& position_ids,
        const Tensor& beam_idx,
        const std::vector<int32_t>& layer_ids,
        const Tensor* visual_embeds = nullptr,
        const Tensor* visual_pos_mask = nullptr,
        const Tensor* audio_features = nullptr,
        const Tensor* audio_pos_mask = nullptr,
        const std::vector<Tensor>* deepstack_embeds = nullptr) {
        auto embeds = model_.embed_tokens().forward(input_ids);
        auto [normed_hidden, concat_hidden] = model_.forward_with_selected_layers(
            embeds, position_ids, beam_idx, layer_ids,
            visual_embeds, visual_pos_mask,
            audio_features, audio_pos_mask,
            deepstack_embeds);
        auto logits = lm_head_.forward(normed_hidden);
        return {logits, concat_hidden};
    }

    Qwen3OmniTextModel& model() { return model_; }
    LMHead& lm_head() { return lm_head_; }

private:
    Qwen3OmniTextConfig cfg_;
    Qwen3OmniTextModel model_;
    LMHead lm_head_;
};

std::shared_ptr<ov::Model> create_qwen3_omni_text_model(
    const Qwen3OmniConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    bool use_inputs_embeds,
    bool enable_multimodal_inputs) {
    PrefixMappedWeightSource thinker_source(source, "thinker.");

    BuilderContext ctx;

    Qwen3OmniTextForCausalLM model(ctx, cfg.text);

    model.packed_mapping().rules.push_back({"model.embed_tokens.", "language_model.embed_tokens.", 0});
    model.packed_mapping().rules.push_back({"model.norm.", "language_model.norm.", 0});
    for (int32_t i = 0; i < cfg.text.num_hidden_layers; ++i) {
        const std::string src_bracket = "model.layers[" + std::to_string(i) + "].";
        const std::string src_dot = "model.layers." + std::to_string(i) + ".";
        const std::string dst = "language_model.layers." + std::to_string(i) + ".";
        model.packed_mapping().rules.push_back({src_bracket, dst, 0});
        model.packed_mapping().rules.push_back({src_dot, dst, 0});
    }

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_missing = true;
    options.report_unmatched = true;
    (void)ov::genai::modeling::weights::load_model(model, thinker_source, finalizer, options);

    // Re-establish lm_head -> embed_tokens tying after weight loading.
    // load_model() may load a separate "lm_head.weight" from safetensors via the
    // unmatched-key path, which calls bind() and breaks the tie that was set in the
    // constructor.  When tie_word_embeddings is true the lm_head must share the
    // embedding weight, so we re-tie here.
    if (cfg.text.tie_word_embeddings) {
        model.get_parameter("lm_head.weight").tie_to(
            model.get_parameter("language_model.embed_tokens.weight"));
    }

    const auto float_type = ov::element::f32;

    auto attention_mask = ctx.parameter(Qwen3OmniTextIO::kAttentionMask,
                                        ov::element::i64,
                                        ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter(Qwen3OmniTextIO::kPositionIds,
                                      ov::element::i64,
                                      ov::PartialShape{3, -1, -1});
    auto beam_idx = ctx.parameter(Qwen3OmniTextIO::kBeamIdx,
                                  ov::element::i32,
                                  ov::PartialShape{-1});

    (void)attention_mask;

    Tensor visual_embeds;
    Tensor visual_pos_mask;
    Tensor audio_features;
    Tensor audio_pos_mask;
    const Tensor* visual_embeds_ptr = nullptr;
    const Tensor* visual_pos_mask_ptr = nullptr;
    const Tensor* audio_features_ptr = nullptr;
    const Tensor* audio_pos_mask_ptr = nullptr;
    std::vector<Tensor> deepstack_inputs;
    const std::vector<Tensor>* deepstack_ptr = nullptr;

    if (enable_multimodal_inputs) {
        visual_embeds = ctx.parameter(Qwen3OmniTextIO::kVisualEmbeds,
                                      float_type,
                                      ov::PartialShape{-1, -1, cfg.text.hidden_size});
        visual_pos_mask = ctx.parameter(Qwen3OmniTextIO::kVisualPosMask,
                                        ov::element::boolean,
                                        ov::PartialShape{-1, -1});
        audio_features = ctx.parameter(Qwen3OmniTextIO::kAudioFeatures,
                                       float_type,
                                       ov::PartialShape{-1, -1, cfg.text.hidden_size});
        audio_pos_mask = ctx.parameter(Qwen3OmniTextIO::kAudioPosMask,
                                       ov::element::boolean,
                                       ov::PartialShape{-1, -1});
        visual_embeds_ptr = &visual_embeds;
        visual_pos_mask_ptr = &visual_pos_mask;
        audio_features_ptr = &audio_features;
        audio_pos_mask_ptr = &audio_pos_mask;

        const size_t deepstack_count = cfg.vision.deepstack_visual_indexes.size();
        deepstack_inputs.reserve(deepstack_count);
        for (size_t i = 0; i < deepstack_count; ++i) {
            const std::string name = std::string(Qwen3OmniTextIO::kDeepstackEmbedsPrefix) + "." + std::to_string(i);
            deepstack_inputs.emplace_back(ctx.parameter(name,
                                                        float_type,
                                                        ov::PartialShape{-1, -1, cfg.text.hidden_size}));
        }
        if (!deepstack_inputs.empty()) {
            deepstack_ptr = &deepstack_inputs;
        }
    }

    Tensor logits;
    if (use_inputs_embeds) {
        auto inputs_embeds = ctx.parameter(Qwen3OmniTextIO::kInputsEmbeds,
                                           float_type,
                                           ov::PartialShape{-1, -1, cfg.text.hidden_size});
        logits = model.forward_embeds(inputs_embeds,
                                      position_ids,
                                      beam_idx,
                                      visual_embeds_ptr,
                                      visual_pos_mask_ptr,
                                      audio_features_ptr,
                                      audio_pos_mask_ptr,
                                      deepstack_ptr);
    } else {
        auto input_ids = ctx.parameter(Qwen3OmniTextIO::kInputIds,
                                       ov::element::i64,
                                       ov::PartialShape{-1, -1});
        logits = model.forward(input_ids,
                               position_ids,
                               beam_idx,
                               visual_embeds_ptr,
                               visual_pos_mask_ptr,
                               audio_features_ptr,
                               audio_pos_mask_ptr,
                               deepstack_ptr);
    }

    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, Qwen3OmniTextIO::kLogits);
    auto ov_model = ctx.build_model({result->output(0)});
    ov_model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    ov_model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});
    return ov_model;
}

std::shared_ptr<ov::Model> create_qwen3_omni_dflash_target_model(
    const Qwen3OmniConfig& cfg,
    const std::vector<int32_t>& target_layer_ids,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    bool enable_multimodal_inputs) {
    PrefixMappedWeightSource thinker_source(source, "thinker.");

    BuilderContext ctx;

    Qwen3OmniTextForCausalLM model(ctx, cfg.text);

    // Weight mapping: safetensors "model.layers[i]." -> internal "language_model.layers.i."
    model.packed_mapping().rules.push_back({"model.embed_tokens.", "language_model.embed_tokens.", 0});
    model.packed_mapping().rules.push_back({"model.norm.", "language_model.norm.", 0});
    for (int32_t i = 0; i < cfg.text.num_hidden_layers; ++i) {
        const std::string src_bracket = "model.layers[" + std::to_string(i) + "].";
        const std::string src_dot = "model.layers." + std::to_string(i) + ".";
        const std::string dst = "language_model.layers." + std::to_string(i) + ".";
        model.packed_mapping().rules.push_back({src_bracket, dst, 0});
        model.packed_mapping().rules.push_back({src_dot, dst, 0});
    }

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_missing = true;
    options.report_unmatched = true;
    (void)ov::genai::modeling::weights::load_model(model, thinker_source, finalizer, options);

    // Re-tie lm_head -> embed_tokens after weight loading
    if (cfg.text.tie_word_embeddings) {
        model.get_parameter("lm_head.weight").tie_to(
            model.get_parameter("language_model.embed_tokens.weight"));
    }

    const auto float_type = ov::element::f32;

    auto input_ids = ctx.parameter(Qwen3OmniTextIO::kInputIds,
                                   ov::element::i64,
                                   ov::PartialShape{-1, -1});
    auto attention_mask = ctx.parameter(Qwen3OmniTextIO::kAttentionMask,
                                        ov::element::i64,
                                        ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter(Qwen3OmniTextIO::kPositionIds,
                                      ov::element::i64,
                                      ov::PartialShape{3, -1, -1});
    auto beam_idx = ctx.parameter(Qwen3OmniTextIO::kBeamIdx,
                                  ov::element::i32,
                                  ov::PartialShape{-1});

    (void)attention_mask;

    Tensor visual_embeds;
    Tensor visual_pos_mask;
    Tensor audio_features;
    Tensor audio_pos_mask;
    const Tensor* visual_embeds_ptr = nullptr;
    const Tensor* visual_pos_mask_ptr = nullptr;
    const Tensor* audio_features_ptr = nullptr;
    const Tensor* audio_pos_mask_ptr = nullptr;
    std::vector<Tensor> deepstack_inputs;
    const std::vector<Tensor>* deepstack_ptr = nullptr;

    if (enable_multimodal_inputs) {
        visual_embeds = ctx.parameter(Qwen3OmniTextIO::kVisualEmbeds,
                                      float_type,
                                      ov::PartialShape{-1, -1, cfg.text.hidden_size});
        visual_pos_mask = ctx.parameter(Qwen3OmniTextIO::kVisualPosMask,
                                        ov::element::boolean,
                                        ov::PartialShape{-1, -1});
        audio_features = ctx.parameter(Qwen3OmniTextIO::kAudioFeatures,
                                       float_type,
                                       ov::PartialShape{-1, -1, cfg.text.hidden_size});
        audio_pos_mask = ctx.parameter(Qwen3OmniTextIO::kAudioPosMask,
                                       ov::element::boolean,
                                       ov::PartialShape{-1, -1});
        visual_embeds_ptr = &visual_embeds;
        visual_pos_mask_ptr = &visual_pos_mask;
        audio_features_ptr = &audio_features;
        audio_pos_mask_ptr = &audio_pos_mask;

        const size_t deepstack_count = cfg.vision.deepstack_visual_indexes.size();
        deepstack_inputs.reserve(deepstack_count);
        for (size_t i = 0; i < deepstack_count; ++i) {
            const std::string name = std::string(Qwen3OmniTextIO::kDeepstackEmbedsPrefix) + "." + std::to_string(i);
            deepstack_inputs.emplace_back(ctx.parameter(name,
                                                        float_type,
                                                        ov::PartialShape{-1, -1, cfg.text.hidden_size}));
        }
        if (!deepstack_inputs.empty()) {
            deepstack_ptr = &deepstack_inputs;
        }
    }

    // DFlash: forward with selected layer capture -> logits + target_hidden
    auto [logits, target_hidden] = model.forward_with_selected_layers(
        input_ids, position_ids, beam_idx, target_layer_ids,
        visual_embeds_ptr, visual_pos_mask_ptr,
        audio_features_ptr, audio_pos_mask_ptr,
        deepstack_ptr);

    auto logits_result = std::make_shared<ov::op::v0::Result>(logits.output());
    auto hidden_result = std::make_shared<ov::op::v0::Result>(target_hidden.output());
    set_name(logits_result, Qwen3OmniTextIO::kLogits);
    set_name(hidden_result, "target_hidden");

    auto ov_model = ctx.build_model({logits_result->output(0), hidden_result->output(0)});
    ov_model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    ov_model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});
    return ov_model;
}

// ============================================================================
// DFlash Combined Draft Model (embed + draft layers + lm_head)
// ============================================================================
// Merges embed_tokens, draft layers, and lm_head into a single graph.
// Target weights (embed_tokens, lm_head) come from the Qwen3-Omni thinker,
// accessed through a PrefixMappedWeightSource that strips the "thinker." prefix.
// Draft weights come from the standalone DFlash draft model.
//
// Inputs:  target_hidden [B, T, ctx_dim], input_ids [B, block_size],
//          position_ids [1, T+block_size]
// Output:  logits [B, block_size, vocab_size]

std::shared_ptr<ov::Model> create_qwen3_omni_dflash_combined_draft_model(
    const Qwen3OmniConfig& omni_cfg,
    const DFlashDraftConfig& draft_cfg,
    ov::genai::modeling::weights::WeightSource& target_source,
    ov::genai::modeling::weights::WeightFinalizer& target_finalizer,
    ov::genai::modeling::weights::WeightSource& draft_source,
    ov::genai::modeling::weights::WeightFinalizer& draft_finalizer) {
    BuilderContext ctx;

    // ── Embedding path (target weights under "model" prefix) ──
    Module embed_root("model", ctx);
    VocabEmbedding embed(ctx, "embed_tokens", &embed_root);
    embed_root.packed_mapping().rules.push_back({"model.language_model.", "model.", 0});
    embed_root.packed_mapping().rules.push_back({"language_model.", "model.", 0});

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_unmatched = false;
    options.report_missing = true;
    ov::genai::modeling::weights::load_model(embed_root, target_source, target_finalizer, options);

    // ── Draft layers (draft weights) ──
    DFlashDraftModel draft_model(ctx, draft_cfg);
    ov::genai::modeling::weights::load_model(draft_model, draft_source, draft_finalizer);

    // ── LM head (target weights, tied to embed_tokens when configured) ──
    Module lm_root("", ctx);
    LMHead head(ctx, "lm_head", &lm_root);
    if (omni_cfg.text.tie_word_embeddings) {
        head.tie_to(embed.weight_param());
        // When tied, skip load_model for lm_root — Qwen3-Omni safetensors
        // store a separate lm_head.weight that differs from embed_tokens.weight
        // even though tie_word_embeddings=True.  load_model would overwrite the
        // tied binding with the wrong tensor.
    } else if (!target_source.has("lm_head.weight")) {
        const std::vector<std::string> embed_candidates = {
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
        };
        std::string embed_weight;
        for (const auto& name : embed_candidates) {
            if (target_source.has(name)) { embed_weight = name; break; }
        }
        if (!embed_weight.empty()) {
            auto tied = target_finalizer.finalize(embed_weight, target_source, ctx.op_context());
            head.weight_param().bind(tied);
        }
    } else {
        ov::genai::modeling::weights::load_model(lm_root, target_source, target_finalizer, options);
    }

    // ── Inputs ──
    const ov::element::Type dtype = ov::element::f32;
    const int64_t ctx_dim = static_cast<int64_t>(draft_cfg.hidden_size) *
                            static_cast<int64_t>(draft_cfg.num_hidden_layers);
    auto target_hidden = ctx.parameter("target_hidden", dtype, ov::PartialShape{-1, -1, ctx_dim});
    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});

    // ── Forward: embed → draft → lm_head (single graph) ──
    auto noise_embedding = embed.forward(input_ids);
    auto draft_hidden = draft_model.forward(target_hidden, noise_embedding, position_ids);
    auto logits = head.forward(draft_hidden);

    // ── Output ──
    auto logits_result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(logits_result, "logits");

    return ctx.build_model({logits_result->output(0)});
}

// ============================================================================
// DFlash Context KV Preprocessing Model
// ============================================================================
// Computes fc(target_hidden) → RMSNorm → K,V projections + KNorm + RoPE
// for all draft layers.  Runs once per verify cycle on newly-accepted tokens.
// Inputs:  target_hidden [1, A, ctx_dim], position_ids [1, A]
// Outputs: k_0..k_N [1, kv_heads, A, head_dim], v_0..v_N (same shape)

std::shared_ptr<ov::Model> create_qwen3_omni_dflash_context_kv_model(
    const DFlashDraftConfig& draft_cfg,
    ov::genai::modeling::weights::WeightSource& draft_source,
    ov::genai::modeling::weights::WeightFinalizer& draft_finalizer) {
    BuilderContext ctx;

    DFlashDraftModel draft_model(ctx, draft_cfg);
    ov::genai::modeling::weights::load_model(draft_model, draft_source, draft_finalizer);

    const ov::element::Type dtype = ov::element::f32;
    const int64_t ctx_dim = static_cast<int64_t>(draft_cfg.hidden_size) *
                            static_cast<int64_t>(draft_cfg.num_hidden_layers);
    auto target_hidden = ctx.parameter("target_hidden", dtype, ov::PartialShape{-1, -1, ctx_dim});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});

    auto kv_pairs = draft_model.build_context_kv(target_hidden, position_ids);

    ov::OutputVector outputs;
    for (size_t i = 0; i < kv_pairs.size(); ++i) {
        auto k_result = std::make_shared<ov::op::v0::Result>(kv_pairs[i].first.output());
        set_name(k_result, "context_k_" + std::to_string(i));
        auto v_result = std::make_shared<ov::op::v0::Result>(kv_pairs[i].second.output());
        set_name(v_result, "context_v_" + std::to_string(i));
        outputs.push_back(k_result->output(0));
        outputs.push_back(v_result->output(0));
    }

    return ctx.build_model(outputs);
}

// ============================================================================
// DFlash Lightweight Draft Step Model
// ============================================================================
// Embed → attention (with pre-computed context KV) → MLP → lm_head.
// Runs once per draft cycle.  Skips fc + context KV computation entirely.
// Inputs:  input_ids [1, B], position_ids [1, B],
//          context_k_0..N [1, kv_heads, T, head_dim], context_v_0..N (same)
// Output:  logits [1, B, vocab_size]

std::shared_ptr<ov::Model> create_qwen3_omni_dflash_step_model(
    const Qwen3OmniConfig& omni_cfg,
    const DFlashDraftConfig& draft_cfg,
    ov::genai::modeling::weights::WeightSource& target_source,
    ov::genai::modeling::weights::WeightFinalizer& target_finalizer,
    ov::genai::modeling::weights::WeightSource& draft_source,
    ov::genai::modeling::weights::WeightFinalizer& draft_finalizer) {
    BuilderContext ctx;

    // ── Embedding (target weights) ──
    Module embed_root("model", ctx);
    VocabEmbedding embed(ctx, "embed_tokens", &embed_root);
    embed_root.packed_mapping().rules.push_back({"model.language_model.", "model.", 0});
    embed_root.packed_mapping().rules.push_back({"language_model.", "model.", 0});

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_unmatched = false;
    options.report_missing = true;
    ov::genai::modeling::weights::load_model(embed_root, target_source, target_finalizer, options);

    // ── Draft layers ──
    DFlashDraftModel draft_model(ctx, draft_cfg);
    ov::genai::modeling::weights::load_model(draft_model, draft_source, draft_finalizer);

    // ── LM head (target weights) ──
    Module lm_root("", ctx);
    LMHead head(ctx, "lm_head", &lm_root);
    if (omni_cfg.text.tie_word_embeddings) {
        head.tie_to(embed.weight_param());
    } else if (!target_source.has("lm_head.weight")) {
        const std::vector<std::string> embed_candidates = {
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
        };
        std::string embed_weight;
        for (const auto& name : embed_candidates) {
            if (target_source.has(name)) { embed_weight = name; break; }
        }
        if (!embed_weight.empty()) {
            auto tied = target_finalizer.finalize(embed_weight, target_source, ctx.op_context());
            head.weight_param().bind(tied);
        }
    } else {
        ov::genai::modeling::weights::load_model(lm_root, target_source, target_finalizer, options);
    }

    // ── Inputs ──
    const ov::element::Type dtype = ov::element::f32;
    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});

    // Context KV inputs: one K,V pair per draft layer.
    const int32_t kv_heads = draft_cfg.num_key_value_heads > 0
                                 ? draft_cfg.num_key_value_heads
                                 : draft_cfg.num_attention_heads;
    const int32_t head_dim = draft_cfg.head_dim > 0
                                 ? draft_cfg.head_dim
                                 : (draft_cfg.hidden_size / draft_cfg.num_attention_heads);
    std::vector<std::pair<Tensor, Tensor>> context_kv;
    for (int32_t i = 0; i < draft_cfg.num_hidden_layers; ++i) {
        auto ck = ctx.parameter("context_k_" + std::to_string(i), dtype,
                                ov::PartialShape{-1, kv_heads, -1, head_dim});
        auto cv = ctx.parameter("context_v_" + std::to_string(i), dtype,
                                ov::PartialShape{-1, kv_heads, -1, head_dim});
        context_kv.push_back({ck, cv});
    }

    // ── Forward: embed → draft_with_cached_kv → lm_head ──
    auto noise_embedding = embed.forward(input_ids);
    auto draft_hidden = draft_model.forward_with_cached_kv(noise_embedding, position_ids, context_kv);
    auto logits = head.forward(draft_hidden);

    auto logits_result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(logits_result, "logits");

    return ctx.build_model({logits_result->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
