// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_5/modeling_qwen3_5_moe.hpp"

#include <openvino/core/except.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/opsets/opset13.hpp>
#include <vector>

#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

namespace {

Tensor dequantize_packed_moe_weight(const Tensor& packed_weight,
                                    const Tensor& scales_e_g_o,
                                    const Tensor& zps_e_g_o,
                                    int32_t num_experts,
                                    int32_t out_features,
                                    int32_t in_features) {
    auto* op_ctx = packed_weight.context();

    // Convert auxiliary tensors from [E, G, O] to [E, O, G] to align with packed weight layout [E, O, G, GS].
    auto perm = ops::const_vec(op_ctx, std::vector<int64_t>{0, 2, 1});
    auto scales_e_o_g = Tensor(std::make_shared<ov::op::v1::Transpose>(scales_e_g_o.output(), perm), op_ctx);
    auto zps_e_o_g = Tensor(std::make_shared<ov::op::v1::Transpose>(zps_e_g_o.output(), perm), op_ctx);

    auto packed_f32 = packed_weight.to(ov::element::f32);
    auto zps_f32 = zps_e_o_g.unsqueeze(-1).to(ov::element::f32);
    auto scales_f32 = scales_e_o_g.unsqueeze(-1).to(ov::element::f32);

    // Dequantize from packed groups: [E, O, G, GS] -> [E, O, I].
    auto dequant_grouped = (packed_f32 - zps_f32) * scales_f32;
    return dequant_grouped.reshape({num_experts, out_features, in_features}, false);
}

}  // namespace

Qwen3_5SparseMoeBlock::Qwen3_5SparseMoeBlock(BuilderContext& ctx,
                                             const std::string& name,
                                             const Qwen3_5TextModelConfig& cfg,
                                             Module* parent)
    : Module(name, ctx, parent),
      hidden_size_(cfg.hidden_size),
      expert_intermediate_size_(cfg.moe_intermediate_size),
      shared_intermediate_size_(cfg.shared_expert_intermediate_size),
      num_experts_(cfg.num_experts),
      top_k_(cfg.num_experts_per_tok > 0 ? cfg.num_experts_per_tok : 1),
      norm_topk_prob_(cfg.norm_topk_prob) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported Qwen3_5 MoE activation: ", cfg.hidden_act);
    }
    if (hidden_size_ <= 0 || expert_intermediate_size_ <= 0 || shared_intermediate_size_ <= 0 || num_experts_ <= 0) {
        OPENVINO_THROW("Invalid Qwen3_5 MoE configuration");
    }
    if (top_k_ <= 0 || top_k_ > num_experts_) {
        OPENVINO_THROW("Invalid Qwen3_5 MoE top-k configuration");
    }

    gate_param_ = &register_parameter("gate.weight");
    experts_gate_up_param_ = &register_parameter("experts.gate_up_proj");
    experts_down_param_ = &register_parameter("experts.down_proj");
    shared_expert_gate_param_ = &register_parameter("shared_expert_gate.weight");
    shared_gate_proj_param_ = &register_parameter("shared_expert.gate_proj.weight");
    shared_up_proj_param_ = &register_parameter("shared_expert.up_proj.weight");
    shared_down_proj_param_ = &register_parameter("shared_expert.down_proj.weight");

    experts_gate_up_param_->set_weight_loader([this](WeightParameter& param,
                                                     weights::WeightSource& source,
                                                     weights::WeightFinalizer& finalizer,
                                                     const std::string& weight_name,
                                                     const std::optional<int>& shard_id) {
        (void)shard_id;
        if (!param.context()) {
            OPENVINO_THROW("WeightParameter has no OpContext: ", param.name());
        }
        auto weight = finalizer.finalize(weight_name, source, *param.context());
        param.bind(weight);

        gate_up_scales_.reset();
        gate_up_zps_.reset();
        if (weight.get_auxiliary("scales") != std::nullopt && weight.get_auxiliary("zps") != std::nullopt) {
            gate_up_scales_ = weight.auxiliary.at("scales");
            gate_up_zps_ = weight.auxiliary.at("zps");
        }
    });

    experts_down_param_->set_weight_loader([this](WeightParameter& param,
                                                  weights::WeightSource& source,
                                                  weights::WeightFinalizer& finalizer,
                                                  const std::string& weight_name,
                                                  const std::optional<int>& shard_id) {
        (void)shard_id;
        if (!param.context()) {
            OPENVINO_THROW("WeightParameter has no OpContext: ", param.name());
        }
        auto weight = finalizer.finalize(weight_name, source, *param.context());
        param.bind(weight);

        down_scales_.reset();
        down_zps_.reset();
        if (weight.get_auxiliary("scales") != std::nullopt && weight.get_auxiliary("zps") != std::nullopt) {
            down_scales_ = weight.auxiliary.at("scales");
            down_zps_ = weight.auxiliary.at("zps");
        }
    });

    shared_gate_proj_param_->set_weight_loader([this](WeightParameter& param,
                                                      weights::WeightSource& source,
                                                      weights::WeightFinalizer& finalizer,
                                                      const std::string& weight_name,
                                                      const std::optional<int>& shard_id) {
        (void)shard_id;
        if (!param.context()) {
            OPENVINO_THROW("WeightParameter has no OpContext: ", param.name());
        }
        auto weight = finalizer.finalize(weight_name, source, *param.context());

        auto shape = weight.primary.output().get_shape();
        std::vector<int64_t> new_shape = {1};
        for (auto dim : shape)
            new_shape.push_back(static_cast<int64_t>(dim));
        auto new_primary = weight.primary.reshape(new_shape, false);

        // Create new FinalizedWeight with reshaped primary
        weights::FinalizedWeight new_weight(new_primary, weight.auxiliary);
        param.bind(new_weight);

        shared_gate_scales_.reset();
        shared_gate_zps_.reset();
        if (weight.get_auxiliary("scales") != std::nullopt && weight.get_auxiliary("zps") != std::nullopt) {
            shared_gate_scales_ = weight.auxiliary.at("scales");
            shared_gate_zps_ = weight.auxiliary.at("zps");
        }
    });

    shared_up_proj_param_->set_weight_loader([this](WeightParameter& param,
                                                    weights::WeightSource& source,
                                                    weights::WeightFinalizer& finalizer,
                                                    const std::string& weight_name,
                                                    const std::optional<int>& shard_id) {
        (void)shard_id;
        if (!param.context()) {
            OPENVINO_THROW("WeightParameter has no OpContext: ", param.name());
        }
        auto weight = finalizer.finalize(weight_name, source, *param.context());

        auto shape = weight.primary.output().get_shape();
        std::vector<int64_t> new_shape = {1};
        for (auto dim : shape)
            new_shape.push_back(static_cast<int64_t>(dim));
        auto new_primary = weight.primary.reshape(new_shape, false);

        weights::FinalizedWeight new_weight(new_primary, weight.auxiliary);
        param.bind(new_weight);

        shared_up_scales_.reset();
        shared_up_zps_.reset();
        if (weight.get_auxiliary("scales") != std::nullopt && weight.get_auxiliary("zps") != std::nullopt) {
            shared_up_scales_ = weight.auxiliary.at("scales");
            shared_up_zps_ = weight.auxiliary.at("zps");
        }
    });

    shared_down_proj_param_->set_weight_loader([this](WeightParameter& param,
                                                      weights::WeightSource& source,
                                                      weights::WeightFinalizer& finalizer,
                                                      const std::string& weight_name,
                                                      const std::optional<int>& shard_id) {
        (void)shard_id;
        if (!param.context()) {
            OPENVINO_THROW("WeightParameter has no OpContext: ", param.name());
        }
        auto weight = finalizer.finalize(weight_name, source, *param.context());

        auto shape = weight.primary.output().get_shape();
        std::vector<int64_t> new_shape = {1};
        for (auto dim : shape)
            new_shape.push_back(static_cast<int64_t>(dim));
        auto new_primary = weight.primary.reshape(new_shape, false);

        weights::FinalizedWeight new_weight(new_primary, weight.auxiliary);
        param.bind(new_weight);

        shared_down_scales_.reset();
        shared_down_zps_.reset();
        if (weight.get_auxiliary("scales") != std::nullopt && weight.get_auxiliary("zps") != std::nullopt) {
            shared_down_scales_ = weight.auxiliary.at("scales");
            shared_down_zps_ = weight.auxiliary.at("zps");
        }
    });
}

const Tensor& Qwen3_5SparseMoeBlock::gate_weight() const {
    if (!gate_param_) {
        OPENVINO_THROW("Qwen3_5SparseMoeBlock gate parameter is not registered");
    }
    return gate_param_->value();
}

const Tensor& Qwen3_5SparseMoeBlock::gate_up_expert_weights() const {
    if (!experts_gate_up_param_) {
        OPENVINO_THROW("Qwen3_5SparseMoeBlock experts.gate_up_proj parameter is not registered");
    }
    return experts_gate_up_param_->value();
}

const Tensor& Qwen3_5SparseMoeBlock::down_expert_weights() const {
    if (!experts_down_param_) {
        OPENVINO_THROW("Qwen3_5SparseMoeBlock experts.down_proj parameter is not registered");
    }
    return experts_down_param_->value();
}

const Tensor& Qwen3_5SparseMoeBlock::shared_expert_gate_weight() const {
    if (!shared_expert_gate_param_) {
        OPENVINO_THROW("Qwen3_5SparseMoeBlock shared_expert_gate parameter is not registered");
    }
    return shared_expert_gate_param_->value();
}

const Tensor& Qwen3_5SparseMoeBlock::shared_gate_proj_weight() const {
    if (!shared_gate_proj_param_) {
        OPENVINO_THROW("Qwen3_5SparseMoeBlock shared_expert.gate_proj parameter is not registered");
    }
    return shared_gate_proj_param_->value();
}

const Tensor& Qwen3_5SparseMoeBlock::shared_up_proj_weight() const {
    if (!shared_up_proj_param_) {
        OPENVINO_THROW("Qwen3_5SparseMoeBlock shared_expert.up_proj parameter is not registered");
    }
    return shared_up_proj_param_->value();
}

const Tensor& Qwen3_5SparseMoeBlock::shared_down_proj_weight() const {
    if (!shared_down_proj_param_) {
        OPENVINO_THROW("Qwen3_5SparseMoeBlock shared_expert.down_proj parameter is not registered");
    }
    return shared_down_proj_param_->value();
}

bool Qwen3_5SparseMoeBlock::can_use_fused_path() const {
    // NOTE:
    // Current gate_up split relies on StridedSlice over low-bit packed constants (u4/i4).
    // That path crashes during ConstantFolding on GPU in this branch.
    // Keep fused path disabled until packed split is implemented without low-bit StridedSlice.
    return true;
}

size_t Qwen3_5SparseMoeBlock::infer_group_size() const {
    const auto gate_up_shape = gate_up_expert_weights().output().get_shape();
    if (gate_up_shape.size() == 4 && gate_up_shape[3] > 0) {
        return gate_up_shape[3];
    }
    return 128;
}

Tensor Qwen3_5SparseMoeBlock::routed_fused(const Tensor& flat_f32) const {
    auto gate_up_w = gate_up_expert_weights();
    auto down_w = down_expert_weights();
    auto gate_up_w_u8 = ops::convert(gate_up_w, ov::element::u8);

    auto split = ops::split(gate_up_w_u8, 2, 1);
    auto gate_exps_w_u8 = split.first;
    auto up_exps_w_u8 = split.second;
    auto gate_exps_w = ops::convert(gate_exps_w_u8, ov::element::u4);
    auto up_exps_w = ops::convert(up_exps_w_u8, ov::element::u4);

    // scale no need convert
    auto split_scale = ops::split(*gate_up_scales_, 2, 2);
    auto gate_exps_scales = split_scale.first;
    auto up_exps_scales = split_scale.second;

    auto split_zp_u8 = ops::convert(*gate_up_zps_, ov::element::u8);
    auto split_zp = ops::split(split_zp_u8, 2, 2);
    auto gate_exps_zps_u8 = split_zp.first;
    auto up_exps_zps_u8 = split_zp.second;
    auto gate_exps_zps = ops::convert(gate_exps_zps_u8, ov::element::u4);
    auto up_exps_zps = ops::convert(up_exps_zps_u8, ov::element::u4);

    // Check if shared experts are fully quantized, otherwise we must fallback for shared part
    bool use_fused_shared = shared_gate_scales_.has_value() && shared_gate_zps_.has_value() &&
                            shared_up_scales_.has_value() && shared_up_zps_.has_value() &&
                            shared_down_scales_.has_value() && shared_down_zps_.has_value();

    if (!use_fused_shared) {
        std::cout << "[ERROR] Cannot get shared experts" << std::endl;
        exit(0);
    }

    auto sh_gate_w = shared_gate_proj_weight();
    auto sh_up_w = shared_up_proj_weight();
    auto sh_down_w = shared_down_proj_weight();
    auto sh_gate_gate_w = shared_expert_gate_weight().to(ov::element::f16);

    auto sh_gate_scales = shared_gate_scales_.value();
    auto sh_gate_zps = shared_gate_zps_.value();
    auto sh_up_scales = shared_up_scales_.value();
    auto sh_up_zps = shared_up_zps_.value();
    auto sh_down_scales = shared_down_scales_.value();
    auto sh_down_zps = shared_down_zps_.value();

    // std::cout << "[DEBUG] sh_gate_w type: " << sh_gate_w.output().get_element_type()
    //           << ", shape: " << sh_gate_w.output().get_shape() << std::endl;
    // std::cout << "[DEBUG] sh_up_w type: " << sh_up_w.output().get_element_type()
    //           << ", shape: " << sh_up_w.output().get_shape() << std::endl;
    // std::cout << "[DEBUG] sh_down_w type: " << sh_down_w.output().get_element_type()
    //           << ", shape: " << sh_down_w.output().get_shape() << std::endl;
    // std::cout << "[DEBUG] sh_gate_gate_w type: " << sh_gate_gate_w.output().get_element_type()
    //           << ", shape: " << sh_gate_gate_w.output().get_shape() << std::endl;
    // std::cout << "[DEBUG] sh_gate_scales: " << (sh_gate_scales).output().get_element_type()
    //           << ", shape: " << (sh_gate_scales).output().get_shape() << std::endl;
    // std::cout << "[DEBUG] sh_gate_zps: " << (sh_gate_zps).output().get_element_type()
    //           << ", shape: " << (sh_gate_zps).output().get_shape() << std::endl;
    // std::cout << "[DEBUG] sh_up_scales: " << (sh_up_scales).output().get_element_type()
    //           << ", shape: " << (sh_up_scales).output().get_shape() << std::endl;
    // std::cout << "[DEBUG] sh_up_zps: " << (sh_up_zps).output().get_element_type()
    //           << ", shape: " << (sh_up_zps).output().get_element_type()
    //           << ", shape: " << (sh_up_zps).output().get_shape() << std::endl;
    // std::cout << "[DEBUG] sh_down_scales: " << (sh_down_scales).output().get_element_type()
    //           << ", shape: " << (sh_down_scales).output().get_element_type() << std::endl;
    // std::cout << "[DEBUG] sh_down_zps: " << (sh_down_zps).output().get_element_type()
    //           << ", shape: " << (sh_down_zps).output().get_shape() << std::endl;

    auto result = ops::moe3gemm_fused_compressed(flat_f32,
                                                 gate_weight(),
                                                 gate_exps_w,
                                                 gate_exps_scales,
                                                 gate_exps_zps,
                                                 up_exps_w,
                                                 up_exps_scales,
                                                 up_exps_zps,
                                                 down_w,
                                                 *down_scales_,
                                                 *down_zps_,
                                                 hidden_size_,
                                                 expert_intermediate_size_,
                                                 num_experts_,
                                                 top_k_,
                                                 infer_group_size(),
                                                 ov::element::f16,
                                                 sh_gate_w,
                                                 sh_gate_scales,
                                                 sh_gate_zps,
                                                 sh_up_w,
                                                 sh_up_scales,
                                                 sh_up_zps,
                                                 sh_down_w,
                                                 sh_down_scales,
                                                 sh_down_zps,
                                                 sh_gate_gate_w);

    // if (!use_fused_shared) {
    //     auto shared_gate = ops::linear(flat_f32, shared_gate_proj_weight().to(ov::element::f32));
    //     auto shared_up = ops::linear(flat_f32, shared_up_proj_weight().to(ov::element::f32));
    //     auto shared_hidden = ops::silu(shared_gate) * shared_up;
    //     auto shared_out = ops::linear(shared_hidden, shared_down_proj_weight().to(ov::element::f32));

    //     auto shared_gate_logits = ops::linear(flat_f32, shared_expert_gate_weight().to(ov::element::f32));
    //     auto shared_gate_sigmoid = Tensor(std::make_shared<ov::op::v0::Sigmoid>(shared_gate_logits.output()),
    //     op_ctx); result = result + (shared_out * shared_gate_sigmoid);
    // }

    return result;
}

Tensor Qwen3_5SparseMoeBlock::routed_fallback(const Tensor& flat_f32) const {
    auto* op_ctx = flat_f32.context();

    auto logits = ops::linear(flat_f32, gate_weight().to(ov::element::f32));
    auto scores = logits.softmax(1);

    auto k_node = ops::const_scalar(op_ctx, static_cast<int64_t>(top_k_));
    auto topk = std::make_shared<ov::op::v11::TopK>(scores.output(),
                                                    k_node,
                                                    -1,
                                                    ov::op::v11::TopK::Mode::MAX,
                                                    ov::op::v11::TopK::SortType::SORT_VALUES,
                                                    ov::element::i64);

    Tensor topk_vals(topk->output(0), op_ctx);
    Tensor topk_idx(topk->output(1), op_ctx);
    if (norm_topk_prob_) {
        auto reduce_axis = ops::const_vec(op_ctx, std::vector<int64_t>{-1});
        auto sum = std::make_shared<ov::op::v1::ReduceSum>(topk_vals.output(), reduce_axis, true);
        topk_vals = topk_vals / Tensor(sum, op_ctx);
    }

    auto zeros = shape::broadcast_to(Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx), shape::of(scores));
    auto scatter_axis = ops::const_scalar(op_ctx, static_cast<int64_t>(1));
    auto scatter = std::make_shared<ov::op::v12::ScatterElementsUpdate>(zeros.output(),
                                                                        topk_idx.output(),
                                                                        topk_vals.output(),
                                                                        scatter_axis);
    Tensor routing(scatter, op_ctx);  // [T, E]
    auto perm = ops::const_vec(op_ctx, std::vector<int64_t>{1, 0});
    auto routing_t = Tensor(std::make_shared<ov::op::v1::Transpose>(routing.output(), perm), op_ctx);
    auto routing_3d = routing_t.unsqueeze(-1);  // [E, T, 1]

    auto flat_3d = flat_f32.unsqueeze(0);
    auto tiled = ops::tensor::tile(flat_3d, {num_experts_, 1, 1});  // [E, T, H]

    Tensor gate_up;
    Tensor down_exps_w;
    const bool has_quant_aux =
        gate_up_scales_.has_value() && gate_up_zps_.has_value() && down_scales_.has_value() && down_zps_.has_value();

    if (has_quant_aux && gate_up_expert_weights().output().get_shape().size() == 4 &&
        down_expert_weights().output().get_shape().size() == 4) {
        // Quantized packed MoE weights: dequantize explicitly for fallback path.
        gate_up = dequantize_packed_moe_weight(gate_up_expert_weights(),
                                               *gate_up_scales_,
                                               *gate_up_zps_,
                                               num_experts_,
                                               2 * expert_intermediate_size_,
                                               hidden_size_);
        down_exps_w = dequantize_packed_moe_weight(down_expert_weights(),
                                                   *down_scales_,
                                                   *down_zps_,
                                                   num_experts_,
                                                   hidden_size_,
                                                   expert_intermediate_size_);
    } else {
        gate_up = gate_up_expert_weights().to(ov::element::f32);
        down_exps_w = down_expert_weights().to(ov::element::f32);
    }

    auto gate_exps_w = ops::slice(gate_up, 0, expert_intermediate_size_, 1, 1);
    auto up_exps_w = ops::slice(gate_up, expert_intermediate_size_, 2 * expert_intermediate_size_, 1, 1);

    auto gate_bmm = ops::matmul(tiled, gate_exps_w, false, true);
    auto up_bmm = ops::matmul(tiled, up_exps_w, false, true);
    auto swiglu = ops::silu(gate_bmm) * up_bmm;
    auto down_bmm = ops::matmul(swiglu, down_exps_w, false, true);

    auto weighted = down_bmm.to(ov::element::f32) * routing_3d;
    auto reduce_axis = ops::const_vec(op_ctx, std::vector<int64_t>{0});
    auto reduced = std::make_shared<ov::op::v1::ReduceSum>(weighted.output(), reduce_axis, false);
    return Tensor(reduced, op_ctx);
}

Tensor Qwen3_5SparseMoeBlock::forward(const Tensor& hidden_states) const {
    auto* op_ctx = hidden_states.context();
    auto input_dtype = hidden_states.dtype();

    auto flat = hidden_states.reshape({-1, hidden_size_});
    auto flat_f32 = flat.to(ov::element::f32);

    if (can_use_fused_path()) {
        auto fused_out = routed_fused(flat_f32);
        auto restored = fused_out.reshape(shape::of(hidden_states), false);
        return restored.to(input_dtype);
    }

    auto routed_out = routed_fallback(flat_f32);

    auto shared_gate = ops::linear(flat_f32, shared_gate_proj_weight().to(ov::element::f32));
    auto shared_up = ops::linear(flat_f32, shared_up_proj_weight().to(ov::element::f32));
    auto shared_hidden = ops::silu(shared_gate) * shared_up;
    auto shared_out = ops::linear(shared_hidden, shared_down_proj_weight().to(ov::element::f32));

    auto shared_gate_logits = ops::linear(flat_f32, shared_expert_gate_weight().to(ov::element::f32));
    auto shared_gate_sigmoid = Tensor(std::make_shared<ov::op::v0::Sigmoid>(shared_gate_logits.output()), op_ctx);
    auto combined = routed_out + (shared_out * shared_gate_sigmoid);
    auto restored = combined.reshape(shape::of(hidden_states), false);
    return restored.to(input_dtype);
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
