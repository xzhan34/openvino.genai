// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/kv_cache.hpp"

#include <iostream>
#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace ops {

std::pair<Tensor, Tensor> append_kv_cache(const Tensor& keys,
                                          const Tensor& values,
                                          const Tensor& beam_idx,
                                          int32_t num_kv_heads,
                                          int32_t head_dim,
                                          const std::string& cache_prefix,
                                          const BuilderContext& ctx) {
    auto* op_ctx = keys.context();
    auto batch = shape::dim(keys, 0);
    auto kv_heads = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(num_kv_heads)});
    auto zero_len = ops::const_vec(op_ctx, std::vector<int64_t>{0});
    auto head_dim_vec = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(head_dim)});
    auto cache_shape = shape::make({batch, kv_heads, zero_len, head_dim_vec});

    auto zero = Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx).to(keys.dtype());
    auto k_init = shape::broadcast_to(zero, cache_shape);
    auto v_init = shape::broadcast_to(zero, cache_shape);

    const std::string k_name = cache_prefix + ".key_cache";
    const std::string v_name = cache_prefix + ".value_cache";

    ov::op::util::VariableInfo k_info{ov::PartialShape{-1, num_kv_heads, -1, head_dim},
                                      keys.dtype(),
                                      k_name};
    auto k_var = std::make_shared<ov::op::util::Variable>(k_info);
    auto k_read = std::make_shared<ov::op::v6::ReadValue>(k_init.output(), k_var);

    ov::op::util::VariableInfo v_info{ov::PartialShape{-1, num_kv_heads, -1, head_dim},
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
    ctx.register_sink(k_assign);
    ctx.register_sink(v_assign);

    return {k_combined, v_combined};
}

}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
