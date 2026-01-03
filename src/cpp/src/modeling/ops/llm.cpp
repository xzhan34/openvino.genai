// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/llm.hpp"

#include <cmath>
#include <vector>

#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace ops {
namespace llm {

std::pair<Tensor, Tensor> rope_cos_sin(const Tensor& positions,
                                       int32_t head_dim,
                                       float rope_theta,
                                       const OpPolicy* policy) {
    (void)policy;
    auto* ctx = positions.context();
    const int32_t half_dim = head_dim / 2;
    std::vector<float> inv_freq(static_cast<size_t>(half_dim));
    for (int32_t i = 0; i < half_dim; ++i) {
        float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
        inv_freq[static_cast<size_t>(i)] = 1.0f / std::pow(rope_theta, exponent);
    }

    auto inv_freq_const = const_vec(ctx, inv_freq);
    auto inv_freq_shape =
        const_vec(ctx, std::vector<int64_t>{1, 1, static_cast<int64_t>(half_dim)});
    Tensor inv_freq_tensor(inv_freq_const, ctx);
    auto inv_freq_reshaped = inv_freq_tensor.reshape(inv_freq_shape, false);

    auto pos_f = positions.to(ov::element::f32);
    auto freqs = pos_f.unsqueeze(2) * inv_freq_reshaped;
    return {freqs.cos(), freqs.sin()};
}

Tensor apply_rope(const Tensor& x,
                  const Tensor& cos,
                  const Tensor& sin,
                  int32_t head_dim,
                  const OpPolicy* policy) {
    (void)policy;
    auto cos_unsq = cos.unsqueeze(1);
    auto sin_unsq = sin.unsqueeze(1);
    int64_t half_dim = head_dim / 2;

    auto x1 = slice(x, 0, half_dim, 1, 3);
    auto x2 = slice(x, half_dim, head_dim, 1, 3);

    auto rot1 = x1 * cos_unsq - x2 * sin_unsq;
    auto rot2 = x1 * sin_unsq + x2 * cos_unsq;
    return concat({rot1, rot2}, 3);
}

Tensor repeat_kv(const Tensor& x, int32_t num_heads, int32_t num_kv_heads, int32_t head_dim) {
    if (num_heads == num_kv_heads) {
        return x;
    }
    auto* ctx = x.context();
    const int32_t repeats = num_heads / num_kv_heads;
    auto unsq = x.unsqueeze(2);

    auto batch = shape::dim(x, 0);
    auto seq = shape::dim(x, 2);

    auto kv_heads = const_vec(ctx, std::vector<int64_t>{static_cast<int64_t>(num_kv_heads)});
    auto rep = const_vec(ctx, std::vector<int64_t>{static_cast<int64_t>(repeats)});
    auto hdim = const_vec(ctx, std::vector<int64_t>{static_cast<int64_t>(head_dim)});

    auto target = shape::make({batch, kv_heads, rep, seq, hdim});
    auto broadcast = shape::broadcast_to(unsq, target);

    auto heads = const_vec(ctx, std::vector<int64_t>{static_cast<int64_t>(num_heads)});
    auto reshape_shape = shape::make({batch, heads, seq, hdim});
    return broadcast.reshape(reshape_shape, false);
}

Tensor causal_mask(const Tensor& scores) {
    auto* ctx = scores.context();
    auto scores_shape = shape::of(scores);
    auto seq = Tensor(shape::dim(scores, 2), ctx).squeeze(0);

    auto idx = range(seq, 0, 1, ov::element::i64);
    auto row = idx.unsqueeze(1);
    auto col = idx.unsqueeze(0);
    auto ge = greater_equal(row, col);

    auto zero = Tensor(const_scalar(ctx, 0.0f), ctx);
    auto neg = Tensor(const_scalar(ctx, -65504.0f), ctx);
    auto mask2d = where(ge, zero, neg);
    auto mask4d = mask2d.unsqueeze({0, 1});
    return shape::broadcast_to(mask4d, scores_shape);
}

Tensor sdpa(const Tensor& q,
            const Tensor& k,
            const Tensor& v,
            float scale,
            int64_t softmax_axis,
            const Tensor* mask,
            bool causal,
            const OpPolicy* policy) {
    (void)policy;
    auto q_scaled = (scale == 1.0f) ? q : q * scale;
    auto scores = matmul(q_scaled, k, false, true);
    auto scores_f32 = scores.to(ov::element::f32);
    if (mask) {
        scores_f32 = scores_f32 + mask->to(ov::element::f32);
    }
    if (causal) {
        scores_f32 = scores_f32 + causal_mask(scores_f32);
    }
    auto probs = scores_f32.softmax(softmax_axis);
    return matmul(probs, v, false, false);
}

}  // namespace llm
}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
