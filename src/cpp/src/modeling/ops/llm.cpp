// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/llm.hpp"

#include <cmath>
#include <vector>

#include <openvino/opsets/opset13.hpp>
#include <ov_ops/rotary_positional_embeddings.hpp>

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
    // Use internal RoPE op directly for optimal GPU performance.
    // This avoids relying on RoPEFusion transformation to match patterns.
    //
    // Input shapes:
    //   x: [batch, heads, seq, head_dim]
    //   cos/sin: [batch, seq, half_dim] where half_dim = head_dim / 2
    //
    // RoPE op expects cos/sin with shape [batch, seq, rotary_ndims] and will
    // handle the rotation internally.

    op::internal::RoPE::Config config;
    config.rotary_ndims = static_cast<size_t>(head_dim);
    config.is_interleaved = false;
    config.input_trans0213 = false;
    config.output_trans0213 = false;

    // Expand cos/sin from half_dim to head_dim by concatenating with themselves
    auto cos_full = concat({cos, cos}, 2);  // [batch, seq, head_dim]
    auto sin_full = concat({sin, sin}, 2);  // [batch, seq, head_dim]

    auto rope_node = std::make_shared<op::internal::RoPE>(
        ov::OutputVector{x.output(), cos_full.output(), sin_full.output()},
        config);

    return Tensor(rope_node, x.context());
}

Tensor apply_rope_interleave(const Tensor& x,
                             const Tensor& cos,
                             const Tensor& sin,
                             int32_t head_dim,
                             const OpPolicy* policy) {
    (void)policy;
    const int32_t half_dim = head_dim / 2;
    auto interleaved = x.reshape({0, 0, 0, half_dim, 2})
                           .permute({0, 1, 2, 4, 3})
                           .reshape({0, 0, 0, head_dim});
    return apply_rope(interleaved, cos, sin, head_dim, policy);
}

Tensor rope_tail(const Tensor& cos_or_sin, const Tensor& q) {
    auto* ctx = cos_or_sin.context();
    auto total_len = shape::dim(cos_or_sin, 1);
    auto q_len = shape::dim(q, 2);

    auto total_len_scalar = Tensor(total_len, ctx).squeeze(0);
    auto q_len_scalar = Tensor(q_len, ctx).squeeze(0);
    auto start = total_len_scalar - q_len_scalar;

    auto indices = range(start, total_len_scalar, 1, ov::element::i64);
    return gather(cos_or_sin, indices, 1);
}

Tensor pad_to_head_dim(const Tensor& x, int32_t head_dim, int32_t target_head_dim) {
    if (target_head_dim <= head_dim) {
        return x;
    }
    auto* ctx = x.context();
    const int32_t pad = target_head_dim - head_dim;

    auto batch = shape::dim(x, 0);
    auto heads = shape::dim(x, 1);
    auto seq = shape::dim(x, 2);
    auto pad_dim = const_vec(ctx, std::vector<int64_t>{static_cast<int64_t>(pad)});
    auto pad_shape = shape::make({batch, heads, seq, pad_dim});

    auto zero = Tensor(const_scalar(ctx, 0.0f), ctx).to(x.dtype());
    auto pad_tensor = shape::broadcast_to(zero, pad_shape);
    return concat({x, pad_tensor}, 3);
}

Tensor slice_to_head_dim(const Tensor& x, int32_t head_dim, int32_t target_head_dim) {
    if (target_head_dim >= head_dim) {
        return x;
    }
    return slice(x, 0, target_head_dim, 1, 3);
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

Tensor causal_mask_from_seq_len(const Tensor& seq_len) {
    auto* ctx = seq_len.context();
    auto idx = range(seq_len, 0, 1, ov::element::i64);
    auto row = idx.unsqueeze(1);
    auto col = idx.unsqueeze(0);
    auto ge = greater_equal(row, col);

    auto zero = Tensor(const_scalar(ctx, 0.0f), ctx);
    auto neg = Tensor(const_scalar(ctx, -65504.0f), ctx);
    auto mask2d = where(ge, zero, neg);
    return mask2d.unsqueeze({0, 1});
}

Tensor causal_mask(const Tensor& scores) {
    auto scores_shape = shape::of(scores);
    auto seq = Tensor(shape::dim(scores, 2), scores.context()).squeeze(0);
    auto mask4d = causal_mask_from_seq_len(seq);
    return shape::broadcast_to(mask4d, scores_shape);
}

Tensor build_kv_causal_mask(const Tensor& q, const Tensor& k) {
    // Build causal mask for KV cache scenario:
    // Q: [batch, heads, q_len, head_dim]
    // K: [batch, heads, kv_len, head_dim]
    // Output: [batch, 1, q_len, kv_len] where mask[i,j]=0 if col_j <= row_i_absolute, else -inf
    //
    // For prefill: q_len=N, kv_len=N, behaves like standard causal mask
    // For decode: q_len=1, kv_len=cache_len+1, allows attending to all positions
    auto* ctx = q.context();

    // Get dimensions as shape [1] tensors
    auto batch = shape::dim(q, 0);  // [1]
    auto q_len = shape::dim(q, 2);  // [1]
    auto kv_len = shape::dim(k, 2); // [1]

    // Squeeze to scalars for Range op
    auto q_len_scalar = Tensor(q_len, ctx).squeeze(0);
    auto kv_len_scalar = Tensor(kv_len, ctx).squeeze(0);

    // Calculate cache_seq_len = kv_len - q_len (as scalar)
    auto cache_len_scalar = Tensor(
        std::make_shared<ov::opset13::Subtract>(kv_len_scalar.output(), q_len_scalar.output())->output(0), ctx);

    // Convert to i32 for Range
    auto cache_len_i32 = Tensor(
        std::make_shared<ov::op::v0::Convert>(cache_len_scalar.output(), ov::element::i32)->output(0), ctx);
    auto q_len_i32 = Tensor(
        std::make_shared<ov::op::v0::Convert>(q_len_scalar.output(), ov::element::i32)->output(0), ctx);
    auto kv_len_i32 = Tensor(
        std::make_shared<ov::op::v0::Convert>(kv_len_scalar.output(), ov::element::i32)->output(0), ctx);

    // Create col indices: [0, 1, 2, ..., kv_len-1] -> [1, kv_len]
    auto col_range = range(kv_len_i32, 0, 1, ov::element::i32);
    auto col_indices = col_range.unsqueeze(0);  // [1, kv_len]

    // Create row indices: [cache_len, cache_len+1, ..., cache_len+q_len-1] -> [q_len, 1]
    // These represent the absolute positions of query tokens
    auto q_len_plus_cache = cache_len_i32 + q_len_i32;
    auto row_range = range(cache_len_i32, q_len_plus_cache, 1, ov::element::i32);
    auto row_indices = row_range.unsqueeze(1);  // [q_len, 1]

    // Causal condition: col <= row (attend to current and past positions)
    auto causal_cond = less_equal(col_indices, row_indices);  // [q_len, kv_len]

    // Build mask: 0 where can attend, -inf where masked
    auto zero_val = Tensor(const_scalar(ctx, 0.0f), ctx);
    auto neg_inf = Tensor(const_scalar(ctx, -65504.0f), ctx);
    auto mask_2d = where(causal_cond, zero_val, neg_inf);  // [q_len, kv_len]

    // Expand to [batch, 1, q_len, kv_len]
    auto mask_4d = mask_2d.unsqueeze({0, 1});

    // Broadcast to batch size
    auto one_val = const_vec(ctx, std::vector<int64_t>{1});
    auto target_shape = shape::make({batch, one_val, q_len, kv_len});
    return shape::broadcast_to(mask_4d, target_shape);
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
    (void)softmax_axis;  // Native SDPA handles this internally
    (void)scale;  // Native SDPA handles scaling internally

    auto* ctx = q.context();

    // Use native ScaledDotProductAttention for optimal GPU performance
    if (mask) {
        auto sdpa_node = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
            q.output(), k.output(), v.output(), mask->output(), causal);
        return Tensor(sdpa_node, ctx);
    } else {
        auto sdpa_node = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
            q.output(), k.output(), v.output(), causal);
        return Tensor(sdpa_node, ctx);
    }
}

}  // namespace llm
}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
