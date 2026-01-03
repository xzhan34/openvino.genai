// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/tests/test_utils.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

namespace ov {
namespace genai {
namespace modeling {
namespace tests {

ov::Tensor make_tensor(const std::vector<float>& data, const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::f32, shape);
    std::memcpy(tensor.data(), data.data(), data.size() * sizeof(float));
    return tensor;
}

std::vector<float> make_seq(size_t n, float start, float step) {
    std::vector<float> out(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        out[i] = start + step * static_cast<float>(i);
    }
    return out;
}

std::vector<float> matmul_ref(const std::vector<float>& a,
                              const std::vector<float>& b,
                              size_t m,
                              size_t k,
                              size_t n) {
    std::vector<float> out(m * n, 0.0f);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t kk = 0; kk < k; ++kk) {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    return out;
}

std::vector<float> matmul_ref_transpose_a(const std::vector<float>& a,
                                          const std::vector<float>& b,
                                          size_t m,
                                          size_t k,
                                          size_t n) {
    std::vector<float> out(m * n, 0.0f);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t kk = 0; kk < k; ++kk) {
                acc += a[kk * m + i] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    return out;
}

std::vector<float> linear_ref(const std::vector<float>& x,
                              const std::vector<float>& w,
                              size_t rows,
                              size_t in_features,
                              size_t out_features) {
    std::vector<float> y(rows * out_features, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t o = 0; o < out_features; ++o) {
            float acc = 0.0f;
            for (size_t i = 0; i < in_features; ++i) {
                acc += x[r * in_features + i] * w[o * in_features + i];
            }
            y[r * out_features + o] = acc;
        }
    }
    return y;
}

std::vector<float> linear_ref_3d(const std::vector<float>& x,
                                 const std::vector<float>& w,
                                 size_t batch,
                                 size_t seq_len,
                                 size_t in_features,
                                 size_t out_features) {
    std::vector<float> out(batch * seq_len * out_features, 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const size_t x_base = (b * seq_len + s) * in_features;
            const size_t y_base = (b * seq_len + s) * out_features;
            for (size_t o = 0; o < out_features; ++o) {
                float acc = 0.0f;
                for (size_t i = 0; i < in_features; ++i) {
                    acc += x[x_base + i] * w[o * in_features + i];
                }
                out[y_base + o] = acc;
            }
        }
    }
    return out;
}

std::vector<float> linear_ref_3d_bias(const std::vector<float>& x,
                                      const std::vector<float>& w,
                                      const std::vector<float>& bias,
                                      size_t batch,
                                      size_t seq_len,
                                      size_t in_features,
                                      size_t out_features) {
    auto out = linear_ref_3d(x, w, batch, seq_len, in_features, out_features);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const size_t base = (b * seq_len + s) * out_features;
            for (size_t o = 0; o < out_features; ++o) {
                out[base + o] += bias[o];
            }
        }
    }
    return out;
}

std::vector<float> mean_ref(const std::vector<float>& x, size_t rows, size_t cols) {
    std::vector<float> out(rows, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        float acc = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            acc += x[r * cols + c];
        }
        out[r] = acc / static_cast<float>(cols);
    }
    return out;
}

std::vector<float> rms_ref(const std::vector<float>& x,
                           const std::vector<float>& weight,
                           size_t rows,
                           size_t cols,
                           float eps) {
    std::vector<float> out(x.size(), 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        float sumsq = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float v = x[r * cols + c];
            sumsq += v * v;
        }
        float mean = sumsq / static_cast<float>(cols);
        float inv = 1.0f / std::sqrt(mean + eps);
        for (size_t c = 0; c < cols; ++c) {
            out[r * cols + c] = x[r * cols + c] * inv * weight[c];
        }
    }
    return out;
}

std::vector<float> embedding_ref(const std::vector<int64_t>& ids,
                                 const std::vector<float>& weight,
                                 size_t rows,
                                 size_t cols,
                                 size_t embed_dim) {
    std::vector<float> y(rows * cols * embed_dim, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            int64_t idx = ids[r * cols + c];
            for (size_t e = 0; e < embed_dim; ++e) {
                y[(r * cols + c) * embed_dim + e] = weight[static_cast<size_t>(idx) * embed_dim + e];
            }
        }
    }
    return y;
}

std::vector<float> add_ref(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(a.size(), 0.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

std::vector<float> mul_ref(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(a.size(), 0.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] * b[i];
    }
    return out;
}

std::vector<float> silu_ref(const std::vector<float>& x) {
    std::vector<float> out(x.size(), 0.0f);
    for (size_t i = 0; i < x.size(); ++i) {
        const float v = x[i];
        out[i] = v / (1.0f + std::exp(-v));
    }
    return out;
}

std::vector<float> mlp_ref(const std::vector<float>& x,
                           const std::vector<float>& gate_w,
                           const std::vector<float>& up_w,
                           const std::vector<float>& down_w,
                           size_t batch,
                           size_t seq_len,
                           size_t hidden,
                           size_t intermediate) {
    auto gate = linear_ref_3d(x, gate_w, batch, seq_len, hidden, intermediate);
    auto up = linear_ref_3d(x, up_w, batch, seq_len, hidden, intermediate);
    auto gated = mul_ref(silu_ref(gate), up);
    return linear_ref_3d(gated, down_w, batch, seq_len, intermediate, hidden);
}

std::vector<float> to_heads_ref(const std::vector<float>& x,
                                size_t batch,
                                size_t seq_len,
                                size_t num_heads,
                                size_t head_dim) {
    std::vector<float> out(batch * num_heads * seq_len * head_dim, 0.0f);
    const size_t hidden = num_heads * head_dim;
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const size_t in_base = (b * seq_len + s) * hidden;
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t out_base = ((b * num_heads + h) * seq_len + s) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    out[out_base + d] = x[in_base + h * head_dim + d];
                }
            }
        }
    }
    return out;
}

std::vector<float> rmsnorm_heads_ref(const std::vector<float>& x,
                                     const std::vector<float>& weight,
                                     size_t batch,
                                     size_t num_heads,
                                     size_t seq_len,
                                     size_t head_dim,
                                     float eps) {
    std::vector<float> out(x.size(), 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                const size_t base = ((b * num_heads + h) * seq_len + s) * head_dim;
                float sumsq = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    const float v = x[base + d];
                    sumsq += v * v;
                }
                const float mean = sumsq / static_cast<float>(head_dim);
                const float inv = 1.0f / std::sqrt(mean + eps);
                for (size_t d = 0; d < head_dim; ++d) {
                    out[base + d] = x[base + d] * inv * weight[d];
                }
            }
        }
    }
    return out;
}

std::vector<float> merge_heads_ref(const std::vector<float>& x,
                                   size_t batch,
                                   size_t seq_len,
                                   size_t num_heads,
                                   size_t head_dim) {
    std::vector<float> out(batch * seq_len * num_heads * head_dim, 0.0f);
    const size_t hidden = num_heads * head_dim;
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const size_t out_base = (b * seq_len + s) * hidden;
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t in_base = ((b * num_heads + h) * seq_len + s) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    out[out_base + h * head_dim + d] = x[in_base + d];
                }
            }
        }
    }
    return out;
}

std::vector<float> apply_rope_ref(const std::vector<float>& x,
                                  const std::vector<int64_t>& positions,
                                  size_t batch,
                                  size_t seq_len,
                                  size_t num_heads,
                                  size_t head_dim,
                                  float rope_theta) {
    const size_t half_dim = head_dim / 2;
    std::vector<float> inv_freq(half_dim, 0.0f);
    for (size_t i = 0; i < half_dim; ++i) {
        float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
        inv_freq[i] = 1.0f / std::pow(rope_theta, exponent);
    }

    std::vector<float> out(x.size(), 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const float pos = static_cast<float>(positions[b * seq_len + s]);
            for (size_t i = 0; i < half_dim; ++i) {
                const float angle = pos * inv_freq[i];
                const float c = std::cos(angle);
                const float si = std::sin(angle);
                for (size_t h = 0; h < num_heads; ++h) {
                    const size_t base = ((b * num_heads + h) * seq_len + s) * head_dim;
                    const float x1 = x[base + i];
                    const float x2 = x[base + i + half_dim];
                    out[base + i] = x1 * c - x2 * si;
                    out[base + i + half_dim] = x1 * si + x2 * c;
                }
            }
        }
    }
    return out;
}

std::vector<float> repeat_kv_ref(const std::vector<float>& x,
                                 size_t batch,
                                 size_t num_heads,
                                 size_t num_kv_heads,
                                 size_t seq_len,
                                 size_t head_dim) {
    if (num_heads == num_kv_heads) {
        return x;
    }
    std::vector<float> out(batch * num_heads * seq_len * head_dim, 0.0f);
    const size_t repeats = num_heads / num_kv_heads;
    for (size_t b = 0; b < batch; ++b) {
        for (size_t kv = 0; kv < num_kv_heads; ++kv) {
            for (size_t r = 0; r < repeats; ++r) {
                const size_t h = kv * repeats + r;
                for (size_t s = 0; s < seq_len; ++s) {
                    const size_t in_base = ((b * num_kv_heads + kv) * seq_len + s) * head_dim;
                    const size_t out_base = ((b * num_heads + h) * seq_len + s) * head_dim;
                    for (size_t d = 0; d < head_dim; ++d) {
                        out[out_base + d] = x[in_base + d];
                    }
                }
            }
        }
    }
    return out;
}

std::vector<float> attention_ref(const std::vector<float>& hidden,
                                 const std::vector<float>& q_w,
                                 const std::vector<float>& q_b,
                                 const std::vector<float>& k_w,
                                 const std::vector<float>& k_b,
                                 const std::vector<float>& v_w,
                                 const std::vector<float>& v_b,
                                 const std::vector<float>& o_w,
                                 const std::vector<float>& o_b,
                                 const std::vector<float>* q_norm_w,
                                 const std::vector<float>* k_norm_w,
                                 const std::vector<int64_t>& positions,
                                 size_t batch,
                                 size_t seq_len,
                                 size_t hidden_size,
                                 size_t num_heads,
                                 size_t num_kv_heads,
                                 size_t head_dim,
                                 float rope_theta,
                                 float rms_norm_eps) {
    const size_t kv_hidden = num_kv_heads * head_dim;
    auto q = linear_ref_3d_bias(hidden, q_w, q_b, batch, seq_len, hidden_size, hidden_size);
    auto k = linear_ref_3d_bias(hidden, k_w, k_b, batch, seq_len, hidden_size, kv_hidden);
    auto v = linear_ref_3d_bias(hidden, v_w, v_b, batch, seq_len, hidden_size, kv_hidden);

    auto qh = to_heads_ref(q, batch, seq_len, num_heads, head_dim);
    auto kh = to_heads_ref(k, batch, seq_len, num_kv_heads, head_dim);
    auto vh = to_heads_ref(v, batch, seq_len, num_kv_heads, head_dim);

    if (q_norm_w) {
        qh = rmsnorm_heads_ref(qh, *q_norm_w, batch, num_heads, seq_len, head_dim, rms_norm_eps);
    }
    if (k_norm_w) {
        kh = rmsnorm_heads_ref(kh, *k_norm_w, batch, num_kv_heads, seq_len, head_dim, rms_norm_eps);
    }

    qh = apply_rope_ref(qh, positions, batch, seq_len, num_heads, head_dim, rope_theta);
    kh = apply_rope_ref(kh, positions, batch, seq_len, num_kv_heads, head_dim, rope_theta);

    auto kh_expanded = repeat_kv_ref(kh, batch, num_heads, num_kv_heads, seq_len, head_dim);
    auto vh_expanded = repeat_kv_ref(vh, batch, num_heads, num_kv_heads, seq_len, head_dim);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> attn_probs(batch * num_heads * seq_len * seq_len, 0.0f);
    std::vector<float> scores(seq_len, 0.0f);

    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                float max_score = -1e30f;
                for (size_t j = 0; j < seq_len; ++j) {
                    if (j > i) {
                        scores[j] = -65504.0f;
                        continue;
                    }
                    float acc = 0.0f;
                    const size_t q_base = ((b * num_heads + h) * seq_len + i) * head_dim;
                    const size_t k_base = ((b * num_heads + h) * seq_len + j) * head_dim;
                    for (size_t d = 0; d < head_dim; ++d) {
                        acc += qh[q_base + d] * kh_expanded[k_base + d];
                    }
                    acc *= scale;
                    scores[j] = acc;
                    if (acc > max_score) {
                        max_score = acc;
                    }
                }
                float sum = 0.0f;
                for (size_t j = 0; j < seq_len; ++j) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum += scores[j];
                }
                const size_t prob_base = ((b * num_heads + h) * seq_len + i) * seq_len;
                for (size_t j = 0; j < seq_len; ++j) {
                    attn_probs[prob_base + j] = scores[j] / sum;
                }
            }
        }
    }

    std::vector<float> context(batch * num_heads * seq_len * head_dim, 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                const size_t prob_base = ((b * num_heads + h) * seq_len + i) * seq_len;
                const size_t ctx_base = ((b * num_heads + h) * seq_len + i) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    float acc = 0.0f;
                    for (size_t j = 0; j < seq_len; ++j) {
                        const size_t v_base = ((b * num_heads + h) * seq_len + j) * head_dim;
                        acc += attn_probs[prob_base + j] * vh_expanded[v_base + d];
                    }
                    context[ctx_base + d] = acc;
                }
            }
        }
    }

    auto merged = merge_heads_ref(context, batch, seq_len, num_heads, head_dim);
    return linear_ref_3d_bias(merged, o_w, o_b, batch, seq_len, hidden_size, hidden_size);
}

void expect_tensor_near(const ov::Tensor& output, const std::vector<float>& expected, float tol) {
    ASSERT_EQ(output.get_size(), expected.size());
    const float* out_data = output.data<const float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(out_data[i], expected[i], tol);
    }
}

}  // namespace tests
}  // namespace modeling
}  // namespace genai
}  // namespace ov
