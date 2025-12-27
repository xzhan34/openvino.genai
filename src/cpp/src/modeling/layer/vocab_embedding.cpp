// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/layer/vocab_embedding.hpp"

#include <openvino/openvino.hpp>

#include "modeling/ops/ops.hpp"

namespace ov {
namespace genai {
namespace modeling {

VocabEmbedding::VocabEmbedding(const Tensor& weight) : weight_(weight) {}

Tensor VocabEmbedding::operator()(const Tensor& ids) const {
    auto ids_i32 = ids.to(ov::element::i32);
    return ops::gather(weight_, ids_i32, 0);
}

}  // namespace modeling
}  // namespace genai
}  // namespace ov

