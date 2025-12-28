// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_utils/gguf_weight_provider.hpp"

#include <openvino/core/except.hpp>

#include "gguf_utils/building_blocks.hpp"

namespace ov {
namespace genai {
namespace gguf {

GGUFWeightProvider::GGUFWeightProvider(const std::unordered_map<std::string, ov::Tensor>& consts,
                                       const std::unordered_map<std::string, gguf_tensor_type>& qtypes,
                                       ov::genai::modeling::OpContext* ctx) :
    consts_(consts),
    qtypes_(qtypes),
    ctx_(ctx) {}

bool GGUFWeightProvider::has(const std::string& key) const {
    return consts_.count(key) > 0;
}

gguf_tensor_type GGUFWeightProvider::resolve_qtype(const std::string& base_key) const {
    // Some GGUFs may ship only F16 weights without explicit quantization params.
    // In that case, the weight behaves as F16 even if qtype metadata is present.
    const bool has_scales = consts_.count(base_key + ".scales") > 0;
    if (!has_scales) {
        return gguf_tensor_type::GGUF_TYPE_F16;
    }

    const auto it = qtypes_.find(base_key + ".qtype");
    if (it == qtypes_.end()) {
        return gguf_tensor_type::GGUF_TYPE_F16;
    }
    return it->second;
}

ov::genai::modeling::Tensor GGUFWeightProvider::get(const std::string& base_key) {
    const auto it_cached = cache_.find(base_key);
    if (it_cached != cache_.end()) {
        return ov::genai::modeling::Tensor(it_cached->second, ctx_);
    }

    OPENVINO_ASSERT(consts_.count(base_key + ".weight") > 0, "Missing tensor: ", base_key, ".weight");

    const auto qtype = resolve_qtype(base_key);
    auto w = make_weights_subgraph(base_key, consts_, qtype, /*reorder=*/false, /*head_size=*/-1);
    cache_.emplace(base_key, w);
    return ov::genai::modeling::Tensor(w, ctx_);
}

}  // namespace gguf
}  // namespace genai
}  // namespace ov

