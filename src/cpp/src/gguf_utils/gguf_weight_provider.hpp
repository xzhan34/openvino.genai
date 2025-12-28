// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>

#include <openvino/openvino.hpp>

#include "gguf_utils/gguf.hpp"
#include "modeling/ops/context.hpp"
#include "modeling/weights/weights.hpp"

namespace ov {
namespace genai {
namespace gguf {

class GGUFWeightProvider : public ov::genai::modeling::weights::IWeightProvider {
public:
    GGUFWeightProvider(const std::unordered_map<std::string, ov::Tensor>& consts,
                       const std::unordered_map<std::string, gguf_tensor_type>& qtypes,
                       ov::genai::modeling::OpContext* ctx);

    bool has(const std::string& key) const override;
    ov::genai::modeling::Tensor get(const std::string& base_key) override;

private:
    gguf_tensor_type resolve_qtype(const std::string& base_key) const;

    const std::unordered_map<std::string, ov::Tensor>& consts_;
    const std::unordered_map<std::string, gguf_tensor_type>& qtypes_;
    ov::genai::modeling::OpContext* ctx_ = nullptr;
    std::unordered_map<std::string, ov::Output<ov::Node>> cache_;
};

}  // namespace gguf
}  // namespace genai
}  // namespace ov

