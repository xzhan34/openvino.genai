// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/weights/weight_loader.hpp"

#include <openvino/core/except.hpp>

namespace {

std::string replace_once(const std::string& input, const std::string& match, const std::string& replace) {
    auto pos = input.find(match);
    if (pos == std::string::npos) {
        return input;
    }
    std::string out = input;
    out.replace(pos, match.size(), replace);
    return out;
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace weights {

void default_weight_loader(Parameter& param,
                           WeightSource& source,
                           WeightMaterializer& materializer,
                           const std::string& weight_name,
                           const std::optional<int>& shard_id) {
    (void)shard_id;
    if (!param.context()) {
        OPENVINO_THROW("Parameter has no OpContext: ", param.name());
    }
    auto weight = materializer.materialize(weight_name, source, *param.context());
    param.bind(weight);
}

void load_model(Module& model, WeightSource& source, WeightMaterializer& materializer) {
    const auto& packed = model.packed_mapping();
    for (const auto& weight_name : source.keys()) {
        bool matched = false;
        for (const auto& rule : packed.rules) {
            if (weight_name.find(rule.match) != std::string::npos) {
                const std::string param_name = replace_once(weight_name, rule.match, rule.replace);
                auto& param = model.get_parameter(param_name);
                if (const auto* loader = param.weight_loader()) {
                    (*loader)(param, source, materializer, weight_name, rule.shard_id);
                } else {
                    default_weight_loader(param, source, materializer, weight_name, rule.shard_id);
                }
                matched = true;
                break;
            }
        }

        if (!matched) {
            auto& param = model.get_parameter(weight_name);
            if (const auto* loader = param.weight_loader()) {
                (*loader)(param, source, materializer, weight_name, std::nullopt);
            } else {
                default_weight_loader(param, source, materializer, weight_name, std::nullopt);
            }
        }
    }
    model.finalize_parameters();
}

}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov
