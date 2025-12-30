// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

#include "modeling/module.hpp"
#include "modeling/weights/weight_finalizer.hpp"
#include "modeling/weights/weight_source.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace weights {

void default_weight_loader(WeightParameter& param,
                           WeightSource& source,
                           WeightFinalizer& finalizer,
                           const std::string& weight_name,
                           const std::optional<int>& shard_id);

void load_model(Module& model, WeightSource& source, WeightFinalizer& finalizer);

}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov
