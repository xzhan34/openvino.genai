// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "modeling/ops/context.hpp"
#include "modeling/ops/tensor.hpp"
#include "modeling/weights/weight_source.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace weights {

class WeightFinalizer {
public:
    virtual ~WeightFinalizer() = default;

    virtual Tensor finalize(const std::string& name, WeightSource& source, OpContext& ctx) = 0;
};

}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov
