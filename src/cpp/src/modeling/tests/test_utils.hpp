// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/core/except.hpp>
#include <openvino/openvino.hpp>

#include "modeling/ops/ops.hpp"
#include "modeling/weights/weight_finalizer.hpp"
#include "modeling/weights/weight_source.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace tests {

class DummyWeightSource : public weights::WeightSource {
public:
    void add(const std::string& name, const ov::Tensor& tensor) {
        if (!weights_.count(name)) {
            keys_.push_back(name);
        }
        weights_[name] = tensor;
    }

    std::vector<std::string> keys() const override {
        return keys_;
    }

    bool has(const std::string& name) const override {
        return weights_.count(name) != 0;
    }

    const ov::Tensor& get_tensor(const std::string& name) const override {
        auto it = weights_.find(name);
        if (it == weights_.end()) {
            OPENVINO_THROW("Unknown weight: ", name);
        }
        return it->second;
    }

private:
    std::unordered_map<std::string, ov::Tensor> weights_;
    std::vector<std::string> keys_;
};

class DummyWeightFinalizer : public weights::WeightFinalizer {
public:
    Tensor finalize(const std::string& name, weights::WeightSource& source, OpContext& ctx) override {
        const auto& tensor = source.get_tensor(name);
        return ops::constant(tensor, &ctx);
    }
};

}  // namespace tests
}  // namespace modeling
}  // namespace genai
}  // namespace ov
