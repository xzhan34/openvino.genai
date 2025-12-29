// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>

#include "modeling/ops/context.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {

class Parameter;

class BuilderContext {
public:
    BuilderContext() = default;

    Tensor parameter(const std::string& name, const ov::element::Type& type, const ov::PartialShape& shape);

    const ov::ParameterVector& parameters() const;
    std::shared_ptr<ov::Model> build_model(const ov::OutputVector& outputs,
                                           const ov::SinkVector& sinks = {}) const;

    OpContext& op_context();
    const OpContext& op_context() const;

    void register_parameter(const std::string& full_name, Parameter* param);
    Parameter* find_parameter(const std::string& full_name) const;
    const std::vector<Parameter*>& registered_parameters() const;

private:
    OpContext op_ctx_;
    ov::ParameterVector inputs_;
    std::unordered_map<std::string, Parameter*> params_by_name_;
    std::vector<Parameter*> params_;
};

}  // namespace modeling
}  // namespace genai
}  // namespace ov
