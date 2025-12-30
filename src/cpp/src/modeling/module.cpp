// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/module.hpp"

#include <openvino/core/except.hpp>

namespace ov {
namespace genai {
namespace modeling {

Module::Module(std::string name, BuilderContext& ctx, Module* parent)
    : ctx_(&ctx), parent_(parent), name_(std::move(name)) {
    if (parent_ && !parent_->full_path().empty()) {
        full_path_ = parent_->full_path() + "." + name_;
    } else {
        full_path_ = name_;
    }
}

const std::string& Module::name() const {
    return name_;
}

const std::string& Module::full_path() const {
    return full_path_;
}

WeightParameter& Module::register_parameter(const std::string& name) {
    if (!ctx_) {
        OPENVINO_THROW("Module has no BuilderContext for parameter registration");
    }
    const std::string full_name = full_path_.empty() ? name : full_path_ + "." + name;
    auto param = std::make_unique<WeightParameter>(full_name, &ctx_->op_context());
    auto* ptr = param.get();
    parameters_.push_back(std::move(param));
    ctx_->register_parameter(full_name, ptr);
    return *ptr;
}

WeightParameter& Module::get_parameter(const std::string& full_name) {
    if (!ctx_) {
        OPENVINO_THROW("Module has no BuilderContext for parameter lookup");
    }
    auto* param = ctx_->find_parameter(full_name);
    if (!param) {
        OPENVINO_THROW("Unknown parameter: ", full_name);
    }
    return *param;
}

BuilderContext& Module::ctx() {
    if (!ctx_) {
        OPENVINO_THROW("Module has no BuilderContext");
    }
    return *ctx_;
}

const BuilderContext& Module::ctx() const {
    if (!ctx_) {
        OPENVINO_THROW("Module has no BuilderContext");
    }
    return *ctx_;
}

PackedMapping& Module::packed_mapping() {
    return packed_mapping_;
}

const PackedMapping& Module::packed_mapping() const {
    return packed_mapping_;
}

void Module::finalize_parameters() {
    if (!ctx_) {
        return;
    }
    for (auto* param : ctx_->registered_parameters()) {
        if (param) {
            param->finalize();
        }
    }
}

}  // namespace modeling
}  // namespace genai
}  // namespace ov
