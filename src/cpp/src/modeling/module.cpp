// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/module.hpp"

#include <algorithm>

#include <openvino/core/except.hpp>

#include "modeling/ops/ops.hpp"

namespace ov {
namespace genai {
namespace modeling {

Parameter::Parameter(std::string name, OpContext* ctx) : name_(std::move(name)), ctx_(ctx) {}

const std::string& Parameter::name() const {
    return name_;
}

OpContext* Parameter::context() const {
    return ctx_;
}

void Parameter::set_weight_loader(WeightLoaderFn fn) {
    weight_loader_ = std::move(fn);
}

const Parameter::WeightLoaderFn* Parameter::weight_loader() const {
    if (!weight_loader_) {
        return nullptr;
    }
    return &weight_loader_;
}

void Parameter::bind(const Tensor& weight) {
    if (tied_to_) {
        tied_to_ = nullptr;
    }
    auto* w_ctx = weight.context();
    if (ctx_ && w_ctx && ctx_ != w_ctx) {
        OPENVINO_THROW("Parameter context does not match weight context: ", name_);
    }
    weight_ = weight;
    bound_ = true;
    shards_.clear();
}

bool Parameter::is_bound() const {
    return bound_ || tied_to_ != nullptr;
}

const Tensor& Parameter::value() const {
    if (tied_to_) {
        return tied_to_->value();
    }
    if (!bound_) {
        OPENVINO_THROW("Parameter not bound: ", name_);
    }
    return weight_;
}

void Parameter::add_shard(int shard_id, const Tensor& shard) {
    if (bound_) {
        OPENVINO_THROW("Parameter already bound: ", name_);
    }
    if (tied_to_) {
        OPENVINO_THROW("Parameter is tied and cannot accept shards: ", name_);
    }
    shards_[shard_id] = shard;
}

void Parameter::finalize() {
    if (bound_ || tied_to_ || shards_.empty()) {
        return;
    }
    if (shards_.size() == 1) {
        bind(shards_.begin()->second);
        return;
    }

    std::vector<int> keys;
    keys.reserve(shards_.size());
    for (const auto& kv : shards_) {
        keys.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());

    std::vector<Tensor> parts;
    parts.reserve(keys.size());
    for (int k : keys) {
        parts.push_back(shards_.at(k));
    }

    auto merged = ops::concat(parts, 0);
    bind(merged);
}

void Parameter::tie_to(Parameter& other) {
    if (&other == this) {
        return;
    }
    tied_to_ = &other;
}

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

Parameter& Module::register_parameter(const std::string& name) {
    if (!ctx_) {
        OPENVINO_THROW("Module has no BuilderContext for parameter registration");
    }
    const std::string full_name = full_path_.empty() ? name : full_path_ + "." + name;
    auto param = std::make_unique<Parameter>(full_name, &ctx_->op_context());
    auto* ptr = param.get();
    parameters_.push_back(std::move(param));
    ctx_->register_parameter(full_name, ptr);
    return *ptr;
}

Parameter& Module::get_parameter(const std::string& full_name) {
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
