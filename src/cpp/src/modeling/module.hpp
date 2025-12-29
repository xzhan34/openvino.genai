// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {

namespace weights {
class WeightSource;
class WeightMaterializer;
}  // namespace weights

struct PackedRule {
    std::string match;
    std::string replace;
    int shard_id = 0;
};

struct PackedMapping {
    std::vector<PackedRule> rules;
};

class Parameter {
public:
    using WeightLoaderFn = std::function<void(Parameter&,
                                              weights::WeightSource&,
                                              weights::WeightMaterializer&,
                                              const std::string& weight_name,
                                              const std::optional<int>& shard_id)>;

    Parameter(std::string name, OpContext* ctx);

    const std::string& name() const;
    OpContext* context() const;

    void set_weight_loader(WeightLoaderFn fn);
    const WeightLoaderFn* weight_loader() const;

    void bind(const Tensor& weight);
    bool is_bound() const;
    const Tensor& value() const;

    void add_shard(int shard_id, const Tensor& shard);
    void finalize();

    void tie_to(Parameter& other);

private:
    std::string name_;
    OpContext* ctx_ = nullptr;
    Tensor weight_;
    bool bound_ = false;
    Parameter* tied_to_ = nullptr;
    std::unordered_map<int, Tensor> shards_;
    WeightLoaderFn weight_loader_;
};

class Module {
public:
    Module() = default;
    Module(std::string name, BuilderContext& ctx, Module* parent = nullptr);
    virtual ~Module() = default;

    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&&) = default;
    Module& operator=(Module&&) = default;

    const std::string& name() const;
    const std::string& full_path() const;

    Parameter& register_parameter(const std::string& name);
    Parameter& get_parameter(const std::string& full_name);

    BuilderContext& ctx();
    const BuilderContext& ctx() const;

    PackedMapping& packed_mapping();
    const PackedMapping& packed_mapping() const;

    void finalize_parameters();

protected:
    BuilderContext* ctx_ = nullptr;
    Module* parent_ = nullptr;
    std::string name_;
    std::string full_path_;
    std::vector<std::unique_ptr<Parameter>> parameters_;
    PackedMapping packed_mapping_;
};

}  // namespace modeling
}  // namespace genai
}  // namespace ov
