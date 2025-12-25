// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <openvino/openvino.hpp>

#include "modeling/ops/context.hpp"

namespace ov {
namespace genai {
namespace modeling {

class Tensor {
public:
    Tensor() = default;
    Tensor(const ov::Output<ov::Node>& value, OpContext* ctx = nullptr);

    const ov::Output<ov::Node>& output() const;
    OpContext* context() const;

    ov::element::Type dtype() const;
    Tensor to(const ov::element::Type& type) const;

    Tensor pow(float exp) const;
    Tensor mean(int64_t axis, bool keepdim = true) const;
    Tensor rsqrt() const;

private:
    ov::Output<ov::Node> value_;
    OpContext* ctx_ = nullptr;
};

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator+(const Tensor& a, float b);
Tensor operator+(float a, const Tensor& b);

Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);

}  // namespace modeling
}  // namespace genai
}  // namespace ov
