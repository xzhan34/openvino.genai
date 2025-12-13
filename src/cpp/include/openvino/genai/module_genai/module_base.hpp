// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/visibility.hpp"
#include "openvino/core/any.hpp"

namespace ov {
namespace genai {
namespace module {

class OPENVINO_GENAI_EXPORTS IModule {
public:
    ~IModule() = default;
    // initialize
    virtual bool initialize() = 0;
};

class OPENVINO_GENAI_EXPORTS IBaseModule : public IModule {
public:
    ~IBaseModule() = default;

    virtual void run() = 0;

    using PTR = std::shared_ptr<IBaseModule>;
};

}  // namespace module
}  // namespace genai
}  // namespace ov
