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
    using PTR = std::shared_ptr<IBaseModule>;

    virtual void run() = 0;

    virtual std::string get_name() = 0;
};

}  // namespace module
}  // namespace genai
}  // namespace ov
