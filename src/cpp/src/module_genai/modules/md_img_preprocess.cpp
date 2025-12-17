// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <thread>

#include "md_img_preprocess.hpp"

namespace ov {
namespace genai {
namespace module {

    ImagePreprocesModule::ImagePreprocesModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {

    }

    inline ov::Tensor get_input_tensor(const std::string& name, std::map<std::string, IBaseModule::InputModule>& inputs, const bool& optional = false) {
        // inputs[name].module_ptr->outputs
        // for (auto& input : inputs) {
        //     if (input.out_port_name == name) {
        //         input.module_ptr->outputs
        //         return input.data.as<ov::Tensor>();
        //     }
        // }
        OPENVINO_ASSERT(!optional, "Can't find input tensor with name " + name);
        return ov::Tensor();
    }

    void ImagePreprocesModule::prepare_inputs() {
        for (auto& input : this->inputs) {
            const auto& parent_port_name = input.first;
            input.second.data = input.second.module_ptr->outputs[parent_port_name].data;
        }
    }

    void ImagePreprocesModule::run() {
        prepare_inputs();

        auto image1_data = this->inputs["image1_data"].data.as<ov::Tensor>();
        auto image2_data = this->inputs["image2_data"].data.as<ov::Tensor>();

        image1_data.data<float>()[0] = 111;
        image2_data.data<float>()[0] = 222;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::cout << "Run: " << ModuleTypeConverter::toString(static_cast<ModuleType>(module_desc->type)) << "["
                  << module_desc->name << "]" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(700));

        auto thw_tensor = ov::Tensor(ov::element::i32, ov::Shape(3));
        thw_tensor.data<int>()[0] = 3;
        thw_tensor.data<int>()[1] = 4;
        thw_tensor.data<int>()[2] = 5;
        this->outputs["raw_data"].data = image1_data;
        this->outputs["thw"].data = thw_tensor;
    }

}  // namespace module
}  // namespace genai
}  // namespace ov
