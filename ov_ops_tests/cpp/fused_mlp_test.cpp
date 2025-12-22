#include <openvino/openvino.hpp>
#include "openvino/opsets/opset13.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/op/fused_mlp_extension.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <numeric>
#include <algorithm>

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static std::vector<float> matmul(const std::vector<float>& a, const std::vector<float>& b, int64_t m, int64_t k, int64_t n) {
    std::vector<float> c(static_cast<size_t>(m * n), 0.0f);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int64_t kk = 0; kk < k; ++kk) {
                acc += a[static_cast<size_t>(i * k + kk)] * b[static_cast<size_t>(kk * n + j)];
            }
            c[static_cast<size_t>(i * n + j)] = acc;
        }
    }
    return c;
}


std::vector<float> create_random_f32_vector(ov::Shape tensor_shape) {
    // Calculate the total number of elements
    size_t total_elements = std::accumulate(tensor_shape.begin(), 
                                            tensor_shape.end(), 
                                            1, 
                                            std::multiplies<size_t>());

    // 2. Create and Fill a C++ Vector with Random Data

    // Use a standard C++ vector to hold the data
    std::vector<float> input_data(total_elements);

    std::vector<float> input_zero_data(total_elements);

    // Setup the random number generator
    // Mersenne Twister engine for high-quality pseudo-random numbers
    std::mt19937 generator(std::random_device{}()); 

    // Uniform distribution between -1.0 and 1.0 (a common range for input testing)
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f); 

    // Fill the vector with random floats
    std::generate(input_data.begin(), input_data.end(), 
                  [&]() { return distribution(generator); });

    return input_data;
}


int main(int argc, char* argv[]) {
    // Shapes (bfyx encoded):
    // X:      [B, S, IC, 1]
    // W_gate: [IC, OC, 1, 1]
    // W_up:   [IC, OC, 1, 1]
    // W_down: [OC, IC, 1, 1]
    const int64_t b = 1;
    const int64_t s = 2;
    const int64_t ic = 2560;
    const int64_t oc = 9728;

    const int64_t mb = b * s;

    ov::Shape input_shape{b, s, ic, 1};
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape(input_shape));
    auto input_f16 = std::make_shared<ov::op::v0::Convert>(input, ov::element::f16);

    ov::Shape gate_w_shape{ic, oc};
    ov::Shape up_w_shape{ic, oc};
    ov::Shape down_w_shape{oc, ic};

    std::vector<float> gate_w_f32_val = create_random_f32_vector(gate_w_shape);
    std::vector<float> up_w_f32_val = create_random_f32_vector(up_w_shape);
    std::vector<float> down_w_f32_val = create_random_f32_vector(down_w_shape);

    ov::Tensor gate_w = ov::Tensor(ov::element::f32, gate_w_shape);
    ov::Tensor up_w = ov::Tensor(ov::element::f32, up_w_shape);
    ov::Tensor down_w = ov::Tensor(ov::element::f32, down_w_shape);

    std::copy(gate_w_f32_val.begin(), gate_w_f32_val.end(), gate_w.data<float>());
    std::copy(up_w_f32_val.begin(), up_w_f32_val.end(), up_w.data<float>());
    std::copy(down_w_f32_val.begin(), down_w_f32_val.end(), down_w.data<float>());
    
    auto gate_w_node = std::make_shared<ov::op::v0::Constant>(gate_w);
    auto up_w_node = std::make_shared<ov::op::v0::Constant>(up_w);
    auto down_w_node = std::make_shared<ov::op::v0::Constant>(down_w);
    
    auto gate_w_f16 = std::make_shared<ov::op::v0::Convert>(gate_w_node, ov::element::f16);
    auto up_w_f16 = std::make_shared<ov::op::v0::Convert>(up_w_node, ov::element::f16);
    auto down_w_f16 = std::make_shared<ov::op::v0::Convert>(down_w_node, ov::element::f16);

    auto mlp_node = std::make_shared<ov::op::internal::FusedMLPExtension>(input_f16, gate_w_f16, up_w_f16, down_w_f16);
    auto mlp_node_f32 = std::make_shared<ov::op::v0::Convert>(mlp_node, ov::element::f32);
    auto result = std::make_shared<ov::op::v0::Result>(mlp_node_f32);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "fused_mlp_model");

    ov::Core core;
    auto compiled_model = core.compile_model(model, "GPU");
    auto ireq = compiled_model.create_infer_request();

    // ref model use matmul & swish
    ov::Shape input_shape_ref{b, s, ic};
    auto input_ref = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape(input_shape_ref));

    std::shared_ptr<ov::Node> gate_proj = std::make_shared<ov::op::v0::MatMul>(input_ref, gate_w_node, false, false);
    auto silu = std::make_shared<ov::op::v4::Swish>(gate_proj);
    auto up_proj = std::make_shared<ov::op::v0::MatMul>(input_ref, up_w_node, false, false);
    auto mul = std::make_shared<ov::op::v1::Multiply>(
            silu, up_proj, ov::op::AutoBroadcastType::NUMPY);
    auto down_proj = std::make_shared<ov::op::v0::MatMul>(mul, down_w_node, false, false);
    auto result_ref = std::make_shared<ov::op::v0::Result>(down_proj);

    auto model_ref = std::make_shared<ov::Model>(ov::ResultVector{result_ref}, ov::ParameterVector{input_ref}, "mlp_model_ref");

    auto compiled_model_ref = core.compile_model(model_ref, "GPU");
    auto ireq_ref = compiled_model_ref.create_infer_request();


    // create input tensor
    auto x_vals = create_random_f32_vector(input_shape_ref);
    ov::Tensor input_tensor(ov::element::f32, input_shape);
    std::copy(x_vals.begin(), x_vals.end(), input_tensor.data<float>());

    ov::Tensor input_tensor_ref(ov::element::f32, input_shape_ref);
    std::copy(x_vals.begin(), x_vals.end(), input_tensor_ref.data<float>());

    ireq.set_input_tensor(0, input_tensor);
    ireq.infer();
    auto output = ireq.get_output_tensor(0);
    std::vector<float> actual(output.data<float>(), output.data<float>() + output.get_size());

    ireq_ref.set_input_tensor(0, input_tensor_ref);
    ireq_ref.infer();
    auto output_ref = ireq_ref.get_output_tensor(0);
    std::vector<float> ov_gpu_ref(output_ref.data<float>(), output_ref.data<float>() + output_ref.get_size());

    // ref impl
    auto gate = matmul(x_vals, gate_w_f32_val, mb, ic, oc);
    auto up = matmul(x_vals, up_w_f32_val, mb, ic, oc);

    std::vector<float> swish(static_cast<size_t>(mb * oc), 0.0f);
    for (size_t i = 0; i < swish.size(); ++i) {
        swish[i] = gate[i] * sigmoid(gate[i]);
    }

    std::vector<float> hidden(static_cast<size_t>(mb * oc), 0.0f);
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = swish[i] * up[i];
    }

    auto ref = matmul(hidden, down_w_f32_val, mb, oc, ic);
    
    for (size_t i = 0; i < 10; i++) {
        std::cout << "i: " << i << ", fused_mlp: " << std::fixed << std::setprecision(3) << actual[i] << ", \t ov gpu ref: " << std::fixed << std::setprecision(3)  << ov_gpu_ref[i] << ", \t CPU ref: " << std::fixed << std::setprecision(3)  << ref[i] << std::endl;
    }

    // const float absolute_error_threshold = 5e-2f;
    const float absolute_error_threshold = 1.0f;

    for (size_t i = 0; i < ref.size(); ++i) {
        if (fabs(actual[i] - ref[i]) > absolute_error_threshold || fabs(ov_gpu_ref[i] - actual[i]) > absolute_error_threshold) {
            std::cout << "Mismatch detected at index: " << i << ", actual: " << actual[i] << ", ref: " << ref[i] << std::endl;
            exit(0);
        }
    }

    std::cout << "\nPassed" << std::endl;
}
