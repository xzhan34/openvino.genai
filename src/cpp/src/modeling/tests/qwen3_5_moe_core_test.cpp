// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/core/except.hpp>
#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_moe.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/quantization_config.hpp"
#include "modeling/weights/weight_loader.hpp"
#include "modeling/weights/weight_source.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"

namespace {

using namespace ov::genai::modeling;
using namespace ov::genai::modeling::models;
using namespace ov::genai::modeling::tests;
using namespace ov::genai::modeling::weights;

struct Qwen35MoeRuntimeShape {
    // Core routed-MoE dimensions aligned with Qwen3 30B-A3B config.
    int32_t hidden_size = 2048;
    int32_t moe_intermediate_size = 768;
    int32_t num_experts = 128;
    int32_t num_experts_per_tok = 8;

    // Qwen3.5 MoE-specific shared expert branch.
    int32_t shared_expert_intermediate_size = 512;

    int64_t batch = 1;
    int64_t seq_len = 16;
};

int32_t parse_env_int(const char* name, int32_t default_value, int32_t min_value) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return default_value;
    }
    try {
        const long long parsed = std::stoll(raw);
        if (parsed < static_cast<long long>(min_value) || parsed > std::numeric_limits<int32_t>::max()) {
            return default_value;
        }
        return static_cast<int32_t>(parsed);
    } catch (...) {
        return default_value;
    }
}

Qwen35MoeRuntimeShape resolve_runtime_shape() {
    Qwen35MoeRuntimeShape s;

    // ULT default uses reduced experts only on CI to keep memory bounded.
    const bool is_ci = std::getenv("CI") != nullptr;
    const int32_t default_experts = is_ci ? 32 : 128;
    s.num_experts = parse_env_int("OV_GENAI_QWEN35_MOE_TEST_NUM_EXPERTS", default_experts, 1);
    s.num_experts_per_tok = std::min<int32_t>(s.num_experts_per_tok, s.num_experts);

    const int32_t seq = parse_env_int("OV_GENAI_QWEN35_MOE_TEST_SEQ_LEN", static_cast<int32_t>(s.seq_len), 1);
    s.seq_len = seq;
    return s;
}

ov::Tensor make_random_f16_tensor(const ov::Shape& shape, uint32_t seed, float scale = 0.02f) {
    ov::Tensor t(ov::element::f16, shape);
    auto* dst = t.data<ov::float16>();
    uint32_t state = seed;
    const size_t n = t.get_size();
    for (size_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        const float unit = static_cast<float>((state >> 8) & 0x00FFFFFFu) / static_cast<float>(0x00FFFFFFu);
        const float value = (unit * 2.0f - 1.0f) * scale;
        dst[i] = ov::float16(value);
    }
    return t;
}

ov::Tensor make_random_f32_input(const Qwen35MoeRuntimeShape& s, uint32_t seed = 0x13579BDFu) {
    ov::Tensor t(ov::element::f32,
                 ov::Shape{static_cast<size_t>(s.batch), static_cast<size_t>(s.seq_len), static_cast<size_t>(s.hidden_size)});
    auto* dst = t.data<float>();
    uint32_t state = seed;
    const size_t n = t.get_size();
    for (size_t i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        const float unit = static_cast<float>((state >> 9) & 0x007FFFFFu) / static_cast<float>(0x007FFFFFu);
        dst[i] = (unit * 2.0f - 1.0f) * 0.5f;
    }
    return t;
}

class ProceduralMoeWeightSource : public WeightSource {
public:
    explicit ProceduralMoeWeightSource(const Qwen35MoeRuntimeShape& s) {
        const size_t e = static_cast<size_t>(s.num_experts);
        const size_t h = static_cast<size_t>(s.hidden_size);
        const size_t i = static_cast<size_t>(s.moe_intermediate_size);
        const size_t si = static_cast<size_t>(s.shared_expert_intermediate_size);

        add_spec("mlp.gate.weight", {e, h}, 101u);
        add_spec("mlp.experts.gate_up_proj", {e, 2 * i, h}, 103u);
        add_spec("mlp.experts.down_proj", {e, h, i}, 107u);
        add_spec("mlp.shared_expert_gate.weight", {1, h}, 109u);
        add_spec("mlp.shared_expert.gate_proj.weight", {si, h}, 113u);
        add_spec("mlp.shared_expert.up_proj.weight", {si, h}, 127u);
        add_spec("mlp.shared_expert.down_proj.weight", {h, si}, 131u);
    }

    std::vector<std::string> keys() const override {
        return keys_;
    }

    bool has(const std::string& name) const override {
        return specs_.find(name) != specs_.end();
    }

    const ov::Tensor& get_tensor(const std::string& name) const override {
        auto it = specs_.find(name);
        OPENVINO_ASSERT(it != specs_.end(), "Unknown procedural MoE weight: ", name);
        if (cached_name_ != name) {
            cached_name_ = name;
            cached_tensor_ = make_random_f16_tensor(it->second.shape, it->second.seed);
        }
        return cached_tensor_;
    }

private:
    struct WeightSpec {
        ov::Shape shape;
        uint32_t seed = 0;
    };

    void add_spec(const std::string& name, ov::Shape shape, uint32_t seed) {
        keys_.push_back(name);
        specs_.emplace(name, WeightSpec{std::move(shape), seed});
    }

    std::vector<std::string> keys_;
    std::unordered_map<std::string, WeightSpec> specs_;
    mutable std::string cached_name_;
    mutable ov::Tensor cached_tensor_;
};

Qwen3_5TextModelConfig make_moe_text_cfg(const Qwen35MoeRuntimeShape& s) {
    Qwen3_5TextModelConfig cfg;
    cfg.hidden_size = s.hidden_size;
    cfg.moe_intermediate_size = s.moe_intermediate_size;
    cfg.shared_expert_intermediate_size = s.shared_expert_intermediate_size;
    cfg.num_experts = s.num_experts;
    cfg.num_experts_per_tok = s.num_experts_per_tok;
    cfg.norm_topk_prob = true;
    cfg.hidden_act = "silu";
    return cfg;
}

bool model_contains_internal_fused_moe(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        const std::string type_name = node->get_type_name();
        if (type_name.find("MOE3GemmFusedCompressed") != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::shared_ptr<ov::Model> build_qwen3_5_moe_only_model(const Qwen35MoeRuntimeShape& s,
                                                         WeightSource& source,
                                                         WeightFinalizer& finalizer) {
    BuilderContext ctx;
    Qwen3_5SparseMoeBlock moe(ctx, "mlp", make_moe_text_cfg(s));
    (void)load_model(moe, source, finalizer);

    auto hidden_states = ctx.parameter("hidden_states",
                                       ov::element::f32,
                                       ov::Shape{static_cast<size_t>(s.batch),
                                                 static_cast<size_t>(s.seq_len),
                                                 static_cast<size_t>(s.hidden_size)});
    auto out = moe.forward(hidden_states);
    return ctx.build_model({out.output()});
}

bool has_gpu_device() {
    ov::Core core;
    for (const auto& device : core.get_available_devices()) {
        if (device.rfind("GPU", 0) == 0) {
            return true;
        }
    }
    return false;
}

ov::Tensor run_model_on_gpu(const std::shared_ptr<ov::Model>& model, const ov::Tensor& input) {
    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, input);
    request.infer();
    return request.get_output_tensor(0);
}

struct TensorDiffStats {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
};

TensorDiffStats compare_tensors(const ov::Tensor& a, const ov::Tensor& b) {
    OPENVINO_ASSERT(a.get_shape() == b.get_shape(), "Tensor shape mismatch in compare_tensors");
    OPENVINO_ASSERT(a.get_element_type() == ov::element::f32 && b.get_element_type() == ov::element::f32,
                    "compare_tensors expects f32 tensors");

    const float* pa = a.data<const float>();
    const float* pb = b.data<const float>();
    const size_t n = a.get_size();

    TensorDiffStats stats;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float d = std::abs(pa[i] - pb[i]);
        stats.max_abs = std::max(stats.max_abs, d);
        sum += static_cast<double>(d);
    }
    stats.mean_abs = static_cast<float>(sum / static_cast<double>(n));
    return stats;
}

class Qwen3_5MoeCoreULT : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        try {
            shape_ = resolve_runtime_shape();
            input_ = make_random_f32_input(shape_);

            {
                ProceduralMoeWeightSource source(shape_);
                DummyWeightFinalizer finalizer;
                fallback_model_ = build_qwen3_5_moe_only_model(shape_, source, finalizer);
                fallback_has_fused_moe_op_ = model_contains_internal_fused_moe(fallback_model_);
            }

            {
                ProceduralMoeWeightSource source(shape_);
                QuantizationConfig qcfg;
                qcfg.mode = QuantizationConfig::Mode::INT4_ASYM;
                qcfg.backup_mode = QuantizationConfig::Mode::INT4_ASYM;
                qcfg.group_size = 128;
                qcfg.selection.exclude_patterns.clear();
                qcfg.selection.quantize_attention = false;
                qcfg.selection.quantize_mlp = true;
                qcfg.selection.quantize_moe = true;
                qcfg.selection.quantize_embeddings = false;
                qcfg.selection.quantize_lm_head = false;
                ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(qcfg);
                fused_model_ = build_qwen3_5_moe_only_model(shape_, source, finalizer);
                fused_has_fused_moe_op_ = model_contains_internal_fused_moe(fused_model_);
            }
        } catch (const std::exception& e) {
            setup_error_ = e.what();
        }
    }

    static void TearDownTestSuite() {
        fallback_model_.reset();
        fused_model_.reset();
    }

    static void assert_setup_ok() {
        ASSERT_TRUE(setup_error_.empty()) << "Suite setup failed: " << setup_error_;
    }

    static void skip_if_no_gpu() {
        if (!has_gpu_device()) {
            GTEST_SKIP() << "GPU device is not available for Qwen3.5 MoE ULT.";
        }
    }

    static Qwen35MoeRuntimeShape shape_;
    static ov::Tensor input_;
    static std::shared_ptr<ov::Model> fallback_model_;
    static std::shared_ptr<ov::Model> fused_model_;
    static bool fallback_has_fused_moe_op_;
    static bool fused_has_fused_moe_op_;
    static std::string setup_error_;
};

Qwen35MoeRuntimeShape Qwen3_5MoeCoreULT::shape_{};
ov::Tensor Qwen3_5MoeCoreULT::input_{};
std::shared_ptr<ov::Model> Qwen3_5MoeCoreULT::fallback_model_{};
std::shared_ptr<ov::Model> Qwen3_5MoeCoreULT::fused_model_{};
bool Qwen3_5MoeCoreULT::fallback_has_fused_moe_op_ = false;
bool Qwen3_5MoeCoreULT::fused_has_fused_moe_op_ = false;
std::string Qwen3_5MoeCoreULT::setup_error_{};

TEST_F(Qwen3_5MoeCoreULT, OpsetMoePathBuildsCompilesAndInfersOnGpu) {
    assert_setup_ok();
    skip_if_no_gpu();

    ASSERT_NE(fallback_model_, nullptr);
    EXPECT_FALSE(fallback_has_fused_moe_op_)
        << "Fallback model should not contain internal fused MoE op.";

    auto out = run_model_on_gpu(fallback_model_, input_);
    EXPECT_EQ(out.get_shape(),
              (ov::Shape{static_cast<size_t>(shape_.batch),
                         static_cast<size_t>(shape_.seq_len),
                         static_cast<size_t>(shape_.hidden_size)}));
}

TEST_F(Qwen3_5MoeCoreULT, FusedMoePathBuildsCompilesAndInfersOnGpu) {
    assert_setup_ok();
    skip_if_no_gpu();

    ASSERT_NE(fused_model_, nullptr);
    EXPECT_TRUE(fused_has_fused_moe_op_)
        << "Fused model should contain internal MOE3GemmFusedCompressed op.";

    auto out = run_model_on_gpu(fused_model_, input_);
    EXPECT_EQ(out.get_shape(),
              (ov::Shape{static_cast<size_t>(shape_.batch),
                         static_cast<size_t>(shape_.seq_len),
                         static_cast<size_t>(shape_.hidden_size)}));
}

TEST_F(Qwen3_5MoeCoreULT, FusedAndOpsetOutputsStayConsistentOnGpu) {
    assert_setup_ok();
    skip_if_no_gpu();

    auto out_fallback = run_model_on_gpu(fallback_model_, input_);
    auto out_fused = run_model_on_gpu(fused_model_, input_);

    ASSERT_EQ(out_fallback.get_shape(), out_fused.get_shape());
    ASSERT_EQ(out_fallback.get_element_type(), ov::element::f32);
    ASSERT_EQ(out_fused.get_element_type(), ov::element::f32);

    const auto stats = compare_tensors(out_fallback, out_fused);
    EXPECT_LE(stats.max_abs, 1.0f);
    EXPECT_LE(stats.mean_abs, 0.15f);
}

}  // namespace
