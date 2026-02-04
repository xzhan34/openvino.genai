// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_tts/modeling_qwen3_tts_code_predictor.hpp"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/layers/lm_head.hpp"
#include "modeling/layers/rms_norm.hpp"
#include "modeling/layers/vocab_embedding.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/llm.hpp"
#include "modeling/ops/nn.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/weights/weight_finalizer.hpp"
#include "modeling/weights/weight_source.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

//===----------------------------------------------------------------------===//
// Code Predictor Model Factories
//===----------------------------------------------------------------------===//

std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    // TODO: Implement in Patch 4 (Code Predictor)
    OPENVINO_THROW("create_qwen3_tts_code_predictor_model not yet implemented");
}

std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_ar_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    int generation_step,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    // TODO: Implement in Patch 4 (Code Predictor)
    OPENVINO_THROW("create_qwen3_tts_code_predictor_ar_model not yet implemented");
}

std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_codec_embed_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    // TODO: Implement in Patch 4 (Code Predictor)
    OPENVINO_THROW("create_qwen3_tts_code_predictor_codec_embed_model not yet implemented");
}

std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_single_codec_embed_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    int codec_layer,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    // TODO: Implement in Patch 4 (Code Predictor)
    OPENVINO_THROW("create_qwen3_tts_code_predictor_single_codec_embed_model not yet implemented");
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
