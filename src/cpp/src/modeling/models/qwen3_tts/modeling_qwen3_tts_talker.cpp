// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_tts/modeling_qwen3_tts_talker.hpp"

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
// Embedding Model Factory
//===----------------------------------------------------------------------===//

std::shared_ptr<ov::Model> create_qwen3_tts_embedding_model(
    const Qwen3TTSTalkerConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    // TODO: Implement in Patch 3 (Talker)
    OPENVINO_THROW("create_qwen3_tts_embedding_model not yet implemented");
}

std::shared_ptr<ov::Model> create_qwen3_tts_codec_embedding_model(
    const Qwen3TTSTalkerConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    // TODO: Implement in Patch 3 (Talker)
    OPENVINO_THROW("create_qwen3_tts_codec_embedding_model not yet implemented");
}

//===----------------------------------------------------------------------===//
// Talker Model Factories
//===----------------------------------------------------------------------===//

std::shared_ptr<ov::Model> create_qwen3_tts_talker_model(
    const Qwen3TTSTalkerConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    // TODO: Implement in Patch 3 (Talker)
    OPENVINO_THROW("create_qwen3_tts_talker_model not yet implemented");
}

std::shared_ptr<ov::Model> create_qwen3_tts_talker_prefill_model(
    const Qwen3TTSTalkerConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    // TODO: Implement in Patch 3 (Talker)
    OPENVINO_THROW("create_qwen3_tts_talker_prefill_model not yet implemented");
}

std::shared_ptr<ov::Model> create_qwen3_tts_talker_decode_model(
    const Qwen3TTSTalkerConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    // TODO: Implement in Patch 3 (Talker)
    OPENVINO_THROW("create_qwen3_tts_talker_decode_model not yet implemented");
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
