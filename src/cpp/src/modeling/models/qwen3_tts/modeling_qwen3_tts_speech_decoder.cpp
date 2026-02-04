// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_tts/modeling_qwen3_tts_speech_decoder.hpp"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
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
// Speech Decoder Model Factory
//===----------------------------------------------------------------------===//

std::shared_ptr<ov::Model> create_qwen3_tts_speech_decoder_model(
    const SpeechDecoderConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    // TODO: Implement in Patch 5 (Speech Decoder)
    OPENVINO_THROW("create_qwen3_tts_speech_decoder_model not yet implemented");
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
