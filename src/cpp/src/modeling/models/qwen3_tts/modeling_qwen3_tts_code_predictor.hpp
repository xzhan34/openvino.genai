// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Qwen3-TTS Code Predictor Module
//
// The Code Predictor generates Layer 1-15 codec tokens in parallel using
// a smaller 5-layer transformer. It takes the hidden states from the Talker
// and autoregressively predicts each of the 15 remaining codec layers.
//
// Architecture: 5-layer transformer with GQA (16 heads, 8 KV heads)
// Input: Talker hidden states projected to 1024 dim + previous codec embeddings
// Output: Codec token logits for layers 1-15
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

#include "modeling/models/qwen3_tts/modeling_qwen3_tts.hpp"

namespace ov {
class Model;
}  // namespace ov

namespace ov {
namespace genai {
namespace modeling {
namespace weights {
class WeightFinalizer;
class WeightSource;
}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov

namespace ov {
namespace genai {
namespace modeling {
namespace models {

//===----------------------------------------------------------------------===//
// Forward Declarations - Code Predictor Modules
//===----------------------------------------------------------------------===//

class Qwen3TTSCodePredictorAttention;
class Qwen3TTSCodePredictorMLP;
class Qwen3TTSCodePredictorDecoderLayer;
class Qwen3TTSCodePredictorModel;
class Qwen3TTSCodePredictorForConditionalGeneration;

//===----------------------------------------------------------------------===//
// Factory Functions - Code Predictor Models
//===----------------------------------------------------------------------===//

// Create Code Predictor model (for Layer 1-15 codec generation)
std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

// Create Code Predictor AR model for specific generation step
// Input:
//   - inputs_embeds: [batch, seq_len, hidden_size] from codec embeddings
//   - position_ids: [batch, seq_len] position indices
// Output:
//   - logits: [batch, seq_len, vocab_size] next token logits
std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_ar_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    int generation_step,  // 0..14 for groups 1..15
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

// Create Code Predictor codec embedding model (for decode phase)
// Gets the sum of all 15 codec embeddings from CodePredictor (layers 1-15)
// Input:
//   - codec_input_0 to codec_input_14: [batch, 1] tokens for each layer
// Output:
//   - codec_embeds_sum: [batch, 1, hidden_size] sum of all embeddings
std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_codec_embed_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

// Create Code Predictor single codec embedding model
// Input: codec_input [batch, 1] token for specific layer
// Output: codec_embed [batch, 1, hidden_size]
std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_single_codec_embed_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    int codec_layer,  // 0..14 for layers 1..15
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
