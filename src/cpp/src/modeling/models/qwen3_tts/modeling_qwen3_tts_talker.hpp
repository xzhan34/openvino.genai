// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Qwen3-TTS Talker Module
//
// The Talker is the main autoregressive model for generating Layer-0 codec
// tokens. It uses mRoPE (multi-dimensional rotary position embedding) for
// temporal position encoding.
//
// Architecture: 28-layer transformer with GQA (16 heads, 8 KV heads)
// Input: text embeddings + codec embeddings -> combined embedding
// Output: Layer-0 codec token logits
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
// Forward Declarations - Talker Modules
//===----------------------------------------------------------------------===//

class Qwen3TTSTalkerAttention;
class Qwen3TTSTalkerMLP;
class Qwen3TTSTalkerDecoderLayer;
class Qwen3TTSTalkerModel;
class Qwen3TTSTalkerForConditionalGeneration;
class Qwen3TTSTextProjection;

//===----------------------------------------------------------------------===//
// Factory Functions - Embedding Models
//===----------------------------------------------------------------------===//

// Create Embedding model (for converting text/codec tokens to embeddings)
// Input:
//   - text_input_ids: [batch, seq] text token IDs
//   - codec_input_ids: [batch, seq] codec token IDs
// Output:
//   - inputs_embeds: [batch, seq, hidden_size] combined embeddings
//                    = text_projection(text_embedding(text_ids)) + codec_embedding(codec_ids)
std::shared_ptr<ov::Model> create_qwen3_tts_embedding_model(
    const Qwen3TTSTalkerConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

// Create codec-only embedding model (for decode phase - no text projection!)
// Input:
//   - codec_input_ids: [batch, 1] codec token ID
// Output:
//   - codec_embeds: [batch, 1, hidden_size] codec embedding
std::shared_ptr<ov::Model> create_qwen3_tts_codec_embedding_model(
    const Qwen3TTSTalkerConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

//===----------------------------------------------------------------------===//
// Factory Functions - Talker Models
//===----------------------------------------------------------------------===//

// Create Talker model (for Layer-0 codec generation)
std::shared_ptr<ov::Model> create_qwen3_tts_talker_model(
    const Qwen3TTSTalkerConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

// Create Talker Prefill model - processes initial sequence, outputs KV caches
// Inputs: inputs_embeds [B, T, hidden], position_ids [3, B, T], attention_mask [B, 1, T, T]
// Outputs: logits [B, T, vocab], hidden_states [B, T, hidden],
//          key_cache_0..N-1 [B, kv_heads, T, head_dim],
//          value_cache_0..N-1 [B, kv_heads, T, head_dim]
std::shared_ptr<ov::Model> create_qwen3_tts_talker_prefill_model(
    const Qwen3TTSTalkerConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

// Create Talker Decode model - generates one token at a time using KV caches
// Inputs: inputs_embeds [B, 1, hidden], position_ids [3, B, 1], attention_mask [B, 1, 1, total_len],
//         past_key_0..N-1 [B, kv_heads, past_len, head_dim],
//         past_value_0..N-1 [B, kv_heads, past_len, head_dim]
// Outputs: logits [B, 1, vocab], hidden_states [B, 1, hidden],
//          key_cache_0..N-1 [B, kv_heads, total_len, head_dim],
//          value_cache_0..N-1 [B, kv_heads, total_len, head_dim]
std::shared_ptr<ov::Model> create_qwen3_tts_talker_decode_model(
    const Qwen3TTSTalkerConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
