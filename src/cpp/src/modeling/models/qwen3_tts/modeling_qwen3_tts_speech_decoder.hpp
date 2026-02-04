// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Qwen3-TTS Speech Decoder Module
//
// The Speech Decoder converts the 16-layer codec tokens into audio waveform.
// It consists of:
//   1. RVQ Dequantizer: Converts codec tokens to latent vectors
//   2. Pre-Transformer: 8-layer transformer with sliding window attention
//   3. Upsampler: Interpolation layers to increase temporal resolution
//   4. ConvNeXt Decoder: Multi-stage convolutional decoder with SnakeBeta
//
// Architecture: RVQ (16 quantizers) -> PreTransformer (8 layers) -> ConvNeXt
// Input: [batch, 16, seq_len] codec tokens
// Output: [batch, 1, num_samples] audio waveform at 24kHz
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
// Forward Declarations - Speech Decoder Modules
//===----------------------------------------------------------------------===//

class RVQDequantizer;
class PreTransformerAttention;
class PreTransformerMLP;
class PreTransformerDecoderLayer;
class PreTransformer;
class SnakeBetaActivation;
class ConvNeXtBlock;
class ResidualUnit;
class DecoderBlock;
class SpeechDecoderModel;

//===----------------------------------------------------------------------===//
// Factory Functions - Speech Decoder Models
//===----------------------------------------------------------------------===//

// Create Speech Decoder model (codes -> audio waveform)
// Input:
//   - codes: [batch, 16, seq_len] codec tokens from all 16 layers
// Output:
//   - audio: [batch, 1, num_samples] audio waveform
//
// Processing pipeline:
//   1. RVQ dequantize: codes -> latent [batch, seq_len, 512]
//   2. Pre-transformer: latent -> features [batch, seq_len, 512]
//   3. Pre-upsample: features -> upsampled [batch, seq_len*4, 512]
//   4. ConvNeXt decode: upsampled -> audio [batch, 1, num_samples]
std::shared_ptr<ov::Model> create_qwen3_tts_speech_decoder_model(
    const SpeechDecoderConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
