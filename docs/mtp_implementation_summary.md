# Qwen3.5 MTP (Multi-Token Prediction) Implementation Summary

## Overview

This document summarizes the initial MTP speculative decoding implementation for Qwen3.5 in the OpenVINO GenAI explicit modeling framework. This serves as the baseline for future performance optimization work.

**Status**: Functionally complete. Speculative decoding works end-to-end on both text and VL modes. Performance is not yet positive—throughput is currently lower than baseline due to the cost of 2-token inference and low hit rates on the 0.8B model.

## Architecture

### MTP Model Pipeline

```
Input: input_ids [B, S], hidden_states [B, S, H]
  ↓
1. mtp.embed_tokens(input_ids)           → [B, S, H]       (shared with main model)
2. mtp.pre_fc_norm_embedding(embed)      → [B, S, H]
   mtp.pre_fc_norm_hidden(hidden)        → [B, S, H]
3. Concat([norm_embed, norm_hidden], -1) → [B, S, 2*H]
4. mtp.fc(concat)                        → [B, S, H]       (linear: 2*H → H)
5. MTP decoder layers (full_attention only, dense MLP, typically 1 layer)
6. mtp.norm(out)                         → [B, S, H]
7. lm_head(norm)                         → [B, S, V]       (shared with main model)
```

### Speculative Decode Loop

```
Given: next_id (confirmed token), mtp_draft_id (MTP's prediction)

1. Feed [next_id, draft_id] as 2-token sequence to main model
2. Main model outputs logits at both positions:
   - logits[0] → verified_id  (what comes after next_id)
   - logits[1] → bonus_id     (what comes after draft_id)
3. Compare verified_id == draft_id:

   ACCEPT:
   - Emit both verified_id and bonus_id (2 tokens per infer)
   - past_len += 2
   - Run MTP on bonus_id's hidden_states → new draft

   REJECT:
   - Emit only verified_id (1 token per infer)
   - Trim main model KV cache by 1 (remove draft's position)
   - Reset MTP state (KV cache invalid after rejection)
   - Run MTP on verified_id's hidden_states → new draft
```

## Changed Files

### Model Definition (6 files)

| File | Changes |
|------|---------|
| `processing_qwen3_5.hpp` | Added `mtp_num_hidden_layers` to `Qwen3_5TextConfig`; added `Qwen3_5MtpIO` struct with I/O names; added `kHiddenStates` to `Qwen3_5TextIO` |
| `processing_qwen3_5.cpp` | Parse `mtp_num_hidden_layers` from JSON config |
| `modeling_qwen3_5_text.hpp` | Added `forward_with_hidden()`, `forward_embeds_with_hidden()` methods; added `output_hidden_states` param to `create_qwen3_5_text_model()`; declared `create_qwen3_5_mtp_model()` |
| `modeling_qwen3_5_text.cpp` | Implemented MTP model graph construction (`create_qwen3_5_mtp_model()`); implemented `forward_with_hidden()` to capture hidden states before LM head; MTP config: full_attention only, dense MLP, no MoE |
| `qwen3_5_weight_specs.hpp` | Declared `build_qwen3_5_mtp_weight_specs()` |
| `qwen3_5_weight_specs.cpp` | Implemented MTP weight specs: `mtp.fc`, `mtp.pre_fc_norm_*`, `mtp.layers[i].*`, shared `embed_tokens` and `lm_head` via PackedMapping rules |

### Sample Application (1 file)

| File | Changes |
|------|---------|
| `modeling_qwen3_5.cpp` | Added `--mtp` / `--mtp-layers` CLI args; `extract_logits_at_pos_f32()` for multi-position logit extraction; `trim_kv_cache_states()` for KV cache trimming on rejection; `run_mtp_draft` lambda for MTP inference; full speculative decode loop with accept/reject paths; MTP metrics output |

### Benchmark Script (1 file, new)

| File | Description |
|------|-------------|
| `bench_mtp.ps1` | PowerShell benchmark script: runs baseline vs MTP across text/VL modes, parses all metrics (TTFT, TPOT, throughput, MTP hit rate, tokens/infer), outputs summary table with speedup calculation |

## Key Implementation Decisions

1. **Fresh tensors per speculative step**: 2-token input tensors are created fresh each iteration rather than pre-allocated, to avoid GPU plugin output reallocation issues when switching between seq_len=1 and seq_len=2.

2. **MTP state reset on rejection**: On draft rejection, MTP KV cache is fully reset (`mtp_request->reset_state()`) rather than trimmed. GPU plugin crashes when `set_state()` is called with CPU tensors on the MTP model after trim. Main model KV trim works correctly.

3. **Output copy before input reset**: After 2-token inference, logits and hidden states are copied to local tensors before resetting inputs back to 1-token shapes, because GPU plugin may invalidate tensor references on input change.

4. **Shared weights**: `embed_tokens` and `lm_head` weights are shared between main model and MTP model via PackedMapping weight loading rules.

## Benchmark Results (Baseline)

**Setup**: Qwen3.5-0.8B | INT4_ASYM group_size=128 | GPU | 64 output tokens | 3 runs averaged

### Text Mode

| Config | TPOT (ms/tok) | Throughput (tok/s) | MTP Hit Rate | Tok/Infer | Delta |
|--------|---------------|--------------------|--------------|-----------|-------|
| Baseline (no MTP) | 15.09 | 66.28 | — | 1.00 | — |
| MTP Speculative | 22.90 | 43.91 | 27.6% | 1.27 | -33.8% |

### VL Mode

| Config | TPOT (ms/tok) | Throughput (tok/s) | MTP Hit Rate | Tok/Infer | Delta |
|--------|---------------|--------------------|--------------|-----------|-------|
| Baseline (no MTP) | 19.90 | 53.53 | — | 1.00 | — |
| MTP Speculative | 22.74 | 44.35 | 38.7% | 1.38 | -17.1% |

## Performance Analysis

MTP speculative decoding currently shows **negative throughput impact** (-17% ~ -34%). Root causes:

| Factor | Impact | Notes |
|--------|--------|-------|
| 2-token infer cost | ~2x single-token | GPU memory-bound; 2 tokens ≈ 2x latency |
| Low hit rate | 28-39% | 0.8B model's single MTP layer is too weak; need >50% to break even |
| KV cache trim overhead | High on rejection | GPU→CPU→GPU roundtrip for state slicing |
| MTP state reset on rejection | Reduces accuracy | Discards speculative context, hurts subsequent predictions |

**Break-even analysis**: With 2x infer cost, the minimum hit rate for net positive throughput is ~50%. At 39% (VL mode best case), we generate 1.38 tokens/infer vs the required ~1.5 tokens/infer.

## Next Steps for Performance Optimization

1. **Reduce 2-token infer cost**: Investigate if GPU plugin can batch 2-token inference more efficiently; explore async MTP execution
2. **Avoid MTP state reset**: Implement proper MTP KV cache trim on GPU to preserve speculative context across rejections
3. **Test with larger models**: 7B+ models typically achieve 60-80% MTP hit rates, making speculative decode profitable
4. **Multi-token speculation (k>1)**: Extend to draft k tokens ahead and verify all at once
5. **Benchmark with longer sequences**: Hit rate may improve with more context
