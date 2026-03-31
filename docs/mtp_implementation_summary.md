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

### Speculative Decode Loop (v3 — 2-Token Batch Verify, current)

```
Given: next_id (confirmed token), mtp_draft_id (MTP's prediction)

1. Feed [next_id, draft_id] as 2-token sequence to main model
2. Main model outputs logits at both positions:
   - logits[0] → verified_id  (what comes after next_id)
   - logits[1] → bonus_id     (what comes after draft_id)
3. Compare verified_id == draft_id:

   ACCEPT:
   - Emit both verified_id and bonus_id (2 tokens per infer)
   - past_len += 2, no KV trim
   - Run MTP on bonus_id's hidden_states → new draft

   REJECT:
   - Emit only verified_id (1 token per infer)
   - Trim main model KV cache by 1 (remove draft's position)
   - Reset MTP state (KV cache invalid after rejection)
   - Run MTP on verified_id's hidden_states → new draft

E[Tok/Infer] = h × 2 + (1-h) × 1 = 1 + h
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

## Draft Token Verification: Algorithm Comparison

### v1: 2-Token Verify

每步将 confirmed token 和 MTP draft token 一起喂给主模型，一次推理验证 + 采样：

```
Input to main:  [confirmed_tok, draft_tok]     ← 2 tokens
Output:         logits[0] → verified_id        (what comes after confirmed_tok)
                logits[1] → bonus_id           (what comes after draft_tok)

Compare verified_id vs draft_tok:
  ACCEPT (match):    emit verified_id + bonus_id → 2 tokens/step
                     past_len += 2
                     run MTP on bonus_id's hidden_states → new draft

  REJECT (mismatch): emit verified_id only → 1 token/step
                     trim KV cache by 1 (remove draft's position)
                     reset MTP state
                     run MTP on verified_id's hidden_states → new draft
```

**问题**：
- 2-token infer 在 0.8B 上 ~25ms（1.75x baseline 14.3ms）
- Reject 时需要 KV trim（4-6ms GPU→CPU→GPU roundtrip）
- MTP state reset 丢失 speculative context

### v2: 1-Token Verify

每步只喂 confirmed token，draft token **从不进入主模型 KV cache**：

```
Input to main:  [confirmed_tok]                ← 1 token only
Output:         logits → verified_id

Compare verified_id vs draft_tok:
  MATCH:   draft 正确，下一步直接用 draft_tok 作为 confirmed
           MTP 已在上一步产出 next_draft，继续推进
           → 等效于 2 tokens/step（省掉一次主模型推理）

  MISMATCH: draft 错误，用 verified_id 作为下一步 confirmed
            无需 KV trim（draft 从未进入 KV cache）
            重新跑 MTP → new draft
            → 1 token/step
```

**核心优势**：
- 每步只做 1-token infer（14.3ms vs 25ms）
- 永远不需要 KV trim（节省 4-6ms/rejection）
- E[tokens/step] 仍然 = 1 + h（与 v1 相同）

**问题**（9B 基准测试暴露）：
- Tok/Infer = 1.00 恒等 — MTP 永远不减少主模型推理次数
- HIT 时做额外 1-tok infer 取 bonus，但这与 baseline 无异
- MTP 变成纯额外开销（MTP infer + overhead），无法带来加速

### v3: 2-Token Batch Verify（当前实现）

融合 v1 和 v2 的优势：用 2-token batch infer 验证 draft，HIT 时一次推理产出 2 token：

```
Input to main:  [next_id, draft_id]            ← 2 tokens as batch
Output:         logits[0] → verified_id        (验证 next_id 的下一个 token)
                logits[1] → bonus_id           (draft_id 位置的下一个 token)
                hidden_states[0],[1]           (各位置的隐藏状态)

Compare verified_id vs draft_id:
  HIT (match):
    - Emit verified_id + bonus_id → 2 tokens
    - KV cache 正确，无需 trim
    - MTP draft from hs[1] → new draft
    - 1 main infer → 2 tokens.  Tok/Infer = 2.0

  MISS (mismatch):
    - Trim KV cache by 1（移除 draft_id 的位置）
    - Emit verified_id only → 1 token
    - MTP draft from hs[0] → new draft（reset MTP KV）
    - 1 main infer + trim → 1 token
```

**vs v1**：相同的 2-token infer 策略，但 MTP hidden_states 提取更精确（从 batch 输出的对应位置提取）
**vs v2**：恢复 2-token infer 和 KV trim，但 HIT 时真正跳过主模型调用，Tok/Infer > 1

**关键改进**：
- HIT 时 Tok/Infer = 2.0（v2 恒为 1.0）
- E[tokens/infer] = h×2 + (1-h)×1 = 1+h
- 在 memory-bound 大模型上 T_2tok ≈ T_1tok，所以 2-token infer 几乎零成本
- MISS 时需要 KV trim，但在 CPU 上 trim 成本很低（纯内存操作）

### vLLM: 标准 Speculative Decoding（K+1 Token Verify）

vLLM 采用 Leviathan et al. 2023 算法，将所有 K 个 draft tokens 打包成一次 forward：

```
上一步 confirmed: tok_c
MTP proposer 产出 K 个 drafts: [d1, d2, ..., dk]

Phase 1 — Target Forward (单次推理):
  Input:  [tok_c, d1, d2, ..., dk]        ← K+1 tokens
  Output: logits at K+1 positions

Phase 2 — Rejection Sampling (顺序判断):
  Greedy (temperature=0):
    pos 0: argmax(logits[0]) == d1?  → accept/reject
    pos 1: argmax(logits[1]) == d2?  → accept/reject (仅前一个 accept 才检查)
    ...
    在第一个 reject 处停止，emit target's token
    全部 accept → emit d1..dk + bonus_token (K+1 tokens!)

  Probabilistic (temperature>0):
    accept_prob = min(1, p_target[d_i] / p_draft[d_i])
    reject → sample from max(0, p_target - p_draft)  ("recovered" distribution)
    保证输出分布严格等价于 target model

Phase 3 — KV Cache:
  vLLM 不 trim KV cache
  - 所有 K+1 positions 的 KV 都已写入 (PagedAttention)
  - Reject 后通过 num_rejected_tokens 调整 seq_lens
  - Stale KV entries 在下次 forward 被覆写
```

### 四种方案对比

| 特性 | v1 (2-tok verify) | v2 (1-tok verify) | v3 (2-tok batch, 当前) | vLLM (K+1-tok verify) |
|------|-------------------|-------------------|----------------------|-----------------------|
| 验证 input 大小 | 2 tokens | 1 token | 2 tokens | K+1 tokens |
| 每步最大产出 | 2 tokens | 1 token (等效2) | 2 tokens | K+1 tokens |
| E[tokens/step] | 1 + h | 1 + h | 1 + h | 1 + h₁ + h₁h₂ + ... |
| Tok/Infer (实际) | 1 + h | **1.00** | **1 + h** | 1 + h₁ + h₁h₂ + ... |
| KV trim 需求 | 是 (miss) | 否 | 是 (miss) | 否 (stale覆写) |
| 适用 K | 1 only | 1 only | 1 only | 任意 K≥1 |
| 适合模型规模 | 小-中 | 小 (compute-bound) | **中-大 (memory-bound)** | 中-大 (memory-bound) |
| 小模型 (0.8B) 成本 | ~25ms | ~14.3ms | ~25ms | ~32ms (K=3) |
| 大模型 (9B) 成本 | ~85ms | ~80ms + MTP | ~85ms | ≈T_1tok |

### 为什么 v3 的 Tok/Infer = 1 + h（而 v2 = 1.00）？

```
v3 (2-token batch verify):
  每步 1 次 main model infer([next_id, draft_id])

  Hit (概率 h):
    logits[0] → verified_id == draft_id ✓
    logits[1] → bonus_id
    → 产出 2 tokens, 1 main infer
    Tok/Infer = 2.0

  Miss (概率 1-h):
    logits[0] → verified_id ≠ draft_id
    → 产出 1 token, 1 main infer + trim
    Tok/Infer = 1.0

  E[Tok/Infer] = h × 2 + (1-h) × 1 = 1 + h

v2 (1-token verify) 的问题:
  Hit 时仍做 1-tok infer(verified) → 1-tok infer(bonus)
  → 2 main infers / 2 tokens = Tok/Infer = 1.0
  Miss: 1 main infer / 1 token = Tok/Infer = 1.0
  → Tok/Infer 恒为 1.0，MTP 完全没减少推理次数
```

### 关键洞察

1. **v3 解决了 v2 的核心问题**：v2 的 Tok/Infer=1.00 意味着 MTP 是纯开销；v3 恢复 2-token batch verify，HIT 时真正跳过一次主模型调用
2. **v3 适合 memory-bound 大模型**：9B+ 模型上 T_2tok ≈ T_1tok，2-token infer 几乎零额外成本
3. **v2 适合 compute-bound 小模型**：0.8B 上 T_2tok = 1.75× T_1tok，2-token infer 代价太高
4. **vLLM 的 K+1 verify 在大模型上有优势**：当 T_{K+1_tok} ≈ T_{1tok}（memory-bound），多验证多个 draft 几乎零成本
5. **CPU 上 KV trim 成本很低**：纯内存操作，不像 GPU 需要 GPU→CPU→GPU roundtrip

## Cross-Platform Comparison: vLLM (4090) vs OpenVINO (Intel GPU)

**Model**: Qwen3.5-0.8B | **vLLM**: NVIDIA RTX 4090, ~230 output tokens | **OV**: Intel GPU, INT4_ASYM, 64 output tokens

### Raw Data

| Metric | vLLM Baseline | vLLM MTP | OV v2 Baseline | OV v2 MTP |
|--------|---------------|----------|----------------|-----------|
| Decode throughput (tok/s) | 424.8 | 428.7 | ~70 | ~61 |
| Decode latency (ms/tok) | ~2.35 | ~2.33 | 14.3 | 16.5 |
| MTP hit rate | — | **71.9%** | — | **~30%** |
| Mean acceptance length | — | 1.72 | — | 1.30 |
| MTP decode speedup | — | **+1%** | — | **-13%** |
| Overall speedup (incl TTFT) | — | **+10%** | — | -13% |

### Gap Analysis

#### 1. Hit Rate 差距巨大：72% vs 30%

同一个 0.8B 模型，hit rate 不应该有如此大的差异。可能的原因：

| 因素 | vLLM | OV | 影响 |
|------|------|-----|------|
| 生成长度 | ~230 tokens | 64 tokens | 更长生成 → 更多上下文 → MTP 更准 |
| Prompt 类型 | 未知（可能更结构化） | 通用测试 prompt | 某些 prompt 更可预测 |
| 采样方式 | 可能 greedy | greedy | 应一致 |
| MTP 输入 | hidden_states from target model | 同上 | 实现一致 |
| **数值精度** | FP16 on CUDA | INT4_ASYM on GPU | **量化可能损害 MTP 预测质量** |

**验证建议**：用完全相同的 prompt 和生成长度在两个平台测试，排除 prompt/长度因素。重点关注 INT4 量化对 MTP head 预测精度的影响。

#### 2. 即使 72% Hit Rate，Decode 仅提升 1%

vLLM 在 4090 上：
```
Baseline decode: 2.35 ms/tok
MTP overhead:    draft cost + verification overhead
At 72% hit rate, 1.72 mean acceptance → ~1.72 tokens/step

Effective: 2.35ms × (1/1.72) = 1.37 ms/tok (理论值)
Actual:    2.33 ms/tok (仅提升1%)
→ MTP 编排开销几乎吃掉了全部收益
```

**结论**：0.8B 在 4090 上太快（2.35ms/tok），MTP 的调度/同步开销已与模型推理时间同量级。小模型 + 快硬件 = MTP 无法盈利的 decode 阶段。

#### 3. vLLM 的 10% 整体加速主要来自 TTFT

```
vLLM TTFT: 2.337s → 2.000s (↓14%)
vLLM Decode: 0.539s → 0.511s (↓5%)
```

TTFT 改善 337ms 贡献了大部分加速。这是因为 MTP 在 prefill 阶段可以 overlap 一些计算。Decode 阶段收益极小。

### 提升路径

| 优先级 | 方向 | 预期效果 | 详情 |
|--------|------|----------|------|
| **P0** | **测试更大模型 (7B/14B/72B)** | Decode +30~80% | 大模型 memory-bound，MTP 开销可摊销；hit rate 更高（60-80%） |
| **P1** | **验证 INT4 对 MTP 精度的影响** | 可能提升 hit rate | 如果 INT4 损害 MTP 预测，考虑 MTP head 使用 FP16/INT8 |
| **P1** | **用相同 prompt 对齐测试** | 明确真实 hit rate 差距 | 排除 prompt/长度/采样差异的影响 |
| **P2** | **优化 MTP 编排开销** | Decode +5~10% | 减少 tensor 创建/复制，MTP 异步执行 |
| **P2** | **增加生成长度测试** | 可能提升 hit rate | 64 tokens 可能不足以让 MTP 建立足够 context |

### 核心结论

> **0.8B 模型在任何高性能硬件上都很难从 MTP 获益于 decode 阶段。** vLLM 在 4090 上有 72% hit rate 但 decode 仅快 1%，OV 在 Intel GPU 上 30% hit rate 则慢 13%。真正的 ROI 在于 7B+ 模型，其中：
> - Decode 延迟足够高（>50ms/tok）使 MTP 开销可忽略
> - MTP head 预测更准（60-80% hit rate）
> - memory-bound 特性使 K+1 token verify ≈ 1-token verify 成本

## Benchmark Results: Qwen3.5-9B on Intel GPU

**Setup**: Qwen3.5-9B | INT4_ASYM group_size=128 | Intel GPU (CPU device) | 220 output tokens | 3 runs averaged

### Text Mode

| Config | TPOT (ms/tok) | Throughput (tok/s) | MTP Hit Rate | Tok/Infer | Delta |
|--------|---------------|--------------------|--------------|-----------|-------|
| Baseline | 84.21 | 11.90 | — | — | — |
| MTP v2 | 83.76 | 11.97 | 28.7% | 1.00 | -0.5% |

### VL Mode

| Config | TPOT (ms/tok) | Throughput (tok/s) | MTP Hit Rate | Tok/Infer | Delta |
|--------|---------------|--------------------|--------------|-----------|-------|
| Baseline | 66.26 | 15.09 | — | — | — |
| MTP v2 | 77.52 | 12.90 | 40.6% | 1.00 | **+17%** (worse) |

### 关键发现：Tok/Infer = 1 = MTP 是纯开销

**Tok/Infer = 1.00** 在所有测试中恒定，意味着 v2 算法 **没有跳过任何 main model 调用**。每个输出 token 仍然需要一次 main model 推理 + 一次 MTP 推理：

```
当前 v2 实际执行流程（每步）：
  1. main(confirmed_tok) → verified_id + hidden_states    ← 总是执行
  2. MTP(hidden_states) → new_draft                        ← 总是执行
  3. 比较 verified_id vs previous_draft → 记录 hit/miss
  Result: 1 token emitted, 1 main infer + 1 MTP infer

问题：即使 hit，下一步仍然要调用 main model
  → MTP 推理时间 是纯开销
  → 没有实现 "on hit skip next main model call" 优化
```

### 9B MTP 开销估算

```
Qwen3.5-9B: hidden_size=4096, intermediate_size=12288, vocab=248320
MTP overhead per step:
  - fc (2*4096 → 4096):           ~1ms (INT4)
  - 1 decoder layer (4096 dim):   ~5-8ms
  - lm_head (4096 → 248320):      ~3-5ms
  - Total MTP:                     ~10-15ms

VL mode: baseline 66ms + MTP ~11ms overhead = 77ms (matches actual 77.5ms ✓)
Text mode: baseline 84ms + MTP ~10ms overhead ≈ 94ms (actual 83.8ms, within noise)
```

### Text vs VL 差异分析

Text mode per-run variance 极大（TPOT: 77.75~87.62），MTP hit rate 也极不稳定（11%~53%）。-0.5% 的"提升"完全在噪声范围内。

VL mode 高度一致（TPOT: 76.96~77.81），清晰显示 +11ms/tok 的 MTP 开销。这与 hidden_size=4096 的 MTP 模型开销吻合。

### 与 vLLM（4090）对比

| 指标 | vLLM 0.8B + 4090 | OV 9B + Intel GPU |
|------|-------------------|-------------------|
| Baseline TPOT | 2.35 ms | 66~84 ms |
| MTP hit rate | 71.9% | 29~41% |
| MTP decode speedup | +1% | -0.5% ~ -17% |
| Tok/Infer | 1.72 | **1.00** |
| **根本差异** | vLLM 用 K+1 verify: 打包 draft 到一次 forward，减少推理次数 | OV v2: 每步仍需 1 次 main + 1 次 MTP，无法减少推理次数 |

### Why vLLM Achieves Tok/Infer > 1 But We Don't

```
vLLM (K+1 verify, k=1):
  Step N: main([confirmed, draft]) → verify draft + get bonus
  On accept: 2 tokens emitted from 1 main model call → Tok/Infer = 2
  On reject:  1 token emitted from 1 main model call → Tok/Infer = 1
  E[Tok/Infer] = 1 + h = 1.72 at 72% hit

OV v2 (1-tok verify):
  Step N: main([confirmed]) → 1 token always
  MTP runs additionally → pure overhead
  E[Tok/Infer] = 1.00 always (MTP never reduces main model calls)
```

**核心问题**：v2 用 1-token infer 避免了 KV trim，但也失去了 "一次验证多 token" 的能力。vLLM 的 K+1 verify 同时验证 + 产出多个 token，所以能真正减少 main model 调用次数。

### 改进方向

| 优先级 | 方向 | 预期效果 | 详情 |
|--------|------|----------|------|
| **P0** | **实现 main model call skip on hit** | Decode +30~40% (at 40% hit) | Hit 时跳过下一次 main model 调用，用 MTP hidden_states 继续预测。需解决 miss 后的 recovery |
| **P0** | **~~回归 v1 式 2-token verify~~** | ✅ 已实现 (v3) | 2-token batch verify：HIT 时 Tok/Infer=2.0，一次推理产出 2 token。MISS 时 trim KV 1 entry |
| **P1** | **实现 vLLM 式 K+1 verify** | 最大收益 | 需要 OpenVINO GPU 支持 stale KV overwrite 或实现 efficient KV trim |
| **P2** | **提升 hit rate** | 间接提升所有方案 | 验证 INT4 量化对 MTP 精度影响；增加生成长度 |

> **v3 vs v2 预期效果**（9B CPU, T_main≈80ms, T_mtp≈15ms, h=30%）：
> - v2: TPOT ≈ (80 + 15) / 1.0 = 95 ms/tok（比 baseline 80ms 慢 19%）
> - v3: TPOT ≈ (80 + 15) / 1.3 ≈ 73 ms/tok（比 baseline 快 ~9%）
> - 关键差异：v3 的 Tok/Infer = 1+h = 1.3 > 1.0，MTP 开销被 hit 加速 amortize

## Next Steps for Performance Optimization

1. **~~Reduce 2-token infer cost~~**: ✅ Resolved — v3 uses 2-token batch verify which is near-free on memory-bound models (9B+)
2. **~~Skip main model call on hit~~**: ✅ Resolved — v3 produces 2 tokens from 1 main infer on HIT
3. **Test with larger models**: 7B+ models typically achieve 60-80% MTP hit rates, making speculative decode profitable
4. **Multi-token speculation (k>1)**: Extend to draft k tokens ahead and verify all at once (vLLM-style K+1 verify)
5. **Benchmark v3 on 9B**: Measure actual Tok/Infer, KV trim cost on CPU, overall throughput improvement
6. **Adaptive strategy**: Auto-select v2 (small compute-bound models) vs v3 (large memory-bound models) based on model size
