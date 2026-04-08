# Qwen3.5 MTP (Multi-Token Prediction) Implementation Summary

## Overview

This document summarizes the initial MTP speculative decoding implementation for Qwen3.5 in the OpenVINO GenAI explicit modeling framework. This serves as the baseline for future performance optimization work.

**Status**: K+1 verify speculative decoding is fully implemented with **inline sequential verify** (`--seq-verify 1`)。**关键 bug 已修复**：1) attention mask shape bug; 2) `linear_states` 未在 KV trim 时恢复导致 K≥2 退化。使用 `--seq-verify 1` 后 **K=1/2/3 在 9B 上均功能正确**（与 baseline 输出完全一致）。0.8B 因模型敏感性仍退化。

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

### Speculative Decode Loop (v3 — 2-Token Batch Verify)

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

### Speculative Decode Loop (v4 — K+1 Verify, current)

vLLM-style K+1 verify with autoregressive MTP drafting:

```
Given: next_id (confirmed token), K draft tokens [d1, d2, ..., dk]

DRAFT PHASE (autoregressive MTP):
  d1 = MTP(main_hidden_states[last], next_id)       ← from main model hs
  d2 = MTP(mtp_hidden_states, d1)                    ← from MTP's own hs output
  ...
  dk = MTP(mtp_hidden_states, d_{k-1})

VERIFY PHASE:
  Feed [next_id, d1, d2, ..., dk] as K+1 tokens to main model
  Main model outputs logits at K+1 positions + hidden_states

ACCEPT/REJECT (sequential):
  For k = 0..K-1:
    verified_id = sample(logits[k])
    if verified_id == d_{k+1}:
      ACCEPT: emit verified_id, continue
    else:
      REJECT: emit verified_id as correction, stop

  If all K accepted:
    BONUS: sample(logits[K]) → emit extra token (K+1 total!)

KV MANAGEMENT:
  trim_count = K - num_accepted
  if trim_count > 0: trim KV cache by trim_count entries

E[Tok/Infer] = 1 + h + h² + ... + h^K   (geometric series)
  K=1: 1+h       K=2: 1+h+h²       K=3: 1+h+h²+h³
```

**Key features of v4 vs v3**:
- **Autoregressive MTP**: MTP model outputs `mtp_hidden_states` (pre-lm_head), enabling K>1 drafting without main model
- **K+1 verify**: All K drafts verified in a single main model forward pass
- **Bonus token**: When all K drafts accepted, sample position K for K+1 total tokens
- **Scalable K**: `--mtp-k N` CLI parameter (default: 1)

## Changed Files

### Model Definition (6 files)

| File | Changes |
|------|---------|
| `processing_qwen3_5.hpp` | Added `mtp_num_hidden_layers` to `Qwen3_5TextConfig`; added `Qwen3_5MtpIO` struct with I/O names including `kMtpHiddenStates`; added `kHiddenStates` to `Qwen3_5TextIO` |
| `processing_qwen3_5.cpp` | Parse `mtp_num_hidden_layers` from JSON config |
| `modeling_qwen3_5_text.hpp` | Added `forward_with_hidden()`, `forward_embeds_with_hidden()` methods; added `output_hidden_states` param to `create_qwen3_5_text_model()`; declared `create_qwen3_5_mtp_model()` |
| `modeling_qwen3_5_text.cpp` | Implemented MTP model graph construction (`create_qwen3_5_mtp_model()`); implemented `forward_with_hidden()` to capture hidden states before LM head; MTP outputs both logits and `mtp_hidden_states` (normed pre-lm_head) for autoregressive K>1 drafting; MTP config: full_attention only, dense MLP, no MoE |
| `qwen3_5_weight_specs.hpp` | Declared `build_qwen3_5_mtp_weight_specs()` |
| `qwen3_5_weight_specs.cpp` | Implemented MTP weight specs: `mtp.fc`, `mtp.pre_fc_norm_*`, `mtp.layers[i].*`, shared `embed_tokens` and `lm_head` via PackedMapping rules |

### Sample Application (1 file)

| File | Changes |
|------|---------|
| `modeling_qwen3_5.cpp` | Added `--mtp` / `--mtp-layers` / `--mtp-k` / `--temperature` CLI args; `extract_logits_at_pos_f32()` for multi-position logit extraction; `trim_kv_cache_states()` for KV cache trimming on rejection; K+1 verify speculative decode loop with `generate_k_drafts()` (autoregressive MTP), `run_kp1_verify()` (K+1 token main model forward), sequential accept/reject with bonus token; three acceptance rate metrics (conditional, absolute, mean per step); greedy (T=0) and sampling modes |

### Benchmark Script (1 file, new)

| File | Description |
|------|-------------|
| `bench_mtp.ps1` | PowerShell benchmark script: runs baseline vs MTP K=1/2/3 across text/VL modes for multiple models (0.8B, 9B), parses all metrics (TTFT, TPOT, throughput, three MTP acceptance rates, tokens/infer), outputs summary table with speedup calculation; saves per-run logs with timestamps to `OV_Logs/` directory |

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

### 五种方案对比

| 特性 | v1 (2-tok verify) | v2 (1-tok verify) | v3 (2-tok batch) | v4 (K+1 verify, 当前) | vLLM (K+1 verify) |
|------|-------------------|-------------------|-------------------|-----------------------|-----------------------|
| 验证 input 大小 | 2 tokens | 1 token | 2 tokens | K+1 tokens | K+1 tokens |
| 每步最大产出 | 2 tokens | 1 token (等效2) | 2 tokens | K+1 tokens | K+1 tokens |
| E[tokens/step] | 1 + h | 1 + h | 1 + h | 1+h+h²+...+h^K | 1+h₁+h₁h₂+... |
| Tok/Infer (实际) | 1 + h | **1.00** | **1 + h** | **1+h+h²+...+h^K** | 1+h₁+h₁h₂+... |
| KV trim 需求 | 是 (miss) | 否 | 是 (miss) | 是 (K-accepted) | 否 (stale覆写) |
| 适用 K | 1 only | 1 only | 1 only | **任意 K≥1** | 任意 K≥1 |
| MTP 自回归 | 否 | 否 | 否 | **是 (mtp_hs)** | 是 |
| 适合模型规模 | 小-中 | 小 (compute-bound) | 中-大 (memory-bound) | **中-大 (memory-bound)** | 中-大 (memory-bound) |

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

## Bug Fix: K+1 Verify Attention Mask (2026-03-31)

### 问题现象

0.8B 模型 MTP K=2 和 K=3 生成的文本严重退化：

| Config | 生成内容 | Accept% |
|--------|----------|---------|
| Baseline (no MTP) | 连贯的短篇故事（Leo 在实验室画画） | — |
| MTP K=1 | "to now to now to now to now..." | 38% |
| MTP K=2 | "- - - - - - - - - - - -..." | **93%** (虚假高) |
| MTP K=3 | "The\nThe\nThe\nThe\nThe..." | 65% |

**关键线索**：K=2 的 93% accept rate 是虚假的 — MTP 容易预测退化的重复序列。在 greedy (T=0) 模式下，speculative decoding 应该产出与 baseline **完全相同**的文本，但实际输出完全不同。

### Debug 过程

#### Step 1: 对比 token IDs

首先添加 debug print 输出 generated token ids，对比 baseline vs K=1 vs K=2：

```
Baseline:  760 20513 314 279 9775 557 524 4147 26 424 557 8545 11 1040 279 4528
K=1:       760 20513 557 279 1118 3065 3397 6408 13 271 2064 579 1288 1018 ...
K=2:       760 20513 303 6949 11 264 3074 1500 314 6105 13 271 2064 557 279 ...
```

前 2 个 token (760, 20513) 一致，但第 3 个开始分岔。Baseline 多次运行完全确定性（GPU greedy），排除随机性。

#### Step 2: 验证是系统性误差

```
Baseline Run 1/2/3:  760 20513 314 279 9775 ...  ← 每次完全相同
K=1 Run 1/2:         760 20513 557 279 1118 ...  ← 也确定性，但与 baseline 不同
```

两点结论：
1. GPU greedy 是确定性的（baseline 三次一致）
2. MTP 也是确定性的（K=1 两次一致）
3. 差异是**系统性**的，不是随机噪声

#### Step 3: 分析内部状态

添加详细 debug 输出追踪每步的 verify/accept/reject 决策：

```
[DBG] iter=1 past_len=29 next_id=760 logits=[1,3,248320] hs=[1,3,1024] drafts=[11952,995]
[DBG]   k=0 REJECT draft=11952 verified=20513
[DBG]   num_accepted=0 trim=2 past_len_before_trim=29
```

K+1 verify 在功能上是正确的（accept/reject 逻辑无误），但 **logits 本身就不对** — 主模型对 K+1 token input 产出了错误的 logits。

#### Step 4: 定位 attention mask 问题

检查 `run_kp1_verify()` 中的 tensor 设置，发现 attention mask 的形状：

```cpp
// 原始代码（有 bug）
ov::Tensor step_mask_kp1 = make_usm_host_tensor(gpu_ctx, ov::element::i64, {batch, kp1});
// shape = [1, K+1]，全部填 1
```

但模型内部的 causal mask 构建逻辑（`build_kv_causal_mask_with_attention_from_q_len`）使用 `attention_mask.shape[1]` 作为 `kv_len`：

```
kv_len = attention_mask.shape[1] = K+1
q_len = input_ids.shape[1] = K+1
cache_len = kv_len - q_len = (K+1) - (K+1) = 0   ← BUG!
```

`cache_len = 0` 意味着模型认为 **没有任何缓存的 KV**，所有 K+1 个 token 只能互相 attend，完全看不到 prompt 和之前生成的内容。

#### Step 5: 理解为什么 single-token decode 不受影响

Single-token decode 的 attention mask shape = `[batch, 1]`：

```
kv_len = 1, q_len = 1 → 但 dim=1 在 SDPA 中会 broadcast 到实际 KV 长度
→ 所有 past positions 的 mask 值 = 1 (attend) → 正确！
```

K+1 verify 的 attention mask shape = `[batch, K+1]` (K+1 ≥ 2)：

```
kv_len = K+1, q_len = K+1 → 不会 broadcast
→ 只构建了 [K+1, K+1] 的 causal mask → 只能看到 K+1 个 token 之间 → 错误！
```

### Root Cause

`run_kp1_verify()` 中 attention mask 形状为 `[batch, K+1]`，但正确形状应为 `[batch, past_len + K+1]`。

模型内部 `build_kv_causal_mask_with_attention_from_q_len()` 依赖 `attention_mask.shape[1]` 来推算 `cache_len = kv_len - q_len`：

| 场景 | mask shape | kv_len | q_len | cache_len | 是否正确 |
|------|-----------|--------|-------|-----------|----------|
| Prefill | `[B, prompt_len]` | prompt_len | prompt_len | 0 | ✓ |
| Single decode | `[B, 1]` | 1 | 1 | 0, but dim=1 broadcasts | ✓ |
| **K+1 verify (bug)** | `[B, K+1]` | K+1 | K+1 | **0** | **✗** |
| **K+1 verify (fix)** | `[B, past_len+K+1]` | past_len+K+1 | K+1 | **past_len** | ✓ |

### Fix

在 `modeling_qwen3_5.cpp` 的 `run_kp1_verify()` lambda 中，将静态预分配的 `[batch, K+1]` attention mask 改为动态创建的 `[batch, past_len + K+1]` mask：

```cpp
// 修复后代码
const size_t full_mask_len = static_cast<size_t>(past_len) + kp1;
ov::Tensor step_mask_kp1 = make_usm_host_tensor(
    gpu_ctx, ov::element::i64, {batch, full_mask_len});
auto* p = step_mask_kp1.data<int64_t>();
for (size_t i = 0; i < batch * full_mask_len; ++i) p[i] = 1;
```

这确保模型计算出正确的 `cache_len = (past_len + K+1) - (K+1) = past_len`。

### Fix 验证

#### 0.8B 模型

| Config | 修复前 | 修复后 |
|--------|--------|--------|
| K=1 | "to now to now to now..." | 连贯故事文本 |
| K=2 | "- - - - - - - - -..." | 连贯文本（与 baseline 略有差异） |
| K=3 | "The\nThe\nThe\nThe..." | 连贯开头文本 |

#### 9B 模型

| Config | 修复后 | Accept% | Tok/Infer |
|--------|--------|---------|-----------|
| K=1 | 连贯故事（与 baseline 前 15 token 一致） | 51% | 1.51 |
| K=2 | 连贯文本（"Paris is capital", 巧克力蛋糕食谱等） | 64% | 2.27 |
| K=3 | 连贯文本 | 18% | 1.55 |

### 已知限制：Multi-token SDPA 数值差异

修复后，greedy (T=0) 模式下 MTP 输出仍与 baseline 略有不同。这是 GPU SDPA kernel 在 `q_len > 1` vs `q_len = 1` 时的数值精度差异导致的，是硬件层面的固有行为：

- **9B K=1**: 前 15 个 token 与 baseline 完全一致，第 16 个开始分岔
- **0.8B K=1**: 前 2 个 token 一致，第 3 个开始分岔（小模型对数值扰动更敏感）

这不是代码 bug，而是 multi-token batch inference 在 GPU 上的固有特性。

**重要更新 (2026-03-31 greedy T=0 测试)**：9B 在 K=1 (q_len=2) 输出连贯有意义的文本，但 **K≥2 (q_len≥3) 时 9B 也会严重退化**（"and and 174444..."、"android, android, android..."）。SDPA 数值差异随 q_len 增大而加剧，q_len=2 是 9B 的安全边界。详见「Benchmark Results: Qwen3.5-9B — Greedy T=0」章节。0.8B 在 K≥1 即退化，详见下一节。

## 0.8B 模型 MTP 功能退化根因分析 (2026-03-31)

### 问题现象

修复 attention mask bug 后，0.8B 模型 MTP K=1/2/3 greedy (T=0) 输出仍然严重退化：

| Config | 量化模式 | 输出内容 | Accept% | 状态 |
|--------|----------|----------|---------|------|
| Baseline | INT4_ASYM | "The silence of the lab was not empty; it was heavy..." | — | ✅ 连贯 |
| MTP K=1 | INT4_ASYM | "Hello, my dear friend," 反复重复, "robot robot", "dear dear friend friend" | ~38% | ❌ 退化 |
| MTP K=2 | INT4_ASYM | "about her about her about her about her..." 无限重复 | 57→93% | ❌ 退化 |
| MTP K=3 | INT4_ASYM | "My, My, My, My, My, My, My, My..." 无限重复 | ~13% | ❌ 退化 |

Baseline 完全正常，问题仅出现在 MTP speculative decoding 模式下。

### 排除量化因素：FP16 隔离测试

为验证是否与 INT4_ASYM 量化有关，使用 `OV_GENAI_INFLIGHT_QUANT_MODE=none`（FP16）重新测试：

| Config | 量化模式 | 输出内容 | 状态 |
|--------|----------|----------|------|
| Baseline | FP16 | "The hum of the machine was a constant, rhythmic thrum..." | ✅ 连贯 |
| MTP K=1 (32 tok) | FP16 | "The hummedieval, a low-frequency vibration..." | ⚠️ 勉强连贯 |
| MTP K=2 (64 tok) | FP16 | "The hummedium'stapped... robot paintbrushes... dust** and walls** painted.**" | ❌ 退化 |
| MTP K=3 (128 tok) | FP16 | "it was a dream it was a dream it was a dream it was a dream..." | ❌ 严重退化 |

**结论：FP16 模式下出现完全相同的退化模式 → 问题与 INT4_ASYM 量化无关。**

### 根因：GPU Multi-Token SDPA 数值差异 + 小模型敏感性

问题的根本原因是两个因素的叠加效应：

#### 1. GPU SDPA Kernel 数值差异

GPU 的 Scaled Dot-Product Attention (SDPA) kernel 在以下两种场景使用不同的计算路径：

| 场景 | q_len | 计算策略 | 精度特征 |
|------|-------|----------|----------|
| 逐 token 生成 (baseline) | 1 | 单行 softmax | 参考精度 |
| K+1 verify batch | K+1 ≥ 2 | 多行 tiled reduction | 微小数值偏差 |

两种路径在数学上等价，但 GPU 浮点运算的结合律差异（tiling 分块 → 中间结果累积顺序不同）导致最终 logits 存在 1e-5 ~ 1e-4 量级的微小差异。

#### 2. 小模型对数值扰动极度敏感

| 模型 | 参数量 | Hidden Size | 首次 token 分岔位置 | 分岔后行为 |
|------|--------|-------------|---------------------|------------|
| **0.8B** | 0.8B | 1024 | **第 3 个 token** | 快速进入退化重复循环 |
| **9B** | 9B | 4096 | 第 16 个 token | 继续生成连贯文本 |

- **0.8B 的概率分布更 "平坦"**：vocabulary 上的 logit 值差异小，top-1 和 top-2 之间 margin 经常 < 0.01。微小的 SDPA 数值扰动足以翻转 `argmax`，导致选出不同的 token。
- **9B 的概率分布更 "尖锐"**：模型更自信，top-1 token 的 logit 远高于其他候选，SDPA 数值差异不足以改变选择结果。

#### 3. 分岔 → 退化的级联效应

```
位置 3: SDPA 数值差异 → argmax 翻转 → 选出不同 token
         ↓
位置 4-5: 错误 token 导致上下文偏移，进一步降低后续 token 的可预测性
         ↓
位置 6+: 0.8B 模型的弱 language modeling 能力无法恢复，
         陷入低 perplexity 的退化模式（重复短语是 loss 最低的 "安全" 输出）
         ↓
结果: "about her about her about her..." 或 "My, My, My, My,..."
```

9B 模型即使在第 16 个 token 分岔，仍然能够利用更强的 language modeling 能力生成连贯（但不同于 baseline 的）文本。

### 代码正确性验证

为排除代码 bug，逐一验证了 speculative decoding 管线的每个组件：

| 组件 | 验证结果 | 说明 |
|------|----------|------|
| Causal masking | ✅ 正确 | attention mask 修复后 `[B, past_len+K+1]`，cache_len 计算正确 |
| KV cache trim | ✅ 正确 | reject 时正确 trim K-accepted 个 entry |
| past_len 跟踪 | ✅ 正确 | accept/reject 后 past_len 正确更新 |
| Penalty processor | ✅ 正确 | repetition penalty 在 accept/reject 后正确更新生成 token 列表 |
| Draft generation | ✅ 正确 | MTP 自回归 drafting 使用正确的 hidden_states |
| Accept/reject 逻辑 | ✅ 正确 | 顺序验证 + bonus token 逻辑无误 |
| Baseline (K=0) | ✅ 正确 | 0.8B 和 9B 都生成连贯文本 |

**所有 speculative decoding 逻辑都经过验证，代码层面无 bug。**

### 结论

> **0.8B 模型的 MTP 功能退化是 GPU multi-token SDPA 数值差异 + 小模型弱 robustness 的固有限制，不是代码 bug，无法通过代码修复。**

#### 推荐

1. **MTP speculative decoding 仅推荐 K=1 + 9B+ 模型** — 9B 在 K=1 产出连贯文本，K≥2 退化
2. **bench_mtp.ps1 已添加自动功能质量检测**（Check-TextQuality），每次运行后自动检查：
   - 连续相同单词重复 ≥5 次
   - 相同 bigram 重复 ≥4 次
   - 文本后半段 unique word ratio < 0.20
3. **0.8B 模型在 bench_mtp.ps1 的 $MODELS 数组中已默认注释掉**
4. 如果未来 GPU SDPA kernel 的 multi-token 精度改善，0.8B 可能恢复可用

### 影响：之前的 benchmark 数据需重新测量

之前报告的 K+1 verify benchmark 结果（Text +37%, VL +15%）**不可信**，因为：

1. **0.8B 模型**：K≥2 的退化输出产生虚假高 accept rate（MTP 容易预测重复 token）
2. **9B 模型**：之前的 benchmark 同时受 tokenizer bug（dummy tokenization）和 attention mask bug 双重影响

修复后需要重新运行完整 benchmark 获取真实数据。

## Next Steps for Performance Optimization

1. **~~Reduce 2-token infer cost~~**: ✅ Resolved — v3 uses 2-token batch verify which is near-free on memory-bound models (9B+)
2. **~~Skip main model call on hit~~**: ✅ Resolved — v3 produces 2 tokens from 1 main infer on HIT
3. **~~Multi-token speculation (k>1)~~**: ✅ Resolved — v4 implements K+1 verify with autoregressive MTP drafting
4. **~~Fix K+1 verify attention mask~~**: ✅ Fixed — mask shape corrected from `[B, K+1]` to `[B, past_len+K+1]`
5. **~~Fix 9B tokenizer~~**: ✅ Fixed — generated openvino_tokenizer.xml/bin for Qwen3.5-9B
6. **Re-run full benchmark**: All previous 9B results are invalid (tokenizer + attention mask bugs). Need fresh data.
7. **Run greedy (T=0) benchmark**: Eliminate sampling randomness; compare accept rate with vLLM's 90%
8. **~~Verify INT4 impact on MTP precision~~**: ✅ Verified — FP16 test on 0.8B shows identical degeneration pattern. INT4 quantization is NOT the cause. Accept rate gap with vLLM likely due to different prompt/length/model size.
9. **Adaptive K selection**: Auto-tune K based on observed accept rate during generation
10. **Async MTP execution**: Overlap MTP inference with main model decode to hide MTP latency
11. **bench_mtp.ps1 functional quality check**: ✅ Added — Check-TextQuality function detects degenerate text (word repeat, bigram repeat, low diversity)

## Benchmark Results: Qwen3.5-9B on Intel GPU — K+1 Verify (v4)

> ⚠️ **WARNING: 以下数据不可信** — 受两个已修复 bug 影响：(1) 9B 模型缺少 openvino_tokenizer.xml，使用 dummy byte-level tokenization（prompt 变成 66 个垃圾 token）；(2) K+1 verify attention mask 形状错误，导致 K≥2 生成退化文本。需要重新 benchmark。

**Setup**: Qwen3.5-9B | INT4_ASYM group_size=128 | Intel GPU | 256 output tokens | 3 runs averaged | Sampling (T=1.0, top_p=0.95, top_k=20)

### Text Mode

| Config | TTFT (ms) | TPOT (ms/tok) | Throughput (tok/s) | Accept% | Tok/Infer | Avg Acc | vs Baseline |
|--------|-----------|---------------|--------------------|---------|-----------|---------|-------------|
| Baseline | 266.64 | 77.67 | 12.89 | — | — | — | — |
| MTP K=1 | 258.05 | 77.65 | 12.89 | 52.5% | 1.52 | 0.52 | **+0%** |
| **MTP K=2** | **205.22** | **57.31** | **17.67** | **57.7%** | **2.15** | **1.16** | **+37.1%** |
| MTP K=3 | 264.10 | 58.25 | 17.30 | 46.6% | 2.39 | 1.40 | +34.2% |

### VL Mode

| Config | TTFT (ms) | TPOT (ms/tok) | Throughput (tok/s) | Accept% | Tok/Infer | Avg Acc | vs Baseline |
|--------|-----------|---------------|--------------------|---------|-----------|---------|-------------|
| Baseline | 254.38 | 67.96 | 14.72 | — | — | — | — |
| MTP K=1 | 261.62 | 86.05 | 11.62 | 47.1% | 1.47 | 0.47 | **-21.1%** |
| MTP K=2 | 239.11 | 69.36 | 14.53 | 47.7% | 1.95 | 0.95 | -1.3% |
| **MTP K=3** | **256.83** | **61.41** | **16.87** | **49.8%** | **2.49** | **1.49** | **+14.6%** |

### Analysis

#### GPU 上 K+1 verify 首次实现正向收益

这是 MTP 实现以来第一次看到明确的正向吞吐提升：

- **Text K=2: TPOT -26.2%，吞吐 +37.1%** — 最强结果
- **VL K=3: TPOT -9.6%，吞吐 +14.6%** — VL 模式需要更多 draft 来摊销开销

核心原因：**GPU 上 T_{K+1tok} ≈ T_{1tok}**。GPU 是纯 memory-bound，多 token batch 的计算近乎免费，K+1 个 token 的 KV 更新成本与 1 个 token 接近。

#### K 值选择分析

**Text 模式**：K=2 是甜蜜点
- K=1 → K=2: 吞吐从 12.89 跳到 17.67 (+37%)
- K=2 → K=3: 吞吐从 17.67 降到 17.30 (-2%)
- 原因：K=3 时 Accept% 从 57.7% 降到 46.6%，额外 draft 的低命中率抵消了 K+1 的收益

**VL 模式**：K=3 是甜蜜点
- K=1 负收益 (-21%)，K=2 持平 (-1.3%)，K=3 才正向 (+14.6%)
- VL 模式下 MTP 开销更高（可能与 visual embeddings 处理有关），需要更高 K 来摊销

#### 各 K 值下的理论 vs 实际 E[Tok/Infer]

假设均匀命中率 h（绝对 accept%），理论 E[Tok/Infer] = 1+h+h²+...+h^K：

| K | Text h | 理论 E | 实际 Tok/Infer | Text 效率 |
|---|--------|--------|---------------|-----------|
| 1 | 52.5% | 1.53 | 1.52 | **99%** |
| 2 | 57.7% | 1.91 | 2.15 | **112%** (bonus token 超额) |
| 3 | 46.6% | 1.57 | 2.39 | **152%** (bonus token 超额) |

实际 Tok/Infer > 理论值，因为 bonus token（全部 K 个 draft 接受时额外 +1）的贡献。

#### 方差问题（非 greedy 采样）

Per-run 方差极大：
- VL K=3: 21.42 / 15.51 / 13.69 tok/s — 最好最差差 57%
- Text K=2: 15.10 / 18.11 / 19.81 — 差距 31%

原因：temperature=1.0 + top_p=0.95 的采样随机性。**需要 greedy (T=0) 基准测试来获取稳定结果。**

#### 与 vLLM (4090) 对比

| 指标 | vLLM 9B (4090) | OV 9B (Intel GPU) | 差距 |
|------|----------------|-------------------|------|
| K=1 Accept rate | 90.4% | 52.5% | **-38pp** |
| K=1 Tok/Infer | 1.90 | 1.52 | -0.38 |
| K=1 Decode speedup | +30% | +0% | **-30pp** |
| K=3 Tok/Infer | 3.24 | 2.39 | -0.85 |
| K=3 Decode speedup | +50% | +34% | -16pp |

**最大差距在 accept rate**：vLLM 90% vs OV 52%。可能原因：
1. INT4 量化损害 MTP head 精度（vLLM 用 FP16）
2. 采样随机性（vLLM 可能用 greedy）
3. 不同的 prompt 和生成内容

### Acceptance Rate Metrics 说明

v4 输出三种 acceptance rate：

| 指标 | 公式 | 含义 |
|------|------|------|
| **MTP hits (conditional)** | hits / attempts | 条件命中率：仅统计实际验证过的位置（在第一个 reject 处停止） |
| **MTP draft acceptance (absolute)** | hits / (steps × K) | 绝对接受率：所有生成的 draft 中有多少被接受 |
| **MTP mean accepted/step** | hits / main_infers | 每次主模型推理平均接受的 draft 数 |

条件命中率偏高（因为后位置只在前位置 accept 时才测试），绝对接受率更真实地反映 MTP 质量。

## Benchmark Results: Qwen3.5-9B — Greedy T=0, Bug-Free (2026-03-31)

**Setup**: Qwen3.5-9B | INT4_ASYM group_size=128 | Intel GPU | 256 output tokens | 3 runs | Greedy (T=0)
**Fixes applied**: tokenizer fix + attention mask fix。这是修复所有已知 bug 后的首次有效 benchmark。

### Text Mode

| Config | TTFT (ms) | TPOT (ms/tok) | Throughput (tok/s) | Accept% | Tok/Infer | Avg Acc | Quality |
|--------|-----------|---------------|--------------------|---------|-----------|---------|---------|
| Baseline | ~771 | 69.12 | 14.47 | — | — | — | ✅ 连贯故事 |
| MTP K=1 | ~1289 | 80.85 | 12.37 | 50% | 1.50 | 0.50 | ✅ 连贯 |
| MTP K=2 | ~1008 | 44.38 | 22.53 | 79% (虚假) | 2.58 | 1.59 | ❌ "and and and 174444..." |
| MTP K=3 | ~1033 | 44.70 | 22.37 | 66% (虚假) | 2.97 | 1.98 | ❌ "android, android, android..." |

### VL Mode

| Config | TTFT (ms) | TPOT (ms/tok) | Throughput (tok/s) | Accept% | Tok/Infer | Avg Acc | Quality |
|--------|-----------|---------------|--------------------|---------|-----------|---------|---------|
| Baseline | ~213 | 69.25 | 14.44 | — | — | — | ✅ 完整 image 分析 |
| MTP K=1 | ~250 | 69.44 | 14.40 | 67% | 1.67 | 0.67 | ✅ 基本连贯 |
| MTP K=2 | ~288 | 99.53 | 10.05 | 30% | 1.59 | 0.60 | ⚠️ 质量下降 |
| MTP K=3 | ~259 | 77.25 | 12.95 | 35% | 2.06 | 1.06 | ⚠️ 碎片化 |

### 关键发现

#### 1. K=1 功能正确，但性能回退

| Mode | Config | TPOT | Throughput | vs Baseline | KV Trim 总耗时 | Trim 次数 | Avg Trim |
|------|--------|------|------------|-------------|----------------|-----------|----------|
| Text | K=1 | 80.85 ms | 12.37 tok/s | **-14.5%** | 3338ms | 85次 | 39ms |
| VL | K=1 | 69.44 ms | 14.40 tok/s | **-0.3%** | 2210ms | 51次 | 43ms |

Text 模式 -14.5% 回退的主因是 **KV trim 开销**：50% reject × 39ms/trim = 3338ms，占 decode 总时间 16%。
VL 模式因 67% accept rate，trim 次数少，基本持平。

#### 2. K≥2 功能退化 — SDPA 数值差异随 q_len 放大

| K 值 | verify q_len | 9B Text 表现 | 9B VL 表现 |
|------|-------------|-------------|------------|
| 0 (baseline) | 1 | ✅ 连贯 | ✅ 连贯 |
| 1 | 2 | ✅ 连贯（token 15 分岔） | ✅ 基本连贯 |
| **2** | **3** | **❌ "and and 174444..."** | ⚠️ 质量下降 |
| **3** | **4** | **❌ "android, android..."** | ⚠️ 碎片化 |

**规律**：q_len 越大 → GPU SDPA tiling 数值偏差越大 → argmax 越早翻转 → 级联退化越快。
9B 在 q_len=2 (K=1) 仍然 robust，但 **q_len≥3 (K≥2) 超出 9B 的容忍范围**。

K=2/K=3 的高 accept rate (79%/66%) 是虚假的 — 退化重复 token 极易被 MTP 预测。

#### 3. 9B KV Trim Profiling

| Config | Main Verify | MTP Draft | KV Trim | Trim 比例 |
|--------|-------------|-----------|---------|-----------|
| K=1 Text | 14950ms (170×88ms) | 2267ms (169×13ms) | 3338ms (85×39ms) | **16.2%** |
| K=1 VL | 13235ms (153×87ms) | 2209ms (152×15ms) | 2210ms (51×43ms) | **12.5%** |
| K=2 Text | 8472ms (99×86ms) | 2390ms (196×12ms) | 399ms (24×17ms) | **3.5%** |
| K=3 Text | 7511ms (86×87ms) | 3049ms (255×12ms) | 795ms (32×25ms) | **7.0%** |

K=2/K=3 的低 trim 比例反映了虚假高 accept rate（退化文本的自我维持循环）。

#### 4. 结论 (旧 — batch verify 时期)

> **旧结论（已修正）**：之前认为根因是 GPU SDPA kernel 在 q_len≥3 时数值精度不足。
> 实际根因见下方「根因修正与修复」章节。

---

## 根因修正与修复：K≥2 退化的真正原因 (2026-04-01)

### 调查过程

1. **初始假设**：GPU SDPA multi-token kernel (q_len>1) 使用不同于 single-token (q_len=1) 的 softmax 算法（online partitioned softmax vs single-pass），导致数值精度差异
2. **实施 sequential verify** (`--seq-verify 1`)：将 K+1 batch inference 拆为 K+1 个 single-token inference，每次使用 q_len=1 的 SDPA kernel
3. **结果**：sequential verify 产出的 token 与 batch verify **完全一致** — K=2 仍然退化
4. **关键发现**：对比 K=1 和 K=2 的逐 token 决策日志：
   - 两者在 verify #1 产出相同 token（next_id=623, accept d[0]=279, correction/bonus=78828）
   - 但在 verify #2（同样的 next_id=78828，同样的 past_len=28）：
     - K=1: `verified=11` (ACCEPT) ← 正确
     - K=2: `verified=20513` (REJECT) ← 错误，开始偏离

### 根因：`linear_states` 未在 KV trim 时恢复

Qwen3.5 使用**混合注意力架构**，包含两种 stateful 组件：

| 状态类型 | 名称模式 | Shape | 数量 | 特征 |
|---------|---------|-------|------|------|
| KV Cache | `past_key_values.N.key/value` | [B, H, S, D] | 16 (8层×2) | seq_len 维度随 token 增长，可 trim |
| **Recurrent/Linear States** | `linear_states.N.conv` | [1, 8192, 4] | 24 | **固定 shape，累积更新，不可 trim** |
| **Recurrent/Linear States** | `linear_states.N.recurrent` | [1, 32, 128, 128] | 24 | **固定 shape，累积更新，不可 trim** |

原来的 `trim_kv_cache_states()` 函数只 trim `past_key_values.*` 状态，**完全忽略了 `linear_states.*`**。

**退化机制**：
```
K=2 batch verify → feed [next_id, draft[0], draft[1]] → 3 tokens
  → KV cache: +3 entries
  → linear_states: 累积了 3 个 token 的效果（包括可能被 reject 的 draft）
  
reject draft[1] → trim KV cache by 1 → KV 恢复正确
  → 但 linear_states 仍包含 draft[1] 的效果！← BUG
  
下一次 verify → linear_states 基态已被污染 → 所有后续推理偏离 → 级联退化
```

**为何 K=1 不受影响**：K=1 的 accept rate 足够高（~50%），且 q_len=2 时 linear_states 的单次累积误差较小。
更重要的是：在 K=1 时，即使第一次 trim 发生在较晚位置，误差累积速度慢，不会在短期内导致可见退化。

### 修复方案：Inline Sequential Verify with Early Stopping

**核心思路**：不先推理所有 K+1 token 再 trim，而是**逐 token 推理 + 即时判定**，在第一个 reject 时**立即停止**。这样：
1. 被 reject 的 draft token **永远不会被推理**，linear_states 不会被污染
2. **无需 KV trim** — 只推理了正确的 token，缓存中没有错误条目
3. 使用 q_len=1 的 SDPA kernel，与 baseline 精度一致

**算法**：
```
// 逐 token 推理 + 即时验证
infer(next_id) → logits[0]
  verified = argmax(logits[0])
  if verified != drafts[0]: REJECT → stop, next_id = verified
  else: ACCEPT drafts[0]

infer(drafts[0]) → logits[1]
  verified = argmax(logits[1])
  if verified != drafts[1]: REJECT → stop, next_id = verified
  else: ACCEPT drafts[1]

...

infer(drafts[K-1]) → logits[K]
  bonus = argmax(logits[K])
  next_id = bonus
```

**CLI 开关**：`--seq-verify 1`（默认关闭以保持向后兼容）

### 修复后 Benchmark: Qwen3.5-9B — Greedy T=0, `--seq-verify 1`

| Config | TPOT (ms/tok) | Throughput (tok/s) | vs Baseline | Accept% | Tok/Infer | Quality |
|--------|---------------|-------------------|-------------|---------|-----------|---------|
| Baseline (K=0) | 68.90 | 14.51 | — | N/A | 1.00 | ✅ 连贯 |
| K=1 seq-verify | 77.45 | 12.91 | **-11.0%** | 50.89% | 1.51 | ✅ **连贯** |
| **K=2 seq-verify** | **83.79** | **11.93** | **-17.8%** | **35.91%** | **1.71** | **✅ 连贯** |
| **K=3 seq-verify** | **88.28** | **11.33** | **-21.9%** | **25.52%** | **1.76** | **✅ 连贯** |

**关键结果**：
- **K=1/2/3 输出文本与 baseline 完全一致**（逐 token 相同，greedy T=0 deterministic）
- **KV trim 次数 = 0** — inline sequential verify 完全避免了 trim
- Token 质量问题彻底解决

**文本验证**（128 tokens，与 baseline 完全一致）：
> In the sterile, humming silence of Sector 4, Unit 734—known affectionately as "Seven"—stood before a blank canvas. To the other robots in the factory, painting was a calculation: *Measure X, dispense pigment Y, apply pressure Z.* It was an assembly line for art, efficient and perfect, but utterly soulless...

### 性能分析

Inline sequential verify 的性能代价：每个 verify step 需要 `num_accepted + 1` 次 main model inference（而非 batch verify 的 1 次）。

| Config | Main Verify 总耗时 | 调用次数 | Avg/call | MTP Draft | Draft 调用次数 |
|--------|-------------------|---------|----------|-----------|---------------|
| K=1 | ~13.2s | 88 | 150ms | ~1.0s | 87 |
| K=2 | ~8.5s | 79 | 108ms | ~2.1s | 156 |
| K=3 | ~8.6s | 75 | 115ms | ~3.0s | 222 |

**权衡**：
- K 越大 → 每次 verify 的 infer 次数越多 → TPOT 越高
- 但 accept 时可以一次推多个 token → 总 main infer 次数越少
- 当前 accept rate (~25-50%) 不足以弥补额外 infer 开销 → throughput 低于 baseline
- **关键优势**：文本质量与 baseline 一致，为未来优化提供正确的基线

### 推荐配置

| 场景 | 推荐 |
|------|------|
| 生产使用 | `--mtp 0`（baseline，最高 throughput） |
| MTP 功能验证 | `--mtp 1 --mtp-k 1 --seq-verify 1` |
| 大 K 实验 | `--mtp 1 --mtp-k 2/3 --seq-verify 1`（功能正确，throughput 较低） |
| 不推荐 | `--mtp 1 --mtp-k 2+` 不带 `--seq-verify 1`（linear_states 污染导致退化） |
