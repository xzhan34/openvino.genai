#!/bin/bash
# =============================================================================
# export_dflash_ir.sh — Export DFlash draft model to OpenVINO IR (XML/BIN)
# =============================================================================
#
# 用法:
#   # Draft-only 导出 (Qwen3-4B 等扁平 config 模型):
#   ./scripts/export_dflash_ir.sh
#   ./scripts/export_dflash_ir.sh /path/to/Qwen3-4B /path/to/DFlash-b16
#   ./scripts/export_dflash_ir.sh /path/to/Qwen3-4B /path/to/DFlash-b16 FP16 INT4_ASYM
#
#   # 导出 target + draft (Qwen3.5 VLM 模型):
#   ./scripts/export_dflash_ir.sh /path/to/Qwen3.5-4B /path/to/DFlash INT4_ASYM INT4_ASYM
#
# 脚本参数:
#   $1  TARGET_MODEL   target 模型目录 (默认: Qwen3-4B)
#   $2  DRAFT_MODEL    DFlash draft 模型目录 (默认: Qwen3-4B-DFlash-b16)
#   $3  TARGET_QUANT   target 权重量化 (默认: FP16)
#                      可选: FP16 / INT4_ASYM / INT4_SYM / INT8_ASYM / INT8_SYM
#   $4  DRAFT_QUANT    draft 权重量化 (默认: INT4_ASYM)
#                      可选: FP16 / INT4_ASYM / INT4_SYM / INT8_ASYM / INT8_SYM
#
# 工作模式:
#   - Qwen3.5 target (有 text_config/vision_config):
#     导出 target + draft + vision(VL) 三个 IR
#   - Qwen3 / 其他扁平 config target (draft-only 模式):
#     自动检测，仅导出 draft model IR（含 embed_tokens + draft layers + lm_head）
#     target model 跳过（因其不兼容 Qwen3.5 的 GatedDeltaNet 架构）
#
# 底层参数说明 (modeling_qwen3_5_dflash 的位置参数):
#   argv[1]  TARGET_MODEL_DIR   target 模型目录，提取 embed_tokens + lm_head
#   argv[2]  DRAFT_MODEL_DIR    draft 模型目录，IR 也保存在此目录
#   argv[3]  PROMPT             导出模式下填 "dummy" 占位（不运行推理）
#   argv[4]  DEVICE             填 "CPU" 占位（导出模式不编译/推理）
#   argv[5]  MAX_NEW_TOKENS     填 "1"（导出模式不推理）
#   argv[6]  BLOCK_SIZE         填 "0" = 使用 config.json 默认值（通常 16）
#   argv[7]  TARGET_QUANT       target 权重精度
#   argv[8]  DRAFT_QUANT        draft 权重精度
#
# 环境变量:
#   OV_GENAI_EXPORT_DFLASH_IR=1  启用导出模式（脚本自动设置）
#
# 输出文件（保存在 DRAFT_MODEL_DIR 下）:
#   Draft-only:
#     dflash_draft_combined.xml/.bin               (FP16)
#     dflash_draft_combined_q4a_b8a_g128.xml/.bin  (INT4_ASYM)
#   Full (Qwen3.5):
#     dflash_target[_q4a_b8a_g128].xml/.bin
#     dflash_draft_combined[_q4a_b8a_g128].xml/.bin
#     dflash_vision.xml/.bin                       (VL 模式)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build/src/cpp/src/modeling/samples"
BINARY="${BUILD_DIR}/modeling_qwen3_5_dflash"

# ── 可配置参数 ──
TARGET_MODEL="${1:-/media/xzhan34/data/models/Qwen3-4B}"
DRAFT_MODEL="${2:-/media/xzhan34/data/zlab_dflash_ws/models/Qwen3-4B-DFlash-b16}"
TARGET_QUANT="${3:-FP16}"        # FP16 / INT4_ASYM / INT4_SYM
DRAFT_QUANT="${4:-INT4_ASYM}"    # FP16 / INT4_ASYM / INT4_SYM

# ── 检查二进制 ──
if [[ ! -x "${BINARY}" ]]; then
    echo "[ERROR] Binary not found: ${BINARY}"
    echo "        Please build first: cmake --build build --target modeling_qwen3_5_dflash -j\$(nproc)"
    exit 1
fi

echo "============================================="
echo "  DFlash IR Export"
echo "============================================="
echo "  Target model:  ${TARGET_MODEL}"
echo "  Draft model:   ${DRAFT_MODEL}"
echo "  Target quant:  ${TARGET_QUANT}"
echo "  Draft quant:   ${DRAFT_QUANT}"
echo "  Output dir:    ${DRAFT_MODEL}"
echo "============================================="

OV_GENAI_EXPORT_DFLASH_IR=1 "${BINARY}" \
    "${TARGET_MODEL}" \
    "${DRAFT_MODEL}" \
    "dummy" \
    "CPU" \
    "1" \
    "0" \
    "${TARGET_QUANT}" \
    "${DRAFT_QUANT}"

echo ""
echo "=== Exported IR files ==="
ls -lh "${DRAFT_MODEL}"/dflash_*.{xml,bin} 2>/dev/null || echo "(no IR files found)"
