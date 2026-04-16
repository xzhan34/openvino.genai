#!/bin/bash
# Run Qwen3-4B DFlash speculative decoding on OpenVINO
# Usage: bash run_qwen3_4b_dflash.sh [PROMPT] [DEVICE] [MAX_TOKENS] [BLOCK_SIZE]
ROOT_DIR=/media/xzhan34/data/zlab_dflash_ws/openvino_dflash
export LD_LIBRARY_PATH="${ROOT_DIR}/openvino/bin/intel64/RelWithDebInfo:${ROOT_DIR}/openvino.genai/build/openvino_genai:${LD_LIBRARY_PATH}"
export OV_TOKENIZERS_LIB_PATH="${ROOT_DIR}/openvino.genai/build/openvino_genai"

BINARY=${ROOT_DIR}/openvino.genai/build/src/cpp/src/modeling/samples/modeling_dflash
TARGET=/media/xzhan34/data/models/Qwen3-4B
DRAFT=/media/xzhan34/data/models/Qwen3-4B-DFlash-b16

PROMPT="${1:-How many positive whole-number divisors does 196 have? Please reason step by step.}"
DEVICE="${2:-GPU}"
MAX_TOKENS="${3:-256}"
BLOCK_SIZE="${4:-16}"

echo "=========================================="
echo " Qwen3-4B DFlash on OpenVINO"
echo " Target: $TARGET"
echo " Draft:  $DRAFT"
echo " Device: $DEVICE"
echo " Max tokens: $MAX_TOKENS"
echo " Block size: $BLOCK_SIZE"
echo "=========================================="

"$BINARY" "$TARGET" "$DRAFT" "$PROMPT" "$DEVICE" "$MAX_TOKENS" "$BLOCK_SIZE"
