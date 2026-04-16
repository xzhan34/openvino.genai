#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OV_DIR="${ROOT_DIR}/openvino"
GENAI_DIR="${SCRIPT_DIR}"
export LD_LIBRARY_PATH="${OV_DIR}/bin/intel64/RelWithDebInfo:${GENAI_DIR}/build/openvino_genai:${LD_LIBRARY_PATH}"
export OV_TOKENIZERS_LIB_PATH="${GENAI_DIR}/build/openvino_genai"

MODEL_DIR=/home/xzhan34/work/openvino.genai.modular-ws/models/Qwen3-Omni-4B-Instruct-multilingual

# Executables
CPP_BIN=${GENAI_DIR}/build/src/cpp/src/modeling/samples/modeling_qwen3_omni
CPP_TTS_BIN=${GENAI_DIR}/build/src/cpp/src/modeling/samples/modeling_qwen3_omni_tts_min

# Ensure libopenvino_tokenizers.so symlink exists next to the C++ binaries
TOKENIZERS_LIB="${GENAI_DIR}/build/openvino_genai/libopenvino_tokenizers.so"
BIN_DIR=$(dirname "${CPP_TTS_BIN}")
if [[ -f "${TOKENIZERS_LIB}" && ! -e "${BIN_DIR}/libopenvino_tokenizers.so" ]]; then
    ln -sf "${TOKENIZERS_LIB}" "${BIN_DIR}/libopenvino_tokenizers.so"
    echo "Created symlink: ${BIN_DIR}/libopenvino_tokenizers.so -> ${TOKENIZERS_LIB}"
fi

cd "${GENAI_DIR}"

# Model and inputs (aligned with qwen3_omni_4b_demo/run_qwen3_omni_dense.py)
TESTDATA_DIR="${GENAI_DIR}/testdata"
#IMG="${TESTDATA_DIR}/cars.jpg"
IMG="${TESTDATA_DIR}/get_started_with_cpp.jpg"
TEST_AUDIO="${TESTDATA_DIR}/cough.wav"

# Case 5 resources (text + image + video + audio -> text + tts)
CASE5_IMAGE="${TESTDATA_DIR}/london.jpg"
CASE5_VIDEO="${TESTDATA_DIR}/rainning.mp4"
CASE5_AUDIO="${TESTDATA_DIR}/thunder-and-rain-sounds.wav"
CASE5_PROMPT_FILE="${TESTDATA_DIR}/prompt.txt"

# Matrix configuration
#DEVICES="CPU,GPU.1"
DEVICES="GPU.1"
PRECISIONS="inf_fp16_kv_int8"
#fp32,inf_fp16_kv_int8,inf_fp32_kv_fp32_w_int4_asym"

# IR dump directory (empty = disabled)
DUMP_IR_DIR=$MODEL_DIR/ir_dumps

# Report output directory
DATESTAMP=$(python -c "from datetime import datetime; print(datetime.now().strftime('%Y%m%d_%H%M'))")
OUTDIR="./reports/qwen3_omni_4b_dense_matrix_${DATESTAMP}"
mkdir -p "${OUTDIR}"


echo "============================================"
echo " Input Parameters"
echo "============================================"
echo "CPP_BIN:      ${CPP_BIN}"
echo "CPP_TTS_BIN:  ${CPP_TTS_BIN}"
echo "MODEL:        ${MODEL_DIR}"
echo "IMG:          ${IMG}"
echo "TEST_AUDIO:   ${TEST_AUDIO}"
echo "CASE5_IMAGE:  ${CASE5_IMAGE}"
echo "CASE5_VIDEO:  ${CASE5_VIDEO}"
echo "CASE5_AUDIO:  ${CASE5_AUDIO}"
echo "CASE5_PROMPT: ${CASE5_PROMPT_FILE}"
echo "DEVICES:      ${DEVICES}"
echo "PRECISIONS:   ${PRECISIONS}"
echo "DUMP_IR_DIR:  ${DUMP_IR_DIR:-<disabled>}"
echo "OUTDIR:       ${OUTDIR}"
echo "============================================"
echo

python tools/qwen3_omni_case_compare.py \
    --model-dir "${MODEL_DIR}" \
    --image "${IMG}" \
    --test-audio "${TEST_AUDIO}" \
    --video "${CASE5_VIDEO}" \
    --case5-audio "${CASE5_AUDIO}" \
    --case5-image "${CASE5_IMAGE}" \
    --case5-prompt-file "${CASE5_PROMPT_FILE}" \
    --cpp-bin "${CPP_BIN}" \
    --cpp-tts-bin "${CPP_TTS_BIN}" \
    --out-json "${OUTDIR}/case_compare_matrix.json" \
    --max-new-tokens 32 \
    --max-video-frames 5 \
    --devices ${DEVICES} \
    --precisions ${PRECISIONS} \
    --timeout 600 \
    ${DUMP_IR_DIR:+--dump-ir-dir "${DUMP_IR_DIR}"} \
    --cpp-only --cases 1
#    --cases 5
