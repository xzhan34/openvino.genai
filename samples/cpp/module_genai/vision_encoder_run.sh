#! /usr/bin/bash
SCRIPT_DIR_EXAMPLE_OV_CPP_RUN="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR_EXAMPLE_OV_CPP_RUN}

source ../../../../python-env/bin/activate
# Based on myown openvino
source ../../../../source_ov.sh

cd ${SCRIPT_DIR_EXAMPLE_OV_CPP_RUN}

./build/module_genai_app ./config_pipeline/config_qwen2_5_vl.yaml
