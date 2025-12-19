#! /usr/bin/bash

SCRIPT_DIR_EXAMPLE_OV_CPP="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR_EXAMPLE_OV_CPP}

source ../../../../python-env/bin/activate
# Based on myown openvino
source ../../../../source_ov.sh

cd ${SCRIPT_DIR_EXAMPLE_OV_CPP}

mkdir -p build
cd build

export CMAKE_PREFIX_PATH="../../../../install/runtime/cmake/"
export OpenVINOGenAI_DIR="../../../../build"

cmake -DCMAKE_BUILD_TYPE=Debug ..
# cmake -DCMAKE_BUILD_TYPE=Release ..
make -j32
