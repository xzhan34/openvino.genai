# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alongside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

file(DOWNLOAD
    https://raw.githubusercontent.com/nothings/stb/f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31/stb_image.h
    ${CMAKE_BINARY_DIR}/stb_image.h
    EXPECTED_HASH MD5=27932e6fb3a2f26aee2fc33f2cb4e696)

add_executable(qwen3_vl_modeling
    qwen3_vl_modeling.cpp
    ../visual_language_chat/load_image.cpp)

target_include_directories(qwen3_vl_modeling PRIVATE
    "${CMAKE_BINARY_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../visual_language_chat"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../src/cpp/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../openvino/thirdparty/json/nlohmann_json/include")

target_link_libraries(qwen3_vl_modeling PRIVATE openvino::genai)

set_target_properties(qwen3_vl_modeling PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS qwen3_vl_modeling
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
