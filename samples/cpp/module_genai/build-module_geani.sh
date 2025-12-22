#OPENVINO=/home/xzhan34/work/openvino.genai.modular-ws/openvino.xzhan34
#GENAI=/home/xzhan34/work/openvino.genai.modular-ws/openvino.genai.xzhan34
#cmake -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Debug -B build_openvino -S $OPENVINO
#cmake --build ./build_openvino/ --config Debug -j 25
#cmake --install ./build_openvino/ --config Debug
#source /home/xzhan34/ov_build_ws/install/setupvars.sh
#cmake -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Debug -B build_openvino_genai -S $GENAI
#cmake --build ./build_openvino_genai/ --config Debug -j 25
#cmake --install ./build_openvino_genai/ --config Debug

#source source /home/xzhan34/ov_build_ws/install/setupvars.sh
#export CMAKE_PREFIX_PATH="/home/xzhan34/ov_build_ws/install/runtime/cmake/"
cmake -DCMAKE_BUILD_TYPE=Debug -B build -S .
cmake --build build -j10
