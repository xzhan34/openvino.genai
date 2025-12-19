# git submodule update --init
OPENVINO=/media/xzhan34/data/modular_genai_ws/openvino
GENAI=/media/xzhan34/data/modular_genai_ws/openvino.genai.xzhan34
cmake -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Debug  -B build_openvino/ -S $OPENVINO
cmake --build build_openvino/ --config Debug -j 10
cmake --install build_openvino/ --config Debug
source /home/xzhan34/ov_build_ws/install/setupvars.sh
cmake -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Debug -B build_openvino_genai/ -S $GENAI
cmake --build build_openvino_genai/ --config Debug -j 10
cmake --install build_openvino_genai/ --config Debug
export CMAKE_PREFIX_PATH="/home/xzhan34/ov_build_ws/install/runtime/cmake/"

# git submodule update --init
# source ./openvino/build/install/setupvars.sh
# cmake --install ./build/ --config Debug --prefix ./install
# cmake -DCMAKE_BUILD_TYPE=Debug -S ./ -B ./build_$1/
# cmake --build ./build_$1/ --config Debug -j 10
#cmake --install ./build_$1/ --config Debug --prefix ./install_$1
# export CMAKE_PREFIX_PATH="$(pwd)/install_$1/runtime/cmake/"

