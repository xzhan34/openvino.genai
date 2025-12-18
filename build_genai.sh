# git submodule update --init
# source ./openvino/build/install/setupvars.sh
cmake -DCMAKE_BUILD_TYPE=Debug -S ./ -B ./build_$1/
cmake --build ./build_$1/ --config Debug -j 10
cmake --install ./build_$1/ --config Debug --prefix ./install_$1
export CMAKE_PREFIX_PATH="$(pwd)/install_$1/runtime/cmake/"

