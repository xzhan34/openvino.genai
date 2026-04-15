SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OV_DIR="$(dirname "$SCRIPT_DIR")/openvino"
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DOpenVINO_DIR="$OV_DIR/build" ..
cmake --build . --config RelWithDebInfo --verbose -j16
cd ..
