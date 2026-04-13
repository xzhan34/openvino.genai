mkdir -p build && cd build
OV_DIR=/media/xzhan34/data/zlab_dflash_ws/openvino_dflash/openvino
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DOpenVINO_DIR="$OV_DIR/build" ..
cmake --build . --config RelWithDebInfo --verbose -j16
cd ..
