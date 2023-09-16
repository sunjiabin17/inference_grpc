#!/bin/bash
# git submodule add https://github.com/grpc/grpc third_party/grpc 



export MY_INSTALL_DIR=`pwd`/third_party/grpc_lib
mkdir -p $MY_INSTALL_DIR
export PATH="$MY_INSTALL_DIR/bin:$PATH"

apt update
apt install -y cmake
apt install -y build-essential autoconf libtool pkg-config

# git clone --recurse-submodules -b v1.58.0 --depth 1 --shallow-submodules https://github.com/grpc/grpc
# cd grpc
git submodule update --init --recursive

# mkdir -p third_party/grpc
# git submodule add https://github.com/grpc/grpc third_party/grpc
# pushd third_party/grpc
# git checkout tags/v1.58.0 && git submodule update --init --recursive
pushd third_party/grpc
git checkout tags/v1.58.0 && git submodule update --init --recursive --depth 1
mkdir -p cmake/build
pushd cmake/build
cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR \
      ../..
make -j8
# make install
# sudo make install
popd
popd