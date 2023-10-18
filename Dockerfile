# run docker build --network=host -t ubuntu-grpc-cpp-server .
# FROM ubuntu:20.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
WORKDIR /grpc_service
ENV WORKDIR=/grpc_service

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

ENV GRPC_DIR=${WORKDIR}/third_party/grpc
ENV OPENCV_DIR=${WORKDIR}/third_party/opencv
ENV GRPC_INSTALL_DIR=${WORKDIR}/libs/grpc_lib
ENV OPENCV_INSTALL_DIR=${WORKDIR}/libs/opencv_lib
ENV TENSORRT_INSTALL_DIR=${WORKDIR}/libs/TensorRT-8.6.1.6 
# RUN mkdir -p ${GRPC_INSTALL_DIR} && mkdir -p ${GRPC_DIR} && mkdir -p ${OPENCV_DIR} && mkdir -p ${OPENCV_INSTALL_DIR}

# COPY src include scripts models protos test CMakeLists.txt ./
COPY src ./src
COPY include ./include
COPY scripts ./scripts
COPY models ./models
COPY protos ./protos
COPY test ./test
COPY CMakeLists.txt ./CMakeLists.txt

# 直接拷贝本地文件，不下载编译
COPY libs/grpc_lib ${GRPC_INSTALL_DIR}
COPY libs/opencv_lib ${OPENCV_INSTALL_DIR}
COPY libs/TensorRT-8.6.1.6 ${TENSORRT_INSTALL_DIR}

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN apt update && apt install -y cmake build-essential autoconf libtool pkg-config git wget clang libc++-dev 
# libprotobuf-dev protobuf-compiler libpng-dev


# host proxy address
# RUN git config --global http.proxy "socks://127.0.0.1:10808"

ENV LD_LIBRARY_PATH=${GRPC_INSTALL_DIR}/lib:${OPENCV_INSTALL_DIR}/lib:${TENSORRT_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}

# RUN /bin/bash -c "cd ${WORKDIR} && \
#     mkdir -p build && \
#     pushd build && \
#     cmake -DBUILD_TRT=1 -DCMAKE_INSTALL_PREFIX=. .. && \
#     make && \
#     popd"


# RUN git clone --recurse-submodules -b v1.58.0 --depth 1 --shallow-submodules \
#         https://github.com/grpc/grpc ${GRPC_DIR}
# RUN /bin/bash -c "cd ${GRPC_DIR} && \
#     mkdir -p cmake/build && \
#     pushd cmake/build && \
#     cmake ../.. \
#             -DgRPC_INSTALL=ON                \
#             -DCMAKE_BUILD_TYPE=Release       \
#             -DgRPC_ABSL_PROVIDER=package     \
#             -DgRPC_CARES_PROVIDER=package    \
#             -DgRPC_PROTOBUF_PROVIDER=package \
#             -DgRPC_RE2_PROVIDER=package      \
#             -DgRPC_SSL_PROVIDER=package      \
#             -DgRPC_ZLIB_PROVIDER=package     \
#             -DCMAKE_INSTALL_PREFIX=${GRPC_INSTALL_DIR} \
#          && \
#     make -j4 && \
#     make install && \
#     popd"

# RUN git clone https://github.com/opencv/opencv.git ${OPENCV_DIR}
# RUN /bin/bash -c "cd ${OPENCV_DIR} && \
#     mkdir -p build && \
#     pushd build && \
#     cmake -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR} .. && \
#     make -j4 && \
#     make install && \
#     popd"

# RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v24.4/protobuf-24.4.tar.gz
# RUN tar -zxvf protobuf-24.4.tar.gz
# RUN /bin/bash -c "cd protobuf-24.4 && \
#     mkdir -p cmake/build && \
#     cd cmake/build && \
#     cmake ../.. \
#             -DCMAKE_BUILD_TYPE=Release       \
#          && \
#     make -j4 && \
#     make install"
