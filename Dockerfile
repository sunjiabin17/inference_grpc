# run docker build --network=host -t ubuntu-grpc-cpp-server .
FROM ubuntu:20.04

WORKDIR /gprc_service
ENV WORKDIR=/gprc_service

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# COPY src include scripts models protos test CMakeLists.txt ./
COPY src ./src
COPY include ./include
COPY scripts ./scripts
COPY models ./models
COPY protos ./protos
COPY test ./test
COPY CMakeLists.txt ./CMakeLists.txt

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN apt update && apt install -y cmake build-essential autoconf libtool pkg-config git wget clang libc++-dev libprotobuf-dev protobuf-compiler libpng-dev

ENV GRPC_DIR=${WORKDIR}/third_party/grpc
ENV OPENCV_DIR=${WORKDIR}/third_party/opencv
ENV GRPC_INSTALL_DIR=${WORKDIR}/libs/grpc_lib
ENV OPENCV_INSTALL_DIR=${WORKDIR}/libs/opencv_lib
# RUN mkdir -p ${GRPC_INSTALL_DIR} && mkdir -p ${GRPC_DIR} && mkdir -p ${OPENCV_DIR} && mkdir -p ${OPENCV_INSTALL_DIR}

ENV PATH=${GRPC_INSTALL_DIR}/bin:${PATH}


# host proxy address
RUN git config --global http.proxy "socks://127.0.0.1:10808"

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
COPY libs/grpc_lib ${GRPC_INSTALL_DIR}

# RUN git clone https://github.com/opencv/opencv.git ${OPENCV_DIR}
# RUN /bin/bash -c "cd ${OPENCV_DIR} && \
#     mkdir -p build && \
#     pushd build && \
#     cmake -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR} .. && \
#     make -j4 && \
#     make install && \
#     popd"
COPY libs/opencv_lib ${OPENCV_INSTALL_DIR}

# RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
# RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda3
# ENV PATH=/miniconda3/bin:$PATH
# RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
# RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
# RUN conda config --set show_channel_urls yes
# RUN echo 'export PATH=/miniconda3/bin:$PATH' >> ~/.bashrc
# RUN echo 'export PATH=/miniconda3/bin:$PATH' >> ~/.bash_profile

# RUN conda init bash
# RUN conda create -n py39 python=3.9 -y 
# RUN /bin/bash -c ". activate py39 && conda install -y opencv grpcio protobuf flask pillow"

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

RUN /bin/bash -c "cd ${WORKDIR} && \
    mkdir -p build && \
    pushd build && \
    cmake .. && \
    make grpc_client && \
    popd"

