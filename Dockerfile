# run docker build --network=host -t ubuntu-grpc-cpp-server .
FROM ubuntu:20.04

WORKDIR /gprc_service

COPY src/ include/ scripts/ ./

ENV GRPC_INSTALL_DIR=${WORKDIR}/third_party/grpc_lib
ENV GRPC_DIR=${WORKDIR}/third_party/grpc
ENV PATH=${GRPC_INSTALL_DIR}/bin:${PATH}
RUN mkdir -p ${GRPC_INSTALL_DIR} && mkdir -p ${GRPC_DIR}

RUN apt update && apt install -y cmake
RUN apt install -y build-essential autoconf libtool pkg-config git

# host proxy address
RUN git config --global http.proxy "socks://127.0.0.1:10808"

RUN git clone --recurse-submodules -b v1.58.0 --depth 1 --shallow-submodules \
        https://github.com/grpc/grpc ${GRPC_DIR}

RUN /bin/bash -c "cd ${GRPC_DIR} && \
    mkdir -p cmake/build && \
    pushd cmake/build && \
    cmake -DgRPC_INSTALL=ON \
        -DgRPC_BUILD_TESTS=OFF \
        -DCMAKE_INSTALL_PREFIX=${GRPC_INSTALL_DIR} \
        ../.. && \
    make -j4 && \
    make install && \
    popd"