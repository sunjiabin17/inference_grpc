# run docker build --network=host -t ubuntu-flask-server .
FROM ubuntu:20.04

WORKDIR /http_service
ENV WORKDIR=/http_service

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# COPY src include scripts models protos test CMakeLists.txt ./
COPY python ./python
COPY protos ./protos
COPY test ./test

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN apt update && apt install -y cmake build-essential autoconf libtool pkg-config git wget clang libc++-dev libprotobuf-dev protobuf-compiler libpng-dev
RUN apt update && apt-get install ffmpeg libsm6 libxext6 -y

RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda3
ENV PATH=/miniconda3/bin:$PATH
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
RUN conda config --set show_channel_urls yes
RUN echo 'export PATH=/miniconda3/bin:$PATH' >> ~/.bashrc
RUN echo 'export PATH=/miniconda3/bin:$PATH' >> ~/.bash_profile

RUN conda init bash
RUN conda create -n py39 python=3.9 -y 
RUN /bin/bash -c ". activate py39 && conda install -y opencv grpcio grpcio-tools protobuf flask requests"



ENV PYTHONPATH ${PYTHONPATH}:${WORKDIR}/protos:${WORKDIR}
