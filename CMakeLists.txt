# add_subdirectory(third_party/grpc)

cmake_minimum_required(VERSION 3.8)

project(GrpcTest C CXX)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/libs/grpc_lib)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/libs/TensorRT_lib)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE "Debug")

# 在 CMake 中查找 gRPC 库和 Protobuf 库
find_package(Protobuf CONFIG REQUIRED)
MESSAGE(STATUS "Using protobuf ${Protobuf_VERSION}")

find_package(gRPC CONFIG REQUIRED)
MESSAGE(STATUS "Using gRPC ${gRPC_VERSION}")

# 包含 gRPC 和 Protobuf 的头文件
include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${_GRPC_GRPCPP_INCLUDE_DIR})

find_package(CUDA REQUIRED)
set(TENSORRT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/TensorRT-8.6.1.6)
set(TENSORRT_INCLUDE_DIRS ${TENSORRT_DIR}/include)

set(CUDA_TOOLKIT_DIR "/usr/local/cuda")
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_DIR}/include)
include_directories(
    ${TENSORRT_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

link_directories(
    ${TENSORRT_DIR}/lib
    ${CUDA_TOOLKIT_DIR}/lib64
)

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# 查找全部.cc 文件
# file(GLOB_RECURSE SRCS src/*.cc)
set(GRPC_SERVER src/grpc_server.cc)
set(GRPC_CLIENT src/grpc_client.cc)


# 添加 gRPC 插件的库
# get_target_property(_GRPC_CPP_PLUGIN_EXECUTABLE gRPC::grpc_cpp_plugin IMPORTED_LOCATION_RELEASE)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

# 生成 gRPC 的 C++ 代码
set(_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/grpc_infer_service.pb.cc")
set(_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/grpc_infer_service.pb.h")
set(_GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/grpc_infer_service.grpc.pb.cc")
set(_GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/grpc_infer_service.grpc.pb.h")
# set(MY_PROTO "${CMAKE_CURRENT_SOURCE_DIR}/protos/grpc_infer_service.proto")
get_filename_component(MY_PROTO "${CMAKE_CURRENT_SOURCE_DIR}/protos/grpc_infer_service.proto" ABSOLUTE)
get_filename_component(MY_PROTO_PATH "${MY_PROTO}" PATH)
MESSAGE(STATUS "Using proto ${MY_PROTO}")


add_custom_command(
    OUTPUT "${_PROTO_SRCS}" "${_PROTO_HDRS}" "${_GRPC_SRCS}" "${_GRPC_HDRS}"
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
      --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
      -I "${MY_PROTO_PATH}"
      --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
      "${MY_PROTO}"
    DEPENDS "${MY_PROTO}"
)

# 添加 gRPC 和 Protobuf 的库
add_executable(grpc_server ${GRPC_SERVER})
add_executable(grpc_client ${GRPC_CLIENT})

add_executable(trt_test src/model_tensorrt.cc)
target_link_libraries(trt_test ${CUDA_LIBRARIES} nvinfer nvonnxparser)

# 方法1链接
# 这里如果不链接grpc_proto会报错，如果链接grpc_proto时不加${_PROTO_SRCS} ${_PROTO_HDRS} ${_GRPC_SRCS} ${_GRPC_HDRS}也会报错
# add_library(grpc_proto ${_PROTO_SRCS} ${_PROTO_HDRS} ${_GRPC_SRCS} ${_GRPC_HDRS})
# target_link_libraries(grpc_proto protobuf::libprotobuf gRPC::grpc++ gRPC::grpc++_reflection)
# target_link_libraries(grpc_server grpc_proto gRPC::grpc++ gRPC::grpc++_reflection protobuf::libprotobuf)
# target_link_libraries(grpc_client grpc_proto gRPC::grpc++ gRPC::grpc++_reflection protobuf::libprotobuf)

# 方法2链接
# 将生成的 gRPC 的 C++ 代码添加到可执行文件中：
target_sources(grpc_server PRIVATE ${_PROTO_SRCS} ${_PROTO_HDRS} ${_GRPC_SRCS} ${_GRPC_HDRS})
target_sources(grpc_client PRIVATE ${_PROTO_SRCS} ${_PROTO_HDRS} ${_GRPC_SRCS} ${_GRPC_HDRS})
target_link_libraries(grpc_server gRPC::grpc++ gRPC::grpc++_reflection protobuf::libprotobuf)
target_link_libraries(grpc_client gRPC::grpc++ gRPC::grpc++_reflection protobuf::libprotobuf)

# 只编译生成proto文件
add_custom_target(proto DEPENDS ${_PROTO_SRCS} ${_PROTO_HDRS} ${_GRPC_SRCS} ${_GRPC_HDRS})

add_library(cxxopts include/cxxopts.hpp)
# 对应的lib都是由模板类头文件.hpp生成的，而其中并没有.cpp或者.c文件。所以需要设置一下链接器的语言为C++，否则会报错
set_target_properties(cxxopts PROPERTIES LINKER_LANGUAGE CXX)
add_executable(cxxopts_test test/cxxopts_test.cc)
target_link_libraries(cxxopts_test cxxopts)
