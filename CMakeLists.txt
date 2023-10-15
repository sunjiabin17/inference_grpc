# add_subdirectory(third_party/grpc)

cmake_minimum_required(VERSION 3.8)

project(InferGRPC CXX)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/libs/grpc_lib)
# set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/libs/TensorRT_lib)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/libs/TensorRT-8.6.1.6)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/libs/opencv_lib)
MESSAGE(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE "Debug")

# ------------------
# 在 CMake 中查找 gRPC 库和 Protobuf 库
find_package(Protobuf CONFIG REQUIRED)
MESSAGE(STATUS "Using protobuf ${Protobuf_VERSION}")

find_package(gRPC CONFIG REQUIRED)
MESSAGE(STATUS "Using gRPC ${gRPC_VERSION}")

# 包含 gRPC 和 Protobuf 的头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/grpc_lib/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/protos)


# 添加 gRPC 插件的库
# get_target_property(_GRPC_CPP_PLUGIN_EXECUTABLE gRPC::grpc_cpp_plugin IMPORTED_LOCATION_RELEASE)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

# 生成 gRPC 的 C++ 代码
set(PROTO_OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/protos)

set(_PROTO_SRCS "${PROTO_OUTPUT}/grpc_infer_service.pb.cc")
set(_PROTO_HDRS "${PROTO_OUTPUT}/grpc_infer_service.pb.h")
set(_GRPC_SRCS "${PROTO_OUTPUT}/grpc_infer_service.grpc.pb.cc")
set(_GRPC_HDRS "${PROTO_OUTPUT}/grpc_infer_service.grpc.pb.h")
# set(MY_PROTO "${CMAKE_CURRENT_SOURCE_DIR}/protos/grpc_infer_service.proto")
get_filename_component(MY_PROTO "${CMAKE_CURRENT_SOURCE_DIR}/protos/grpc_infer_service.proto" ABSOLUTE)
get_filename_component(MY_PROTO_PATH "${MY_PROTO}" PATH)
MESSAGE("MY_PROTO = ${MY_PROTO}")
MESSAGE("MY_PROTO_PATH = ${MY_PROTO_PATH}")

add_custom_command(
    OUTPUT "${_PROTO_SRCS}" "${_PROTO_HDRS}" "${_GRPC_SRCS}" "${_GRPC_HDRS}"
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --grpc_out "${PROTO_OUTPUT}"
      --cpp_out "${PROTO_OUTPUT}"
      -I "${MY_PROTO_PATH}"
      --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
      "${MY_PROTO}"
    DEPENDS "${MY_PROTO}"
)

if(BUILD_TRT)
    # ------------------
    # 添加cuda依赖、tensorrt依赖
    find_package(CUDA REQUIRED)
    set(TENSORRT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/TensorRT-8.6.1.6)
    set(TENSORRT_INCLUDE_DIRS ${TENSORRT_DIR}/include)

    set(CUDA_TOOLKIT_DIR "/usr/local/cuda")
    set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_DIR}/include)
    include_directories(
        ${TENSORRT_INCLUDE_DIRS}
        ${TENSORRT_DIR}/samples/common
        ${CUDA_INCLUDE_DIRS}
    )

    link_directories(
        ${TENSORRT_DIR}/lib
        ${CUDA_TOOLKIT_DIR}/lib64
    )
endif()

# ------------------
# 添加opencv依赖
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status: " ${OpenCV_FOUND})
message(STATUS "OpenCV include dirs: " ${OpenCV_INCLUDE_DIRS})

include_directories(${OpenCV_INCLUDE_DIRS})

# ------------------
# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# 查找全部.cc 文件
# file(GLOB_RECURSE SRCS src/*.cc)
set(GRPC_SERVER src/grpc_server.cc)
set(GRPC_CLIENT src/grpc_client.cc)
set(INFER_SERVER src/img_classify_tensorrt.cc)
set(CXX_OPTS include/cxxopts.hpp)
set(BASE64 src/base64.cc)
set(HELPER src/helper.cc)

if(BUILD_TRT)
    add_library(lib_infer_server ${INFER_SERVER})
    target_link_libraries(lib_infer_server ${CUDA_LIBRARIES} ${OpenCV_LIBS} nvinfer nvonnxparser nvparsers)
endif()

# 对应的lib都是由模板类头文件.hpp生成的，而其中并没有.cpp或者.c文件。所以需要设置一下链接器的语言为C++，否则会报错
add_library(lib_cxxopts ${CXX_OPTS})
set_target_properties(lib_cxxopts PROPERTIES LINKER_LANGUAGE CXX)

add_library(lib_base64 ${BASE64})

add_library(lib_helper ${HELPER})
target_link_libraries(lib_helper lib_base64 ${OpenCV_LIBS})

# 添加 gRPC 和 Protobuf 的库
add_executable(grpc_server ${GRPC_SERVER})
add_executable(grpc_client ${GRPC_CLIENT})

# 方法1链接
# 这里如果不链接grpc_proto会报错，如果链接grpc_proto时不加${_PROTO_SRCS} ${_PROTO_HDRS} ${_GRPC_SRCS} ${_GRPC_HDRS}也会报错
add_library(lib_grpc_proto ${_PROTO_SRCS} ${_PROTO_HDRS} ${_GRPC_SRCS} ${_GRPC_HDRS})
target_link_libraries(lib_grpc_proto protobuf::libprotobuf gRPC::grpc++ gRPC::grpc++_reflection)

if(BUILD_TRT)
    target_link_libraries(grpc_server lib_grpc_proto lib_infer_server lib_cxxopts lib_base64 lib_helper)
endif()

target_link_libraries(grpc_client lib_grpc_proto lib_cxxopts lib_base64 lib_helper)

# 方法2链接
# 将生成的 gRPC 的 C++ 代码添加到可执行文件中：
# target_sources(grpc_server PRIVATE ${_PROTO_SRCS} ${_PROTO_HDRS} ${_GRPC_SRCS} ${_GRPC_HDRS})
# target_sources(grpc_client PRIVATE ${_PROTO_SRCS} ${_PROTO_HDRS} ${_GRPC_SRCS} ${_GRPC_HDRS})
# target_link_libraries(grpc_server lib_infer_server lib_cxxopts gRPC::grpc++ gRPC::grpc++_reflection protobuf::libprotobuf)
# target_link_libraries(grpc_client lib_infer_server lib_cxxopts gRPC::grpc++ gRPC::grpc++_reflection protobuf::libprotobuf)

# 只编译生成proto文件
add_custom_target(proto DEPENDS ${_PROTO_SRCS} ${_PROTO_HDRS} ${_GRPC_SRCS} ${_GRPC_HDRS})

# add_executable(test test/test.cc)
# target_link_libraries(test lib_grpc_proto lib_infer_server lib_cxxopts lib_base64 lib_helper gRPC::grpc++ gRPC::grpc++_reflection protobuf::libprotobuf)

install(TARGETS grpc_server grpc_client DESTINATION bin)

# install libnvinfer_plugin.so.8
# install(FILES ${TENSORRT_DIR}/lib/libnvinfer_plugin.so.8 DESTINATION lib)
# install(DIRECTORY ${TENSORRT_DIR}/lib DESTINATION .)
# install libopencv_imgproc.so.408
# install(FILES ${OpenCV_LIBS} DESTINATION lib)



# get_cmake_property(_variableNames VARIABLES)
# list (SORT _variableNames)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()
