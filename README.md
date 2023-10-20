# inference_grpc
grpc, tensorrt

## docker build

`docker build --network=host -t ubuntu-grpc-cpp-server .`

`docker build -f Dockerfile.flask --network=host -t ubuntu-flask-server .`

## docker run

### run http flask server
`docker run -it --network=host --rm --name ubuntu-flask-server ubuntu-flask-server`

or 

`docker run -it --network=host --rm --name ubuntu-flask-server ubuntu-flask-server python python/flask_server.py $http_port $grpc_address`

### run grpc infer server

`docker run -it --network=host --gpus=all --privileged --rm -v ${PWD}:/grpc_service --name ubuntu-grpc-cpp-server ubuntu-grpc-cpp-server`

or 

`docker run -it --network=host --gpus=all --privileged --rm -v ${PWD}:/grpc_service --name ubuntu-grpc-cpp-server ubuntu-grpc-cpp-server grpc_server --port=50051 --onnx_file=/grpc_service/models/densenet_onnx/model.onnx --label_file=/grpc_service/models/densenet_onnx/densenet_labels.txt --engine_file=/grpc_service/models/densenet_onnx/densenet.engine`

## bugs
### cmake can't find cuda in docker 11.8.0-cudnn8-runtime-ubuntu22.04  
> https://stackoverflow.com/questions/56889376/opencv-cmake-cant-find-cuda-while-building-image-for-docker

finally someone from the opencv repository pointed out that the problem I was having was due to the image I was using:
FROM nvidia/cuda:9.0-cudnn7-runtime
The runtime version usually doesn't contain the SDK files, the fix was to change -runtime for -develop.

### ImportError: libGL.so.1: cannot open shared object file: No such file or directory

> https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo

```docker
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
````

### nvinfer1::createInferBuilder return nullptr or

or load engine causes core dump

because I forget to specify `--gpus=all` when run docker



### /usr/bin/ld: warning: libpng16.so.16, needed by ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0, not found (try using -rpath or -rpath-link)

```
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_set_interlace_handling@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_set_IHDR@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_get_io_ptr@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_get_eXIf_1@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_set_longjmp_fn@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_set_gray_to_rgb@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_set_bgr@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_get_valid@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_set_rgb_to_gray@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_set_swap@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_destroy_read_struct@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_set_palette_to_rgb@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_get_tRNS@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_set_packing@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_read_end@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_set_read_fn@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_write_end@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_set_expand_gray_1_2_4_to_8@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_set_write_fn@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_create_write_struct@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_error@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_destroy_write_struct@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_set_strip_16@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_create_read_struct@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_set_tRNS_to_alpha@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_set_compression_level@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_set_filter@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_init_io@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_get_IHDR@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_set_compression_strategy@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_write_info@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_create_info_struct@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_read_update_info@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_write_image@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_read_image@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_read_info@PNG16_0'
/usr/bin/ld: ../libs/opencv_lib/lib/libopencv_imgcodecs.so.4.8.0: undefined reference to `png_set_strip_alpha@PNG16_0'
collect2: error: ld returned 1 exit status
```

try `apt-get install -y libpng16-16`


## TODO

- build libs grpc, proto and opencv in dockerfile, download tensorrt in dockerfile