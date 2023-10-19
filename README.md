# inference_grpc
grpc, tensorrt


## bugs
### cmake can't find cuda in docker 11.8.0-cudnn8-runtime-ubuntu22.04  
> https://stackoverflow.com/questions/56889376/opencv-cmake-cant-find-cuda-while-building-image-for-docker

finally someone from the opencv repository pointed out that the problem I was having was due to the image I was using:
FROM nvidia/cuda:9.0-cudnn7-runtime
The runtime version usually doesn't contain the SDK files, the fix was to change -runtime for -develop.

