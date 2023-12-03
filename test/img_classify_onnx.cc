#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "helper.h"

int main() {
    // 初始化ONNX Runtime环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    // 设置模型路径
    const std::string modelPath = "/home/tars/projects/code/inference_grpc/models/densenet_onnx/model.onnx";

    // 创建ONNX Runtime会话
    Ort::SessionOptions sessionOptions;
    Ort::Session session(env, modelPath.c_str(), sessionOptions);

    // 加载图片
    std::string img_file = "/home/tars/projects/code/inference_grpc/test/cat.jpg";
    cv::Mat img = cv::imread(img_file);
    if (img.empty()) {
        LOG_ERROR("Image is empty");
        return -1;
    }
    LOG_INFO("Image size: " << img.size());
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(224, 224));
    LOG_INFO("resized image size: " << resized_img.size());
    
    resized_img.convertTo(resized_img, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> img_channels(3);
    cv::split(resized_img, img_channels);

    // 获取输入节点信息
    std::vector<const char*> inputNames{"data_0"};
    const int batch_size = 1;
    const int channels = 3;
    const int height = 224;
    const int width = 224;
    std::vector<int64_t> input_shape{batch_size, channels, height, width};
    std::vector<float> inputs;
    inputs.resize(batch_size * channels * height * width);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(inputs.data() + c * height * width, img_channels[c].data, height * width * sizeof(float));
    }

    std::vector<Ort::Value> inputTensors;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memory_info, inputs.data(), inputs.size(), input_shape.data(), input_shape.size()));

    // 获取输出节点信息
    std::vector<const char*> outputNames{"fc6_1"};
    std::vector<int64_t> output_shape{1, 1000, 1, 1};
    std::vector<float> outputs;
    outputs.resize(1000);
    std::vector<Ort::Value> outputTensors;
    outputTensors.push_back(Ort::Value::CreateTensor<float>(memory_info, outputs.data(), outputs.size(), output_shape.data(), output_shape.size()));

    // 执行推理
    session.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), 1, outputNames.data(), outputTensors.data(), 1);

    // 获取输出结果
    int64_t result = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end())); 
    std::cout << "result = " << result << std::endl;


    // 打印预测结果
    std::vector<std::string> labels;
    std::string label_file = "/home/tars/projects/code/inference_grpc/models/densenet_onnx/densenet_labels.txt";

    auto load_label = [&labels, &label_file]() {
        std::ifstream file(label_file);
        if (!file.is_open()) {
            LOG_ERROR("Could not open label file");
            return 1;
        }
        std::string line;
        while (std::getline(file, line)) {
            labels.push_back(line);
        }
        file.close();
    };
    load_label();


    std::cout << "预测结果: " << labels[result] << std::endl;

    return 0;
}
