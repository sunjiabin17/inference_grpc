#include "img_classify_tensorrt.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>


void Logger::log(Severity severity, const char *msg) noexcept {
    if (severity <= Severity::kERROR) {
        std::cout << msg << std::endl;
    }
}

InferenceEngine::InferenceEngine(const std::string& mode, const unsigned int& max_batchsize, 
            const std::string& onnx_file, const std::string& engine_file, const std::string label_file) :
    _runtime(nullptr),
    _engine(nullptr),
    _logger(Logger()),
    _stream(nullptr),
    _mode(mode),
    _max_batchsize(max_batchsize),
    _onnx_file(onnx_file),
    _engine_file(engine_file),
    _label_file(label_file)
{
    init();
}

int InferenceEngine::init() {
    if (load_labels() != 0) {
        std::cout << "Could not load labels" << std::endl;
        return 1;
    }
    std::ifstream ifs(_engine_file);
    if (ifs.good()) {
        std::cout << "deserializing engine..." << std::endl;
        return deserialize_engine();
    } else {
        return build();
    }
    return 0;
}

int InferenceEngine::destroy() {
    // _engine->destroy();
    // _runtime->destroy();
    return 0;
}

int InferenceEngine::load_labels() {
        std::ifstream file(_label_file);
    if (!file.is_open()) {
        std::cout << "Could not open label file" << std::endl;
        return 1;
    }
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
    file.close();
    return 0;
}

int InferenceEngine::build() {
    auto builder = my_unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(_logger));
    if (builder == nullptr) {
        std::cout << "Could not create builder" << std::endl;
        return 1;
    }

	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = my_unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (network == nullptr) {
        std::cout << "Could not create network" << std::endl;
        return 1;
    }

    auto config = my_unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (config == nullptr) {
        std::cout << "Could not create builder config" << std::endl;
        return 1;
    }

    auto onnx_parser = my_unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, _logger));
    if (onnx_parser == nullptr) {
        std::cout << "Could not create parser" << std::endl;
        return 1;
    }
    if (!onnx_parser->parseFromFile(_onnx_file.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        std::cout << "Could not parse ONNX file" << std::endl;
        return 1;
    }

    if (_mode == "fp16") {
        if (!builder->platformHasFastFp16()) {
            std::cout << "Platform does not support fast FP16" << std::endl;
            return 1;
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    else if (_mode == "int8") {
        if (!builder->platformHasFastInt8()) {
            std::cout << "Platform does not support Int8" << std::endl;
            return 1;
        }
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
    }
    else if (_mode == "tf32") {
        if (!builder->platformHasTf32()) {
            std::cout << "Platform does not support Tf32" << std::endl;
            return 1;
        }
        config->setFlag(nvinfer1::BuilderFlag::kTF32);
    }
    else {
        std::cout << "Unknown mode" << std::endl;
        return 1;
    }


    auto ret = cudaStreamCreate(&_stream);
    if (ret != cudaSuccess) {
        std::cout << "Could not create stream" << std::endl;
        return 1;
    }
    config->setProfileStream(_stream);


    // builder->setMaxBatchSize(max_batchsize);

    // const auto input = network->getInput(0);
    // const auto output = network->getOutput(0);
    // const auto inputname = input->getName();
    // const auto inputdims = input->getDimensions();

    // nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    // if (profile == nullptr) {
    //     std::cout << "Could not create optimization profile" << std::endl;
    //     return 1;
    // }
    // profile->setDimensions(inputname, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, 224, 224));
    // profile->setDimensions(inputname, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3, 224, 224));
    // profile->setDimensions(inputname, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(max_batchsize, 3, 224, 224));
    // config->addOptimizationProfile(profile);
    
    my_unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (plan == nullptr) {
        std::cout << "Could not build serialized network" << std::endl;
        return 1;
    }
    FILE* f = fopen(_engine_file.c_str(), "wb");
    if (f == nullptr) {
        std::cout << "Could not open file for writing" << std::endl;
        return 1;
    }
    fwrite(plan->data(), 1, plan->size(), f);
    fclose(f);

    _runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(_logger));
    if (_runtime == nullptr) {
        std::cout << "Could not create runtime" << std::endl;
        return 1;
    }
    _engine = std::shared_ptr<nvinfer1::ICudaEngine>(_runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());

    if (_engine == nullptr) {
        std::cout << "Could not create engine" << std::endl;
        return 1;
    }
    assert(network->getNbInputs() == 1);
    _input_dims = network->getInput(0)->getDimensions();
    assert(_input_dims.nbDims == 4);
    
    assert(network->getNbOutputs() == 1);
    _output_dims = network->getOutput(0)->getDimensions();
    assert(_output_dims.nbDims == 4);

    return 0;
}


int InferenceEngine::deserialize_engine() {
    std::ifstream file(_engine_file, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        _logger.log(nvinfer1::ILogger::Severity::kERROR, "Could not read file");
        // std::cout << "Could not read file" << std::endl;
        return 1;
    }
    if (_runtime == nullptr) {
        _runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(_logger));
    }
    if (_runtime == nullptr) {
        std::cout << "Could not create runtime" << std::endl;
        return 1;
    }

    if (_engine == nullptr) {
        _engine = std::shared_ptr<nvinfer1::ICudaEngine>(_runtime->deserializeCudaEngine(buffer.data(), size));
    }
    if (_engine == nullptr) {
        _logger.log(nvinfer1::ILogger::Severity::kERROR, "Could not deserialize engine");
        // std::cout << "Could not deserialize engine" << std::endl;
        return 1;
    }
    _input_dims = _engine->getBindingDimensions(0);
    _output_dims = _engine->getBindingDimensions(1);

    // auto ret = cudaStreamCreate(&_stream);
    // if (ret != cudaSuccess) {
    //     std::cout << "Could not create stream" << std::endl;
    //     return 1;
    // }
    return 0;
}

int InferenceEngine::infer(void* input, void* output) {
    auto context = my_unique_ptr<nvinfer1::IExecutionContext>(_engine->createExecutionContext());
    if (context == nullptr) {
        std::cout << "Could not create execution context" << std::endl;
        return 1;
    }

    // for (int i = 0; i < _engine->getNbBindings(); ++i) {
    //     std::cout << "binding name: " << _engine->getBindingName(i) << std::endl;
    //     std::cout << "binding index: " << _engine->getBindingIndex(_engine->getBindingName(i)) << std::endl;
    //     std::cout << "binding data type: " << static_cast<int>(_engine->getBindingDataType(i)) << std::endl;
    //     std::cout << "binding dims: " << _engine->getBindingDimensions(i).nbDims << std::endl;
    //     auto dims = _engine->getBindingDimensions(i);
    //     if (_engine->bindingIsInput(i)) {
    //         std::cout << "binding is input" << std::endl;
    //         context->setBindingDimensions(i, dims);
    //     }
    //     else {
    //         std::cout << "binding is output" << std::endl;
    //     }
    // }
    if (!context->allInputDimensionsSpecified()) {
        std::cout << "Not all input dimensions are specified" << std::endl;
        return 1;
    }

    const int input_size = _max_batchsize * _input_dims.d[1] * _input_dims.d[2] * _input_dims.d[3];
    const int output_size = _max_batchsize * _output_dims.d[1] * _output_dims.d[2] * _output_dims.d[3];
    // std::cout << "input size: " << input_size << std::endl;
    // std::cout << "output size: " << output_size << std::endl;

    float* input_data = static_cast<float*>(input);
    float* output_data = static_cast<float*>(output);


    void* buffers[2];
    buffers[0] = input_data;
    buffers[1] = output_data;

    if (_stream == nullptr) {
        auto ret = cudaStreamCreate(&_stream);
        if (ret != cudaSuccess) {
            std::cout << "Could not create stream" << std::endl;
            return 1;
        }
    }
    void* d_input, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpyAsync(d_input, input_data, input_size * sizeof(float), cudaMemcpyHostToDevice, _stream);
    cudaMemcpyAsync(d_output, output_data, output_size * sizeof(float), cudaMemcpyHostToDevice, _stream);

    context->setTensorAddress("data_0", d_input);
    context->setTensorAddress("fc6_1", d_output);
    
    context->enqueueV3(_stream);

    cudaStreamSynchronize(_stream);    

    cudaMemcpyAsync(output_data, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost, _stream);
    cudaStreamSynchronize(_stream);

    // std::cout << "output: " << std::endl;
    // for (int i = 0; i < output_size; ++i) {
    //     std::cout << output_data[i] << " ";
    // }
    // std::cout << std::endl;


    const int batch_size = 1;

    return 0;
}
// // test
// int main(int argc, char** argv) {
//     std::string onnx_file = "/home/tars/projects/code/inference_grpc/models/densenet_onnx/model.onnx";
//     std::string label_file = "/home/tars/projects/code/inference_grpc/models/densenet_onnx/densenet_labels.txt";
//     std::string engine_file = "/home/tars/projects/code/inference_grpc/build/model.engine";
//     InferenceEngine engine("tf32", 1, onnx_file, engine_file, label_file);
    
//     // engine.build();
//     engine.deserialize_engine();

//     std::string img_path("/home/tars/projects/code/inference_grpc/test/cat.jpg");
//     cv::Mat img = cv::imread(img_path);
//     if (img.empty()) {
//         std::cout << "Could not read image" << std::endl;
//         return 1;
//     }
//     std::cout << "image size: " << img.size() << std::endl;
//     // 将图像缩放到 224x224 大小
//     cv::Mat resized_img;
//     cv::resize(img, resized_img, cv::Size(224, 224));
    

//     // 将图像转换为 TensorRT 引擎输入格式
//     const int batch_size = 1;
//     const int channels = 3;
//     const int height = 224;
//     const int width = 224;
//     const int input_size = batch_size * channels * height * width;
    
//     float* input_data = new float[input_size];
//     for (int c = 0; c < channels; ++c) {
//         for (int h = 0; h < height; ++h) {
//             for (int w = 0; w < width; ++w) {
//                 int idx = c * height * width + h * width + w;
//                 input_data[idx] = resized_img.at<cv::Vec3b>(h, w)[c] / 255.0f;
//             }
//         }
//     }

//     float* output_data = new float[batch_size * 1000];
    
//     engine.infer(input_data, output_data);

//     float* max_element = std::max_element(output_data, output_data + 1000);
//     int max_idx = max_element - output_data;
//     std::cout << "max element: " << *max_element << std::endl;
//     std::cout << "max index: " << max_idx << std::endl;
//     std::cout << "label: " << engine.get_label(max_idx) << std::endl;

//     return 0;
// }