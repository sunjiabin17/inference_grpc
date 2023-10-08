#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <opencv2/opencv.hpp>



struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;

class Logger : public nvinfer1::ILogger {
public:
    void log (Severity severity, const char* msg) noexcept override;
};

class InferenceEngine {
public:
    InferenceEngine(const std::string& mode, const unsigned int& max_batchsize, 
            const std::string& onnx_file, const std::string& engine_file, const bool is_compile=false);

    bool build();

    Logger _logger;

private:
    nvinfer1::Dims _input_dims;
    nvinfer1::Dims _output_dims;
    cudaStream_t _stream;

    std::shared_ptr<nvinfer1::IRuntime> _runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> _engine;

    std::string _mode;
    unsigned int _max_batchsize;
    std::string _onnx_file;
    std::string _engine_file;

};

void Logger::log(Severity severity, const char *msg) noexcept {
    if (severity <= Severity::kERROR) {
        std::cout << msg << std::endl;
    }
}

InferenceEngine::InferenceEngine(const std::string& mode, const unsigned int& max_batchsize, 
            const std::string& onnx_file, const std::string& engine_file, const bool is_compile) :
    _engine(nullptr),
    _logger(Logger()),
    _stream(nullptr),
    _mode(mode),
    _max_batchsize(max_batchsize),
    _onnx_file(onnx_file),
    _engine_file(engine_file)
{
}

bool InferenceEngine::build() {
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(_logger));
    if (builder == nullptr) {
        std::cout << "Could not create builder" << std::endl;
        return 1;
    }

	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (network == nullptr) {
        std::cout << "Could not create network" << std::endl;
        return 1;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (config == nullptr) {
        std::cout << "Could not create builder config" << std::endl;
        return 1;
    }

    auto onnx_parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, _logger));
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


    // auto ret = cudaStreamCreate(&_stream);
    // if (ret != cudaSuccess) {
    //     std::cout << "Could not create stream" << std::endl;
    //     return 1;
    // }
    // config->setProfileStream(_stream);


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
    
    SampleUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
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


int main(int argc, char** argv) {
    std::string model_path = "/home/tars/projects/code/inference_grpc/models/densenet_onnx/model.onnx";
    InferenceEngine engine("tf32", 1, model_path, "model.engine", true);
    engine.build();
}
