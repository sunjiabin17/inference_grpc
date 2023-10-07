#pragma once

#include <string>
#include <vector>
#include <memory>

#include <NvInfer.h>
// #include <buffers.h>

class Vocab;


class Logger : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept override;
};


class Engine {
public:
    Engine(const std::string& mode, const unsigned int& max_batchsize, 
            const std::string& onnx_file, const std::string& engine_file, const bool is_compile=false);
    ~Engine();
    
    int compile(const std::string& mode, const unsigned int& max_batchsize, 
            const std::string& onnx_file, const std::string& engine_file);
    int deSerializeEngine(const std::string& enginefile);
private:
    nvinfer1::ICudaEngine* _engine;
    Logger _logger;
    cudaStream_t _stream;
};

class Classifier {
public:
    Classifier(std::shared_ptr<nvinfer1::ICudaEngine> engine);
    ~Classifier();
    int infer(const void* input, void* output);
private:
    std::shared_ptr<nvinfer1::ICudaEngine> _engine;
    cudaStream_t _stream;
};
