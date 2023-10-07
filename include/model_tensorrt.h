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
    Engine(const std::string& model_path, const std::string& device);
    bool load_model(const std::string& model_path);
    
    bool compile(const std::string& mode, const unsigned int& max_batchsize, 
            const std::string& onnx_file, const std::string& engine_file);
    bool load_network(const std::string& enginefile);
    std::string inference(const std::string& sentence);
private:
    std::shared_ptr<nvinfer1::ICudaEngine> _engine;
    std::shared_ptr<nvinfer1::IExecutionContext> _context;
    std::shared_ptr<nvinfer1::IRuntime> _runtime;
    Logger _logger;
    cudaStream_t _stream;
};
