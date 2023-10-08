#pragma once

#include <string>
#include <vector>
#include <memory>

#include <NvInfer.h>
// #include <buffers.h>


class Logger : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept override;
};


class InferenceEngine {
public:
    InferenceEngine(const std::string& mode, const unsigned int& max_batchsize, 
            const std::string& onnx_file, const std::string& engine_file, const bool is_compile=false);
    ~InferenceEngine();
    
    int compile(const std::string& mode, const unsigned int& max_batchsize, 
            const std::string& onnx_file, const std::string& engine_file);
    int deSerializeEngine(const std::string& enginefile);
private:
    std::shared_ptr<nvinfer1::ICudaEngine> _engine;
    std::shared_ptr<nvinfer1::IRuntime> _runtime;
    Logger _logger;
    cudaStream_t _stream;

    nvinfer1::Dims _input_dims;
    nvinfer1::Dims _output_dims;
};

// class Classifier {
// public:
//     Classifier(std::shared_ptr<InferenceEngine> engine, const std::string& label_file);
//     ~Classifier();
//     int infer(const void* input, void* output);
//     int load_labels(const std::string& label_file);
// private:
//     std::shared_ptr<InferenceEngine> _engine;
//     nvinfer1::IExecutionContext* _context;
//     cudaStream_t _stream;

//     std::vector<std::string> _labels;
// };
