
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <opencv2/opencv.hpp>

#include "engine.h"

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using my_unique_ptr = std::unique_ptr<T, InferDeleter>;

class Logger : public nvinfer1::ILogger {
public:
    void log (Severity severity, const char* msg) noexcept override;
};

class InferenceEngine : public Engine {
public:
    InferenceEngine(const std::string& mode, const unsigned int& max_batchsize, 
            const std::string& onnx_file, const std::string& engine_file, const std::string label_file);
    
    int build();
    int deserialize_engine();
    virtual int init() override;
    virtual int infer(void* input, void* output) override;
    virtual int destroy() override;
    int load_labels();
    std::string get_label(const int& idx) {
        if (idx < 0 || idx >= labels.size()) {
            return "Unknown";
        }
        return labels[idx];
    }

    Logger _logger;

private:
    nvinfer1::Dims _input_dims;
    nvinfer1::Dims _output_dims;
    cudaStream_t _stream;

    /**
     * Error Code 3: API Usage Error (Parameter check failed at: 
     * runtime/rt/runtime.cpp::~Runtime::346, condition: mEngineCounter.use_count() == 1. 
     * Destroying a runtime before destroying deserialized engines created by the runtime leads to undefined behavior.
     * runtime声明放在engine之前
    */
    std::shared_ptr<nvinfer1::IRuntime> _runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> _engine;

    std::string _mode;
    unsigned int _max_batchsize;
    std::string _onnx_file;
    std::string _engine_file;
    std::string _label_file;
    std::vector<std::string> labels;
};
