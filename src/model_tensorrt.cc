#include "model_tensorrt.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <unordered_set>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
// #include <buffers.h>
// #include <common.h>



void Logger::log(Severity severity, const char *msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

Engine::Engine(const std::string& mode, const unsigned int& max_batchsize, 
            const std::string& onnx_file, const std::string& engine_file, const bool is_compile) :
    _engine(nullptr),
    _logger(Logger()),
    _stream(nullptr)
{
    // if engine file exists
    if (std::ifstream(engine_file) and is_compile == false) {
        std::cout << "Engine file exists, deserialize: " << engine_file << std::endl;
        if (deSerializeEngine(engine_file) != 0) {
            std::cout << "Could not deserialize engine" << std::endl;
            exit(1);
        }
    } else {
        std::cout << "compile onnx file: " << onnx_file << std::endl;
        if (compile(mode, max_batchsize, onnx_file, engine_file) != 0) {
            std::cout << "Could not compile engine" << std::endl;
            exit(1);
        }
        std::cout << "deserialize engine file: " << engine_file << std::endl;
        if (deSerializeEngine(engine_file) != 0) {
            std::cout << "Could not deserialize engine" << std::endl;
            exit(1);
        }
    }
    std::cout << "Engine init done" << std::endl;
}

Engine::~Engine() {
    if (_engine != nullptr) {
        _engine->destroy();
    }
}


int Engine::compile(const std::string& mode, const unsigned int& max_batchsize, 
            const std::string& onnx_file, const std::string& engine_file) {
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(_logger));
    if (builder == nullptr) {
        std::cout << "Could not create builder" << std::endl;
        return 1;
    }

    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    // config->setFlag(nvinfer1::BuilderFlag::kTF32);

    if (mode == "fp16") {
        if (!builder->platformHasFastFp16()) {
            std::cout << "Platform does not support fast FP16" << std::endl;
            return 1;
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    else if (mode == "int8") {
        if (!builder->platformHasFastInt8()) {
            std::cout << "Platform does not support Int8" << std::endl;
            return 1;
        }
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
    }
    else if (mode == "tf32") {
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
    std::unique_ptr<nvinfer1::INetworkDefinition> network;
    std::unique_ptr<nvonnxparser::IParser> onnx_parser;

	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));

    onnx_parser.reset(nvonnxparser::createParser(*network, _logger));
    if (onnx_parser == nullptr) {
        std::cout << "Could not create parser" << std::endl;
        return 1;
    }
    if (!onnx_parser->parseFromFile(onnx_file.c_str(), 1)) {
        std::cout << "Could not parse ONNX file" << std::endl;
        return 1;
    }
    builder->setMaxBatchSize(max_batchsize);

    const auto input = network->getInput(0);
    const auto output = network->getOutput(0);
    const auto inputname = input->getName();
    const auto inputdims = input->getDimensions();

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    if (profile == nullptr) {
        std::cout << "Could not create optimization profile" << std::endl;
        return 1;
    }
    profile->setDimensions(inputname, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, 224, 224));
    profile->setDimensions(inputname, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3, 224, 224));
    profile->setDimensions(inputname, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(max_batchsize, 3, 224, 224));
    config->addOptimizationProfile(profile);
    
    std::unique_ptr<nvinfer1::IHostMemory> host_memory(builder->buildSerializedNetwork(*network, *config));
    if (host_memory == nullptr) {
        std::cout << "Could not build serialized network" << std::endl;
        return 1;
    }
    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(_logger)};
    if (runtime == nullptr) {
        std::cout << "Could not create runtime" << std::endl;
        return 1;
    }
    std::unique_ptr<nvinfer1::ICudaEngine> engine{runtime->deserializeCudaEngine(host_memory->data(), host_memory->size())};

    if (engine == nullptr) {
        std::cout << "Could not create engine" << std::endl;
        return 1;
    }
    std::unique_ptr<nvinfer1::IHostMemory> model_data(engine->serialize());
    if (model_data == nullptr) {
        std::cout << "Could not serialize engine" << std::endl;
        return 1;
    }

    FILE* f = fopen(engine_file.c_str(), "wb");
    if (f == nullptr) {
        std::cout << "Could not open file for writing" << std::endl;
        return 1;
    }
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    return 0;
}


int Engine::deSerializeEngine(const std::string& enginefile) {
    std::ifstream file(enginefile, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cout << "Could not read file" << std::endl;
        return 1;
    }
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(_logger);
    // std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(_logger)};
    if (runtime == nullptr) {
        std::cout << "Could not create runtime" << std::endl;
        return 1;
    }
    // _engine.reset(runtime->deserializeCudaEngine(buffer.data(), size));
    _engine = runtime->deserializeCudaEngine(buffer.data(), size);
    if (_engine == nullptr) {
        std::cout << "Could not deserialize engine" << std::endl;
        return 1;
    }

    auto ret = cudaStreamCreate(&_stream);
    if (ret != cudaSuccess) {
        std::cout << "Could not create stream" << std::endl;
        return 1;
    }
    return 0;
}


// std::string Engine::inference(const std::string& sentence) {
//     return "";
//     // for (int i = 0; i < _engine->getNbBindings(); ++i) {
//     //     auto dims = _engine->getBindingDimensions(i);
//     //     if (_engine->bindingIsInput(i)) {
//     //         _context->setBindingDimensions(i, dims);
//     //     }
//     // }
//     // if (!_context->allInputDimensionsSpecified()) {
//     //     std::cout << "Input dimensions not specified" << std::endl;
//     //     return "";
//     // }
    
//     // const int& pad_idx = _source_vocab->get_idx(_pad);
//     // const int& bos_idx = _source_vocab->get_idx(_bos);
//     // const int& eos_idx = _source_vocab->get_idx(_eos);
//     // const int& unk_idx = _source_vocab->get_idx(_unk);
    
//     // std::vector<std::string> tokens = sentence2tokens(preprocess(sentence));
//     // std::vector<int> enc_ids = tokens2ids(tokens, _source_vocab);
//     // enc_ids.push_back(eos_idx);
//     // std::vector<int> dec_ids = {_target_vocab->get_idx(_bos)};

//     // samplesCommon::ManagedBuffer enc_ids_buff;
//     // samplesCommon::ManagedBuffer dec_ids_buff;
//     // samplesCommon::ManagedBuffer enc_self_mask_buff;
//     // samplesCommon::ManagedBuffer dec_self_mask_buff;
//     // samplesCommon::ManagedBuffer enc_dec_mask_buff;
//     // samplesCommon::ManagedBuffer output_buff;

//     // nvinfer1::Dims2 input_dims = {1, _max_len};
//     // nvinfer1::Dims3 mask_dims = {1, _max_len, _max_len};
//     // nvinfer1::Dims3 output_dims = {1, _max_len, _target_vocab->get_vocab_size()};
    
//     // enc_ids_buff.hostBuffer.resize(input_dims);
//     // enc_ids_buff.deviceBuffer.resize(input_dims);
//     // dec_ids_buff.hostBuffer.resize(input_dims);
//     // dec_ids_buff.deviceBuffer.resize(input_dims);
//     // enc_self_mask_buff.hostBuffer.resize(mask_dims);
//     // enc_self_mask_buff.deviceBuffer.resize(mask_dims);
//     // dec_self_mask_buff.hostBuffer.resize(mask_dims);
//     // dec_self_mask_buff.deviceBuffer.resize(mask_dims);
//     // enc_dec_mask_buff.hostBuffer.resize(mask_dims);
//     // enc_dec_mask_buff.deviceBuffer.resize(mask_dims);
//     // output_buff.hostBuffer.resize(output_dims);
//     // output_buff.deviceBuffer.resize(output_dims);

//     // // encoder process
//     // auto* enc_ids_data_buffer = static_cast<int*>(enc_ids_buff.hostBuffer.data());
//     // for (int i = 0; i < _max_len; ++i) {
//     //     if (i < enc_ids.size()) {
//     //         enc_ids_data_buffer[i] = enc_ids[i];
//     //     } else {
//     //         enc_ids_data_buffer[i] = pad_idx;
//     //     }
//     // }
//     // // for (int i = 0; i < _max_len; ++i) {
//     // //     std::cout << enc_ids_data_buffer[i] << " ";
//     // // }
//     // // std::cout << std::endl;

//     // auto* enc_self_mask_data_buffer = static_cast<bool*>(enc_self_mask_buff.hostBuffer.data());
//     // for (int i = 0; i < _max_len; ++i) {
//     //     for (int j = 0; j < _max_len; ++j) {
//     //         if (j < enc_ids.size()) {
//     //             enc_self_mask_data_buffer[i * _max_len + j] = false;
//     //         } else {
//     //             enc_self_mask_data_buffer[i * _max_len + j] = true;
//     //         }
//     //     }
//     // }

//     // // for (int i = 0; i < _max_len; ++i) {
//     // //     for (int j = 0; j < _max_len; ++j) {
//     // //         std::cout << enc_self_mask_data_buffer[i * _max_len + j] << " ";
//     // //     }
//     // //     std::cout << std::endl;
//     // // }
//     // std::vector<std::string> inference_tokens;
//     // std::string next_tokens;
//     // for (int step = 0; step < _max_len-1 and next_tokens != _eos; ++step) {
//     //     auto* dec_ids_data_buffer = static_cast<int*>(dec_ids_buff.hostBuffer.data());
//     //     for (int i = 0; i < _max_len; ++i) {
//     //         if (i < dec_ids.size()) {
//     //             dec_ids_data_buffer[i] = dec_ids[i];
//     //         } else {
//     //             dec_ids_data_buffer[i] = pad_idx;
//     //         }
//     //     }
        
//     //     auto* dec_self_mask_data_buffer = static_cast<bool*>(dec_self_mask_buff.hostBuffer.data());
//     //     for (int i = 0; i < _max_len; ++i) {
//     //         for (int j = 0; j < _max_len; ++j) {
//     //             if (j < dec_ids.size() || j <= i) {
//     //                 dec_self_mask_data_buffer[i * _max_len + j] = false;
//     //             } else {
//     //                 dec_self_mask_data_buffer[i * _max_len + j] = true;
//     //             }
//     //         }
//     //     }
        
//     //     auto* dec_enc_mask_data_buffer = static_cast<bool*>(enc_dec_mask_buff.hostBuffer.data());
//     //     for (int i = 0; i < _max_len; ++i) {
//     //         for (int j = 0; j < _max_len; ++j) {
//     //             if (j < enc_ids.size()) {
//     //                 dec_enc_mask_data_buffer[i * _max_len + j] = false;
//     //             } else {
//     //                 dec_enc_mask_data_buffer[i * _max_len + j] = true;
//     //             }
//     //         }
//     //     }
        
//     //     cudaMemcpyAsync(enc_ids_buff.deviceBuffer.data(), enc_ids_buff.hostBuffer.data(), enc_ids_buff.hostBuffer.nbBytes(), cudaMemcpyHostToDevice, _stream);
//     //     cudaMemcpyAsync(dec_ids_buff.deviceBuffer.data(), dec_ids_buff.hostBuffer.data(), dec_ids_buff.hostBuffer.nbBytes(), cudaMemcpyHostToDevice, _stream);
//     //     cudaMemcpyAsync(enc_self_mask_buff.deviceBuffer.data(), enc_self_mask_buff.hostBuffer.data(), enc_self_mask_buff.hostBuffer.nbBytes(), cudaMemcpyHostToDevice, _stream);
//     //     cudaMemcpyAsync(dec_self_mask_buff.deviceBuffer.data(), dec_self_mask_buff.hostBuffer.data(), dec_self_mask_buff.hostBuffer.nbBytes(), cudaMemcpyHostToDevice, _stream);
//     //     cudaMemcpyAsync(enc_dec_mask_buff.deviceBuffer.data(), enc_dec_mask_buff.hostBuffer.data(), enc_dec_mask_buff.hostBuffer.nbBytes(), cudaMemcpyHostToDevice, _stream);
//     //     // cudaMemcpy(enc_ids_buff.deviceBuffer.data(), enc_ids_buff.hostBuffer.data(), enc_ids_buff.hostBuffer.nbBytes(), cudaMemcpyHostToDevice);
//     //     // cudaMemcpy(dec_ids_buff.deviceBuffer.data(), dec_ids_buff.hostBuffer.data(), dec_ids_buff.hostBuffer.nbBytes(), cudaMemcpyHostToDevice);
//     //     // cudaMemcpy(enc_self_mask_buff.deviceBuffer.data(), enc_self_mask_buff.hostBuffer.data(), enc_self_mask_buff.hostBuffer.nbBytes(), cudaMemcpyHostToDevice);
//     //     // cudaMemcpy(dec_self_mask_buff.deviceBuffer.data(), dec_self_mask_buff.hostBuffer.data(), dec_self_mask_buff.hostBuffer.nbBytes(), cudaMemcpyHostToDevice);
//     //     // cudaMemcpy(enc_dec_mask_buff.deviceBuffer.data(), enc_dec_mask_buff.hostBuffer.data(), enc_dec_mask_buff.hostBuffer.nbBytes(), cudaMemcpyHostToDevice);

//     //     std::vector<void*> prediction_bindings = {
//     //         enc_ids_buff.deviceBuffer.data(),
//     //         dec_ids_buff.deviceBuffer.data(),
//     //         // enc_self_mask_buff.deviceBuffer.data(),
//     //         // dec_self_mask_buff.deviceBuffer.data(),
//     //         // enc_dec_mask_buff.deviceBuffer.data(),
//     //         output_buff.deviceBuffer.data()
//     //     };
//     //     bool status = _context->enqueueV2(prediction_bindings.data(), _stream, nullptr);
//     //     if (!status) {
//     //         std::cout << "Could not enqueue" << std::endl;
//     //         return "";
//     //     }
//     //     cudaMemcpyAsync(output_buff.hostBuffer.data(), output_buff.deviceBuffer.data(), output_buff.hostBuffer.nbBytes(), cudaMemcpyDeviceToHost, _stream);
//     //     // cudaMemcpyAsync(output_buff.hostBuffer.data(), output_buff.deviceBuffer.data(), output_buff.hostBuffer.nbBytes(), cudaMemcpyDeviceToHost);
//     //     auto* host_output_buffer = static_cast<float*>(output_buff.hostBuffer.data());

//     //     // cudaStreamSynchronize(_stream);
//     //     // std::cout << step << std::endl;
//     //     // for (int i = 0; i < _max_len; ++i) {
//     //     //     for (int j = 0; j < _target_vocab->get_vocab_size(); ++j) {
//     //     //         std::cout << host_output_buffer[i * _target_vocab->get_vocab_size() + j] << " ";
//     //     //     }
//     //     //     std::cout << std::endl;
//     //     // }
//     //     int max_idx = 0;
//     //     for (int j = 0; j < _target_vocab->get_vocab_size(); ++j) {
//     //         if (host_output_buffer[step * _target_vocab->get_vocab_size() + j] > host_output_buffer[step * _target_vocab->get_vocab_size() + max_idx]) {
//     //             max_idx = j;
//     //         }
//     //     }
//     //     dec_ids.push_back(max_idx);
//     //     next_tokens = _target_vocab->get_token(max_idx);
//     //     inference_tokens.push_back(next_tokens);
//     //     // std::cout << _target_vocab->get_token(max_idx) << std::endl;
//     // }
//     // if (inference_tokens.size() > 0) {
//     //     inference_tokens.pop_back();
//     // }
//     // return tokens2sentence(inference_tokens);
// }

Classifier::Classifier(std::shared_ptr<nvinfer1::ICudaEngine> engine) :
    _engine(engine),
    _stream(nullptr)
{
    auto ret = cudaStreamCreate(&_stream);
    if (ret != cudaSuccess) {
        std::cout << "Could not create stream" << std::endl;
        exit(1);
    }
    std::cout << "Classifier init done" << std::endl;
}

Classifier::~Classifier() {
    cudaStreamDestroy(_stream);
}

int Classifier::infer(const void* input, void* output) {

    return 0;
}


// test
int main(int argc, char** argv) {
    std::string model_path = "/home/tars/projects/code/inference_grpc/models/densenet_onnx/model.onnx";
    std::string device = "cuda";

    std::unique_ptr<Engine> engine = std::make_unique<Engine>(
        "tf32", 1, model_path, "model.engine", false);

    return 0;
}