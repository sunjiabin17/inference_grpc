#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <ctime>

#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <opencv2/opencv.hpp>

#include "grpc_infer_service.pb.h"
#include "grpc_infer_service.grpc.pb.h"

#include "img_classify_tensorrt.h"
#include "engine.h"
#include "cxxopts.hpp"
#include "helper.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using grpc_infer_service::Request;
using grpc_infer_service::Response;
using grpc_infer_service::InferenceService;

// Logic and data behind the server's behavior.
class InferenceServiceImpl final : public InferenceService::Service {
public:
    InferenceServiceImpl(const std::string& mode, const unsigned int& max_batchsize, 
            const std::string& onnx_file, const std::string& engine_file, const std::string label_file)
            : _infer_engine(nullptr) {
        // 初始化推理引擎
        _infer_engine = std::make_unique<InferenceEngine>(mode, max_batchsize, onnx_file, engine_file, label_file);
    }

private:
    Status GetImgClsResult(ServerContext* context, const Request* request,
        Response* response) override {
        // std::string prefix("Hello ");
        // // get current time, format it as HH:MM:SS
        // std::time_t t = std::time(nullptr);
        // std::tm *tm = std::localtime(&t);
        // std::stringstream ss;
        // ss << std::put_time(tm, "%H:%M:%S");
        // std::string time_str = ss.str();
        // response->set_message(prefix + request->name() + " at " + time_str);
        // return Status::OK;
        std::string base64_img_str = request->name();
        cv::Mat img;
        base64_to_image(base64_img_str, img);
        if (img.empty()) {
            LOG_ERROR("Could not read image");
            return Status::CANCELLED;
        }        

        // 将图像转换为 TensorRT 引擎输入格式
        const int batch_size = 1;
        const int channels = 3;
        const int height = 224;
        const int width = 224;
        const int input_size = batch_size * channels * height * width;
        
        float* input_data = new float[input_size];
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int idx = c * height * width + h * width + w;
                    input_data[idx] = img.at<cv::Vec3b>(h, w)[c] / 255.0f;
                }
            }
        }

        float* output_data = new float[batch_size * 1000];
        
        _infer_engine->infer(input_data, output_data);

        float* max_element = std::max_element(output_data, output_data + 1000);
        int max_idx = max_element - output_data;
        // std::cout << "max element: " << *max_element << std::endl;
        // std::cout << "max index: " << max_idx << std::endl;
        // std::cout << "label: " << _infer_engine->get_label(max_idx) << std::endl;
        InferenceEngine* infer_engine = dynamic_cast<InferenceEngine*>(_infer_engine.get());
        response->set_message("result: " + infer_engine->get_label(max_idx));
        return Status::OK;
    }

private:
    std::unique_ptr<Engine> _infer_engine;
};

void RunServer(uint16_t port, const std::string& mode, const unsigned int& max_batchsize, 
        const std::string& onnx_file, const std::string& engine_file, const std::string label_file) {
    std::string server_address("0.0.0.0:" + std::to_string(port));
    InferenceServiceImpl service(mode, max_batchsize, onnx_file, engine_file, label_file);

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    builder.RegisterService(&service);
    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    LOG_INFO("Server listening on " + server_address);

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
}

int main(int argc, char** argv) {
    uint16_t port = 50051;
    std::string onnx_file = "/home/tars/projects/code/inference_grpc/models/densenet_onnx/model.onnx";
    std::string label_file = "/home/tars/projects/code/inference_grpc/models/densenet_onnx/densenet_labels.txt";
    std::string engine_file = "/home/tars/projects/code/inference_grpc/build/model.engine";
    std::string mode = "tf32";
    unsigned int max_batchsize = 1;
    
    cxxopts::Options options("gRPC Server", "gRPC Server for Inference"); 
    options.add_options()
        ("p,port", "Port", cxxopts::value<uint16_t>()->default_value(std::to_string(port)))
        ("onnx_file", "ONNX File", cxxopts::value<std::string>()->default_value(onnx_file))
        ("label_file", "Label File", cxxopts::value<std::string>()->default_value(label_file))
        ("engine_file", "Engine File", cxxopts::value<std::string>()->default_value(engine_file))
        ("mode", "Mode", cxxopts::value<std::string>()->default_value(mode))
        ("max_batchsize", "Max Batch Size", cxxopts::value<unsigned int>()->default_value(std::to_string(max_batchsize)))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);
    
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    port = result["port"].as<uint16_t>();
    onnx_file = result["onnx_file"].as<std::string>();
    label_file = result["label_file"].as<std::string>();
    engine_file = result["engine_file"].as<std::string>();
    mode = result["mode"].as<std::string>();
    max_batchsize = result["max_batchsize"].as<unsigned int>();

    RunServer(port, mode, max_batchsize, onnx_file, engine_file, label_file);
    return 0;
}