#include <iostream>
#include <string>
#include <fstream>
#include <grpcpp/grpcpp.h>
#include <opencv2/opencv.hpp>

#include "grpc_infer_service.pb.h"
#include "grpc_infer_service.grpc.pb.h"
#include "cxxopts.hpp"
#include "helper.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using grpc_infer_service::Request;
using grpc_infer_service::Response;
using grpc_infer_service::InferenceService;

class InferenceClient {
public:
    InferenceClient(std::shared_ptr<Channel> channel)
        : stub_(InferenceService::NewStub(channel)) {}

    // Assembles the client's payload, sends it and presents the response back
    // from the server.
    std::string GetImgClsResult(const std::string& data) {
        // Data we are sending to the server.
        Request request;
        request.set_name(data);

        // Container for the data we expect from the server.
        Response response;

        // Context for the client. It could be used to convey extra information to
        // the server and/or tweak certain RPC behaviors.
        ClientContext context;

        // The actual RPC.
        Status status = stub_->GetImgClsResult(&context, request, &response);

        // Act upon its status.
        if (status.ok()) {
            return response.message();
        }
        else {
            std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
            return "RPC failed";
        }
    }

private:
    std::unique_ptr<InferenceService::Stub> stub_;
};


int main(int argc, char** argv) {
    uint16_t port = 50051;
    std::string img_file = "/home/tars/projects/code/inference_grpc/test/cat.jpg";
    
    cxxopts::Options options("gRPC Server", "gRPC Server for Inference"); 
    options.add_options()
        ("p,port", "Port", cxxopts::value<uint16_t>()->default_value(std::to_string(port)))
        ("img", "Image File", cxxopts::value<std::string>()->default_value(img_file))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);
    
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    port = result["port"].as<uint16_t>();
    img_file = result["img"].as<std::string>();
    std::cout << "img_file: " << img_file << std::endl;
    
    cv::Mat img = cv::imread(img_file);
    if (img.empty()) {
        std::cout << "Image is empty" << std::endl;
        return -1;
    }
    std::cout << "Image size: " << img.size() << std::endl;

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(224, 224));
    std::string img_base64;
    image_to_base64(resized_img, img_base64);

    std::string target = "localhost:" + std::to_string(port);
    InferenceClient client(
        grpc::CreateChannel(target, grpc::InsecureChannelCredentials()));
    
    std::string response = client.GetImgClsResult(img_base64);
    std::cout << "gRPC receive: " << response << std::endl;

    return 0;
}