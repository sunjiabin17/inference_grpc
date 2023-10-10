#include <iostream>
#include <string>
#include <sstream>
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
#include "cxxopts.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using grpc_infer_service::Request;
using grpc_infer_service::Response;
using grpc_infer_service::InferenceService;

// Logic and data behind the server's behavior.
class InferenceServiceImpl final : public InferenceService::Service {
    InferenceServiceImpl(const std::string& mode, const unsigned int& max_batchsize, 
            const std::string& onnx_file, const std::string& engine_file, const std::string label_file) {
        _infer_engine = std::make_unique<InferenceEngine>(mode, max_batchsize, onnx_file, engine_file, label_file);
    }
    Status GetImgClsResult(ServerContext* context, const Request* request,
                    Response* response) override;

private:
    std::unique_ptr<InferenceEngine> _infer_engine;

};
