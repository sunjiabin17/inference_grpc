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

#include "my_grpc_service.pb.h"
#include "my_grpc_service.grpc.pb.h"
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using my_grpc_service::HelloRequest;
using my_grpc_service::HelloResponse;
using my_grpc_service::HelloWorld;

// Logic and data behind the server's behavior.
class HelloWorldServiceImpl final : public HelloWorld::Service {
    Status SayHello(ServerContext* context, const HelloRequest* request,
                    HelloResponse* response) override {
        std::string prefix("Hello ");
        // get current time, format it as HH:MM:SS
        std::time_t t = std::time(nullptr);
        std::tm *tm = std::localtime(&t);
        std::stringstream ss;
        ss << std::put_time(tm, "%H:%M:%S");
        std::string time_str = ss.str();
        response->set_message(prefix + request->name() + " at " + time_str);
        return Status::OK;
    }

};

void RunServer(uint16_t port) {
    std::string server_address("0.0.0.0:" + std::to_string(port));
    HelloWorldServiceImpl service;

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    builder.RegisterService(&service);
    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
}

int main(int argc, char** argv) {
    uint16_t port = 50051;
    RunServer(port);
    return 0;
}