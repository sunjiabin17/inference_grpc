#include <iostream>
#include <string>

#include <grpcpp/grpcpp.h>

// #include "my_grpc_service.pb.h"
#include "my_grpc_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using my_grpc_service::Request;
using my_grpc_service::Response;
using my_grpc_service::MyGrpcService;

class HelloWorldClient {
public:
    HelloWorldClient(std::shared_ptr<Channel> channel)
        : stub_(MyGrpcService::NewStub(channel)) {}

    // Assembles the client's payload, sends it and presents the response back
    // from the server.
    std::string GetResult(const std::string& user) {
        // Data we are sending to the server.
        Request request;
        request.set_name(user);

        // Container for the data we expect from the server.
        Response response;

        // Context for the client. It could be used to convey extra information to
        // the server and/or tweak certain RPC behaviors.
        ClientContext context;

        // The actual RPC.
        Status status = stub_->GetResult(&context, request, &response);

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
    std::unique_ptr<MyGrpcService::Stub> stub_;
};

int main(int argc, char** argv) {
    std::string target_str = "localhost:50051";
    HelloWorldClient client(
        grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
    
    std::string user("world");
    std::string response = client.GetResult(user);
    std::cout << "Greeter received: " << response << std::endl;

    return 0;
}