#include "cxxopts.hpp"
#include <iostream>

// ref: https://github.com/jarro2783/cxxopts
int main(int argc, char *argv[]) {
    cxxopts::Options options("MyProgram", "One line description of MyProgram");

    options.add_options()
        ("d,debug", "Enable debugging") // a bool parameter
        ("i,integer", "Int param", cxxopts::value<int>())
        ("f,file", "File name", cxxopts::value<std::string>())
        ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"));

    auto result = options.parse(argc, argv);

    bool debug = result["debug"].as<bool>();
    int integer = result["integer"].as<int>();
    std::string file = result["file"].as<std::string>();
    bool verbose = result["verbose"].as<bool>();

    std::cout << "debug = " << debug << std::endl;
    std::cout << "integer = " << integer << std::endl;
    std::cout << "file = " << file << std::endl;
    std::cout << "verbose = " << verbose << std::endl;

    return 0;
}