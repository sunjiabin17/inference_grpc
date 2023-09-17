#include <argparse/argparse.hpp>

int main(int argc, char *argv[]) {
    argparse::ArgumentParser program("program_name");

    program.add_argument("square")
        .help("display the square of a given integer")
        .scan<'e', float>();

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    auto input = program.get<float>("square");
    std::cout << (input) << typeid(input).name() << std::endl;

    return 0;
}