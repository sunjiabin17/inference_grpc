#include "helper.h"
#include <fstream>
#include <vector>
#include <iostream>


int main(int argc, char** argv) {
    std::string img_file(argv[1]);
    cv::Mat img = cv::imread(img_file);
    if (img.empty()) {
        std::cout << "Image is empty" << std::endl;
        return -1;
    }
    std::cout << "Image size: " << img.size() << std::endl;
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(224, 224));
    cv::imwrite("resized.jpg", resized_img);
    std::string img_base64;
    std::cout << "encode image to base64 string" << std::endl;
    image_to_base64(resized_img, img_base64);
    std::cout << img_base64.size() << std::endl;
    cv::Mat img_decode;
    std::cout << "decode base64 string to image" << std::endl;
    base64_to_image(img_base64, img_decode);
    cv::imwrite("decode.jpg", img_decode);
}