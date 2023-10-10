#include "helper.h"
#include <fstream>
#include <vector>
#include <iostream>

#include "base64.h"




// convert image to base64 string
int image_to_base64(cv::Mat& img, std::string& base64_result) {
    std::vector<uchar> buffer;
    cv::imencode(".jpg", img, buffer);
    std::string img_base64 = base64_encode(buffer.data(), buffer.size(), false);
    base64_result = img_base64;
    return 0;
}

int base64_to_image(const std::string& base64_result, cv::Mat& img) {
    std::string img_base64 = base64_decode(base64_result);
    std::vector<uchar> data(img_base64.begin(), img_base64.end());
    img = cv::imdecode(data, cv::IMREAD_COLOR);
    return 0;
}


// int main() {
//     std::string img_file = "/home/tars/projects/code/inference_grpc/test/cat.jpg";
//     cv::Mat img = cv::imread(img_file);
//     if (img.empty()) {
//         std::cout << "Image is empty" << std::endl;
//         return -1;
//     }
//     std::cout << "Image size: " << img.size() << std::endl;
//     cv::Mat resized_img;
//     cv::resize(img, resized_img, cv::Size(224, 224));
//     cv::imwrite("resized.jpg", resized_img);
//     std::string img_base64;
//     std::cout << "encode image to base64 string" << std::endl;
//     image_to_base64(resized_img, img_base64);
//     std::cout << img_base64.size() << std::endl;
//     cv::Mat img_decode;
//     std::cout << "decode base64 string to image" << std::endl;
//     base64_to_image(img_base64, img_decode);
//     cv::imwrite("decode.jpg", img_decode);
// }