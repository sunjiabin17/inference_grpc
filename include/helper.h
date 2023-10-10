#pragma once

#include <opencv2/opencv.hpp>
#include <string>

int image_to_base64(cv::Mat& img, std::string& base64_result);

int base64_to_image(const std::string& base64_result, cv::Mat& img);