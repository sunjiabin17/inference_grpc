#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <ctime>

#define LOG_INFO(msg) \
    do { \
        std::time_t t = std::time(nullptr); \
        std::cout << "[" << std::put_time(std::localtime(&t), "%H:%M:%S") << "] " \
            << "[INFO] " << __FILE__ << ":" << __LINE__ << "\t" << msg << std::endl; \
    } while (0)

#define LOG_ERROR(msg) \
    do { \
        std::time_t t = std::time(nullptr); \
        std::cerr << "[" << std::put_time(std::localtime(&t), "%H:%M:%S") << "] " \
            << "[ERROR] " << __FILE__ << ":" << __LINE__ << "\t" << msg << std::endl; \
    } while (0)

int image_to_base64(cv::Mat& img, std::string& base64_result);

int base64_to_image(const std::string& base64_result, cv::Mat& img);