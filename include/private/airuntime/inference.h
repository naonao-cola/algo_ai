#ifndef __INFERENCE_H__
#define __INFERENCE_H__

#include <string>
#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "./AIRuntimeDefines.h"

namespace Algo {
    using json = nlohmann::json;
    struct Box {
        /// @brief yolo
        float left;
        float top;
        float right;
        float bottom;
        float confidence;

        /// @brief segmentation
        std::vector<cv::Point> contour;

        /// @brief classification && obb
        float center_x, center_y, width, height, angle;
        int class_label;

        /// @brief OCR
        //rec 
        float ocr_ret_score;
        std::string ocr_ret_str;
        std::vector<std::pair<std::string, double>> rec_conf_res;
        std::vector<float>                          rec_ratio_res;
        std::vector<std::vector<cv::Point>>         ocr_single_pos;
        std::vector<int>                            ocr_char_index;
        //det
        std::vector<std::vector<int>> ocr_det;
        cv::Mat detMat;
        /// @brief MSAE
        cv::Mat                       msae_img;

        /// @brief Anomalib
        cv::Mat heatMap;
        //Box() = default;
        Box()
        {
            left = -1;
            top = -1;
            right = -1;
            bottom = -1;
            confidence = -1;
            center_x = -1;
            center_y = -1;
            width    = -1;
            height   = -1;
            ocr_ret_score = -1;
            ocr_ret_str   = "";
        }
        Box(float left, float top, float right, float bottom, float confidence, int class_label):left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label) {}
        Box(float score, std::string const& str) :left(0.f), top(0.f), right(0.f), bottom(0.f), confidence(0.f), class_label(0), ocr_ret_score(score), ocr_ret_str(str) {}
        
        Box(float center_x, float center_y, float width, float height, float angle, float confidence, int class_label)
        :center_x(center_x), center_y(center_y), width(width), height(height), angle(angle), confidence(confidence), class_label(class_label){}
    };

    template<typename Type=Box>
    class BoxArrayT :public std::vector<Type>{
    public:
        long long total_time{-1};
        long long pre_time{-1};
        long long host_time{-1};
        long long infer_time{-1};
        cv::Mat index_mat;
        cv::Mat prob_mat;
        Type& operator[](const int i) {
            return *(std::vector<Type>::begin() + i);
        }
    
    };

    using BoxArray = BoxArrayT<Box>;

    class Infer {
    public:
        virtual std::shared_future<BoxArray> commit(const cv::Mat& image) = 0;
        virtual std::vector<std::shared_future<BoxArray>> commits(const std::vector<cv::Mat>& images) = 0;
        virtual json infer_info() = 0;
        virtual bool set_param(const json& config) = 0;
    };
};

#endif // __INFERENCE_H__