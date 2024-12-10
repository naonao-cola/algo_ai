
#ifndef __ORT_UTILITY_H__
#define __ORT_UTILITY_H__
#if USE_ORT

#include "../../airuntime/inference.h"
#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"
#include <string>
#include <vector>

namespace ort::utils {
using Box      = Algo::Box;
using BoxArray = Algo::BoxArray;

enum
{
    CHW = 0,
    HWC = 1
};

std::wstring to_wstring(const std::string& str);

void permute(const cv::Mat& mat, float* tensor_value, unsigned int data_format = CHW);

cv::Mat normalize(const cv::Mat& mat, const float* mean, const float* scale);
cv::Mat normalize(const cv::Mat& mat, float mean, float scale);
void    normalize(const cv::Mat& inmat, cv::Mat& outmat, float mean, float scale);
void    normalize_inplace(cv::Mat& mat_inplace, float mean, float scale);
void    normalize_inplace(cv::Mat& mat_inplace, const float mean[3], const float scale[3]);

float iou_of(const Box& t_lsh, const Box& t_rsh);
void  blending_nms(BoxArray& input, BoxArray& output, float iou_threshold, unsigned int topk);
void  offset_nms(BoxArray& input, BoxArray& output, float iou_threshold, unsigned int topk);
void  hard_nms(BoxArray& input, BoxArray& output, float iou_threshold, unsigned int topk);

void resize_img_type0(const cv::Mat& img, cv::Mat& resize_img, int max_size_len, float& ratio_h, float& ratio_w);
/*
带了除255的归一化
*/
void normalize_img_type0(cv::Mat* im, const std::vector<float>& mean, const std::vector<float>& scale, const bool is_scale);

}  // namespace ort::utils
#endif  // !__ORT_UTILITY_H__


#endif //USE_ORT