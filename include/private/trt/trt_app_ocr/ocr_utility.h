#ifndef __OCR_UTILITY__
#define __OCR_UTILITY__

#include <future>
#include <math.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


#include "./clipper.h"
namespace OCR {
namespace utility {
// 预处理
void                     permute(const cv::Mat* im, float* data);
void                     permute_batch(const std::vector<cv::Mat> imgs, float* data);
void                     normalize(cv::Mat* im, const std::vector<float>& mean, const std::vector<float>& scale, const bool is_scale);
void                     resize_img_type0(const cv::Mat& img, cv::Mat& resize_img, int max_size_len, float& ratio_h, float& ratio_w);
void                     crnn_resize_img(const cv::Mat& img, cv::Mat& resize_img, float wh_ratio, const std::vector<int>& rec_image_shape);
void                     cls_resize_img(const cv::Mat& img, cv::Mat& resize_img, const std::vector<int>& rec_image_shape);
std::vector<std::string> read_dict(const std::string& path);
std::vector<int>         argsort(const std::vector<float>& array);

template <class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

// 后处理

template <class T>
inline T clamp(T x, T min, T max)
{
    if (x > max)
        return max;
    if (x < min)
        return min;
    return x;
}
float           BoxScoreFast(std::vector<std::vector<float>> box_array, cv::Mat pred);
void            GetContourArea(const std::vector<std::vector<float>>& box, float unclip_ratio, float& distance);
cv::RotatedRect UnClip(std::vector<std::vector<float>> box, const float& unclip_ratio);

bool                                       XsortFp32(std::vector<float> a, std::vector<float> b);
std::vector<std::vector<float>>            Mat2Vector(cv::Mat mat);
std::vector<std::vector<float>>            GetMiniBoxes(cv::RotatedRect box, float& ssid);
float                                      PolygonScoreAcc(std::vector<cv::Point> contour, cv::Mat pred);
std::vector<std::vector<std::vector<int>>> BoxesFromBitmap(const cv::Mat pred, const cv::Mat bitmap, const float& box_thresh, const float& det_db_unclip_ratio, const bool& use_polygon_score);
bool                                       XsortInt(std::vector<int> a, std::vector<int> b);
std::vector<std::vector<int>>              OrderPointsClockwise(std::vector<std::vector<int>> pts);
std::vector<std::vector<std::vector<int>>> FilterTagDetRes(std::vector<std::vector<std::vector<int>>> boxes, float ratio_h, float ratio_w, cv::Mat srcimg);
cv::Mat GetRotateCropImage(const cv::Mat& srcimage, std::vector<std::vector<int>> box);

}  // namespace utility
};  // namespace OCR

#endif  // !__OCR_UTILITY__
