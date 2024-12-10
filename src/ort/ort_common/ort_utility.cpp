#ifdef _WIN32
#if USE_ORT
#include "../../../include/private/ort/ort_common/ort_utility.h"
#include "../../../include/private/airuntime/inference.h"
#include <locale.h>

namespace ort::utils {

std::wstring to_wstring(const std::string& str)
{
    unsigned len = str.size() * 2;
    setlocale(LC_CTYPE, "");
    auto* p = new wchar_t[len];
    mbstowcs(p, str.c_str(), len);
    std::wstring wstr(p);
    delete[] p;
    return wstr;
}

void permute(const cv::Mat& mat, float* tensor_value, unsigned int data_format)
{
    const unsigned int rows     = mat.rows;
    const unsigned int cols     = mat.cols;
    const unsigned int channels = mat.channels();

    cv::Mat mat_ref;
    if (mat.type() != CV_32FC(channels))
        mat.convertTo(mat_ref, CV_32FC(channels));
    else
        mat_ref = mat;

    // CXHXW
    if (data_format == CHW) {
        for (int i = 0; i < channels; ++i) {
            cv::extractChannel(
                mat, cv::Mat(rows, cols, CV_32FC1, tensor_value + i * rows * cols),
                i);
        }
        return;
    }
    if (data_format == HWC) {
        std::memcpy(tensor_value, mat.data, rows * cols * channels * sizeof(float));
        return;
    }
}

cv::Mat normalize(const cv::Mat& mat, const float* mean, const float* scale)
{
    cv::Mat mat_copy;
    if (mat.type() != CV_32FC3)
        mat.convertTo(mat_copy, CV_32FC3);
    else
        mat_copy = mat.clone();
    for (unsigned int i = 0; i < mat_copy.rows; ++i) {
        cv::Vec3f* p = mat_copy.ptr<cv::Vec3f>(i);
        for (unsigned int j = 0; j < mat_copy.cols; ++j) {
            p[j][0] = (p[j][0] - mean[0]) * scale[0];
            p[j][1] = (p[j][1] - mean[1]) * scale[1];
            p[j][2] = (p[j][2] - mean[2]) * scale[2];
        }
    }
    return mat_copy;
}

cv::Mat normalize(const cv::Mat& mat, float mean, float scale)
{
    cv::Mat matf;
    if (mat.type() != CV_32FC3)
        mat.convertTo(matf, CV_32FC3);
    else
        matf = mat;  // reference
    return (matf - mean) * scale;
}
void normalize(const cv::Mat& inmat, cv::Mat& outmat, float mean, float scale)
{
    outmat = ort::utils::normalize(inmat, mean, scale);
}

void normalize_inplace(cv::Mat& mat_inplace, float mean, float scale)
{
    if (mat_inplace.type() != CV_32FC3)
        mat_inplace.convertTo(mat_inplace, CV_32FC3);
    ort::utils::normalize(mat_inplace, mat_inplace, mean, scale);
}

void normalize_inplace(cv::Mat& mat_inplace, const float mean[3], const float scale[3])
{
    if (mat_inplace.type() != CV_32FC3)
        mat_inplace.convertTo(mat_inplace, CV_32FC3);
    for (unsigned int i = 0; i < mat_inplace.rows; ++i) {
        cv::Vec3f* p = mat_inplace.ptr<cv::Vec3f>(i);
        for (unsigned int j = 0; j < mat_inplace.cols; ++j) {
            p[j][0] = (p[j][0] - mean[0]) * scale[0];
            p[j][1] = (p[j][1] - mean[1]) * scale[1];
            p[j][2] = (p[j][2] - mean[2]) * scale[2];
        }
    }
}

float iou_of(const Box& t_lsh, const Box& t_rsh)
{

    float inner_x1 = std::max(t_lsh.left, t_rsh.left);
    float inner_y1 = std::max(t_lsh.top, t_rsh.top);
    float inner_x2 = std::min(t_lsh.right, t_rsh.right);
    float inner_y2 = std::min(t_lsh.bottom, t_rsh.bottom);
   

    float inner_h = inner_y2 - inner_y1 + 1.0f;
    float inner_w = inner_x2 - inner_x1 + 1.0f;

    if (inner_h <= 0.f || inner_w <= 0.f)
        return std::numeric_limits<float>::min();
    float inner_area = inner_h * inner_w;
    float lsh_area   = std::max(0.0f, (t_lsh.right - t_lsh.left)) * std::max(0.0f, (t_lsh.bottom - t_lsh.top));
    float rsh_area = std::max(0.0f, (t_rsh.right - t_rsh.left)) *std::max(0.0f, (t_rsh.bottom - t_rsh.top));
    return float(inner_area / (lsh_area + rsh_area - inner_area));
}

void blending_nms(BoxArray& input, BoxArray& output, float iou_threshold, unsigned int topk)
{
    if (input.empty())
        return;
    std::sort(input.begin(), input.end(), [](const Box& a, const Box& b) {
        return a.confidence > b.confidence;
    });

    const unsigned int box_num = input.size();
    std::vector<int>   merged(box_num, 0);

    unsigned int count = 0;
    for (unsigned int i = 0; i < box_num; ++i) {
        if (merged[i])
            continue;
        BoxArray buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        for (unsigned int j = i + 1; j < box_num; ++j) {
            if (merged[j])
                continue;
            float iou = iou_of(input[i], input[j]);
            if (iou > iou_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }

        float total = 0.f;
        for (unsigned int k = 0; k < buf.size(); ++k) {
            total += std::exp(buf[k].confidence);
        }
        Box rects;
        for (unsigned int l = 0; l < buf.size(); ++l) {
            float rate = std::exp(buf[l].confidence) / total;
            rects.left += buf[l].left * rate;
            rects.top += buf[l].top * rate;
            rects.right += buf[l].right * rate;
            rects.bottom += buf[l].bottom * rate;
            rects.confidence += buf[l].confidence * rate;
        }

        output.push_back(rects);

        // keep top k
        count += 1;
        if (count >= topk)
            break;
    }
}

void offset_nms(BoxArray& input, BoxArray& output, float iou_threshold, unsigned int topk)
{
    if (input.empty())
        return;
    std::sort(input.begin(), input.end(), [](const Box& a, const Box& b) {
        return a.confidence > b.confidence;
    });
    const unsigned int box_num = input.size();
    std::vector<int>   merged(box_num, 0);

    const float offset = 4096.f;
    /** Add offset according to classes.
     * That is, separate the boxes into categories, and each category performs its
     * own NMS operation. The same offset will be used for those predicted to be
     * of the same category. Therefore, the relative positions of boxes of the
     * same category will remain unchanged. Box of different classes will be
     * farther away after offset, because offsets are different. In this way, some
     * overlapping but different categories of entities are not filtered out by
     * the NMS. Very clever!
     */
    for (unsigned int i = 0; i < box_num; ++i) {
        input[i].left += static_cast<float>(input[i].class_label) * offset;
        input[i].top += static_cast<float>(input[i].class_label) * offset;
        input[i].right += static_cast<float>(input[i].class_label) * offset;
        input[i].bottom += static_cast<float>(input[i].class_label) * offset;
    }

    unsigned int count = 0;
    for (unsigned int i = 0; i < box_num; ++i) {
        if (merged[i])
            continue;
        BoxArray buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        for (unsigned int j = i + 1; j < box_num; ++j) {
            if (merged[j])
                continue;

            float iou = iou_of(input[i], input[j]);

            if (iou > iou_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        output.push_back(buf[0]);

        // keep top k
        count += 1;
        if (count >= topk)
            break;
    }

    /** Substract offset.*/
    if (!output.empty()) {
        for (unsigned int i = 0; i < output.size(); ++i) {
            output[i].left -= static_cast<float>(output[i].class_label) * offset;
            output[i].top -= static_cast<float>(output[i].class_label) * offset;
            output[i].right -= static_cast<float>(output[i].class_label) * offset;
            output[i].bottom -= static_cast<float>(output[i].class_label) * offset;
        }
    }
}

void hard_nms(BoxArray& input, BoxArray& output, float iou_threshold, unsigned int topk)
{
    if (input.empty())
        return;
    std::sort(input.begin(), input.end(), [](const Box& a, const Box& b) {
        return a.confidence > b.confidence;
    });
    const unsigned int box_num = input.size();
    std::vector<int>   merged(box_num, 0);

    unsigned int count = 0;
    for (unsigned int i = 0; i < box_num; ++i) {
        if (merged[i])
            continue;
        BoxArray buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        for (unsigned int j = i + 1; j < box_num; ++j) {
            if (merged[j])
                continue;

            float iou = iou_of(input[i], input[j]);

            if (iou > iou_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        output.push_back(buf[0]);

        // keep top k
        count += 1;
        if (count >= topk)
            break;
    }
}

// 图像的宽和高处理为32的倍数后输出，最长不会超出最大预设值
void resize_img_type0(const cv::Mat& img, cv::Mat& resize_img, int max_size_len, float& ratio_h, float& ratio_w)
{
    int   w      = img.cols;
    int   h      = img.rows;
    float ratio  = 1.f;
    int   max_wh = w >= h ? w : h;
    // 处理图像最长处不能超出预设值
    if (max_wh > max_size_len) {
        if (h > w) {
            ratio = float(max_size_len) / float(h);
        }
        else {
            ratio = float(max_size_len) / float(w);
        }
    }
    int resize_h = int(float(h) * ratio);
    int resize_w = int(float(w) * ratio);
    // 除32，余下的超过16补全为32，不足16的长度舍弃
    resize_h = (std::max)(int(round(float(resize_h) / 32) * 32), 32);
    resize_w = (std::max)(int(round(float(resize_w) / 32) * 32), 32);

    cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
    ratio_h = float(resize_h) / float(h);
    ratio_w = float(resize_w) / float(w);
}
void normalize_img_type0(cv::Mat* im, const std::vector<float>& mean, const std::vector<float>& scale, const bool is_scale)
{
    double e = 1.0;
    if (is_scale) {
        e /= 255.0;
    }
    (*im).convertTo(*im, CV_32FC3, e);
    std::vector<cv::Mat> bgr_channels(3);
    cv::split(*im, bgr_channels);
    for (auto i = 0; i < bgr_channels.size(); i++) {
        bgr_channels[i].convertTo(bgr_channels[i], CV_32FC1, 1.0 * scale[i], (0.0 - mean[i]) * scale[i]);
    }
    cv::merge(bgr_channels, *im);
}
}  // namespace ort::utils

#endif //USE_ORT
#elif __linux__
#endif