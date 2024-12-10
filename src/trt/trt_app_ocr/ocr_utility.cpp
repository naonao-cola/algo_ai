

#include "../../../include/private/trt/trt_app_ocr/ocr_utility.h"
#include <future>
#include <math.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#ifdef _WIN32
#elif __linux__
#include <fstream>
#endif
// #include "clipper."
namespace OCR {
namespace utility {
// �˲��ֲο��ٷ��ĵ�
// Ŀ¼�ṹΪpaddleocr/deploy/cpp_infer/src
void permute(const cv::Mat* im, float* data)
{
    int rh = im->rows;
    int rw = im->cols;
    int rc = im->channels();
    for (int i = 0; i < rc; ++i) {
        cv::extractChannel(*im, cv::Mat(rh, rw, CV_32FC1, data + i * rh * rw), i);
    }
}

void permute_batch(const std::vector<cv::Mat> imgs, float* data)
{
    for (int j = 0; j < imgs.size(); j++) {
        int rh = imgs[j].rows;
        int rw = imgs[j].cols;
        int rc = imgs[j].channels();
        for (int i = 0; i < rc; ++i) {
            cv::extractChannel(
                imgs[j], cv::Mat(rh, rw, CV_32FC1, data + (j * rc + i) * rh * rw), i);
        }
    }
}

void normalize(cv::Mat* im, const std::vector<float>& mean, const std::vector<float>& scale, const bool is_scale)
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

// ͼ��Ŀ��͸ߴ���Ϊ32�ı��������������ᳬ�����Ԥ��ֵ
void resize_img_type0(const cv::Mat& img, cv::Mat& resize_img, int max_size_len, float& ratio_h, float& ratio_w)
{
    int   w      = img.cols;
    int   h      = img.rows;
    float ratio  = 1.f;
    int   max_wh = w >= h ? w : h;
    // ����ͼ��������ܳ���Ԥ��ֵ
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
    // ��32�����µĳ���16��ȫΪ32������16�ĳ�������
    resize_h = (std::max)(int(round(float(resize_h) / 32) * 32), 32);
    resize_w = (std::max)(int(round(float(resize_w) / 32) * 32), 32);

    cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
    ratio_h = float(resize_h) / float(h);
    ratio_w = float(resize_w) / float(w);
}

// ��ͼƬ���߱ȴ���Ԥ��ֵ������Ϊ [imgH�� imgH * wh_ratio]
// ��ͼƬ���߱�С��Ԥ��ֵ������Ϊ [imgH�� imgH * wh_ratio]�����Ȳ��㣬���Ҳ�������
void crnn_resize_img(const cv::Mat& img, cv::Mat& resize_img, float wh_ratio, const std::vector<int>& rec_image_shape)
{
    int imgC, imgH, imgW;
    imgC = rec_image_shape[0];
    imgH = rec_image_shape[1];
    imgW = rec_image_shape[2];

    imgW = int(imgH * wh_ratio);

    float ratio = float(img.cols) / float(img.rows);
    int   resize_w, resize_h;

    if (ceilf(imgH * ratio) > imgW)
        resize_w = imgW;
    else
        resize_w = int(ceilf(imgH * ratio));

    cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f, cv::INTER_LINEAR);
    cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0, int(imgW - resize_img.cols), cv::BORDER_CONSTANT, { 127, 127, 127 });
}

void cls_resize_img(const cv::Mat& img, cv::Mat& resize_img, const std::vector<int>& rec_image_shape)
{
    int imgC, imgH, imgW;
    imgC = rec_image_shape[0];
    imgH = rec_image_shape[1];
    imgW = rec_image_shape[2];

    float ratio = float(img.cols) / float(img.rows);
    int   resize_w;
    if (ceilf(imgH * ratio) > imgW)
        resize_w = imgW;
    else
        resize_w = int(ceilf(imgH * ratio));

    cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f, cv::INTER_LINEAR);
    if (resize_w < imgW) {
        cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0, imgW - resize_w, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }
}

std::vector<std::string> read_dict(const std::string& path)
{
    std::ifstream            in(path);
    std::string              line;
    std::vector<std::string> m_vec;
    if (in) {
        while (getline(in, line)) {
            m_vec.push_back(line);
        }
    }
    else {
        std::cout << "no such label file: " << path << ", exit the program..."
                  << std::endl;
        exit(1);
    }
    return m_vec;
}
std::vector<int> argsort(const std::vector<float>& array)
{
    const int        array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i)
        array_index[i] = i;

    std::sort(
        array_index.begin(), array_index.end(),
        [&array](int pos1, int pos2) { return (array[pos1] < array[pos2]); });
    return array_index;
}

inline float clampf(float x, float min, float max)
{
    if (x > max)
        return max;
    if (x < min)
        return min;
    return x;
}
inline int _max(int a, int b) { return a >= b ? a : b; }

inline int _min(int a, int b) { return a >= b ? b : a; }

float BoxScoreFast(std::vector<std::vector<float>> box_array, cv::Mat pred)
{
    auto array  = box_array;
    int  width  = pred.cols;
    int  height = pred.rows;

    float box_x[4] = { array[0][0], array[1][0], array[2][0], array[3][0] };
    float box_y[4] = { array[0][1], array[1][1], array[2][1], array[3][1] };

    int xmin = clamp(int(std::floor(*(std::min_element(box_x, box_x + 4)))), 0, width - 1);
    int xmax = clamp(int(std::ceil(*(std::max_element(box_x, box_x + 4)))), 0, width - 1);
    int ymin = clamp(int(std::floor(*(std::min_element(box_y, box_y + 4)))), 0, height - 1);
    int ymax = clamp(int(std::ceil(*(std::max_element(box_y, box_y + 4)))), 0, height - 1);

    cv::Mat mask;
    mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

    cv::Point root_point[4];
    root_point[0]           = cv::Point(int(array[0][0]) - xmin, int(array[0][1]) - ymin);
    root_point[1]           = cv::Point(int(array[1][0]) - xmin, int(array[1][1]) - ymin);
    root_point[2]           = cv::Point(int(array[2][0]) - xmin, int(array[2][1]) - ymin);
    root_point[3]           = cv::Point(int(array[3][0]) - xmin, int(array[3][1]) - ymin);
    const cv::Point* ppt[1] = { root_point };
    int              npt[]  = { 4 };
    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

    cv::Mat croppedImg;
    pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
        .copyTo(croppedImg);

    auto score = cv::mean(croppedImg, mask)[0];
    return score;
}

void GetContourArea(const std::vector<std::vector<float>>& box, float unclip_ratio, float& distance)
{
    int   pts_num = 4;
    float area    = 0.0f;
    float dist    = 0.0f;
    for (int i = 0; i < pts_num; i++) {
        area += box[i][0] * box[(i + 1) % pts_num][1] -
                box[i][1] * box[(i + 1) % pts_num][0];
        dist += sqrtf((box[i][0] - box[(i + 1) % pts_num][0]) * (box[i][0] - box[(i + 1) % pts_num][0]) + (box[i][1] - box[(i + 1) % pts_num][1]) * (box[i][1] - box[(i + 1) % pts_num][1]));
    }
    area = fabs(float(area / 2.0));

    distance = area * unclip_ratio / dist;
}

cv::RotatedRect UnClip(std::vector<std::vector<float>> box, const float& unclip_ratio)
{
    float distance = 1.0;

    GetContourArea(box, unclip_ratio, distance);

    ClipperLib::ClipperOffset offset;
    ClipperLib::Path          p;
    p << ClipperLib::IntPoint(int(box[0][0]), int(box[0][1]))
      << ClipperLib::IntPoint(int(box[1][0]), int(box[1][1]))
      << ClipperLib::IntPoint(int(box[2][0]), int(box[2][1]))
      << ClipperLib::IntPoint(int(box[3][0]), int(box[3][1]));
    offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

    ClipperLib::Paths soln;
    offset.Execute(soln, distance);
    std::vector<cv::Point2f> points;

    for (int j = 0; j < soln.size(); j++) {
        for (int i = 0; i < soln[soln.size() - 1].size(); i++) {
            points.emplace_back(soln[j][i].X, soln[j][i].Y);
        }
    }
    cv::RotatedRect res;
    if (points.size() <= 0) {
        res = cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f(1, 1), 0);
    }
    else {
        res = cv::minAreaRect(points);
    }
    return res;
}

bool XsortFp32(std::vector<float> a, std::vector<float> b)
{
    if (a[0] != b[0])
        return a[0] < b[0];
    return false;
}

std::vector<std::vector<float>> Mat2Vector(cv::Mat mat)
{
    std::vector<std::vector<float>> img_vec;
    std::vector<float>              tmp;

    for (int i = 0; i < mat.rows; ++i) {
        tmp.clear();
        for (int j = 0; j < mat.cols; ++j) {
            tmp.push_back(mat.at<float>(i, j));
        }
        img_vec.push_back(tmp);
    }
    return img_vec;
}

std::vector<std::vector<float>> GetMiniBoxes(cv::RotatedRect box, float& ssid)
{
    ssid = std::max(box.size.width, box.size.height);

    cv::Mat points;
    cv::boxPoints(box, points);

    auto array = Mat2Vector(points);
    std::sort(array.begin(), array.end(), XsortFp32);

    std::vector<float> idx1 = array[0];
    std::vector<float> idx2 = array[1];
    std::vector<float> idx3 = array[2];
    std::vector<float> idx4 = array[3];
    if (array[3][1] <= array[2][1]) {
        idx2 = array[3];
        idx3 = array[2];
    }
    else {
        idx2 = array[2];
        idx3 = array[3];
    }
    if (array[1][1] <= array[0][1]) {
        idx1 = array[1];
        idx4 = array[0];
    }
    else {
        idx1 = array[0];
        idx4 = array[1];
    }

    array[0] = idx1;
    array[1] = idx2;
    array[2] = idx3;
    array[3] = idx4;

    return array;
}

float PolygonScoreAcc(std::vector<cv::Point> contour, cv::Mat pred)
{
    int                width  = pred.cols;
    int                height = pred.rows;
    std::vector<float> box_x;
    std::vector<float> box_y;
    for (int i = 0; i < contour.size(); ++i) {
        box_x.push_back(contour[i].x);
        box_y.push_back(contour[i].y);
    }

    int xmin =
        clamp(int(std::floor(*(std::min_element(box_x.begin(), box_x.end())))), 0, width - 1);
    int xmax =
        clamp(int(std::ceil(*(std::max_element(box_x.begin(), box_x.end())))), 0, width - 1);
    int ymin =
        clamp(int(std::floor(*(std::min_element(box_y.begin(), box_y.end())))), 0, height - 1);
    int ymax =
        clamp(int(std::ceil(*(std::max_element(box_y.begin(), box_y.end())))), 0, height - 1);

    cv::Mat mask;
    mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

    cv::Point* rook_point = new cv::Point[contour.size()];

    for (int i = 0; i < contour.size(); ++i) {
        rook_point[i] = cv::Point(int(box_x[i]) - xmin, int(box_y[i]) - ymin);
    }
    const cv::Point* ppt[1] = { rook_point };
    int              npt[]  = { int(contour.size()) };

    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

    cv::Mat croppedImg;
    pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
        .copyTo(croppedImg);
    float score = cv::mean(croppedImg, mask)[0];

    delete[] rook_point;
    return score;
}

std::vector<std::vector<std::vector<int>>>
BoxesFromBitmap(const cv::Mat pred, const cv::Mat bitmap, const float& box_thresh, const float& det_db_unclip_ratio, const bool& use_polygon_score)
{
    const int min_size       = 3;
    const int max_candidates = 1000;

    int width  = bitmap.cols;
    int height = bitmap.rows;

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i>              hierarchy;

    cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    int num_contours =
        contours.size() >= max_candidates ? max_candidates : contours.size();

    std::vector<std::vector<std::vector<int>>> boxes;

    for (int _i = 0; _i < num_contours; _i++) {
        if (contours[_i].size() <= 2) {
            continue;
        }
        float           ssid;
        cv::RotatedRect box   = cv::minAreaRect(contours[_i]);
        auto            array = GetMiniBoxes(box, ssid);

        auto box_for_unclip = array;
        // end get_mini_box

        if (ssid < min_size) {
            continue;
        }

        float score;
        if (use_polygon_score)
            /* compute using polygon*/
            score = PolygonScoreAcc(contours[_i], pred);
        else
            score = BoxScoreFast(array, pred);

        if (score < box_thresh)
            continue;

        // start for unclip
        cv::RotatedRect points = UnClip(box_for_unclip, det_db_unclip_ratio);
        if (points.size.height < 1.001 && points.size.width < 1.001) {
            continue;
        }
        // end for unclip

        cv::RotatedRect clipbox   = points;
        auto            cliparray = GetMiniBoxes(clipbox, ssid);

        if (ssid < min_size + 2)
            continue;

        int                           dest_width  = pred.cols;
        int                           dest_height = pred.rows;
        std::vector<std::vector<int>> intcliparray;

        for (int num_pt = 0; num_pt < 4; num_pt++) {
            std::vector<int> a{ int(clampf(roundf(cliparray[num_pt][0] / float(width) * float(dest_width)), 0, float(dest_width))),
                                int(clampf(roundf(cliparray[num_pt][1] / float(height) * float(dest_height)), 0, float(dest_height))) };
            intcliparray.push_back(a);
        }
        boxes.push_back(intcliparray);
    }  // end for
    return boxes;
}

bool XsortInt(std::vector<int> a, std::vector<int> b)
{
    if (a[0] != b[0])
        return a[0] < b[0];
    return false;
}

std::vector<std::vector<int>>
OrderPointsClockwise(std::vector<std::vector<int>> pts)
{
    std::vector<std::vector<int>> box = pts;
    std::sort(box.begin(), box.end(), XsortInt);

    std::vector<std::vector<int>> leftmost  = { box[0], box[1] };
    std::vector<std::vector<int>> rightmost = { box[2], box[3] };

    if (leftmost[0][1] > leftmost[1][1])
        std::swap(leftmost[0], leftmost[1]);

    if (rightmost[0][1] > rightmost[1][1])
        std::swap(rightmost[0], rightmost[1]);

    std::vector<std::vector<int>> rect = { leftmost[0], rightmost[0], rightmost[1],
                                           leftmost[1] };
    return rect;
}

std::vector<std::vector<std::vector<int>>>
FilterTagDetRes(std::vector<std::vector<std::vector<int>>> boxes, float ratio_h, float ratio_w, cv::Mat srcimg)
{
    int oriimg_h = srcimg.rows;
    int oriimg_w = srcimg.cols;

    std::vector<std::vector<std::vector<int>>> root_points;
    for (int n = 0; n < boxes.size(); n++) {
        boxes[n] = OrderPointsClockwise(boxes[n]);
        for (int m = 0; m < boxes[0].size(); m++) {
            boxes[n][m][0] /= ratio_w;
            boxes[n][m][1] /= ratio_h;

            boxes[n][m][0] = int(_min(_max(boxes[n][m][0], 0), oriimg_w - 1));
            boxes[n][m][1] = int(_min(_max(boxes[n][m][1], 0), oriimg_h - 1));
        }
    }

    for (int n = 0; n < boxes.size(); n++) {
        int rect_width;
        int rect_height;
        cv::Point left_top { boxes[n][0][0], boxes[n][0][1] }; 
        cv::Point right_top{ boxes[n][1][0], boxes[n][1][1] }; 
        cv::Point right_bot{ boxes[n][2][0], boxes[n][2][1] }; 
        cv::Point left_bot { boxes[n][3][0], boxes[n][3][1] }; 
        rect_width  = int(sqrt(pow(boxes[n][0][0] - boxes[n][1][0], 2) + pow(boxes[n][0][1] - boxes[n][1][1], 2)));
        rect_height = int(sqrt(pow(boxes[n][0][0] - boxes[n][3][0], 2) + pow(boxes[n][0][1] - boxes[n][3][1], 2)));
        if (rect_width <= 4 || rect_height <= 4)
            continue;
        root_points.push_back(boxes[n]);
    }
    return root_points;
}
cv::Mat GetRotateCropImage(const cv::Mat& srcimage, std::vector<std::vector<int>> box)
{
    std::vector<int> x_vec{ box[0][0], box[1][0], box[2][0], box[3][0] };
    std::vector<int> y_vec{ box[0][1], box[1][1], box[2][1], box[3][1] };
    int              x_min = *std::min_element(x_vec.begin(), x_vec.end());
    int              x_max = *std::max_element(x_vec.begin(), x_vec.end());

    int y_min = *std::min_element(y_vec.begin(), y_vec.end());
    int y_max = *std::max_element(y_vec.begin(), y_vec.end());
    if (x_max - x_min < 3 || y_max - y_min < 3)
        return cv::Mat();

    cv::Mat image;
    srcimage.copyTo(image);
    std::vector<std::vector<int>> points = box;

    int x_collect[4] = { box[0][0], box[1][0], box[2][0], box[3][0] };
    int y_collect[4] = { box[0][1], box[1][1], box[2][1], box[3][1] };
    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));

    cv::Mat img_crop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

    for (int i = 0; i < points.size(); i++) {
        points[i][0] -= left;
        points[i][1] -= top;
    }

    int img_crop_width = int(sqrt(pow(points[0][0] - points[1][0], 2) + pow(points[0][1] - points[1][1], 2)));
    int img_crop_height = int(sqrt(pow(points[0][0] - points[3][0], 2) + pow(points[0][1] - points[3][1], 2)));

    cv::Point2f pts_std[4];
    pts_std[0] = cv::Point2f(0., 0.);
    pts_std[1] = cv::Point2f(img_crop_width, 0.);
    pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
    pts_std[3] = cv::Point2f(0.f, img_crop_height);

    cv::Point2f pointsf[4];
    pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
    pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
    pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
    pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

    cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

    cv::Mat dst_img;
    cv::warpPerspective(img_crop, dst_img, M, cv::Size(img_crop_width, img_crop_height), cv::BORDER_REPLICATE);

    if (float(dst_img.rows) >= float(dst_img.cols) * 1.5) {
        cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
        cv::transpose(dst_img, srcCopy);
        cv::flip(srcCopy, srcCopy, 0);
        return srcCopy;
    }
    else {
        return dst_img;
    }
}

}  // namespace utility
};  // namespace OCR
