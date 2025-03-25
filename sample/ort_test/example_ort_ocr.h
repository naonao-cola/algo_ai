
#include <algorithm>
#include <corecrt.h>
#include <vector>

#include "../../include/private/airuntime/logger.h"
#include "../../include/private/ort/ort_models.h"


cv::Mat GetRotateCropImage(const cv::Mat& srcimage, std::vector<std::vector<int>> box)
{
    cv::Mat image;
    srcimage.copyTo(image);
    std::vector<std::vector<int>> points = box;

    int x_collect[4] = { box[0][0], box[1][0], box[2][0], box[3][0] };
    int y_collect[4] = { box[0][1], box[1][1], box[2][1], box[3][1] };
    int left         = int(*std::min_element(x_collect, x_collect + 4));
    int right        = int(*std::max_element(x_collect, x_collect + 4));
    int top          = int(*std::min_element(y_collect, y_collect + 4));
    int bottom       = int(*std::max_element(y_collect, y_collect + 4));

    cv::Mat img_crop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

    for (int i = 0; i < points.size(); i++) {
        points[i][0] -= left;
        points[i][1] -= top;
    }

    int img_crop_width  = int(sqrt(pow(points[0][0] - points[1][0], 2) + pow(points[0][1] - points[1][1], 2)));
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


inline std::vector<cv::Mat> test_ort_ocr_det()
{
    std::cout << "test_trt_ocr_det" << std::endl;
    std::string engine_patn =
        R"(D:\work\INSP\gtmc\vin\code\vin_algo\build\windows\x64\release\Model\)";
    std::string                  engine_name = R"(det_new.onnx)";
    std::shared_ptr<Algo::Infer> infer =
        ort::ocr::det::create_infer(engine_patn + engine_name, 0, 0, 0, 0, 0, 0, 0);

    if (infer == nullptr) {
        LOG_INFOE("can not load engine:{}", engine_patn + engine_name);
        return std::vector<cv::Mat>();
    }

    std::string          img_path = R"(D:\work\INSP\gtmc\vin\code\vin_algo\build\windows\x64\release\test_data\vincode\tayin.JPG)";
    std::vector<cv::Mat> img_vec;
    cv::Mat              img_src = cv::imread(img_path);
    img_vec.emplace_back(img_src);

    auto rstlists = infer->commits(img_vec);

    int                  img_count = 0;
    std::vector<cv::Mat> ret_img;
    for (auto item : rstlists) {
        auto ret = item.get();
        std::cout << "img index: " << img_count
                  << " return counts: " << ret.size() << std::endl;
        for (int i = 0; i < ret.size(); i++) {
            std::cout << "ret index: " << i << "" << std::endl;
            std::cout << "left top: " << ret[i].ocr_det[0][0] << " "
                      << ret[i].ocr_det[0][1] << std::endl;
            std::cout << "right top: " << ret[i].ocr_det[1][0] << " "
                      << ret[i].ocr_det[1][1] << std::endl;
            std::cout << "right bottom: " << ret[i].ocr_det[2][0] << " "
                      << ret[i].ocr_det[2][1] << std::endl;
            std::cout << "left bottom: " << ret[i].ocr_det[3][0] << " "
                      << ret[i].ocr_det[3][1] << std::endl;
            cv::Mat crop_img = GetRotateCropImage(img_src, ret[i].ocr_det);
            ret_img.emplace_back(crop_img);
        }
        img_count++;
    }
    return ret_img;
}

inline void test_ort_ocr_rec()
{
    std::cout << "test_trt_ocr_rec" << std::endl;
    std::string model_path =
        R"(D:\work\1.INSP\gtmc\vin_algo\Model\)";
    std::string model_name = R"(rec_s_1x3x48x640.onnx)";
    std::string label_file = R"(D:\work\1.INSP\gtmc\vin_algo\Model\ppocr_keys_v1.txt)";
    std::shared_ptr<Algo::Infer> infer =
       ort::ocr::rec::create_infer(model_path + model_name, label_file, 0, 0.5);

    if (infer == nullptr) {
        return;
    }
    cv::Mat              recimg   = cv::imread(R"(D:\work\1.INSP\gtmc\vin_data\ocr_train_data\engine_ng\20240913033603.jpg_EN72161__M20F_20240913113450899_1.jpg)");
    //std::vector<cv::Mat> img_list = test_ort_ocr_det();

    std::vector<cv::Mat> img_list;
    img_list.push_back(recimg);
    auto rstlists = infer->commits(img_list);
    for (auto item : rstlists) {
        auto ret = item.get();
        std::for_each(ret.begin(), ret.end(), [](const Algo::Box& box) {
            std::cout << "score: " << box.ocr_ret_score
                      << " string: " << box.ocr_ret_str << std::endl;
            for (auto ch_idx : box.ocr_char_index)
            {
                std::cout<<"char:"<<ch_idx<<std::endl;
            }
        });
    }

    return;
}

inline void test_ort_ocr()
{
    // test_ort_ocr_det();
    test_ort_ocr_rec();
    return;
}