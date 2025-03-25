#ifdef _WIN32
#include <corecrt.h>
#elif __linux__

#endif

#include <vector>
#include "../../include/private/airuntime/logger.h"
#include "../../include/private/trt/trt_models.h"


cv::Mat GetRotateCropImage(const cv::Mat& srcimage, std::vector<std::vector<int>> box)
{
    

    std::vector<int> x_vec{ box[0][0], box[1][0], box[2][0], box[3][0] };
    std::vector<int> y_vec{ box[0][1], box[1][1], box[2][1], box[3][1] };
    int x_min = *std::min_element(x_vec.begin(), x_vec.end());
    int x_max = *std::max_element(x_vec.begin(), x_vec.end());

    int y_min = *std::min_element(y_vec.begin(), y_vec.end());
    int y_max = *std::max_element(y_vec.begin(), y_vec.end());
    if (x_max - x_min < 3 || y_max - y_min < 3)
        return cv::Mat();
    
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

inline std::vector<cv::Mat> test_trt_ocr_det()
{
    std::string                  engine_patn = R"(E:\demo\rep\AIFramework\models\ort_models\ch_PP-OCRv4_det_infer\reshape\)";
    std::string                  engine_name = R"(det.trt.engine)";
    std::shared_ptr<Algo::Infer> infer =
        OCR::det::create_infer(engine_patn + engine_name, 0, 0, 0, 0, 0, 0, 0);
    if (infer == nullptr) {
        LOG_INFOE("can not load engine:{}", engine_patn + engine_name);
        return std::vector<cv::Mat>();
    }

    std::string          img_path = R"(C:\Users\Administrator\Desktop\333.jpg)";
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
            if (!crop_img.empty())
                ret_img.emplace_back(crop_img);
        }
        img_count++;
    }
    return ret_img;
}

inline void test_trt_ocr_rec()
{
    std::cout << "test_trt_ocr_rec" << std::endl;
    std::string model_path = R"(E:\project\EFEM\tv_algorithm\build\windows\x64\release\ort_models\ch_PP-OCRv4_rec_infer\reshape\)";
    std::string model_name = R"(rec.trt.engine)";

    std::shared_ptr<Algo::Infer> infer =
        OCR::rec::create_infer(model_path + model_name, 0, 0.25,R"(E:\project\EFEM\tv_algorithm\build\windows\x64\release\ort_models\ppocr_keys_v1.txt)");
    if (infer == nullptr) {
        return;
    }

    cv::Mat recimg = cv::imread("E:\\project\\EFEM\\tv_algorithm\\build\\windows\\x64\\release\\test_algo\\ocr.png");
    // std::vector<cv::Mat> img_list = test_trt_ocr_det();
    std::vector<cv::Mat> img_list = {recimg};

    if (img_list.size()>0){
        auto rstlists = infer->commits(img_list);
        for (auto item : rstlists) {
            auto ret = item.get();
            for (int i = 0; i < ret.size(); i++) {
                std::cout << "score: " << ret[i].ocr_ret_score
                          << " string: " << ret[i].ocr_ret_str << std::endl;
            }
        }
    }


    return;
}

//�������������������Ҫ������ת
inline std::vector<cv::Mat> test_trt_ocr_cls(std::vector<cv::Mat> img_vec)
{
    std::cout << "test_trt_ocr_cls" << std::endl;
    std::string model_path = R"(E:\demo\rep\AIFramework\models\ort_models\ch_ppocr_mobile_v2.0_cls_infer\reshape_onnx\)";
    std::string model_name = R"(cls.trt.engine)";
    float                        confidence_threshold = 0.9;
    std::shared_ptr<Algo::Infer> infer =OCR::cls::create_infer(model_path + model_name, confidence_threshold);
    if (infer == nullptr) {
        LOG_INFOE("can not load engine:{}", model_path + model_name);
        return std::vector<cv::Mat>();
    }
    auto                 rstlists = infer->commits(img_vec);
    std::vector<cv::Mat> ret_img;
    int                  count = 0;
    for (auto item : rstlists) {
        auto ret = item.get();
        for (int i = 0; i < ret.size(); i++) {
            std::cout << "label: " << ret[i].class_label
                      << " score: " << ret[i].confidence << std::endl;
            if (ret[i].class_label % 2 == 1 && ret[i].confidence >= confidence_threshold) {
                cv::rotate(img_vec[count], img_vec[count], 1);
            }

        }
        ret_img.emplace_back(img_vec[count]);
        count++;
    }
    return ret_img;
}
inline void test_trt_ocr()
{
    // build_model();
    //test_trt_ocr_det();
    test_trt_ocr_rec();
    /*std::vector<cv::Mat> img_vec;
    img_vec.emplace_back(cv::imread(R"(E:\demo\rep\AIFramework\data\test_img\h\xuexiao.jpg)"));
    test_trt_ocr_cls(img_vec);*/

    return;
}
