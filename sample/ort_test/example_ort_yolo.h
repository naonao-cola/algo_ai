

#include "opencv2/opencv.hpp"
#include <vector>

#include "../../include/private/airuntime/logger.h"
#include "../../include/private/ort/ort_models.h"
#include "../../include/public/AIRuntimeUtils.h"
#include "./ort_test_utility.h"

inline int test_ort_yolo()
{
    // 模型文件。输入一个， 1*3*640*640 。
    /*
    * 模型不同可能需要修改
    outputs number: 4
    pred: 1*25200*85
    output2: 1*3*80*80*85
    output3: 1*3*40*40*85
    output4: 1*3*20*20*85
    */
    std::string testDir   = R"(E:\AIFramework\models\ort_models\)";
    std::string modelPath = testDir + "yolov5s.onnx";
    float       nms_thr   = 0.25f;

    auto infer = ort::yolo::create_infer(modelPath, ort::yolo::Type::V5);
    if (infer == nullptr) {
        LOG_INFOE("Can not load model : {}", modelPath);
        return -1;
    }
    LOG_INFO("Load model successful! {}", modelPath);

    json config = {
        { "confidence_threshold", 0.25 },
        { "nms_threshold", 0.45 },
        { "max_objects", 1024 },
        { "max_batch_size", 1 },
    };
    infer->set_param(config);

    std::string             img_dir = R"(E:\AIFramework\data\test_img\coco128\train2017\*.jpg)";
    std::vector<cv::String> img_paths;
    std::vector<cv::Mat>    mat_lists;
    cv::glob(img_dir, img_paths);
    if (img_paths.empty()) {
        LOG_INFO("Not found any image from : {}", img_dir);
        return -1;
    }
    for (auto path : img_paths) {
        mat_lists.emplace_back(cv::imread(path));
    }
    TimeCost time_cost;
    time_cost.start();
    auto        rstlists = infer->commits(mat_lists);
    int         cnt      = 0;
    std::string save_dir = R"(E:\AIFramework\data\test_rsh\)";
    for (auto item : rstlists) {
        auto boxes  = item.get();
        auto rstMat = draw_rst(mat_lists[cnt], boxes);
        char save_name[200];
        sprintf(save_name, "%s%d.jpg", save_dir.c_str(), cnt);
        cv::imwrite(save_name, rstMat);
        cnt++;
        LOG_INFO("pre_time: {}, infer_time: {}, host_time: {}, total: {}", boxes.pre_time, boxes.infer_time, boxes.host_time, boxes.total_time);
    }
    long long total_cost   = time_cost.get_cost_time();
    double    per_cost_img = total_cost / mat_lists.size();

    /* LOG_INFO("===================summary=====================");
     LOG_INFO("total cost: {}, cnt: {}, per_cost_img: {} ms", total_cost, mat_lists.size(), per_cost_img);
     LOG_INFO("resource cost: {}", infer->infer_info().dump());*/
    return 0;
}

inline int test_ort_yolo8()
{
    // 模型文件。输入一个， 1*3*640*640 。
    /*
    * 模型不同可能需要修改
    outputs number: 4
    pred: 1*25200*85
    output2: 1*3*80*80*85
    output3: 1*3*40*40*85
    output4: 1*3*20*20*85
    */
    std::string testDir   = R"(E:\demo\rep\AIFramework\models\ort_models\)";
    std::string modelPath = testDir + "tolov8_collet.onnx";
    float       nms_thr   = 0.25f;

    auto infer = ort::yolo8::create_infer(modelPath);
    if (infer == nullptr) {
        LOG_INFOE("Can not load model : {}", modelPath);
        return -1;
    }
    LOG_INFO("Load model successful! {}", modelPath);

    json config = {
        { "confidence_threshold", 0.45 },
        { "nms_threshold", 0.6 },
        { "max_objects", 200000 },
        { "max_batch_size", 1 },
    };
    infer->set_param(config);

    std::string             img_path = R"(E:\demo\rep\AIFramework\data\test_img\20231115-141947.jpg)";
    std::vector<cv::Mat>    mat_lists;
    mat_lists.emplace_back(cv::imread(img_path));

   
    TimeCost time_cost;
    time_cost.start();
    auto        rstlists = infer->commits(mat_lists);
    int         cnt      = 0;
    std::string save_dir = R"(E:\AIFramework\data\test_rsh\)";
    for (auto item : rstlists) {
        auto boxes  = item.get();
        auto rstMat = draw_rst(mat_lists[cnt], boxes);
        char save_name[200];
        sprintf(save_name, "%s%d.jpg", save_dir.c_str(), cnt);
        cv::imwrite(save_name, rstMat);
        cnt++;
        LOG_INFO("pre_time: {}, infer_time: {}, host_time: {}, total: {}", boxes.pre_time, boxes.infer_time, boxes.host_time, boxes.total_time);
    }
    long long total_cost   = time_cost.get_cost_time();
    double    per_cost_img = total_cost / mat_lists.size();

    /* LOG_INFO("===================summary=====================");
     LOG_INFO("total cost: {}, cnt: {}, per_cost_img: {} ms", total_cost, mat_lists.size(), per_cost_img);
     LOG_INFO("resource cost: {}", infer->infer_info().dump());*/
    return 0;
}