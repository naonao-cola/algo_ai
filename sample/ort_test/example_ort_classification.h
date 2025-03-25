#include "opencv2/opencv.hpp"
#include <vector>

#include "../../include/private/airuntime/logger.h"
#include "../../include/private/ort/ort_models.h"
#include "../../include/public/AIRuntimeUtils.h"
#include "./ort_test_utility.h"

inline int test_ort_classification()
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
    std::string testDir   = R"(D:\work\1.INSP\gtmc\char_cls_train\)";
    std::string modelPath = testDir + "model_1x3x224x224.onnx";

    auto infer = ort::classification::create_infer(modelPath);
    if (infer == nullptr) {
        LOG_INFOE("Can not load model : {}", modelPath);
        return -1;
    }
    LOG_INFO("Load model successful! {}", modelPath);

    json config = {
        { "confidence_threshold", 0.8 }
    };
    infer->set_param(config);

    std::string             img_dir = R"(D:\work\1.INSP\gtmc\vin_data\test_rec\ng_all\ng_char_test_img\test\*.jpg)";
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
    std::string save_dir = R"(D:\work\1.INSP\gtmc\char_cls_train\\test_result\)";

    for (auto item : rstlists) {
        auto boxes  = item.get();
        for (int i = 0; i < boxes.size();i++) {
            std::cout << "label: " << boxes[i].class_label << " confidence: "<<boxes[i].confidence <<"  file:" << img_paths[cnt] << std::endl;
        }
        cnt++;
        //LOG_INFO("pre_time: {}, infer_time: {}, host_time: {}, total: {}", boxes.pre_time, boxes.infer_time, boxes.host_time, boxes.total_time);
    }
    long long total_cost   = time_cost.get_cost_time();
    double    per_cost_img = total_cost / mat_lists.size();

    /* LOG_INFO("===================summary=====================");
     LOG_INFO("total cost: {}, cnt: {}, per_cost_img: {} ms", total_cost, mat_lists.size(), per_cost_img);
     LOG_INFO("resource cost: {}", infer->infer_info().dump());*/
    return 0;
}