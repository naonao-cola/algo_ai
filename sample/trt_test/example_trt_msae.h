

#include "opencv2/opencv.hpp"
#include <vector>

#include "../../include/private/airuntime/logger.h"
#include "../../include/private/trt/trt_common/time_cost.h"
#include "../../include/private/trt/trt_models.h"
#include "./trt_test_utility.h"

int test_mase()
{
    std::string testDir   = R"(E:\project\ai_inference\Model\)";
    std::string modelPath = testDir + "msae_hgz_a.trtmodel";
    float       nms_thr   = 0.25f;

    auto infer = msae::create_infer(modelPath,0);
    

    if (infer == nullptr) {
        LOG_INFOE("Can not load model : {}", modelPath);
        return -1;
    }
    LOG_INFO("Load model successful! {}", modelPath);

    std::string             img_dir = R"(E:\project\ai_inference\test_img\*.jpg)";
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
    TRT::TimeCost time_cost;
    time_cost.start();
    auto        rstlists = infer->commits(mat_lists);
    int         cnt      = 0;
    std::string save_dir = "../../data/test_rst/";
    for (auto item : rstlists) {
      /*  auto boxes  = item.get();
        auto rstMat = draw_rst(mat_lists[cnt], boxes);
        char save_name[200];
        sprintf(save_name, "%s%d.jpg", save_dir.c_str(), cnt);
        cv::imwrite(save_name, rstMat);
        cnt++;
        LOG_INFO("pre_time: {}, infer_time: {}, host_time: {}, total: {}", boxes.pre_time, boxes.infer_time, boxes.host_time, boxes.total_time);*/
        auto boxes = item.get();
        cv::Mat tmp_ret = boxes[0].msae_img;
    }
    long long total_cost   = time_cost.get_cost_time();
    double    per_cost_img = total_cost / mat_lists.size();
    LOG_INFO("===================summary=====================");
    LOG_INFO("total cost: {}, cnt: {}, per_cost_img: {} ms", total_cost, mat_lists.size(), per_cost_img);
    LOG_INFO("resource cost: {}", infer->infer_info().dump());
}