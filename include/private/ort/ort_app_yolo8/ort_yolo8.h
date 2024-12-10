/**
 * @FilePath     : /AIFramework/include/private/ort/ort_app_yolo8/ort_yolo8.h
 * @Description  :
 * @Author       : naonao 1319144981@qq.com
 * @Version      : 0.0.1
 * @LastEditors  : naonao 1319144981@qq.com
 * @LastEditTime : 2023-11-15 14:04:00
 * @Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2023.
**/

#ifndef __ORT_YOLO8_H__
#define __ORT_YOLO8_H__

#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "../../airuntime/inference.h"

namespace ort::yolo8 {

std::shared_ptr<Algo::Infer> create_infer(
    const std::string& onnx_file,
    float              confidence_threshold = 0.25f,
    float              nms_threshold        = 0.45f,
    int                max_objects          = 1024);
}  // namespace ort::yolo8

#endif  // !__ORT_YOLO_H__
