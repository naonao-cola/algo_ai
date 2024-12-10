#ifndef __ORT_YOLO_H__
#define __ORT_YOLO_H__

#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "../../airuntime/inference.h"

namespace ort::yolo {
enum class Type : int
{
    V5 = 0,
    X  = 1,
    V3 = V5
};

std::shared_ptr<Algo::Infer> create_infer(
    const std::string& onnx_file,
    Type               type,
    float              confidence_threshold = 0.25f,
    float              nms_threshold        = 0.45f,
    int                max_objects          = 1024);
}  // namespace ort::yolo

#endif  // !__ORT_YOLO_H__
