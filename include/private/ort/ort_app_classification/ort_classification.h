#ifndef __ORT_CLASSIFICATION_H__
#define __ORT_CLASSIFICATION_H__

#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "../../airuntime/inference.h"

namespace ort::classification {


std::shared_ptr<Algo::Infer> create_infer(const std::string& onnx_file, float confidence_threshold = 0.25f,int gpuid =0);
}  // namespace ort::yolo

#endif  // !__ORT_CLASSIFICATION_H__
