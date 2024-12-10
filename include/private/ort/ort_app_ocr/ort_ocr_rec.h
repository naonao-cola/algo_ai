#ifndef __ORT_OCR_REC_H__
#define __ORT_OCR_REC_H__

#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "../../airuntime/inference.h"

namespace ort::ocr::rec {

std::shared_ptr<Algo::Infer> create_infer(
    const std::string& onnx_file,
    const std::string& label_file,
    int                gpuid,
    float              confidence_threshold = 0.25f);
}  // namespace ort::ocr::det

#endif  // !__ORT_OCR_REC_H__
