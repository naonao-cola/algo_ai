#ifndef __ORT_OCR_DET_H__
#define __ORT_OCR_DET_H__

#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "../../airuntime/inference.h"

namespace ort::ocr::det {

std::shared_ptr<Algo::Infer> create_infer(
    const std::string& onnx_file,
    int                gpuid,
    int kernel_size, 
    bool use_dilation, 
    bool enable_detmat,
    float det_db_box_thresh,
    float det_db_unclip_ratio,
    int max_side_len
    );
}  // namespace ort::ocr::det

#endif  // !__ORT_OCR_DET_H__
