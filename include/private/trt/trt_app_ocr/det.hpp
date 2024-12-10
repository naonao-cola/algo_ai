#ifndef __OCR_DEC_H__
#define __OCR_DEC_H__

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include "../../airuntime/inference.h"


namespace OCR {
    namespace det {
std::shared_ptr<Algo::Infer> create_infer(const std::string& engine_file, int gpuid, int kernel_size, bool use_dilation, bool enable_detmat, float det_db_box_thresh, float det_db_unclip_ratio, int max_side_len);
    }//namespace det
}; // namespace OCR

#endif // !__OCR_DEC_H__
