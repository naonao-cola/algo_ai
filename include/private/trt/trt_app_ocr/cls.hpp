#ifndef __OCR_CLS_H__
#define __OCR_CLS_H__

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include "../../airuntime/inference.h"

namespace OCR {
    namespace cls {
        std::shared_ptr<Algo::Infer> create_infer(const std::string& engine_file,int gpuid,float confidence_threshold = 0.25f);

    }//namespace cls
}; // namespace OCR
#endif // !__OCR_CLS_H__
