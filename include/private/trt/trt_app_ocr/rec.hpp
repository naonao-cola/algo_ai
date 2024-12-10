#ifndef __OCR_REC_H__
#define __OCR_REC_H__

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include "../../airuntime/inference.h"

namespace OCR {
    namespace rec {
        using namespace std;
        using namespace Algo;
        std::shared_ptr<Algo::Infer> create_infer(const std::string& engine_file, int gpuid, float confidence_threshold = 0.25f, std::string label_file = "");
    
    }//namespace rec
}; // namespace OCR
#endif // !__OCR_REC_H__
