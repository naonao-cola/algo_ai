#ifndef __TRT_ANOMALIB_H__
#define __TRT_ANOMALIB_H__

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include "../../airuntime/inference.h"

namespace anomalib {
   
    using namespace std;
    using namespace Algo;
    std::shared_ptr<Algo::Infer> create_infer(const std::string& engine_file, int gpuid, float confidence_threshold = 0.25f, float nms_threshold = 0.5f, int max_objects = 1024);

}; // namespace msae
#endif // !__TRT_ANOMALIB_H__
