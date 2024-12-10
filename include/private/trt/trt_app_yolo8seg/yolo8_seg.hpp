#ifndef __TRT_YOLO8_SEG_H__
#define __TRT_YOLO8_SEG_H__

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include "../../airuntime/inference.h"

namespace yolo8_seg {
   
    using namespace std;
    using namespace Algo;
    std::shared_ptr<Algo::Infer> create_infer(const std::string& engine_file, int gpuid, float confidence_threshold = 0.25f, float nms_threshold = 0.5f, int max_objects = 1024, float seg_thr = 0.5, bool maxAreaCont = true, std::vector<std::vector<int>> dims={});

}; // namespace yolo8
#endif // !__TRT_YOLO8_H__
